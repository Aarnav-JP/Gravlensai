"""
Hyperparameter Optimization (HPO) for Domain Adaptation using Optuna.

Finds the optimal balance between task performance (simulated data)
and domain adaptation confidence (real HST data).
"""

import argparse
import json
import os
from pathlib import Path

import optuna
import optuna.visualization as vis
import torch
from torch.utils.data import DataLoader

from gravlensai.data.dataset import HSTDataset, SimulatedLensDataset
from gravlensai.evaluate.metrics import classifier_metrics
from gravlensai.models.classifier import LensClassifier
from gravlensai.utils.reproducibility import set_global_seed
import importlib.util

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

domain_adapt = _load_module("domain_adapt", "scripts/06_domain_adapt.py")
run_dann_adaptation = domain_adapt.run_dann_adaptation
run_mmd_adaptation = domain_adapt.run_mmd_adaptation

research_suite = _load_module("research_suite", "scripts/09_research_suite.py")
_collect_classifier_outputs = research_suite._collect_classifier_outputs
_collect_hst_probs = research_suite._collect_hst_probs
_summarise_hst_predictions = research_suite._summarise_hst_predictions


def create_objective(
    base_state_dict: dict,
    sim_train_loader: DataLoader,
    sim_val_loader: DataLoader,
    hst_loader: DataLoader,
    device: torch.device,
    method: str,
    epochs: int,
):
    def objective(trial: optuna.Trial) -> float:
        # 1. Suggest Hyperparameters
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        pos_weight = trial.suggest_float("pos_weight", 1.0, 5.0)
        
        # 2. Reset model to baseline
        model = LensClassifier()
        model.load_state_dict(base_state_dict)
        model.to(device)
        
        # 3. Run Adaptation
        if method == "dann":
            lambda_dann = trial.suggest_float("lambda_dann", 1e-3, 10.0, log=True)
            run_dann_adaptation(
                model,
                sim_train_loader,
                hst_loader,
                device,
                epochs=epochs,
                lr=lr,
                lambda_dann=lambda_dann,
                pos_weight=pos_weight,
            )
        else:
            lambda_mmd = trial.suggest_float("lambda_mmd", 1e-3, 10.0, log=True)
            run_mmd_adaptation(
                model,
                sim_train_loader,
                hst_loader,
                device,
                epochs=epochs,
                lr=lr,
                lambda_mmd=lambda_mmd,
                pos_weight=pos_weight,
            )
            
        # 4. Evaluate Unsupervised Fitness
        model.eval()
        
        # Evaluate on sim val (ensures no catastrophic forgetting)
        y_true, y_prob = _collect_classifier_outputs(model, sim_val_loader, device)
        sim_metrics = classifier_metrics(y_true, y_prob)
        auc_sim = sim_metrics.get("auc_roc", 0.0)
        
        # Evaluate entropy on real HST data (measures confidence on target domain)
        hst_probs = _collect_hst_probs(model, hst_loader, device)
        hst_summary = _summarise_hst_predictions(hst_probs)
        entropy_hst = hst_summary.get("mean_entropy", 1.0)
        
        # Fitness: Maximize AUC while penalizing high entropy (confusion) on HST
        # We weigh entropy heavily since domain adaptation should make the model confident
        fitness = float(auc_sim - (0.15 * entropy_hst))
        
        # Log metrics to trial for inspection later
        trial.set_user_attr("auc_sim", auc_sim)
        trial.set_user_attr("entropy_hst", entropy_hst)
        
        return fitness

    return objective


def main():
    parser = argparse.ArgumentParser("Optuna HPO for Domain Adaptation")
    parser.add_argument("--method", choices=["dann", "mmd"], default="dann")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/hpo")
    args = parser.parse_args()

    set_global_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load Baseline Checkpoint ───────────────────────────────
    ckpt_path = "results/models/classifier_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing classifier checkpoint at {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_state = ckpt["model_state"]
    
    # ── Prepare Data ───────────────────────────────────────────
    batch_size = 32
    sim_train = SimulatedLensDataset("data/simulated/", split="train", task="classify", augment=True)
    sim_val = SimulatedLensDataset("data/simulated/", split="val", task="classify", augment=False)
    hst_data = HSTDataset("data/raw/hst/Abell_2744/")
    
    sim_train_loader = DataLoader(sim_train, batch_size=batch_size, shuffle=True)
    sim_val_loader = DataLoader(sim_val, batch_size=256, shuffle=False)
    hst_loader = DataLoader(hst_data, batch_size=batch_size, shuffle=True)
    
    print(f"Running Optuna {args.method.upper()} HPO ({args.n_trials} trials, {args.epochs} epochs/trial)")
    
    # ── Run Optuna ─────────────────────────────────────────────
    # We want to MAXIMIZE the fitness score
    study = optuna.create_study(
        study_name=f"domain_adapt_{args.method}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    objective = create_objective(
        base_state, sim_train_loader, sim_val_loader, hst_loader, device, args.method, args.epochs
    )
    
    study.optimize(objective, n_trials=args.n_trials)
    
    # ── Save Results ───────────────────────────────────────────
    print("\n" + "="*50)
    print("HPO Complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best fitness: {study.best_value:.4f}")
    print(f"Best AUC: {study.best_trial.user_attrs['auc_sim']:.4f}")
    print(f"Best Entropy: {study.best_trial.user_attrs['entropy_hst']:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*50)
    
    # Save params
    with open(out_dir / f"best_params_{args.method}.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
        
    # Generate Visualizations
    fig_hist = vis.plot_optimization_history(study)
    fig_hist.write_image(str(out_dir / f"optuna_history_{args.method}.png"))
    
    fig_param = vis.plot_param_importances(study)
    fig_param.write_image(str(out_dir / f"optuna_importances_{args.method}.png"))
    

if __name__ == "__main__":
    main()
