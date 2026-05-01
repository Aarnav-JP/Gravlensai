"""Run full GravLensAI pipeline in one command."""

import argparse
import subprocess
import sys

from gravlensai.utils.config import load_yaml_section
from gravlensai.utils.reproducibility import set_global_seed


def run_step(cmd: list[str], title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")
    subprocess.run(cmd, check=True)


def main(args):
    cfg = load_yaml_section(args.config, 'pipeline')

    seed = args.seed if args.seed is not None else int(cfg.get('seed', 42))
    set_global_seed(seed)

    generate_data = args.generate_data if args.generate_data is not None else bool(cfg.get('generate_data', True))
    train_classifier = args.train_classifier if args.train_classifier is not None else bool(cfg.get('train_classifier', True))
    train_regressor = args.train_regressor if args.train_regressor is not None else bool(cfg.get('train_regressor', True))
    evaluate = args.evaluate if args.evaluate is not None else bool(cfg.get('evaluate', True))

    simulation_config = cfg.get('simulation_config', 'configs/simulation.yaml')
    classifier_config = cfg.get('classifier_config', 'configs/classifier.yaml')
    regressor_config = cfg.get('regressor_config', 'configs/regressor.yaml')
    evaluate_config = cfg.get('evaluate_config', 'configs/evaluate.yaml')

    data_dir = args.data_dir or cfg.get('data_dir', 'data/simulated/')
    models_dir = args.models_dir or cfg.get('models_dir', 'results/models/')
    output_dir = args.output_dir or cfg.get('output_dir', 'results/')

    py = sys.executable

    if generate_data:
        run_step(
            [py, 'scripts/01_generate_simulations.py', '--config', simulation_config, '--output', data_dir, '--seed', str(seed)],
            'STEP 1/4: Generate Simulated Data',
        )

    if train_classifier:
        run_step(
            [py, 'scripts/03_train_classifier.py', '--config', classifier_config, '--data_dir', data_dir, '--output', models_dir, '--seed', str(seed)],
            'STEP 2/4: Train Classifier',
        )

    if train_regressor:
        run_step(
            [py, 'scripts/04_train_regressor.py', '--config', regressor_config, '--data_dir', data_dir, '--output', models_dir, '--seed', str(seed)],
            'STEP 3/4: Train Regressor',
        )

    if evaluate:
        run_step(
            [py, 'scripts/05_evaluate.py', '--config', evaluate_config, '--data_dir', data_dir, '--models_dir', models_dir, '--output', output_dir, '--seed', str(seed)],
            'STEP 4/4: Evaluate Models',
        )

    print("\nPipeline complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full GravLensAI pipeline')
    parser.add_argument('--config', default='configs/pipeline.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--models_dir', default=None)
    parser.add_argument('--output_dir', default=None)

    parser.add_argument('--generate_data', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--train_classifier', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--train_regressor', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction, default=None)

    main(parser.parse_args())
