"""
GravLensAI — Interactive Gradio Demo
=====================================
One-click web interface for gravitational lens detection and parameter
estimation with uncertainty quantification and Grad-CAM interpretability.

Launch:
    python app.py

Deploy to Hugging Face Spaces:
    gradio deploy
"""

import os
import argparse
import torch
import numpy as np
import cv2
import gradio as gr

from gravlensai.models.classifier import LensClassifier
from gravlensai.models.regressor import LensParameterRegressor
from gravlensai.simulate.lens_generator import LensImageGenerator
from gravlensai.evaluate.grad_cam import GradCAM, get_target_layer
from gravlensai.utils.config import load_yaml_section
from gravlensai.evaluate.ood import OODDetector
from gravlensai.models.ensemble import EnsembleClassifier
from gravlensai.models.baselines import LightweightLensCNN
import joblib

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available()
                       else 'cpu')

classifier = LensClassifier()
regressor = LensParameterRegressor()
# BUMPED OOD entropy_threshold to 0.98 for ViT calibration
ood_detector = OODDetector(entropy_threshold=0.98, snr_threshold=1.0)
ensemble = None
lightweight_cnn = None


def resolve_model_paths(args) -> tuple[str, str]:
    cfg = load_yaml_section(args.config, 'app')

    clf_path = (
        args.classifier_path
        or os.getenv('GRAVLENS_CLASSIFIER_PATH')
        or cfg.get('classifier_path')
        or 'results/models/classifier_best.pt'
    )
    reg_path = (
        args.regressor_path
        or os.getenv('GRAVLENS_REGRESSOR_PATH')
        or cfg.get('regressor_path')
        or 'results/models/regressor_best.pt'
    )
    return str(clf_path), str(reg_path)


def load_models(classifier_path: str, regressor_path: str) -> None:
    global ensemble, lightweight_cnn
    
    # Load checkpoints with weights_only=False since these are our own trusted models
    # saved during training (not from untrusted sources)
    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=DEVICE, weights_only=False)
        # Use strict=False to handle architecture changes (e.g., new feature_projection layer)
        classifier.load_state_dict(ckpt['model_state'], strict=False)
        print(f"\u2713 Classifier loaded from {classifier_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"! Classifier checkpoint not found: {classifier_path}")

    if os.path.exists(regressor_path):
        ckpt = torch.load(regressor_path, map_location=DEVICE, weights_only=False)
        regressor.load_state_dict(ckpt['model_state'], strict=False)
        print(f"\u2713 Regressor loaded from {regressor_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"! Regressor checkpoint not found: {regressor_path}")

    classifier.to(DEVICE).eval()
    regressor.to(DEVICE).eval()
    
    # Try loading Ensemble components
    try:
        if os.path.exists("results/models/ensemble_xgb.json"):
            lightweight_cnn = LightweightLensCNN()
            lightweight_cnn.load_state_dict(torch.load("results/models/lightweight_cnn.pt", map_location=DEVICE, weights_only=False))
            rf = joblib.load("results/models/rf_baseline.joblib")
            lr = joblib.load("results/models/lr_baseline.joblib")
            
            ensemble = EnsembleClassifier(classifier, lightweight_cnn, rf, lr, device=DEVICE)
            ensemble.load("results/models/ensemble_xgb.json")
            print("✓ XGBoost Meta-Ensemble loaded")
    except Exception as e:
        print(f"! Failed to load ensemble (skipping fallback): {e}")


# ── Helper functions ──────────────────────────────────────────────────────

def preprocess_image(image_np):
    """Normalise and convert to model input tensor using global training stats."""
    from gravlensai.data.normalisation import arcsinh_normalise
    if image_np.ndim == 3:
        image_np = image_np.mean(axis=2)  # RGB → grayscale
        
    # Updated stats computed from the new lenstronomy-only Kaggle dataset
    global_stats = {'softening': 620.9315, 'mean': 1.1255, 'std': 0.9555}
    img = arcsinh_normalise(image_np.astype(np.float32), stats=global_stats)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    return tensor.to(DEVICE)


def generate_grad_cam(model, tensor):
    """Generate Grad-CAM heatmap overlay."""
    target_layer = get_target_layer(model)
    cam = GradCAM(model, target_layer)
    heatmap = cam(tensor)
    return heatmap

def create_overlay_numpy(image_np, heatmap_np):
    """Create raw image and overlay using native OpenCV for Gradio UI."""
    from gravlensai.data.normalisation import arcsinh_stretch
    disp_img = arcsinh_stretch(image_np)
    vmin, vmax = np.percentile(disp_img, [1, 99])
    if vmax == vmin:
        vmax += 1e-5

    disp_img = np.clip((disp_img - vmin) / (vmax - vmin), 0, 1)
    disp_img_8u = (disp_img * 255).astype(np.uint8)
    
    # Base input image returned as grayscale RGB
    input_rgb = cv2.cvtColor(disp_img_8u, cv2.COLOR_GRAY2RGB)
    
    # Heatmap is 0-1 float, convert to 0-255
    heatmap_8u = (heatmap_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # Convert to RGB for Gradio
    
    # Overlay
    overlay = cv2.addWeighted(input_rgb, 0.5, heatmap_color, 0.5, 0)
    return input_rgb, overlay


def build_results_html(prob, clf_mean, clf_std, entropy, is_ood, ood_reason, params_mean, params_std, ensemble_prob=None, threshold=0.5):
    """Generate beautiful HTML cards for the results."""
    
    final_prob = ensemble_prob if ensemble_prob is not None else prob
    is_lens = final_prob >= threshold
    
    # Colors
    lens_color = "#00ff41" if is_lens else "#ff4444"
    lens_text = "LENS DETECTED" if is_lens else "NON-LENS"
    
    html = f"""
    <div style='background: rgba(26, 26, 46, 0.7); backdrop-filter: blur(10px); padding: 20px; border-radius: 16px; border: 1px solid #30305a;'>
        <h2 style='color: {lens_color}; font-weight: 800; font-size: 24px; text-transform: uppercase; margin-top: 0;'>{lens_text}</h2>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;'>
            <div style='background: rgba(13, 17, 23, 0.6); padding: 15px; border-radius: 12px; border: 1px solid #2d3748;'>
                <p style='color: #a0aec0; margin: 0; font-size: 12px; text-transform: uppercase;'>ViT Confidence</p>
                <p style='color: white; font-size: 18px; margin: 5px 0 0 0; font-family: monospace;'>{(clf_mean*100):.1f}% ± {(clf_std*100):.1f}%</p>
            </div>
            <div style='background: rgba(13, 17, 23, 0.6); padding: 15px; border-radius: 12px; border: 1px solid #2d3748;'>
                <p style='color: #a0aec0; margin: 0; font-size: 12px; text-transform: uppercase;'>Entropy (Uncertainty)</p>
                <p style='color: white; font-size: 18px; margin: 5px 0 0 0; font-family: monospace;'>{entropy:.3f} bits</p>
            </div>
        </div>
    """
    
    if is_ood:
        html += f"""
        <div style='background: rgba(255, 68, 68, 0.15); padding: 15px; border-radius: 12px; border: 1px solid rgba(255, 68, 68, 0.4); margin-top: 15px;'>
            <p style='color: #ff4444; font-weight: bold; margin: 0;'>⚠️ OUT-OF-DISTRIBUTION WARNING</p>
            <p style='color: #ffbaba; margin: 5px 0 0 0; font-size: 13px;'>{ood_reason}</p>
        </div>
        """
        
    if is_lens and not is_ood:
        param_names = ['Einstein Radius (θ_E)', 'Ellipticity (e1)', 'Ellipticity (e2)', 'Shear (γ1)', 'Shear (γ2)', 'Dark Matter Mass']
        param_units = ['arcsec', '', '', '', '', 'log(M_☉)']
        
        html += "<h3 style='color: #cbd5e1; margin-top: 25px; border-bottom: 1px solid #30305a; padding-bottom: 8px;'>6-Dimensional Physics Regressor</h3>"
        html += "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;'>"
        
        for name, mean, std, unit in zip(param_names, params_mean, params_std, param_units):
            # Highlight Dark matter mass
            color = "#a78bfa" if "Dark Matter" in name else "white"
            html += f"""
            <div style='background: rgba(13, 17, 23, 0.4); padding: 10px 15px; border-radius: 8px; border-left: 3px solid #667eea;'>
                <p style='color: #94a3b8; margin: 0; font-size: 11px; text-transform: uppercase;'>{name}</p>
                <p style='color: {color}; font-size: 15px; font-family: monospace; margin: 3px 0 0 0;'>{mean:+.3f} ± {std:.3f} {unit}</p>
            </div>
            """
        html += "</div>"
        
    html += "</div>"
    return html


# ── Gradio inference functions ────────────────────────────────────────────

def analyze_image(image_np, n_mc_samples, threshold):
    """Full analysis: classification + parameters + uncertainty + Grad-CAM."""
    if image_np is None:
        return None, None, "<p>Please upload an image or generate a simulation.</p>"

    tensor = preprocess_image(image_np)

    # Classification with uncertainty
    with torch.no_grad():
        prob = classifier.predict_proba(tensor).item()

    unc = classifier.predict_with_uncertainty(tensor, n_forward=int(n_mc_samples))
    clf_mean = unc['mean'].item()
    clf_std = unc['std'].item()
    entropy = unc['entropy'].item()
    
    # Check for OOD
    if image_np.ndim == 3:
        display_img = image_np.mean(axis=2)
    else:
        display_img = image_np
        
    is_ood, ood_reason = ood_detector.detect(display_img, entropy)
    
    # Optional Ensemble
    ensemble_prob = None
    if ensemble is not None:
        try:
            ensemble_prob = ensemble.predict_proba(tensor)[0]
        except Exception:
            pass

    # Parameter estimation with uncertainty
    reg_unc = regressor.predict_with_uncertainty(tensor, n_forward=int(n_mc_samples))
    params_mean = reg_unc['mean'].cpu().detach().numpy()[0]
    params_std = reg_unc['std'].cpu().detach().numpy()[0]

    # Grad-CAM (Fallback to Lightweight CNN since ViT lacks deep Conv2d layers)
    try:
        heatmap = generate_grad_cam(classifier, tensor)
    except Exception:
        heatmap = generate_grad_cam(lightweight_cnn, tensor) if lightweight_cnn else np.zeros((64, 64))

    # Generate Image Visuals
    input_rgb, gradcam_overlay = create_overlay_numpy(display_img, heatmap)
    
    # Scale up the 64x64 images to 256x256 for the UI so they don't look tiny
    input_rgb = cv2.resize(input_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
    gradcam_overlay = cv2.resize(gradcam_overlay, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Generate HTML Stats
    stats_html = build_results_html(prob, clf_mean, clf_std, entropy, is_ood, ood_reason, params_mean, params_std, ensemble_prob, threshold)

    return input_rgb, gradcam_overlay, stats_html


def generate_simulation(sim_type, seed):
    """Generate a simulated image and analyse it."""
    gen = LensImageGenerator(seed=int(seed))

    if sim_type == "Lensed Galaxy":
        img, params = gen.generate_lens()
        gt_html = f"""
        <div style='background: rgba(13, 17, 23, 0.6); padding: 15px; border-radius: 8px; font-family: monospace; font-size: 13px; color: #cbd5e1; border: 1px solid #30305a;'>
            <p style='color: #a78bfa; font-weight: bold; margin-top: 0;'>Ground Truth Parameters:</p>
            θ_E: {params.einstein_radius:.4f}<br>
            e1:  {params.ellipticity_e1:.4f}<br>
            e2:  {params.ellipticity_e2:.4f}<br>
            γ1:  {params.shear_g1:.4f}<br>
            γ2:  {params.shear_g2:.4f}<br>
            <span style='color: white;'>M_sub: {params.subhalo_mass:.4f} log(M_☉)</span>
        </div>
        """
    else:
        img, params = gen.generate_nonlens()
        gt_html = """
        <div style='background: rgba(13, 17, 23, 0.6); padding: 15px; border-radius: 8px; font-family: monospace; font-size: 13px; color: #cbd5e1; border: 1px solid #30305a;'>
            <p style='color: #a78bfa; font-weight: bold; margin: 0;'>Ground Truth:</p>
            Non-lensed Empty Galaxy Field
        </div>
        """

    return img, gt_html


# ── Gradio UI ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
body {
    background-color: #05070e !important;
}
.gradio-container {
    background-color: #05070e !important;
    background-image: radial-gradient(circle at top right, #1a1a3a, transparent 400px), 
                      radial-gradient(circle at bottom left, #0d1b2a, transparent 400px) !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
button.primary, .primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.2s ease !important;
    color: white !important;
}
button.primary:hover, .primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
}
.gr-panel {
    background: rgba(13, 17, 23, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
}
.gr-box {
    background-color: transparent !important;
}
h1, h2, h3, p, span {
    color: #f1f5f9 !important;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="GravLensAI — Dark Matter Detector",
) as demo:

    gr.HTML("""
    <div style='text-align: center; margin-bottom: 30px; margin-top: 20px;'>
        <h1 style='font-size: 42px; font-weight: 900; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            GravLensAI Dark Matter Detector
        </h1>
        <p style='font-size: 16px; color: #94a3b8; max-width: 600px; margin: 0 auto;'>
            Vision Transformer & 6D Regressor pipeline for Einstein Ring detection and subhalo mass quantification.
        </p>
    </div>
    """)

    with gr.Tabs():
        # ── Tab 1: Image Analysis ─────────────────────────────────────
        with gr.TabItem("🔍 Analyze Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload Image (64×64 grayscale)",
                        type="numpy",
                        elem_classes="gr-panel"
                    )
                    n_mc = gr.Slider(
                        5, 50, value=30, step=5,
                        label="MC Dropout Samples (Uncertainty)",
                    )
                    threshold_slider = gr.Slider(
                        0.1, 0.99, value=0.75, step=0.01,
                        label="Classification Threshold (P > threshold = LENS)",
                    )
                    analyze_btn = gr.Button(
                        "🔬 Analyze with ViT", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    with gr.Row():
                        out_img1 = gr.Image(label="Input Tensor", interactive=False, elem_classes="gr-panel")
                        out_img2 = gr.Image(label="Grad-CAM Attention Map", interactive=False, elem_classes="gr-panel")
                    output_html = gr.HTML()

            analyze_btn.click(
                fn=analyze_image,
                inputs=[image_input, n_mc, threshold_slider],
                outputs=[out_img1, out_img2, output_html],
            )

        # ── Tab 2: Simulation ─────────────────────────────────────────
        with gr.TabItem("🌌 Simulate & Predict"):
            gr.HTML("""
            <p style='color: #94a3b8; margin-bottom: 20px;'>
            Generate synthetic gravitational lens images using the physics-based simulation engine (lenstronomy).
            </p>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    sim_type = gr.Radio(
                        ["Lensed Galaxy", "Non-lensed Galaxy"],
                        value="Lensed Galaxy",
                        label="Simulation Configuration",
                    )
                    seed_input = gr.Number(
                        value=42, label="Random Seed", precision=0
                    )
                    threshold_slider_sim = gr.Slider(
                        0.1, 0.99, value=0.75, step=0.01,
                        label="Classification Threshold",
                    )
                    sim_btn = gr.Button(
                        "🎲 Generate & Analyze", variant="primary", size="lg"
                    )
                    gt_html = gr.HTML()

                with gr.Column(scale=2):
                    with gr.Row():
                        sim_out_img1 = gr.Image(label="Lenstronomy Render", interactive=False, elem_classes="gr-panel")
                        sim_out_img2 = gr.Image(label="ViT Attention Map", interactive=False, elem_classes="gr-panel")
                    sim_results_html = gr.HTML()

            def simulate_and_analyze(sim_type, seed, threshold):
                img, gt_html_str = generate_simulation(sim_type, seed)
                img1, img2, results_html = analyze_image(img, 30, threshold)
                return img1, img2, results_html, gt_html_str

            sim_btn.click(
                fn=simulate_and_analyze,
                inputs=[sim_type, seed_input, threshold_slider_sim],
                outputs=[sim_out_img1, sim_out_img2, sim_results_html, gt_html],
            )

    gr.HTML("""
    <div style='text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.05);'>
        <p style='color: #475569; font-size: 13px;'>
            <strong>GravLensAI v2.0</strong> &nbsp;·&nbsp; Vision Transformer (ViT-Tiny) Classifier &nbsp;·&nbsp; 6-Dimensional Regressor &nbsp;·&nbsp; MC Dropout Uncertainty
        </p>
    </div>
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch GravLensAI Gradio app")
    parser.add_argument('--config', default='configs/app.yaml')
    parser.add_argument('--classifier_path', default=None)
    parser.add_argument('--regressor_path', default=None)
    parser.add_argument('--server_name', default=None)
    parser.add_argument('--server_port', type=int, default=None)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    app_cfg = load_yaml_section(args.config, 'app')
    clf_path, reg_path = resolve_model_paths(args)
    load_models(clf_path, reg_path)

    server_name = args.server_name or os.getenv('GRAVLENS_SERVER_NAME') or app_cfg.get('server_name', '127.0.0.1')
    server_port = args.server_port or int(os.getenv('GRAVLENS_SERVER_PORT', app_cfg.get('server_port', 7860)))
    share = args.share or str(os.getenv('GRAVLENS_SHARE', str(app_cfg.get('share', False)))).lower() in {'1', 'true', 'yes'}

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        css=CUSTOM_CSS,
        theme=gr.themes.Monochrome(font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    )
