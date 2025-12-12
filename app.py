"""
DeepSense: UI Design using Streamlit

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import json
import io
import tempfile
from fpdf import FPDF

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import DeepSenseMultiModalDetector
from preprocessing import SimpleRegionExtractor
from detector import AdvancedDeepfakeDetector, AdvancedAnalysisResult

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="DeepSense - Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Dark Teal/Slate Theme
st.markdown("""
<style>
    /* Main header - Dark Teal Gradient */
    .main-header {
        background: linear-gradient(135deg, #1a535c 0%, #2d6a4f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(26,83,92,0.4);
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Section headers - Slate/Charcoal */
    .section-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Secondary section header - Teal */
    .section-header-cyan {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Result boxes */
    .result-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(5,150,105,0.4);
    }
    .result-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(220,38,38,0.4);
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .result-type {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .result-confidence {
        font-size: 1rem;
        opacity: 0.95;
    }

    /* Status indicators */
    .status-normal {
        background: #059669;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-suspicious {
        background: #d97706;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-fake {
        background: #dc2626;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Warning/success/danger text boxes */
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #92400e;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #059669;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #065f46;
        margin: 0.5rem 0;
    }
    .danger-box {
        background: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #991b1b;
        margin: 0.5rem 0;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        color: white;
        font-weight: 600;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Analyze button - Teal */
    .stButton>button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(13,148,136,0.4);
    }

    /* Hide footer only, keep menu for settings */
    footer {visibility: hidden;}

    /* Tab styling - Slate */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Model Loading
# =============================================================================

@st.cache_resource
def load_dl_model():
    model = DeepSenseMultiModalDetector(num_classes=2, dropout=0.5)
    checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pth"
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, True, checkpoint.get('best_val_acc', 0), checkpoint.get('best_val_auc', 0)
        except:
            return model, False, 0, 0
    return model, False, 0, 0

@st.cache_resource
def load_region_extractor():
    return SimpleRegionExtractor()

@st.cache_resource
def load_heuristic_detector():
    return AdvancedDeepfakeDetector()


# =============================================================================
# Helper Functions
# =============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_with_dl_model(model, image_np, extractor, threshold=0.6):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Set seed for reproducibility
    set_seed(42)

    # Ensure model is in eval mode
    model.eval()

    result = extractor.extract(image_np)
    if result['success']:
        iris_crop = result['iris_crop']
        full_face = result['full_face']
    else:
        iris_crop = cv2.resize(image_np, (64, 64))
        full_face = cv2.resize(image_np, (224, 224))

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    iris_tensor = transform(image=iris_crop)['image'].unsqueeze(0)
    face_tensor = transform(image=full_face)['image'].unsqueeze(0)

    with torch.no_grad():
        binary_pred, _ = model(iris_tensor, face_tensor)
        probs = torch.softmax(binary_pred, dim=1)
        fake_prob = probs[0][1].item()
        real_prob = probs[0][0].item()
        is_fake = fake_prob > threshold
        confidence = fake_prob if is_fake else real_prob

    return {
        'prediction': 'Fake' if is_fake else 'Real',
        'is_fake': is_fake,
        'confidence': confidence * 100,
        'fake_probability': fake_prob * 100,
        'real_probability': real_prob * 100,
        'threshold': threshold * 100,
        'iris_crop': iris_crop,
        'full_face': full_face
    }


def create_probability_chart(real_prob, fake_prob):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    bars = ax.bar(['Real', 'Fake'], [real_prob, fake_prob],
                  color=['#059669', '#dc2626'], alpha=0.9, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, [real_prob, fake_prob]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12, color='white')

    ax.set_ylim(0, 110)
    ax.axhline(y=60, color='gray', linestyle='--', linewidth=1.5, label='Decision Boundary')
    ax.set_ylabel('Probability (%)', fontsize=10, color='white')
    ax.set_title('Deep Learning Model Prediction', fontsize=12, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    return fig


def get_status_badge(score, thresholds=(0.3, 0.5)):
    if score < thresholds[0]:
        return '<span class="status-normal">Normal</span>'
    elif score < thresholds[1]:
        return '<span class="status-suspicious">Suspicious</span>'
    else:
        return '<span class="status-fake">Likely Fake</span>'


def save_image_to_temp(img_array, prefix="img"):
    """Save numpy array image to temporary file and return the path."""
    if img_array is None:
        return None
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=prefix)
    if len(img_array.shape) == 2:
        # Grayscale
        Image.fromarray(img_array).save(temp_file.name)
    else:
        Image.fromarray(img_array).save(temp_file.name)
    return temp_file.name


def generate_pdf_report(dl_result, heuristic_result, is_fake, combined_conf, original_image=None):
    """Generate a comprehensive PDF report with all images and metrics."""
    pdf = FPDF()
    pdf.add_page()
    temp_files = []  # Track temp files for cleanup

    # Title - Dark Teal
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(26, 83, 92)  # #1a535c
    pdf.cell(0, 15, 'DeepSense Detection Report', ln=True, align='C')

    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
    pdf.ln(10)

    # Main Result Box
    pdf.set_font('Arial', 'B', 18)
    if is_fake:
        pdf.set_fill_color(220, 38, 38)  # #dc2626
        pdf.set_text_color(255, 255, 255)
        result_text = "FAKE DETECTED"
    else:
        pdf.set_fill_color(5, 150, 105)  # #059669
        pdf.set_text_color(255, 255, 255)
        result_text = "REAL IMAGE"

    pdf.cell(0, 15, result_text, ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Combined Confidence: {combined_conf:.1f}%', ln=True, align='C')
    if is_fake:
        pdf.cell(0, 8, f'Fake Type: {heuristic_result.fake_type}', ln=True, align='C')
    pdf.ln(5)

    # =========================================================================
    # OVERVIEW IMAGES (Original, Face Region, Annotated)
    # =========================================================================
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(44, 62, 80)  # #2c3e50
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, '  Image Overview', ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    overview_images = [
        ("Original", heuristic_result.original_image),
        ("Face Region", heuristic_result.face_region),
        ("Detection Result", heuristic_result.annotated_image),
    ]

    x_start = 10
    img_width = 60
    for i, (caption, img) in enumerate(overview_images):
        if img is not None:
            temp_path = save_image_to_temp(img, f"overview_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=x_start + i * 65, y=pdf.get_y(), w=img_width)
    pdf.ln(50)

    # Image captions
    pdf.set_font('Arial', '', 9)
    for i, (caption, _) in enumerate(overview_images):
        pdf.set_x(x_start + i * 65)
        pdf.cell(img_width, 5, caption, align='C')
    pdf.ln(10)

    # =========================================================================
    # Deep Learning Results Section
    # =========================================================================
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(44, 62, 80)  # #2c3e50
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, '  Deep Learning Model Results', ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.ln(3)

    pdf.cell(95, 8, f"Prediction: {dl_result['prediction']}", ln=False)
    pdf.cell(95, 8, f"Confidence: {dl_result['confidence']:.1f}%", ln=True)
    pdf.cell(95, 8, f"Real Probability: {dl_result['real_probability']:.1f}%", ln=False)
    pdf.cell(95, 8, f"Fake Probability: {dl_result['fake_probability']:.1f}%", ln=True)
    pdf.ln(5)

    # =========================================================================
    # Heuristic Analysis Section
    # =========================================================================
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(44, 62, 80)  # #2c3e50
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, '  Heuristic Analysis Results', ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.ln(3)

    pdf.cell(95, 8, f"Result: {'FAKE' if heuristic_result.is_fake else 'REAL'}", ln=False)
    pdf.cell(95, 8, f"Confidence: {heuristic_result.confidence*100:.1f}%", ln=True)
    pdf.cell(95, 8, f"Fake Type: {heuristic_result.fake_type}", ln=False)
    pdf.cell(95, 8, f"Processing Time: {heuristic_result.processing_time*1000:.0f} ms", ln=True)
    pdf.ln(5)

    # =========================================================================
    # IRIS ANALYSIS - NEW PAGE
    # =========================================================================
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(13, 148, 136)  # #0d9488
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, '  Iris Pattern Analysis', ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.ln(3)

    pdf.cell(63, 8, f"Gradient IoU: {heuristic_result.gradient_iou:.3f}", ln=False)
    pdf.cell(63, 8, f"Left Pupil IoU: {heuristic_result.left_pupil_iou:.3f}", ln=False)
    pdf.cell(63, 8, f"Right Pupil IoU: {heuristic_result.right_pupil_iou:.3f}", ln=True)
    pdf.cell(63, 8, f"Pupil Asymmetry: {heuristic_result.pupil_asymmetry:.3f}", ln=False)
    pdf.cell(63, 8, f"Min Pupil IoU: {heuristic_result.min_pupil_iou:.3f}", ln=False)
    pdf.cell(63, 8, f"Avg Pupil IoU: {heuristic_result.avg_pupil_iou:.3f}", ln=True)
    pdf.ln(5)

    # Iris Images - Row 1 (Raw Iris)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, "Raw Iris Extractions:", ln=True)
    iris_imgs1 = [
        ("Left Iris", heuristic_result.left_iris),
        ("Right Iris", heuristic_result.right_iris),
        ("Left + Contour", heuristic_result.left_iris_with_contour),
        ("Right + Contour", heuristic_result.right_iris_with_contour),
    ]
    img_w = 45
    y_pos = pdf.get_y()
    for i, (cap, img) in enumerate(iris_imgs1):
        if img is not None:
            temp_path = save_image_to_temp(img, f"iris1_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=10 + i * 48, y=y_pos, w=img_w)
    pdf.ln(38)
    pdf.set_font('Arial', '', 8)
    for i, (cap, _) in enumerate(iris_imgs1):
        pdf.set_x(10 + i * 48)
        pdf.cell(img_w, 4, cap, align='C')
    pdf.ln(8)

    # Iris Images - Row 2 (Gradients)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, "Gradient Analysis:", ln=True)
    iris_imgs2 = [
        ("Left Gradient", heuristic_result.left_gradient),
        ("Right Gradient", heuristic_result.right_gradient),
        ("Left Heatmap", heuristic_result.left_gradient_heatmap),
        ("Right Heatmap", heuristic_result.right_gradient_heatmap),
    ]
    y_pos = pdf.get_y()
    for i, (cap, img) in enumerate(iris_imgs2):
        if img is not None:
            temp_path = save_image_to_temp(img, f"iris2_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=10 + i * 48, y=y_pos, w=img_w)
    pdf.ln(38)
    pdf.set_font('Arial', '', 8)
    for i, (cap, _) in enumerate(iris_imgs2):
        pdf.set_x(10 + i * 48)
        pdf.cell(img_w, 4, cap, align='C')
    pdf.ln(8)

    # Iris Images - Row 3 (Difference/Overlay)
    iris_imgs3 = [
        ("Difference Map", heuristic_result.gradient_difference_map),
        ("Gradient Overlay", heuristic_result.gradient_overlay),
    ]
    y_pos = pdf.get_y()
    for i, (cap, img) in enumerate(iris_imgs3):
        if img is not None:
            temp_path = save_image_to_temp(img, f"iris3_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=10 + i * 48, y=y_pos, w=img_w)
    pdf.ln(38)
    pdf.set_font('Arial', '', 8)
    for i, (cap, _) in enumerate(iris_imgs3):
        pdf.set_x(10 + i * 48)
        pdf.cell(img_w, 4, cap, align='C')
    pdf.ln(10)

    # =========================================================================
    # FACE-SWAP DETECTION - NEW PAGE
    # =========================================================================
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(13, 148, 136)  # #0d9488
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, '  Face-Swap Artifact Detection', ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.ln(3)

    pdf.cell(63, 8, f"Blending Score: {heuristic_result.blending_score:.3f}", ln=False)
    pdf.cell(63, 8, f"Skin Tone Score: {heuristic_result.skin_tone_score:.3f}", ln=False)
    pdf.cell(63, 8, f"Lighting Score: {heuristic_result.lighting_score:.3f}", ln=True)
    pdf.cell(63, 8, f"Edge Artifacts: {heuristic_result.edge_artifact_score:.3f}", ln=False)
    pdf.cell(63, 8, f"Color Mismatch: {heuristic_result.color_mismatch_score:.3f}", ln=True)
    pdf.ln(5)

    # Face-Swap Images - Row 1
    swap_imgs1 = [
        ("Blending Map", heuristic_result.blending_map),
        ("Blending Heatmap", heuristic_result.blending_map_heatmap),
        ("Edge Map", heuristic_result.edge_map),
        ("Edge Heatmap", heuristic_result.edge_map_heatmap),
    ]
    y_pos = pdf.get_y()
    for i, (cap, img) in enumerate(swap_imgs1):
        if img is not None:
            temp_path = save_image_to_temp(img, f"swap1_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=10 + i * 48, y=y_pos, w=img_w)
    pdf.ln(40)
    pdf.set_font('Arial', '', 8)
    for i, (cap, _) in enumerate(swap_imgs1):
        pdf.set_x(10 + i * 48)
        pdf.cell(img_w, 4, cap, align='C')
    pdf.ln(8)

    # Face-Swap Images - Row 2
    swap_imgs2 = [
        ("Skin Tone Map", heuristic_result.skin_tone_map),
        ("Lighting Map", heuristic_result.lighting_map),
        ("Face Boundary", heuristic_result.face_boundary_mask),
        ("Color Histogram", heuristic_result.color_histogram_comparison),
    ]
    y_pos = pdf.get_y()
    for i, (cap, img) in enumerate(swap_imgs2):
        if img is not None:
            temp_path = save_image_to_temp(img, f"swap2_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=10 + i * 48, y=y_pos, w=img_w)
    pdf.ln(40)
    pdf.set_font('Arial', '', 8)
    for i, (cap, _) in enumerate(swap_imgs2):
        pdf.set_x(10 + i * 48)
        pdf.cell(img_w, 4, cap, align='C')
    pdf.ln(10)

    # =========================================================================
    # AI-GENERATED DETECTION - NEW PAGE
    # =========================================================================
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(13, 148, 136)  # #0d9488
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, '  AI-Generated Detection', ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.ln(3)

    pdf.cell(63, 8, f"Frequency Score: {heuristic_result.frequency_score:.3f}", ln=False)
    pdf.cell(63, 8, f"Noise Pattern: {heuristic_result.noise_pattern_score:.3f}", ln=False)
    pdf.cell(63, 8, f"Texture Score: {heuristic_result.texture_score:.3f}", ln=True)
    pdf.cell(63, 8, f"Symmetry Score: {heuristic_result.symmetry_score:.3f}", ln=False)
    pdf.cell(63, 8, f"Reflection Score: {heuristic_result.reflection_score:.3f}", ln=True)
    pdf.ln(5)

    # AI-Generated Images - Row 1
    ai_imgs1 = [
        ("FFT Frequency Map", heuristic_result.frequency_map),
        ("Frequency Heatmap", heuristic_result.frequency_map_heatmap),
        ("Noise Map", heuristic_result.noise_map),
        ("Noise Heatmap", heuristic_result.noise_map_heatmap),
    ]
    y_pos = pdf.get_y()
    for i, (cap, img) in enumerate(ai_imgs1):
        if img is not None:
            temp_path = save_image_to_temp(img, f"ai1_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=10 + i * 48, y=y_pos, w=img_w)
    pdf.ln(40)
    pdf.set_font('Arial', '', 8)
    for i, (cap, _) in enumerate(ai_imgs1):
        pdf.set_x(10 + i * 48)
        pdf.cell(img_w, 4, cap, align='C')
    pdf.ln(8)

    # AI-Generated Images - Row 2
    ai_imgs2 = [
        ("Texture Map", heuristic_result.texture_map),
        ("Symmetry Comparison", heuristic_result.symmetry_comparison),
        ("Left Eye Reflection", heuristic_result.left_eye_reflection),
        ("Right Eye Reflection", heuristic_result.right_eye_reflection),
    ]
    y_pos = pdf.get_y()
    for i, (cap, img) in enumerate(ai_imgs2):
        if img is not None:
            temp_path = save_image_to_temp(img, f"ai2_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=10 + i * 48, y=y_pos, w=img_w)
    pdf.ln(40)
    pdf.set_font('Arial', '', 8)
    for i, (cap, _) in enumerate(ai_imgs2):
        pdf.set_x(10 + i * 48)
        pdf.cell(img_w, 4, cap, align='C')
    pdf.ln(10)

    # =========================================================================
    # COLOR CONSISTENCY - NEW PAGE
    # =========================================================================
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(13, 148, 136)  # #0d9488
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, '  Face-Body Color Consistency', ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.ln(3)

    score = heuristic_result.face_body_color_score
    pdf.cell(0, 8, f"Face-Body Color Score: {score:.3f}", ln=True)
    if score < 0.3:
        pdf.set_text_color(5, 150, 105)  # #059669
        pdf.cell(0, 8, "Status: Colors are consistent - likely real image", ln=True)
    elif score < 0.5:
        pdf.set_text_color(217, 119, 6)  # #d97706
        pdf.cell(0, 8, "Status: Slight color mismatch detected - possible manipulation", ln=True)
    else:
        pdf.set_text_color(220, 38, 38)  # #dc2626
        pdf.cell(0, 8, "Status: Significant color mismatch - strong manipulation indicator", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Color Analysis Images
    color_imgs = [
        ("Face Region", heuristic_result.face_region_color),
        ("Neck Region", heuristic_result.neck_region_color),
        ("Comparison", heuristic_result.face_body_comparison),
        ("Color Heatmap", heuristic_result.face_body_heatmap),
    ]
    y_pos = pdf.get_y()
    for i, (cap, img) in enumerate(color_imgs):
        if img is not None:
            temp_path = save_image_to_temp(img, f"color_{i}_")
            if temp_path:
                temp_files.append(temp_path)
                pdf.image(temp_path, x=10 + i * 48, y=y_pos, w=img_w)
    pdf.ln(40)
    pdf.set_font('Arial', '', 8)
    for i, (cap, _) in enumerate(color_imgs):
        pdf.set_x(10 + i * 48)
        pdf.cell(img_w, 4, cap, align='C')
    pdf.ln(10)

    # =========================================================================
    # DETECTED INCONSISTENCIES
    # =========================================================================
    if heuristic_result.inconsistencies:
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(255, 193, 7)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, '  Detected Inconsistencies', ln=True, fill=True)
        pdf.set_font('Arial', '', 11)
        pdf.ln(3)

        for inc in heuristic_result.inconsistencies:
            pdf.multi_cell(0, 7, f"- {inc}")
        pdf.ln(5)

    # =========================================================================
    # FOOTER
    # =========================================================================
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 8, 'DeepSense Multi-Modal Deepfake Detection System', ln=True, align='C')
    pdf.cell(0, 6, 'This report was automatically generated for forensic analysis purposes.', ln=True, align='C')

    # Generate PDF bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')

    # Cleanup temp files
    import os
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

    return pdf_bytes


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Load models
    dl_model, model_loaded, best_acc, best_auc = load_dl_model()
    extractor = load_region_extractor()
    heuristic_detector = load_heuristic_detector()

    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    with st.sidebar:
        # Display Settings
        st.markdown("### ⚙️ Display Settings")
        st.markdown("**Sections to Show**")
        show_dl = st.checkbox("Deep Learning Results", value=True)
        show_heuristic = st.checkbox("Heuristic Analysis", value=True)
        show_images = st.checkbox("All Visualization Images", value=True)
        show_radar = st.checkbox("Radar Charts", value=True)

        st.markdown("---")

        # Score Interpretation
        st.markdown("### 📋 Score Interpretation")
        st.markdown("""
        | Score | Status |
        |-------|--------|
        | < 0.3 | 🟢 Normal |
        | 0.3-0.5 | 🟡 Suspicious |
        | > 0.5 | 🔴 Likely Fake |
        """)

        st.markdown("---")

        # Detection Methods
        st.markdown("### 🔬 Detection Methods")
        st.markdown("**Deep Learning:**")
        st.markdown("- 5-Branch CNN")
        st.markdown("- Iris + Face Analysis")
        st.markdown("- FFT + Texture + Blending")
        st.markdown("")
        st.markdown("**Heuristic:**")
        st.markdown("- Iris Pattern Analysis")
        st.markdown("- Face-Swap Artifacts")
        st.markdown("- AI-Generated Fingerprints")
        st.markdown("- Color Consistency")

    # ==========================================================================
    # MAIN CONTENT
    # ==========================================================================

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🔍 DeepSense: Deepfake Detection</h1>
            <p>Multi-Modal Analysis with Deep Learning & Heuristic Detection</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload Section
    st.markdown('<div class="section-header">📷 Upload Image for Analysis</div>', unsafe_allow_html=True)
    st.caption("Choose an image to analyze...")

    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'bmp'], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # Show uploaded image (smaller)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(image_np, caption=f"Uploaded Image ({image_np.shape[1]}x{image_np.shape[0]})", width=350)

        # Analyze button
        if st.button("🔍 ANALYZE IMAGE", use_container_width=True):
            with st.spinner("Analyzing image for deepfake artifacts..."):
                progress = st.progress(0, text="Starting analysis...")
                progress.progress(20, text="Running Deep Learning model...")
                dl_result = predict_with_dl_model(dl_model, image_np, extractor, threshold=0.6)
                progress.progress(60, text="Running Heuristic analysis...")
                heuristic_result = heuristic_detector.detect(image_np)
                progress.progress(100, text="Analysis Complete!")
                st.session_state['dl_result'] = dl_result
                st.session_state['heuristic_result'] = heuristic_result

    # ==========================================================================
    # RESULTS DISPLAY
    # ==========================================================================
    if 'dl_result' in st.session_state:
        dl_result = st.session_state['dl_result']
        heuristic_result = st.session_state['heuristic_result']
        is_fake = dl_result['is_fake'] or heuristic_result.is_fake
        combined_conf = (dl_result['confidence'] + heuristic_result.confidence * 100) / 2

        # Detection Result Section
        st.markdown('<div class="section-header">🎯 DETECTION RESULT</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            if is_fake:
                st.markdown(f"""
                    <div class="result-fake">
                        <div class="result-title">⚠️ FAKE DETECTED</div>
                        <div class="result-type">Type: {heuristic_result.fake_type}</div>
                        <div class="result-confidence">Confidence: {combined_conf:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-real">
                        <div class="result-title">✅ REAL IMAGE</div>
                        <div class="result-type">No manipulation detected</div>
                        <div class="result-confidence">Confidence: {combined_conf:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**DL Prediction**")
                st.markdown(f"### {dl_result['prediction']}")
                delta_color = "inverse" if dl_result['is_fake'] else "normal"
                st.caption(f"↑ {dl_result['confidence']:.1f}% confidence")
            with c2:
                st.markdown("**Heuristic Prediction**")
                st.markdown(f"### {'FAKE' if heuristic_result.is_fake else 'REAL'}")
                st.caption(f"↑ {heuristic_result.confidence*100:.1f}% confidence")

        with col2:
            fig = create_probability_chart(dl_result['real_probability'], dl_result['fake_probability'])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Deep Learning Model Results
        if show_dl:
            st.markdown('<div class="section-header">🧠 Deep Learning Model Results</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Prediction", dl_result['prediction'])
            with c2:
                st.metric("Confidence", f"{dl_result['confidence']:.1f}%")
            with c3:
                st.metric("Real Probability", f"{dl_result['real_probability']:.1f}%")
            with c4:
                st.metric("Fake Probability", f"{dl_result['fake_probability']:.1f}%")

        # Heuristic Analysis Results
        if show_heuristic:
            st.markdown('<div class="section-header">🔬 Heuristic Analysis Results</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Result", "FAKE" if heuristic_result.is_fake else "REAL")
            with c2:
                st.metric("Confidence", f"{heuristic_result.confidence*100:.1f}%")
            with c3:
                st.metric("Fake Type", heuristic_result.fake_type)
            with c4:
                st.metric("Processing Time", f"{heuristic_result.processing_time*1000:.0f} ms")

            # Analysis Tabs
            tabs = st.tabs(["👁️ Iris Analysis", "🔄 Face-Swap Detection", "🤖 AI-Generated Detection", "🎨 Color Consistency", "📊 Metrics Radar"])

            # TAB 1: IRIS ANALYSIS
            with tabs[0]:
                st.markdown("### 👁️ Iris Pattern Analysis")
                st.caption("Analyzes pupil shape, gradient patterns, and bilateral symmetry between eyes.")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Gradient IoU", f"{heuristic_result.gradient_iou:.3f}", help="Similarity between left and right iris gradients")
                    st.metric("Left Pupil IoU", f"{heuristic_result.left_pupil_iou:.3f}")
                with c2:
                    st.metric("Right Pupil IoU", f"{heuristic_result.right_pupil_iou:.3f}")
                    st.metric("Min Pupil IoU", f"{heuristic_result.min_pupil_iou:.3f}")
                with c3:
                    asym = heuristic_result.pupil_asymmetry
                    st.metric("Pupil Asymmetry", f"{asym:.3f}")
                    if asym < 0.3:
                        st.markdown('<span class="status-normal">↑ Normal</span>', unsafe_allow_html=True)
                    elif asym < 0.45:
                        st.markdown('<span class="status-suspicious">↑ Suspicious</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-fake">↑ Abnormal</span>', unsafe_allow_html=True)
                    st.metric("Avg Pupil IoU", f"{heuristic_result.avg_pupil_iou:.3f}")

                if show_images:
                    st.markdown('<div class="section-header-cyan">🖼️ Iris Visualization Images (10 images)</div>', unsafe_allow_html=True)

                    st.markdown("**Raw Iris Extractions:**")
                    cols = st.columns(4)
                    imgs1 = [
                        ("Left Iris", heuristic_result.left_iris),
                        ("Right Iris", heuristic_result.right_iris),
                        ("Left + Contour", heuristic_result.left_iris_with_contour),
                        ("Right + Contour", heuristic_result.right_iris_with_contour),
                    ]
                    for i, (cap, img) in enumerate(imgs1):
                        with cols[i]:
                            if img is not None:
                                st.image(img, caption=cap, use_container_width=True)

                    st.markdown("**Gradient Analysis:**")
                    cols = st.columns(4)
                    imgs2 = [
                        ("Left Gradient", heuristic_result.left_gradient),
                        ("Right Gradient", heuristic_result.right_gradient),
                        ("Left Heatmap", heuristic_result.left_gradient_heatmap),
                        ("Right Heatmap", heuristic_result.right_gradient_heatmap),
                    ]
                    for i, (cap, img) in enumerate(imgs2):
                        with cols[i]:
                            if img is not None:
                                st.image(img, caption=cap, use_container_width=True)

                    cols = st.columns(4)
                    with cols[0]:
                        if heuristic_result.gradient_difference_map is not None:
                            st.image(heuristic_result.gradient_difference_map, caption="Difference Map", use_container_width=True)
                    with cols[1]:
                        if heuristic_result.gradient_overlay is not None:
                            st.image(heuristic_result.gradient_overlay, caption="Gradient Overlay", use_container_width=True)

            # TAB 2: FACE-SWAP DETECTION
            with tabs[1]:
                st.markdown("### 🔄 Face-Swap Artifact Detection")
                st.caption("Detects blending boundaries, skin tone mismatches, and edge artifacts.")

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("Blending", f"{heuristic_result.blending_score:.3f}")
                with c2:
                    st.metric("Skin Tone", f"{heuristic_result.skin_tone_score:.3f}")
                with c3:
                    st.metric("Lighting", f"{heuristic_result.lighting_score:.3f}")
                with c4:
                    st.metric("Edge Artifacts", f"{heuristic_result.edge_artifact_score:.3f}")
                with c5:
                    st.metric("Color Mismatch", f"{heuristic_result.color_mismatch_score:.3f}")

                if show_images:
                    st.markdown('<div class="section-header-cyan">🖼️ Face-Swap Visualization Images</div>', unsafe_allow_html=True)

                    cols = st.columns(4)
                    swap_imgs = [
                        ("Blending Map", heuristic_result.blending_map),
                        ("Blending Heatmap", heuristic_result.blending_map_heatmap),
                        ("Edge Map", heuristic_result.edge_map),
                        ("Edge Heatmap", heuristic_result.edge_map_heatmap),
                    ]
                    for i, (cap, img) in enumerate(swap_imgs):
                        with cols[i]:
                            if img is not None:
                                st.image(img, caption=cap, use_container_width=True)

                    cols = st.columns(4)
                    swap_imgs2 = [
                        ("Skin Tone Map", heuristic_result.skin_tone_map),
                        ("Lighting Map", heuristic_result.lighting_map),
                        ("Face Boundary", heuristic_result.face_boundary_mask),
                        ("Color Histogram", heuristic_result.color_histogram_comparison),
                    ]
                    for i, (cap, img) in enumerate(swap_imgs2):
                        with cols[i]:
                            if img is not None:
                                st.image(img, caption=cap, use_container_width=True)

            # TAB 3: AI-GENERATED DETECTION
            with tabs[2]:
                st.markdown("### 🤖 AI-Generated Image Detection")
                st.caption("Analyzes frequency domain artifacts, noise patterns, and texture consistency.")

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("Frequency", f"{heuristic_result.frequency_score:.3f}")
                with c2:
                    st.metric("Noise Pattern", f"{heuristic_result.noise_pattern_score:.3f}")
                with c3:
                    st.metric("Texture", f"{heuristic_result.texture_score:.3f}")
                with c4:
                    st.metric("Symmetry", f"{heuristic_result.symmetry_score:.3f}")
                with c5:
                    st.metric("Reflection", f"{heuristic_result.reflection_score:.3f}")

                if show_images:
                    st.markdown('<div class="section-header-cyan">🖼️ AI-Generated Visualization Images</div>', unsafe_allow_html=True)

                    cols = st.columns(4)
                    ai_imgs = [
                        ("FFT Frequency Map", heuristic_result.frequency_map),
                        ("Frequency Heatmap", heuristic_result.frequency_map_heatmap),
                        ("Noise Map", heuristic_result.noise_map),
                        ("Noise Heatmap", heuristic_result.noise_map_heatmap),
                    ]
                    for i, (cap, img) in enumerate(ai_imgs):
                        with cols[i]:
                            if img is not None:
                                st.image(img, caption=cap, use_container_width=True)

                    cols = st.columns(4)
                    ai_imgs2 = [
                        ("Texture Map", heuristic_result.texture_map),
                        ("Symmetry Comparison", heuristic_result.symmetry_comparison),
                        ("Left Eye Reflection", heuristic_result.left_eye_reflection),
                        ("Right Eye Reflection", heuristic_result.right_eye_reflection),
                    ]
                    for i, (cap, img) in enumerate(ai_imgs2):
                        with cols[i]:
                            if img is not None:
                                st.image(img, caption=cap, use_container_width=True)

            # TAB 4: COLOR CONSISTENCY
            with tabs[3]:
                st.markdown("### 🎨 Face-Body Color Consistency")
                st.caption("Compares skin tones between face and body regions.")

                score = heuristic_result.face_body_color_score
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Face-Body Score", f"{score:.3f}")
                with col2:
                    if score < 0.3:
                        st.markdown('<div class="success-box">✅ Colors are consistent - likely real image</div>', unsafe_allow_html=True)
                    elif score < 0.5:
                        st.markdown('<div class="warning-box">⚠️ Slight color mismatch detected - possible manipulation</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="danger-box">❌ Significant color mismatch - strong manipulation indicator</div>', unsafe_allow_html=True)

                if show_images:
                    st.markdown('<div class="section-header-cyan">🖼️ Color Analysis Images</div>', unsafe_allow_html=True)

                    cols = st.columns(4)
                    color_imgs = [
                        ("Face Region", heuristic_result.face_region_color),
                        ("Neck Region", heuristic_result.neck_region_color),
                        ("Comparison", heuristic_result.face_body_comparison),
                        ("Color Heatmap", heuristic_result.face_body_heatmap),
                    ]
                    for i, (cap, img) in enumerate(color_imgs):
                        with cols[i]:
                            if img is not None:
                                st.image(img, caption=cap, use_container_width=True)

            # TAB 5: RADAR CHARTS
            with tabs[4]:
                if show_radar:
                    st.markdown("### 📊 Detection Metrics Radar")
                    st.caption("This chart shows how suspicious each aspect of the image is. Higher values (closer to edge) = more likely fake.")

                    # Explanation for layman
                    with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                        st.markdown("""
                        **Understanding the Radar Chart:**

                        Each point on the chart represents a different way we check if an image might be fake:

                        - **Iris**: Checks if the eyes look natural. Real eyes have consistent patterns; fake ones often don't match between left and right.

                        - **Blending**: Looks for "seams" where a face might have been pasted onto another image. Real photos have smooth transitions.

                        - **Frequency**: Analyzes invisible patterns in the image. AI-generated images often have unusual patterns that cameras don't produce.

                        - **Texture**: Checks if skin and surfaces look realistic. Fake images sometimes have unnaturally smooth or repetitive textures.

                        - **Color**: Compares skin tone between face and neck. If someone's face was swapped, the colors often don't match the body.

                        **How to read it:**
                        - Values closer to **center (0)** = Looks normal/real
                        - Values closer to **edge (1)** = Looks suspicious/fake
                        - If multiple metrics are high, the image is more likely manipulated.
                        """)

                    # Create radar chart - smaller size
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        categories = ['Iris', 'Blending', 'Frequency', 'Texture', 'Color']
                        values = [
                            1 - heuristic_result.gradient_iou,
                            heuristic_result.blending_score,
                            heuristic_result.frequency_score,
                            heuristic_result.texture_score,
                            heuristic_result.face_body_color_score
                        ]

                        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                        fig.patch.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')

                        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                        values_plot = values + [values[0]]
                        angles += angles[:1]

                        ax.plot(angles, values_plot, 'o-', linewidth=2, color='#0d9488')
                        ax.fill(angles, values_plot, alpha=0.25, color='#0d9488')
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(categories, color='white', size=9)
                        ax.set_ylim(0, 1)
                        ax.tick_params(colors='white')
                        ax.spines['polar'].set_color('white')
                        ax.set_title('Fake Indicator Scores', color='white', size=11, fontweight='bold')

                        st.pyplot(fig, use_container_width=True)
                        plt.close()

        # Overview Section
        st.markdown('<div class="section-header">📷 Annotated Overview</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[0]:
            if heuristic_result.original_image is not None:
                st.image(heuristic_result.original_image, caption="Original", use_container_width=True)
        with cols[1]:
            if heuristic_result.face_region is not None:
                st.image(heuristic_result.face_region, caption="Face Region", use_container_width=True)
        with cols[2]:
            if heuristic_result.annotated_image is not None:
                st.image(heuristic_result.annotated_image, caption="Detection Result", use_container_width=True)

        # Detected Issues
        if heuristic_result.inconsistencies:
            st.markdown('<div class="section-header">⚠️ Detected Inconsistencies</div>', unsafe_allow_html=True)
            for inc in heuristic_result.inconsistencies:
                st.markdown(f'<div class="warning-box">{inc}</div>', unsafe_allow_html=True)

        # Export - Single comprehensive PDF with all images and metrics
        st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(dl_result, heuristic_result, is_fake, combined_conf)
        st.download_button(
            "📕 Download Complete Report (PDF)",
            pdf_bytes,
            f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "application/pdf",
            use_container_width=True
        )
        st.caption("The PDF report includes all images, metrics, and analysis from the detection.")


if __name__ == "__main__":
    main()
