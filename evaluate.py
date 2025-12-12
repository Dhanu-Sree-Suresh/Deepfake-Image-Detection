"""
DeepSense: Evaluates the trained model and generates:
- Test metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- All visualization plots
- PDF report

"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import DeepSenseMultiModalDetector
from preprocessing import SimpleRegionExtractor
from pdf_report import create_training_report, PDFReportGenerator


# =============================================================================
# Dataset (same as train.py)
# =============================================================================

class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection evaluation."""

    def __init__(self, data_dir, mode='test', use_cache=True):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.use_cache = use_cache
        self.extractor = SimpleRegionExtractor()

        # Test transforms (no augmentation)
        self.iris_transform = A.Compose([
            A.Resize(64, 64),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.face_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.samples = []
        self._load_samples()
        self.cache = {}

    def _load_samples(self):
        """Load all image paths and labels"""
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, 0))

        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        cache_key = str(img_path)
        if self.use_cache and cache_key in self.cache:
            iris_crop, full_face = self.cache[cache_key]
        else:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = self.extractor.extract(image)

            if result['success']:
                iris_crop = result['iris_crop']
                full_face = result['full_face']
            else:
                iris_crop = cv2.resize(image, (64, 64))
                full_face = cv2.resize(image, (224, 224))

            if self.use_cache:
                self.cache[cache_key] = (iris_crop, full_face)

        iris_transformed = self.iris_transform(image=iris_crop)['image']
        face_transformed = self.face_transform(image=full_face)['image']

        return {
            'iris_crop': iris_transformed,
            'full_face': face_transformed,
            'label': torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_model(model, data_loader, device):
    """Evaluate the model on given data"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            iris_crop = batch['iris_crop'].to(device)
            full_face = batch['full_face'].to(device)
            label = batch['label'].to(device)

            binary_pred, _ = model(iris_crop, full_face)

            probs = torch.softmax(binary_pred, dim=1)
            _, predicted = binary_pred.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def generate_all_plots(history: dict, test_results: dict, output_dir: Path):
    """Generate all visualization plots."""

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating visualization plots in: {plots_dir}")

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Check if we have training history
    has_history = history and 'train_loss' in history and len(history['train_loss']) > 0

    if has_history:
        epochs = range(1, len(history['train_loss']) + 1)

        # 1. Loss Curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
        plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
        plt.title('DeepSense - Training & Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / '01_loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [1/10] Loss curve saved")

        # 2. Accuracy Curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
        plt.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        plt.title('DeepSense - Training & Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / '02_accuracy_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [2/10] Accuracy curve saved")

        # 3. AUC Curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['val_auc'], 'g-^', label='Validation AUC-ROC', linewidth=2, markersize=8)
        plt.axhline(y=max(history['val_auc']), color='r', linestyle='--', label=f'Best AUC: {max(history["val_auc"]):.4f}')
        plt.title('DeepSense - Validation AUC-ROC Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('AUC-ROC Score', fontsize=12)
        plt.legend(loc='lower right', fontsize=11)
        plt.ylim(0.4, 1.0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / '03_auc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [3/10] AUC curve saved")
    else:
        print("  [1-3/10] No training history - skipping epoch-based plots")

    # 4. ROC Curve (always generated)
    fpr, tpr, thresholds = roc_curve(test_results['labels'], test_results['probs'])
    roc_auc = test_results['auc']

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'DeepSense (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.3)
    plt.title('DeepSense - ROC Curve (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
                label=f'Optimal Threshold: {optimal_threshold:.3f}', zorder=5)
    plt.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(plots_dir / '04_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4/10] ROC curve saved")

    # 5. Precision-Recall Curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        test_results['labels'], test_results['probs']
    )
    avg_precision = average_precision_score(test_results['labels'], test_results['probs'])

    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, 'g-', linewidth=2,
             label=f'DeepSense (AP = {avg_precision:.4f})')
    plt.fill_between(recall_curve, precision_curve, alpha=0.3, color='green')
    plt.axhline(y=test_results['precision'], color='r', linestyle='--',
                label=f'Precision @ Threshold 0.5: {test_results["precision"]:.4f}')
    plt.title('DeepSense - Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(plots_dir / '05_precision_recall_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [5/10] Precision-Recall curve saved")

    # 6. Confusion Matrix
    cm = test_results['confusion_matrix']

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                annot_kws={'size': 20}, cbar_kws={'label': 'Count'})
    plt.title('DeepSense - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=12, color='gray')

    plt.tight_layout()
    plt.savefig(plots_dir / '06_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [6/10] Confusion matrix saved")

    # 7. Metrics Comparison Bar Chart
    metrics = {
        'Accuracy': test_results['accuracy'] / 100,
        'Precision': test_results['precision'],
        'Recall': test_results['recall'],
        'F1-Score': test_results['f1'],
        'AUC-ROC': test_results['auc']
    }

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics)))
    bars = plt.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black', linewidth=1.2)

    for bar, value in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('DeepSense - Test Set Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / '07_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [7/10] Metrics comparison saved")

    # 8. Combined Training History (if available)
    if has_history:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DeepSense - Training History Overview', fontsize=16, fontweight='bold', y=1.02)

        axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train', markersize=5)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Val', markersize=5)
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train', markersize=5)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Val', markersize=5)
        axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, history['val_auc'], 'g-^', label='Val AUC', markersize=6)
        axes[1, 0].axhline(y=max(history['val_auc']), color='r', linestyle='--',
                           label=f'Best: {max(history["val_auc"]):.4f}')
        axes[1, 0].set_title('Validation AUC-ROC', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Random')
        axes[1, 1].fill_between(fpr, tpr, alpha=0.2)
        axes[1, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / '08_training_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [8/10] Training overview saved")
    else:
        print("  [8/10] No training history - skipping training overview")

    # 9. Score Distribution
    plt.figure(figsize=(12, 6))

    real_probs = [p for p, l in zip(test_results['probs'], test_results['labels']) if l == 0]
    fake_probs = [p for p, l in zip(test_results['probs'], test_results['labels']) if l == 1]

    plt.hist(real_probs, bins=50, alpha=0.7, label='Real Images', color='green', edgecolor='black')
    plt.hist(fake_probs, bins=50, alpha=0.7, label='Fake Images', color='red', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')

    plt.title('DeepSense - Prediction Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Probability (Fake)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / '09_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [9/10] Score distribution saved")

    # 10. Radar Chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    values = [
        test_results['accuracy'] / 100,
        test_results['precision'],
        test_results['recall'],
        test_results['f1'],
        test_results['auc']
    ]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'b-', linewidth=2, label='DeepSense')
    ax.fill(angles, values, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('DeepSense - Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)

    for angle, value, cat in zip(angles[:-1], values[:-1], categories):
        ax.annotate(f'{value:.3f}', xy=(angle, value), xytext=(angle, value + 0.1),
                   ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(plots_dir / '10_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [10/10] Radar chart saved")

    print(f"\nAll plots saved to: {plots_dir}")
    return plots_dir


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print("=" * 60)
    print("DEEPSENSE MODEL - EVALUATION")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    checkpoint_dir = base_dir / "checkpoints"

    # Find checkpoint
    checkpoint_path = checkpoint_dir / "best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / "multimodal_best.pth"

    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found in {checkpoint_dir}")
        print("Please train the model first using: python train.py")
        return

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get training history if available
    history = checkpoint.get('history', {})
    config = checkpoint.get('config', {})
    best_val_auc = checkpoint.get('best_val_auc', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0)
    trained_epochs = checkpoint.get('epoch', 0) + 1

    print(f"\nCheckpoint Info:")
    print(f"  Trained Epochs: {trained_epochs}")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print()

    # Create model
    print("Creating model...")
    model = DeepSenseMultiModalDetector(
        num_classes=2,
        dropout=config.get('dropout', 0.5)
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Load dataset
    print("Loading dataset...")
    full_dataset = DeepfakeDataset(data_dir, mode='test', use_cache=True)
    print(f"Total samples: {len(full_dataset)}")

    # Count labels
    real_count = sum(1 for _, label in full_dataset.samples if label == 0)
    fake_count = sum(1 for _, label in full_dataset.samples if label == 1)
    print(f"  Real: {real_count}")
    print(f"  Fake: {fake_count}")
    print()

    # Use same split as training (seed=42)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset Split (same as training):")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print()

    # Test loader
    batch_size = config.get('batch_size', 16)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluate
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_results = evaluate_model(model, test_loader, device)

    print(f"\n--- Classification Results (Real vs Fake) ---")
    print(f"Test Accuracy:  {test_results['accuracy']:.2f}%")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall:    {test_results['recall']:.4f}")
    print(f"Test F1-Score:  {test_results['f1']:.4f}")
    print(f"Test AUC-ROC:   {test_results['auc']:.4f}")

    print("\nConfusion Matrix:")
    cm = test_results['confusion_matrix']
    print(f"              Predicted")
    print(f"           Real    Fake")
    print(f"Actual Real  {cm[0][0]:<6} {cm[0][1] if len(cm[0]) > 1 else 0}")
    print(f"       Fake  {cm[1][0] if len(cm) > 1 else 0:<6} {cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0}")

    print("\nClassification Report:")
    print(classification_report(
        test_results['labels'],
        test_results['preds'],
        target_names=['Real', 'Fake']
    ))

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 60)

    try:
        plots_dir = generate_all_plots(history, test_results, base_dir)
        print(f"\nAll plots generated successfully!")
    except Exception as e:
        print(f"\n[WARNING] Could not generate plots: {e}")
        import traceback
        traceback.print_exc()

    # Generate PDF report
    print("\n" + "=" * 60)
    print("GENERATING PDF REPORT")
    print("=" * 60)

    model_info = {
        'model_name': 'DeepSense Multi-Modal Deepfake Detector',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_iris_size': '64x64',
        'input_face_size': '224x224',
        'num_branches': 5,
        'feature_dimensions': 576,
        'trained_epochs': trained_epochs
    }

    report_config = {
        'data_dir': str(data_dir),
        'epochs': trained_epochs,
        'batch_size': batch_size,
        'device': str(device),
        'real_samples': real_count,
        'fake_samples': fake_count,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset)
    }

    # Add config from checkpoint if available
    for k, v in config.items():
        if k not in report_config:
            report_config[k] = str(v) if isinstance(v, Path) else v

    try:
        pdf_path = create_training_report(
            config=report_config,
            history=history,
            test_results=test_results,
            model_info=model_info,
            output_dir=str(base_dir / "reports")
        )
        print(f"\nPDF Report generated successfully!")
        print(f"Location: {pdf_path}")
    except Exception as e:
        print(f"\n[WARNING] Could not generate PDF report: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: DeepSense Multi-Modal Deepfake Detector")
    print(f"Trained Epochs: {trained_epochs}")
    print(f"Best Val AUC (during training): {best_val_auc:.4f}")
    print(f"Best Val Acc (during training): {best_val_acc:.2f}%")
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_results['accuracy']:.2f}%")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  F1-Score:  {test_results['f1']:.4f}")
    print(f"  AUC-ROC:   {test_results['auc']:.4f}")
    print(f"\nOutput Files:")
    print(f"  Plots: {base_dir / 'plots'}")
    print(f"  Report: {base_dir / 'reports'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
