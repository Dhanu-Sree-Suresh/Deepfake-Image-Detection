"""
DeepSense: Train the model (deep learning part)
"""
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import json
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

from model import DeepSenseMultiModalDetector, MultiTaskLoss, FocalLoss
from preprocessing import SimpleRegionExtractor
from pdf_report import create_training_report, PDFReportGenerator


# =============================================================================
# Dataset
# =============================================================================

class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection.

    Binary Classification:
    - Real images (label: 0)
    - Fake images (label: 1)

    Directory structure:
    data/
    ├── real/           # Real images
    └── fake/           # Fake images (AI-generated or face-swapped)
    """

    def __init__(self, data_dir, mode='train', use_cache=True):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.use_cache = use_cache

        # Region extractor for preprocessing
        self.extractor = SimpleRegionExtractor()

        # Augmentations
        if mode == 'train':
            self.iris_transform = A.Compose([
                A.Resize(64, 64),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            self.face_transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
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

        # Load samples
        self.samples = []
        self._load_samples()

        # Cache for extracted regions
        self.cache = {}

    def _load_samples(self):
        """Load all image paths and labels"""

        # Load real images (label = 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, 0))

        # Load fake images (label = 1)
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Check cache
        cache_key = str(img_path)
        if self.use_cache and cache_key in self.cache:
            iris_crop, full_face = self.cache[cache_key]
        else:
            # Load image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract regions
            result = self.extractor.extract(image)

            if result['success']:
                iris_crop = result['iris_crop']
                full_face = result['full_face']
            else:
                # Fallback: use resized full image
                iris_crop = cv2.resize(image, (64, 64))
                full_face = cv2.resize(image, (224, 224))

            # Cache if enabled
            if self.use_cache:
                self.cache[cache_key] = (iris_crop, full_face)

        # Apply transforms
        iris_transformed = self.iris_transform(image=iris_crop)['image']
        face_transformed = self.face_transform(image=full_face)['image']

        return {
            'iris_crop': iris_transformed,
            'full_face': face_transformed,
            'label': torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# Loss Function
# =============================================================================

class BinaryClassificationLoss(nn.Module):
    """
    Binary classification loss for Real vs Fake detection.
    """

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 2) - Real vs Fake logits
            target: (B,) - Labels (0=Real, 1=Fake)
        """
        loss = nn.functional.cross_entropy(
            pred, target,
            label_smoothing=self.label_smoothing
        )
        return loss


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        iris_crop = batch['iris_crop'].to(device)
        full_face = batch['full_face'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass (only use binary output)
        binary_pred, _ = model(iris_crop, full_face)

        # Compute loss
        loss = criterion(binary_pred, label)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = binary_pred.max(1)
        correct += predicted.eq(label).sum().item()
        total += label.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            iris_crop = batch['iris_crop'].to(device)
            full_face = batch['full_face'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            binary_pred, _ = model(iris_crop, full_face)

            # Compute loss
            loss = criterion(binary_pred, label)
            total_loss += loss.item()

            # Collect predictions
            probs = torch.softmax(binary_pred, dim=1)
            _, predicted = binary_pred.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'auc': auc,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


def evaluate_test(model, test_loader, device):
    """Final evaluation on test set"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
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

    # Confusion matrix
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
    """
    Generate all training visualization plots.

    Plots generated:
    1. Training & Validation Loss Curve
    2. Training & Validation Accuracy Curve
    3. Validation AUC Curve
    4. ROC Curve
    5. Precision-Recall Curve
    6. Confusion Matrix Heatmap
    7. Metrics Comparison Bar Chart
    8. Combined Training History
    9. Learning Rate Schedule (if available)
    10. Class Distribution
    """

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating visualization plots in: {plots_dir}")

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    epochs = range(1, len(history['train_loss']) + 1)

    # ==========================================================================
    # 1. Training & Validation Loss Curve
    # ==========================================================================
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

    # ==========================================================================
    # 2. Training & Validation Accuracy Curve
    # ==========================================================================
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

    # ==========================================================================
    # 3. Validation AUC Curve
    # ==========================================================================
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

    # ==========================================================================
    # 4. ROC Curve
    # ==========================================================================
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

    # Add optimal threshold point
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
                label=f'Optimal Threshold: {optimal_threshold:.3f}', zorder=5)
    plt.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(plots_dir / '04_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4/10] ROC curve saved")

    # ==========================================================================
    # 5. Precision-Recall Curve
    # ==========================================================================
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

    # ==========================================================================
    # 6. Confusion Matrix Heatmap
    # ==========================================================================
    cm = test_results['confusion_matrix']

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                annot_kws={'size': 20}, cbar_kws={'label': 'Count'})
    plt.title('DeepSense - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Add percentages
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

    # ==========================================================================
    # 7. Metrics Comparison Bar Chart
    # ==========================================================================
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

    # Add value labels on bars
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

    # ==========================================================================
    # 8. Combined Training History (2x2 subplot)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DeepSense - Training History Overview', fontsize=16, fontweight='bold', y=1.02)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train', markersize=5)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Val', markersize=5)
    axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train', markersize=5)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Val', markersize=5)
    axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUC
    axes[1, 0].plot(epochs, history['val_auc'], 'g-^', label='Val AUC', markersize=6)
    axes[1, 0].axhline(y=max(history['val_auc']), color='r', linestyle='--',
                       label=f'Best: {max(history["val_auc"]):.4f}')
    axes[1, 0].set_title('Validation AUC-ROC', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ROC Curve
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

    # ==========================================================================
    # 9. Score Distribution Histogram
    # ==========================================================================
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

    # ==========================================================================
    # 10. Radar Chart of Metrics
    # ==========================================================================
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    values = [
        test_results['accuracy'] / 100,
        test_results['precision'],
        test_results['recall'],
        test_results['f1'],
        test_results['auc']
    ]
    values += values[:1]  # Close the radar chart

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'b-', linewidth=2, label='DeepSense')
    ax.fill(angles, values, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('DeepSense - Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)

    # Add value labels
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
# Main Training
# =============================================================================

def main():
    print("=" * 60)
    print("DEEPFAKE DETECTION MODEL - TRAINING")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    print("Classification: Real vs Fake (Binary)")
    print()

    # Configuration
    config = {
        'data_dir': Path(__file__).parent / "data",
        'epochs': 15,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'dropout': 0.5,
        'label_smoothing': 0.1,
        'use_cache': True
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data directory: {config['data_dir']}")
    print()

    # Create dataset
    print("Loading dataset...")
    full_dataset = DeepfakeDataset(
        config['data_dir'],
        mode='train',
        use_cache=config['use_cache']
    )
    print(f"Total samples: {len(full_dataset)}")

    # Count labels
    real_count = sum(1 for _, label in full_dataset.samples if label == 0)
    fake_count = sum(1 for _, label in full_dataset.samples if label == 1)

    print(f"  Real: {real_count}")
    print(f"  Fake: {fake_count}")
    print()

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print()

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Create model
    print("Creating model...")
    model = DeepSenseMultiModalDetector(
        num_classes=2,
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Loss, optimizer, scheduler
    criterion = BinaryClassificationLoss(
        label_smoothing=config['label_smoothing']
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # Training loop
    best_val_auc = 0.0
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"Starting training for {config['epochs']} epochs...")
    print("-" * 60)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        # Print metrics
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%, "
              f"Val AUC: {val_metrics['auc']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_val_acc = val_metrics['accuracy']
            print(f"  *** New best model! AUC: {best_val_auc:.4f} ***")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_auc': best_val_auc,
                'best_val_acc': best_val_acc,
                'history': history,
                'config': config
            }, checkpoint_dir / "best_model.pth")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(checkpoint_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluate_test(model, test_loader, device)

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

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_results['labels'],
        test_results['preds'],
        target_names=['Real', 'Fake']
    ))

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation AUC:      {best_val_auc:.4f}")
    print(f"Final Test Accuracy:      {test_results['accuracy']:.2f}%")
    print(f"Final Test AUC:           {test_results['auc']:.4f}")
    print(f"\nModel saved to: {checkpoint_dir / 'best_model.pth'}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # =========================================================================
    # GENERATE VISUALIZATION PLOTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 60)

    try:
        plots_dir = generate_all_plots(history, test_results, Path(__file__).parent)
        print(f"\nAll plots generated successfully!")
    except Exception as e:
        print(f"\n[WARNING] Could not generate plots: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # GENERATE PDF REPORT
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING PDF REPORT")
    print("=" * 60)

    # Prepare model info
    model_info = {
        'model_name': 'DeepSense Multi-Modal Deepfake Detector',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_iris_size': '64x64',
        'input_face_size': '224x224',
        'num_branches': 5,
        'feature_dimensions': 576
    }

    # Prepare config for report (convert Path to string)
    report_config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    report_config['device'] = str(device)
    report_config['real_samples'] = real_count
    report_config['fake_samples'] = fake_count
    report_config['train_samples'] = len(train_dataset)
    report_config['val_samples'] = len(val_dataset)
    report_config['test_samples'] = len(test_dataset)

    # Generate PDF report
    try:
        pdf_path = create_training_report(
            config=report_config,
            history=history,
            test_results=test_results,
            model_info=model_info,
            output_dir=str(Path(__file__).parent / "reports")
        )
        print(f"\nPDF Report generated successfully!")
        print(f"Location: {pdf_path}")
    except Exception as e:
        print(f"\n[WARNING] Could not generate PDF report: {e}")
        print("Training completed successfully, but PDF generation failed.")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
