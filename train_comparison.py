"""
DeepSense: Comparison Study - ML vs Deep Learning for Deepfake Detection
=========================================================================
Models: SVM, XGBoost, Random Forest, EfficientNet-B0
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import warnings
import pickle
import json
from tqdm import tqdm

# ML Libraries
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Run: pip install xgboost")

# Deep Learning
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# PDF Report
from src.pdf_report import PDFReportGenerator

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'data_dir': Path(__file__).parent / 'data',
    'output_dir': Path(__file__).parent / 'comparison_results',
    'image_size': 64,  # For feature extraction
    'face_size': 224,  # For EfficientNet
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
    'batch_size': 32,

    # EfficientNet training
    'epochs': 15,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,

    # ML models
    'svm_kernel': 'rbf',
    'svm_C': 10.0,
    'rf_n_estimators': 200,
    'rf_max_depth': 20,
    'xgb_n_estimators': 200,
    'xgb_max_depth': 6,
    'xgb_learning_rate': 0.1,
}

# =============================================================================
# Feature Extraction for ML Models
# =============================================================================

class FeatureExtractor:
    """
    Extract features from images for ML models.
    Features include:
    - Color histograms (RGB, HSV)
    - Texture features (LBP-like)
    - Edge features (Gradient)
    - Statistical features
    - FFT-based frequency features
    """

    def __init__(self, image_size: int = 64):
        self.image_size = image_size

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract all features from an image"""
        if image is None:
            return np.zeros(self.get_feature_dim())

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))

        features = []

        # 1. Color Histogram Features (48 features)
        features.extend(self._color_histogram(image))

        # 2. Texture Features (59 features)
        features.extend(self._texture_features(image))

        # 3. Edge/Gradient Features (36 features)
        features.extend(self._edge_features(image))

        # 4. Statistical Features (15 features)
        features.extend(self._statistical_features(image))

        # 5. FFT Frequency Features (32 features)
        features.extend(self._fft_features(image))

        return np.array(features, dtype=np.float32)

    def get_feature_dim(self) -> int:
        """Return total feature dimension"""
        return 48 + 59 + 36 + 15 + 32  # 190 features

    def _color_histogram(self, image: np.ndarray) -> list:
        """Extract color histogram features"""
        features = []

        # RGB histograms (16 bins each = 48 features)
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            features.extend(hist)

        return features

    def _texture_features(self, image: np.ndarray) -> list:
        """Extract texture features using simplified LBP-like approach"""
        features = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # GLCM-like features using co-occurrence
        # Calculate gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)

        # Histogram of gradients (HOG-like) - 36 features
        mag_hist = np.histogram(magnitude.flatten(), bins=18, range=(0, 255))[0]
        mag_hist = mag_hist / (mag_hist.sum() + 1e-7)
        features.extend(mag_hist)

        angle_hist = np.histogram(angle.flatten(), bins=18, range=(-np.pi, np.pi))[0]
        angle_hist = angle_hist / (angle_hist.sum() + 1e-7)
        features.extend(angle_hist)

        # Laplacian variance (texture sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(laplacian.var())

        # Local variance in blocks (4x4 grid = 16 features)
        block_size = self.image_size // 4
        for i in range(4):
            for j in range(4):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                features.append(block.var())

        # Entropy-like features (8 features)
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        features.extend(hist)

        return features

    def _edge_features(self, image: np.ndarray) -> list:
        """Extract edge features"""
        features = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)

        # Edge histogram in regions (3x3 grid = 9 features)
        block_size = self.image_size // 3
        for i in range(3):
            for j in range(3):
                block = edges[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                features.append(np.sum(block > 0) / block.size)

        # Sobel magnitudes per channel (RGB = 3 x 2 = 6 features)
        for i in range(3):
            channel = image[:, :, i]
            gx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            features.append(np.mean(np.abs(gx)))
            features.append(np.mean(np.abs(gy)))

        # Edge direction histogram (20 features)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(gy, gx).flatten()
        angle_hist = np.histogram(angles, bins=20, range=(-np.pi, np.pi))[0]
        angle_hist = angle_hist / (angle_hist.sum() + 1e-7)
        features.extend(angle_hist)

        return features

    def _statistical_features(self, image: np.ndarray) -> list:
        """Extract statistical features"""
        features = []

        # Per-channel statistics (RGB = 3 channels x 5 stats = 15 features)
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.median(channel))
            features.append(float(np.percentile(channel, 25)))
            features.append(float(np.percentile(channel, 75)))

        return features

    def _fft_features(self, image: np.ndarray) -> list:
        """Extract FFT frequency features"""
        features = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2D FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Log magnitude
        magnitude = np.log1p(magnitude)

        # Radial frequency bands (8 bands)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)

        for r_start in range(0, max_radius, max_radius // 8):
            r_end = r_start + max_radius // 8
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), r_end, 1, -1)
            if r_start > 0:
                cv2.circle(mask, (cx, cy), r_start, 0, -1)
            band_energy = np.sum(magnitude * mask) / (np.sum(mask) + 1e-7)
            features.append(band_energy)

        # Quadrant energies (4 quadrants)
        q1 = magnitude[:cy, :cx].mean()
        q2 = magnitude[:cy, cx:].mean()
        q3 = magnitude[cy:, :cx].mean()
        q4 = magnitude[cy:, cx:].mean()
        features.extend([q1, q2, q3, q4])

        # High vs Low frequency ratio
        low_freq = magnitude[cy-8:cy+8, cx-8:cx+8].mean()
        high_freq = magnitude.mean()
        features.append(low_freq / (high_freq + 1e-7))

        # Angular frequency distribution (19 features to make total 32)
        for angle in range(0, 360, 20):
            rad = np.radians(angle)
            x_end = int(cx + max_radius * np.cos(rad))
            y_end = int(cy + max_radius * np.sin(rad))
            x_end = max(0, min(w-1, x_end))
            y_end = max(0, min(h-1, y_end))

            # Sample along the line
            num_points = max_radius // 2
            x_coords = np.linspace(cx, x_end, num_points).astype(int)
            y_coords = np.linspace(cy, y_end, num_points).astype(int)
            x_coords = np.clip(x_coords, 0, w-1)
            y_coords = np.clip(y_coords, 0, h-1)
            line_energy = magnitude[y_coords, x_coords].mean()
            features.append(line_energy)

        # Trim to 32 features
        features = features[:32]
        while len(features) < 32:
            features.append(0.0)

        return features


# =============================================================================
# Dataset for Feature Extraction
# =============================================================================

def load_dataset(data_dir: Path, image_size: int = 64):
    """Load all images and labels"""
    images = []
    labels = []
    paths = []

    # Real images (label = 0)
    real_dir = data_dir / 'real'
    if real_dir.exists():
        for img_path in real_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    labels.append(0)
                    paths.append(img_path)

    # Fake images (label = 1)
    fake_dir = data_dir / 'fake'
    if fake_dir.exists():
        for img_path in fake_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    labels.append(1)
                    paths.append(img_path)

    print(f"Loaded {len(images)} images")
    print(f"  Real: {sum(1 for l in labels if l == 0)}")
    print(f"  Fake: {sum(1 for l in labels if l == 1)}")

    return images, np.array(labels), paths


def extract_features(images: list, feature_extractor: FeatureExtractor) -> np.ndarray:
    """Extract features from all images"""
    features = []

    print("Extracting features...")
    for img in tqdm(images, desc="Feature Extraction"):
        feat = feature_extractor.extract(img)
        features.append(feat)

    return np.array(features)


# =============================================================================
# EfficientNet-B0 Model
# =============================================================================

class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 for binary classification"""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)

        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # 1280 for B0

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class EfficientNetDataset(Dataset):
    """Dataset for EfficientNet training"""

    def __init__(self, images: list, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, torch.tensor(label, dtype=torch.long)


def get_efficientnet_transforms(mode: str = 'train', image_size: int = 224):
    """Get transforms for EfficientNet"""
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
            A.GaussNoise(var_limit=(5, 25), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


# =============================================================================
# Training Functions
# =============================================================================

def train_svm(X_train, y_train, X_val, y_val, config: dict) -> dict:
    """Train SVM classifier"""
    print("\n" + "=" * 60)
    print("TRAINING SVM")
    print("=" * 60)

    start_time = datetime.now()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train SVM
    model = SVC(
        kernel=config['svm_kernel'],
        C=config['svm_C'],
        probability=True,
        random_state=config['random_seed'],
        verbose=True
    )

    print(f"Training SVM with kernel={config['svm_kernel']}, C={config['svm_C']}")
    model.fit(X_train_scaled, y_train)

    training_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]

    results = {
        'model': model,
        'scaler': scaler,
        'training_time': training_time,
        'accuracy': accuracy_score(y_val, y_pred) * 100,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_prob),
        'confusion_matrix': confusion_matrix(y_val, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print(f"\nSVM Results:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  AUC-ROC:   {results['auc']:.4f}")
    print(f"  Training Time: {training_time:.2f}s")

    return results


def train_random_forest(X_train, y_train, X_val, y_val, config: dict) -> dict:
    """Train Random Forest classifier"""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    start_time = datetime.now()

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=config['rf_n_estimators'],
        max_depth=config['rf_max_depth'],
        random_state=config['random_seed'],
        n_jobs=-1,
        verbose=1
    )

    print(f"Training Random Forest with n_estimators={config['rf_n_estimators']}, max_depth={config['rf_max_depth']}")
    model.fit(X_train, y_train)

    training_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results = {
        'model': model,
        'scaler': None,
        'training_time': training_time,
        'accuracy': accuracy_score(y_val, y_pred) * 100,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_prob),
        'confusion_matrix': confusion_matrix(y_val, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_importance': model.feature_importances_
    }

    print(f"\nRandom Forest Results:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  AUC-ROC:   {results['auc']:.4f}")
    print(f"  Training Time: {training_time:.2f}s")

    return results


def train_xgboost(X_train, y_train, X_val, y_val, config: dict) -> dict:
    """Train XGBoost classifier"""
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    if not XGBOOST_AVAILABLE:
        print("[ERROR] XGBoost not available. Skipping...")
        return None

    start_time = datetime.now()

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=config['xgb_n_estimators'],
        max_depth=config['xgb_max_depth'],
        learning_rate=config['xgb_learning_rate'],
        random_state=config['random_seed'],
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=1
    )

    print(f"Training XGBoost with n_estimators={config['xgb_n_estimators']}, max_depth={config['xgb_max_depth']}")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    training_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results = {
        'model': model,
        'scaler': None,
        'training_time': training_time,
        'accuracy': accuracy_score(y_val, y_pred) * 100,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_prob),
        'confusion_matrix': confusion_matrix(y_val, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_importance': model.feature_importances_
    }

    print(f"\nXGBoost Results:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  AUC-ROC:   {results['auc']:.4f}")
    print(f"  Training Time: {training_time:.2f}s")

    return results


def train_efficientnet(train_loader, val_loader, config: dict, device: torch.device) -> dict:
    """Train EfficientNet-B0 classifier"""
    print("\n" + "=" * 60)
    print("TRAINING EFFICIENTNET-B0")
    print("=" * 60)

    start_time = datetime.now()

    # Create model
    model = EfficientNetClassifier(num_classes=2, pretrained=True)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    best_val_auc = 0
    best_model_state = None

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_probs)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()

    training_time = (datetime.now() - start_time).total_seconds()

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    results = {
        'model': model,
        'history': history,
        'training_time': training_time,
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'y_pred': all_preds,
        'y_prob': all_probs,
        'total_params': total_params
    }

    print(f"\nEfficientNet-B0 Results:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  AUC-ROC:   {results['auc']:.4f}")
    print(f"  Training Time: {training_time:.2f}s")

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_comparison_results(results: dict, output_dir: Path):
    """Generate comparison visualizations"""

    output_dir.mkdir(exist_ok=True, parents=True)

    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    # 1. Bar chart comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Comparison - All Metrics', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        values = [results[m][metric] if metric != 'accuracy' else results[m][metric]/100 for m in models]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax.bar(models, values, color=colors)
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_ylim(0, 1.1 if metric != 'accuracy' else 1.1)
        ax.set_ylabel('Score')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Training time comparison
    ax = axes[1, 2]
    times = [results[m]['training_time'] for m in models]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, times, color=colors)
    ax.set_title('TRAINING TIME (seconds)', fontweight='bold')
    ax.set_ylabel('Time (s)')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, model_name in enumerate(models):
        if 'y_prob' in results[model_name] and results[model_name]['y_prob'] is not None:
            # Need true labels for ROC curve - we'll use validation labels
            # For now, just plot AUC values as text
            pass

    # Since we don't have access to y_true here, create a summary bar chart instead
    aucs = [results[m]['auc'] for m in models]
    bars = ax.barh(models, aucs, color=colors[:len(models)])
    ax.set_xlabel('AUC-ROC Score', fontsize=12)
    ax.set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)

    for bar, val in zip(bars, aucs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', ha='left', va='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'auc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        cm = results[model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        axes[idx].set_title(f'{model_name}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Radar Chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, model_name in enumerate(models):
        values = [
            results[model_name]['accuracy'] / 100,
            results[model_name]['precision'],
            results[model_name]['recall'],
            results[model_name]['f1'],
            results[model_name]['auc']
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to: {output_dir}")


def generate_comparison_report(results: dict, config: dict, output_dir: Path) -> str:
    """Generate PDF comparison report"""

    report = PDFReportGenerator(str(output_dir))
    report.title = "DeepSense - Model Comparison Study"

    # Header
    report.add_section("DEEPSENSE MODEL COMPARISON STUDY")
    report.add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.add_line("")
    report.add_line("Models Compared:")
    report.add_line("  1. Support Vector Machine (SVM)")
    report.add_line("  2. Random Forest")
    report.add_line("  3. XGBoost")
    report.add_line("  4. EfficientNet-B0 (Deep Learning)")

    # Dataset Info
    report.add_section("DATASET INFORMATION")
    report.add_metric("Train Ratio", f"{config['train_ratio']*100:.0f}%")
    report.add_metric("Validation Ratio", f"{config['val_ratio']*100:.0f}%")
    report.add_metric("Test Ratio", f"{config['test_ratio']*100:.0f}%")
    report.add_metric("Random Seed", config['random_seed'])

    # Results per model
    for model_name, model_results in results.items():
        report.add_section(f"{model_name.upper()} RESULTS")
        report.add_metric("Accuracy", f"{model_results['accuracy']:.2f}%")
        report.add_metric("Precision", f"{model_results['precision']:.4f}")
        report.add_metric("Recall", f"{model_results['recall']:.4f}")
        report.add_metric("F1-Score", f"{model_results['f1']:.4f}")
        report.add_metric("AUC-ROC", f"{model_results['auc']:.4f}")
        report.add_metric("Training Time", f"{model_results['training_time']:.2f} seconds")

        if 'total_params' in model_results:
            report.add_metric("Parameters", f"{model_results['total_params']:,}")

        report.add_confusion_matrix(model_results['confusion_matrix'])

    # Comparison Summary
    report.add_section("COMPARISON SUMMARY")

    # Create comparison table
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Time(s)"]
    rows = []
    for model_name, model_results in results.items():
        rows.append([
            model_name,
            f"{model_results['accuracy']:.2f}%",
            f"{model_results['precision']:.4f}",
            f"{model_results['recall']:.4f}",
            f"{model_results['f1']:.4f}",
            f"{model_results['auc']:.4f}",
            f"{model_results['training_time']:.1f}"
        ])

    report.add_table(headers, rows)

    # Best model
    report.add_section("BEST MODEL ANALYSIS")

    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    best_auc = max(results.items(), key=lambda x: x[1]['auc'])
    fastest = min(results.items(), key=lambda x: x[1]['training_time'])

    report.add_line("")
    report.add_line(f"  Best Accuracy:  {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.2f}%)")
    report.add_line(f"  Best F1-Score:  {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    report.add_line(f"  Best AUC-ROC:   {best_auc[0]} ({best_auc[1]['auc']:.4f})")
    report.add_line(f"  Fastest:        {fastest[0]} ({fastest[1]['training_time']:.2f}s)")

    # Recommendations
    report.add_section("RECOMMENDATIONS")
    report.add_line("")

    if best_auc[0] == 'EfficientNet-B0':
        report.add_line("  EfficientNet-B0 achieved the best overall performance.")
        report.add_line("  Recommended for production deployment where accuracy is critical.")
    else:
        report.add_line(f"  {best_auc[0]} achieved competitive results with faster training.")
        report.add_line("  Consider this model for resource-constrained environments.")

    report.add_line("")
    report.add_line("  Trade-offs:")
    report.add_line("  - Deep Learning (EfficientNet): Higher accuracy, longer training, GPU required")
    report.add_line("  - ML Models (SVM/RF/XGB): Faster training, interpretable, CPU-friendly")

    # Generate PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = report.generate_pdf(f"comparison_report_{timestamp}")

    return pdf_path


# =============================================================================
# Main Function
# =============================================================================

def main():
    print("=" * 60)
    print("DEEPFAKE DETECTION - MODEL COMPARISON STUDY")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)

    # Load dataset
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    images, labels, paths = load_dataset(CONFIG['data_dir'])

    # Split data
    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
        np.arange(len(images)), labels,
        train_size=CONFIG['train_ratio'],
        random_state=CONFIG['random_seed'],
        stratify=labels
    )

    X_val_idx, X_test_idx, y_val, y_test = train_test_split(
        X_temp_idx, y_temp,
        train_size=CONFIG['val_ratio'] / (CONFIG['val_ratio'] + CONFIG['test_ratio']),
        random_state=CONFIG['random_seed'],
        stratify=y_temp
    )

    print(f"\nData Split:")
    print(f"  Train: {len(X_train_idx)} samples")
    print(f"  Val:   {len(X_val_idx)} samples")
    print(f"  Test:  {len(X_test_idx)} samples")

    # Get image subsets
    train_images = [images[i] for i in X_train_idx]
    val_images = [images[i] for i in X_val_idx]
    test_images = [images[i] for i in X_test_idx]

    # Extract features for ML models
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION FOR ML MODELS")
    print("=" * 60)

    feature_extractor = FeatureExtractor(image_size=CONFIG['image_size'])
    print(f"Feature dimension: {feature_extractor.get_feature_dim()}")

    X_train_features = extract_features(train_images, feature_extractor)
    X_val_features = extract_features(val_images, feature_extractor)
    X_test_features = extract_features(test_images, feature_extractor)

    # Store all results
    all_results = {}

    # =========================================================================
    # Train SVM
    # =========================================================================
    svm_results = train_svm(X_train_features, y_train, X_test_features, y_test, CONFIG)
    all_results['SVM'] = svm_results

    # =========================================================================
    # Train Random Forest
    # =========================================================================
    rf_results = train_random_forest(X_train_features, y_train, X_test_features, y_test, CONFIG)
    all_results['Random Forest'] = rf_results

    # =========================================================================
    # Train XGBoost
    # =========================================================================
    if XGBOOST_AVAILABLE:
        xgb_results = train_xgboost(X_train_features, y_train, X_test_features, y_test, CONFIG)
        if xgb_results:
            all_results['XGBoost'] = xgb_results

    # =========================================================================
    # Train EfficientNet-B0
    # =========================================================================
    print("\n" + "=" * 60)
    print("PREPARING EFFICIENTNET-B0 DATA")
    print("=" * 60)

    # Create datasets for EfficientNet
    train_transform = get_efficientnet_transforms('train', CONFIG['face_size'])
    val_transform = get_efficientnet_transforms('val', CONFIG['face_size'])

    train_dataset = EfficientNetDataset(train_images, y_train, train_transform)
    val_dataset = EfficientNetDataset(val_images, y_val, val_transform)
    test_dataset = EfficientNetDataset(test_images, y_test, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    efficientnet_results = train_efficientnet(train_loader, test_loader, CONFIG, device)
    all_results['EfficientNet-B0'] = efficientnet_results

    # =========================================================================
    # Generate Comparison Visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 60)

    plot_comparison_results(all_results, CONFIG['output_dir'])

    # =========================================================================
    # Generate PDF Report
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING PDF REPORT")
    print("=" * 60)

    pdf_path = generate_comparison_report(all_results, CONFIG, CONFIG['output_dir'])

    # =========================================================================
    # Save Models
    # =========================================================================
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    models_dir = CONFIG['output_dir'] / 'models'
    models_dir.mkdir(exist_ok=True)

    # Save ML models
    with open(models_dir / 'svm_model.pkl', 'wb') as f:
        pickle.dump({'model': all_results['SVM']['model'], 'scaler': all_results['SVM']['scaler']}, f)
    print(f"  Saved: svm_model.pkl")

    with open(models_dir / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump({'model': all_results['Random Forest']['model']}, f)
    print(f"  Saved: random_forest_model.pkl")

    if 'XGBoost' in all_results:
        with open(models_dir / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump({'model': all_results['XGBoost']['model']}, f)
        print(f"  Saved: xgboost_model.pkl")

    # Save EfficientNet
    torch.save(all_results['EfficientNet-B0']['model'].state_dict(), models_dir / 'efficientnet_b0.pth')
    print(f"  Saved: efficientnet_b0.pth")

    # Save results summary as JSON
    summary = {}
    for model_name, model_results in all_results.items():
        summary[model_name] = {
            'accuracy': model_results['accuracy'],
            'precision': model_results['precision'],
            'recall': model_results['recall'],
            'f1': model_results['f1'],
            'auc': model_results['auc'],
            'training_time': model_results['training_time']
        }

    with open(CONFIG['output_dir'] / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: results_summary.json")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON STUDY COMPLETE")
    print("=" * 60)

    print("\n  FINAL RESULTS SUMMARY")
    print("  " + "-" * 56)
    print(f"  {'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
    print("  " + "-" * 56)

    for model_name, model_results in all_results.items():
        print(f"  {model_name:<20} {model_results['accuracy']:.2f}%{'':<6} "
              f"{model_results['f1']:.4f}{'':<6} {model_results['auc']:.4f}")

    print("  " + "-" * 56)

    # Best model
    best_model = max(all_results.items(), key=lambda x: x[1]['auc'])
    print(f"\n  BEST MODEL: {best_model[0]} (AUC: {best_model[1]['auc']:.4f})")

    print(f"\n  Output Directory: {CONFIG['output_dir']}")
    print(f"  PDF Report: {pdf_path}")
    print(f"\n  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
