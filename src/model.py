"""
DeepSense: Deepfake Detector
=========================================
A comprehensive multi-modal deep learning framework for deepfake detection
that combines multiple detection strategies to identify:
1. Fully AI-generated faces (StyleGAN, Midjourney, etc.)
2. Face-swapped images (DeepFaceLab, FaceSwap, Roop, etc.)

Features analyzed:
- Iris texture patterns (for AI-generated detection)
- Face boundary artifacts (for face-swap detection)
- Blending inconsistencies (for face-swap detection)
- Frequency domain artifacts (for both types)
- Color/lighting consistency (for both types)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on discriminative regions"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention for feature recalibration"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =============================================================================
# Branch 1: Iris Analysis (for AI-generated detection)
# =============================================================================

class IrisBranch(nn.Module):
    """
    Analyzes iris texture patterns.
    Effective for detecting fully AI-generated faces.
    """

    def __init__(self, out_features=256):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.attention = SpatialAttention()

        self.final = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, out_features)
        )

    def forward(self, iris_image):
        """
        Args:
            iris_image: (B, 3, 64, 64) - Cropped iris region
        Returns:
            features: (B, out_features)
        """
        x = self.features(iris_image)
        x = self.attention(x)
        x = self.final(x)
        return x


# =============================================================================
# Branch 2: Face Boundary Analysis (for face-swap detection)
# =============================================================================

class BoundaryBranch(nn.Module):
    """
    Analyzes face boundary regions for blending artifacts.
    Effective for detecting face-swapped images.
    """

    def __init__(self, out_features=128):
        super().__init__()

        # High-pass filter to enhance edges
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.cbam = CBAM(256)

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_features)
        )

    def forward(self, boundary_region):
        """
        Args:
            boundary_region: (B, 3, 128, 128) - Face boundary region
        Returns:
            features: (B, out_features)
        """
        x = self.edge_enhance(boundary_region)
        x = self.features(x)
        x = self.cbam(x)
        x = self.final(x)
        return x


# =============================================================================
# Branch 3: Blending Artifact Detection
# =============================================================================

class BlendingBranch(nn.Module):
    """
    Detects color and texture blending artifacts.
    Uses multi-scale analysis to find inconsistencies.
    """

    def __init__(self, out_features=64):
        super().__init__()

        # Small-scale blending detection
        self.small_scale = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Medium-scale blending detection
        self.medium_scale = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Large-scale blending detection
        self.large_scale = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_features)
        )

    def forward(self, full_face):
        """
        Args:
            full_face: (B, 3, 224, 224) - Full face image
        Returns:
            features: (B, out_features)
        """
        small = self.small_scale(full_face)
        medium = self.medium_scale(full_face)
        large = self.large_scale(full_face)

        combined = torch.cat([small, medium, large], dim=1)
        return self.fusion(combined)


# =============================================================================
# Branch 4: Frequency Domain Analysis
# =============================================================================

class FrequencyBranch(nn.Module):
    """
    Analyzes frequency domain for GAN artifacts and blending traces.
    Works for both AI-generated and face-swapped images.
    """

    def __init__(self, out_features=64):
        super().__init__()

        self.freq_conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),  # magnitude + phase
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(128, out_features)
        )

    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W) - Input image
        Returns:
            features: (B, out_features)
        """
        # Convert to grayscale
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]

        # Compute FFT
        fft = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft)

        # Extract magnitude and phase
        magnitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)

        # Log transform magnitude for better dynamic range
        magnitude = torch.log1p(magnitude)

        # Normalize
        magnitude = (magnitude - magnitude.mean(dim=(-2, -1), keepdim=True)) / (
            magnitude.std(dim=(-2, -1), keepdim=True) + 1e-8
        )
        phase = phase / 3.14159  # Normalize to [-1, 1]

        # Stack and process
        freq_input = torch.stack([magnitude, phase], dim=1)
        return self.freq_conv(freq_input)


# =============================================================================
# Branch 5: Color/Lighting Consistency Analysis
# =============================================================================

class ColorConsistencyBranch(nn.Module):
    """
    Analyzes color and lighting consistency across the face.
    Detects mismatched lighting in face-swaps.
    """

    def __init__(self, out_features=64):
        super().__init__()

        # Encode face region
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )

        # Encode context (hair, background near face)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )

        # Compare face vs context
        self.comparator = nn.Sequential(
            nn.Linear(64 * 16 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features)
        )

    def forward(self, face_region, context_region):
        """
        Args:
            face_region: (B, 3, 112, 112) - Inner face
            context_region: (B, 3, 112, 112) - Outer context
        Returns:
            features: (B, out_features)
        """
        face_feat = self.face_encoder(face_region)
        context_feat = self.context_encoder(context_region)
        combined = torch.cat([face_feat, context_feat], dim=1)
        return self.comparator(combined)


# =============================================================================
# Branch 6: Texture Gradient Analysis
# =============================================================================

class TextureGradientBranch(nn.Module):
    """
    Analyzes texture gradients across the face.
    Detects unnatural texture transitions in manipulated images.
    """

    def __init__(self, out_features=64):
        super().__init__()

        # Sobel-like gradient detection
        self.gradient_x = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.gradient_y = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)

        # Initialize with Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.gradient_x.weight.data = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.gradient_y.weight.data = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # Freeze gradient kernels
        self.gradient_x.weight.requires_grad = False
        self.gradient_y.weight.requires_grad = False

        # Process gradients
        self.processor = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # 6 = gradient_x (3ch) + gradient_y (3ch)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(128, out_features)
        )

    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W) - Input image
        Returns:
            features: (B, out_features)
        """
        grad_x = self.gradient_x(image)
        grad_y = self.gradient_y(image)
        gradients = torch.cat([grad_x, grad_y], dim=1)
        return self.processor(gradients)


# =============================================================================
# Main Multi-Modal Model
# =============================================================================

class MultiModalDeepfakeDetector(nn.Module):
   
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        # Feature dimensions
        self.iris_dim = 256
        self.boundary_dim = 128
        self.blend_dim = 64
        self.freq_dim = 64
        self.color_dim = 64
        self.texture_dim = 64

        # Branch 1: Iris Analysis (AI-generated detection)
        self.iris_branch = IrisBranch(out_features=self.iris_dim)

        # Branch 2: Boundary Analysis (face-swap detection)
        self.boundary_branch = BoundaryBranch(out_features=self.boundary_dim)

        # Branch 3: Blending Artifact Detection
        self.blending_branch = BlendingBranch(out_features=self.blend_dim)

        # Branch 4: Frequency Domain Analysis
        self.frequency_branch = FrequencyBranch(out_features=self.freq_dim)

        # Branch 5: Color Consistency Analysis
        self.color_branch = ColorConsistencyBranch(out_features=self.color_dim)

        # Branch 6: Texture Gradient Analysis
        self.texture_branch = TextureGradientBranch(out_features=self.texture_dim)

        # Total features
        total_features = (self.iris_dim + self.boundary_dim + self.blend_dim +
                         self.freq_dim + self.color_dim + self.texture_dim)

        # Feature fusion with attention
        self.fusion_attention = nn.Sequential(
            nn.Linear(total_features, total_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_features // 4, total_features),
            nn.Sigmoid()
        )

        # Main fusion network
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6)
        )

        # Output heads
        # Head 1: Binary classification (Real vs Fake)
        self.binary_classifier = nn.Linear(128, num_classes)

        # Head 2: Type classification (Real, AI-Generated, Face-Swap)
        self.type_classifier = nn.Linear(128, 3)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, iris_crop, full_face, boundary_region, face_center, face_context):
     
        # Extract features from each branch
        iris_feat = self.iris_branch(iris_crop)
        boundary_feat = self.boundary_branch(boundary_region)
        blend_feat = self.blending_branch(full_face)
        freq_feat = self.frequency_branch(full_face)
        color_feat = self.color_branch(face_center, face_context)
        texture_feat = self.texture_branch(full_face)

        # Concatenate all features
        combined = torch.cat([
            iris_feat, boundary_feat, blend_feat,
            freq_feat, color_feat, texture_feat
        ], dim=1)

        # Apply attention-based weighting
        attention = self.fusion_attention(combined)
        weighted = combined * attention

        # Fuse features
        fused = self.fusion(weighted)

        # Output predictions
        binary_output = self.binary_classifier(fused)
        type_output = self.type_classifier(fused)

        # Return branch features for analysis/visualization
        branch_features = {
            'iris': iris_feat,
            'boundary': boundary_feat,
            'blending': blend_feat,
            'frequency': freq_feat,
            'color': color_feat,
            'texture': texture_feat,
            'fused': fused
        }

        return binary_output, type_output, branch_features


class DeepSenseMultiModalDetector(nn.Module):
   
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        # Branch 1: Iris Analysis
        self.iris_branch = IrisBranch(out_features=256)

        # Branch 2: Full Face Analysis (replaces boundary)
        self.face_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )

        # Branch 3: Blending Analysis
        self.blending_branch = BlendingBranch(out_features=64)

        # Branch 4: Frequency Analysis
        self.frequency_branch = FrequencyBranch(out_features=64)

        # Branch 5: Texture Gradient Analysis
        self.texture_branch = TextureGradientBranch(out_features=64)

        # Fusion
        total_features = 256 + 128 + 64 + 64 + 64  # 576

        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6)
        )

        # Classifiers
        self.binary_classifier = nn.Linear(128, num_classes)
        self.type_classifier = nn.Linear(128, 3)

    def forward(self, iris_crop, full_face):
        """
        Args:
            iris_crop: (B, 3, 64, 64)
            full_face: (B, 3, 224, 224)
        """
        iris_feat = self.iris_branch(iris_crop)
        face_feat = self.face_branch(full_face)
        blend_feat = self.blending_branch(full_face)
        freq_feat = self.frequency_branch(full_face)
        texture_feat = self.texture_branch(full_face)

        combined = torch.cat([
            iris_feat, face_feat, blend_feat, freq_feat, texture_feat
        ], dim=1)

        fused = self.fusion(combined)

        binary_output = self.binary_classifier(fused)
        type_output = self.type_classifier(fused)

        return binary_output, type_output


# =============================================================================
# Loss Functions
# =============================================================================

class MultiTaskLoss(nn.Module):
    
    def __init__(self, binary_weight=1.0, type_weight=0.5, label_smoothing=0.1):
        super().__init__()
        self.binary_weight = binary_weight
        self.type_weight = type_weight
        self.label_smoothing = label_smoothing

    def forward(self, binary_pred, type_pred, binary_target, type_target):
        """
        Args:
            binary_pred: (B, 2) - Real vs Fake logits
            type_pred: (B, 3) - Real, AI-Gen, Face-Swap logits
            binary_target: (B,) - Binary labels (0=Real, 1=Fake)
            type_target: (B,) - Type labels (0=Real, 1=AI-Gen, 2=Face-Swap)
        """
        # Binary loss with label smoothing
        binary_loss = F.cross_entropy(
            binary_pred, binary_target,
            label_smoothing=self.label_smoothing
        )

        # Type classification loss
        type_loss = F.cross_entropy(
            type_pred, type_target,
            label_smoothing=self.label_smoothing
        )

        # Combined loss
        total_loss = self.binary_weight * binary_loss + self.type_weight * type_loss

        return total_loss, binary_loss, type_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Useful when one type of fake is more common than another.
    """

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# Utility Functions
# =============================================================================

def get_model(model_type='deepsense', num_classes=2, **kwargs):
    if model_type == 'full':
        return MultiModalDeepfakeDetector(num_classes=num_classes, **kwargs)
    elif model_type in ['deepsense', 'simplified']:
        return DeepSenseMultiModalDetector(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Alias for backward compatibility
SimplifiedMultiModalDetector = DeepSenseMultiModalDetector


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
