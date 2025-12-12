"""
Data Preprocessing and Dataset Classes for Deepfake Detection
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp


class SimpleRegionExtractor:
    """
    Extracts face regions 
    Uses MediaPipe Face Mesh to detect facial landmarks and extract:
    - Full face (224x224)
    - Iris/Eye region (64x64)
    """

    # MediaPipe Face Mesh indices for eye regions
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, image: np.ndarray) -> Dict:
        """
        Extract face regions from image.

        Args:
            image: RGB image (numpy array)

        Returns:
            Dictionary with:
            - 'success': Whether extraction was successful
            - 'full_face': Full face region (224x224)
            - 'iris_crop': Iris/eye region (64x64)
            - 'landmarks': Face landmarks if detected
        """
        result = {
            'success': False,
            'full_face': None,
            'iris_crop': None,
            'landmarks': None
        }

        if image is None:
            return result

        h, w = image.shape[:2]

        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Process with MediaPipe
        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            # Fallback: use full image
            result['full_face'] = cv2.resize(image, (224, 224))
            result['iris_crop'] = cv2.resize(image, (64, 64))
            return result

        landmarks = results.multi_face_landmarks[0]
        result['landmarks'] = landmarks

        # Get face bounding box from landmarks
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding
        padding = int((x_max - x_min) * 0.1)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Extract full face
        face_region = image[y_min:y_max, x_min:x_max]
        if face_region.size > 0:
            result['full_face'] = cv2.resize(face_region, (224, 224))
        else:
            result['full_face'] = cv2.resize(image, (224, 224))

        # Extract iris/eye region
        iris_crop = self._extract_eye_region(image, landmarks, w, h)
        if iris_crop is not None:
            result['iris_crop'] = iris_crop
        else:
            # Fallback to center face crop
            result['iris_crop'] = cv2.resize(face_region, (64, 64)) if face_region.size > 0 else cv2.resize(image, (64, 64))

        result['success'] = True
        return result

    def _extract_eye_region(self, image: np.ndarray, landmarks, w: int, h: int) -> Optional[np.ndarray]:
        """Extract eye/iris region from detected landmarks"""
        try:
            # Get left eye landmarks
            left_eye_points = []
            for idx in self.LEFT_EYE_INDICES:
                lm = landmarks.landmark[idx]
                left_eye_points.append((int(lm.x * w), int(lm.y * h)))

            # Get right eye landmarks
            right_eye_points = []
            for idx in self.RIGHT_EYE_INDICES:
                lm = landmarks.landmark[idx]
                right_eye_points.append((int(lm.x * w), int(lm.y * h)))

            # Calculate bounding boxes
            left_xs = [p[0] for p in left_eye_points]
            left_ys = [p[1] for p in left_eye_points]
            right_xs = [p[0] for p in right_eye_points]
            right_ys = [p[1] for p in right_eye_points]

            # Combined eye region
            all_xs = left_xs + right_xs
            all_ys = left_ys + right_ys

            x_min, x_max = min(all_xs), max(all_xs)
            y_min, y_max = min(all_ys), max(all_ys)

            # Add padding
            pad_x = int((x_max - x_min) * 0.2)
            pad_y = int((y_max - y_min) * 0.3)

            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w, x_max + pad_x)
            y_max = min(h, y_max + pad_y)

            eye_region = image[y_min:y_max, x_min:x_max]

            if eye_region.size > 0:
                return cv2.resize(eye_region, (64, 64))

        except Exception:
            pass

        return None

    def close(self):
        """Release resources"""
        if self.face_mesh:
            self.face_mesh.close()


def get_transforms(mode: str = 'train', image_size: int = 64) -> A.Compose:
    
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3
            ),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(5, 25)),
                A.ISONoise(intensity=(0.1, 0.3)),
            ], p=0.2),
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


class DeepfakeDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 mode: str = 'train',
                 transform: Optional[A.Compose] = None,
                 image_size: int = 64):
        
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform or get_transforms(mode, image_size)
        self.image_size = image_size

        # Load image paths and labels
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples ({mode} set)")
        print(f"  Real: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"  Fake: {sum(1 for _, l in self.samples if l == 1)}")

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load image paths and labels"""
        samples = []

        # Real images (label = 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append((img_path, 0))

        # Fake images (label = 1)
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append((img_path, 1))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return image, torch.tensor(label, dtype=torch.long)


def create_data_loaders(data_dir: str,
                        batch_size: int = 32,
                        num_workers: int = 0,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        seed: int = 42,
                        image_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Create full dataset
    full_dataset = DeepfakeDataset(data_dir, mode='train', image_size=image_size)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Data loaders created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader
