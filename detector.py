"""
DeepSense: Heuristic Analysis
"""
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from scipy.fft import fft2, fftshift
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AdvancedAnalysisResult:
    """Container for comprehensive deepfake analysis results"""
    # Overall result
    is_fake: bool = False
    confidence: float = 0.0
    fake_type: str = "Unknown"  # "Real", "AI-Generated", "Face-Swap", "Unknown"

    # Iris Analysis (existing)
    gradient_iou: float = 0.0
    left_pupil_iou: float = 0.0
    right_pupil_iou: float = 0.0
    pupil_asymmetry: float = 0.0
    min_pupil_iou: float = 0.0
    avg_pupil_iou: float = 0.0

    # Face-Swap Detection Scores
    blending_score: float = 0.0  # Higher = more blending artifacts
    skin_tone_score: float = 0.0  # Higher = more inconsistent
    lighting_score: float = 0.0  # Higher = more inconsistent
    edge_artifact_score: float = 0.0  # Higher = more artifacts
    color_mismatch_score: float = 0.0  # Higher = more mismatch

    # AI-Generated Detection Scores
    frequency_score: float = 0.0  # Higher = more GAN artifacts
    noise_pattern_score: float = 0.0  # Higher = inconsistent noise
    texture_score: float = 0.0  # Higher = unnatural texture
    symmetry_score: float = 0.0  # Higher = too symmetric (AI artifact)
    reflection_score: float = 0.0  # Higher = inconsistent reflections

    # =====================================================
    # ALL VISUALIZATION IMAGES
    # =====================================================

    # Basic Images
    annotated_image: Optional[np.ndarray] = None
    original_image: Optional[np.ndarray] = None
    face_region: Optional[np.ndarray] = None

    # Iris Analysis Images (10 images)
    left_iris: Optional[np.ndarray] = None
    right_iris: Optional[np.ndarray] = None
    left_iris_with_contour: Optional[np.ndarray] = None
    right_iris_with_contour: Optional[np.ndarray] = None
    left_gradient: Optional[np.ndarray] = None
    right_gradient: Optional[np.ndarray] = None
    left_gradient_heatmap: Optional[np.ndarray] = None
    right_gradient_heatmap: Optional[np.ndarray] = None
    gradient_difference_map: Optional[np.ndarray] = None
    gradient_overlay: Optional[np.ndarray] = None

    # Face-Swap Detection Images (8 images)
    blending_map: Optional[np.ndarray] = None
    blending_map_heatmap: Optional[np.ndarray] = None
    edge_map: Optional[np.ndarray] = None
    edge_map_heatmap: Optional[np.ndarray] = None
    skin_tone_map: Optional[np.ndarray] = None
    lighting_map: Optional[np.ndarray] = None
    face_boundary_mask: Optional[np.ndarray] = None
    color_histogram_comparison: Optional[np.ndarray] = None

    # AI-Generated Detection Images (8 images)
    frequency_map: Optional[np.ndarray] = None
    frequency_map_heatmap: Optional[np.ndarray] = None
    noise_map: Optional[np.ndarray] = None
    noise_map_heatmap: Optional[np.ndarray] = None
    texture_map: Optional[np.ndarray] = None
    symmetry_comparison: Optional[np.ndarray] = None
    left_eye_reflection: Optional[np.ndarray] = None
    right_eye_reflection: Optional[np.ndarray] = None

    # Face-Body Color Consistency Images (5 images)
    face_body_color_score: float = 0.0  # Higher = more mismatch (FAKE)
    face_region_color: Optional[np.ndarray] = None
    neck_region_color: Optional[np.ndarray] = None
    face_body_comparison: Optional[np.ndarray] = None
    face_body_heatmap: Optional[np.ndarray] = None
    color_distribution_chart: Optional[np.ndarray] = None

    # Details
    inconsistencies: List[str] = field(default_factory=list)
    detection_details: Dict = field(default_factory=dict)
    status: str = "Error"
    processing_time: float = 0.0


class AdvancedDeepfakeDetector:
    
    # MediaPipe landmarks
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356]
    LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149]
    RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378]
    NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5]
    # Chin/jaw landmarks for neck detection
    CHIN = [152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 400, 378, 379, 365, 397]
    JAW_LINE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

    # Sobel kernels
    SOBEL_H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    SOBEL_V = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    def __init__(self, iris_size: Tuple[int, int] = (64, 64)):
        """Initialize the detector."""
        self.iris_size = iris_size

        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect(self, image: np.ndarray) -> AdvancedAnalysisResult:
        
        import time
        start_time = time.time()

        result = AdvancedAnalysisResult()
        h, w = image.shape[:2]

        # Step 1: Detect face and get landmarks
        mp_results = self.face_mesh.process(image)

        if not mp_results.multi_face_landmarks:
            result.status = "No face detected"
            result.processing_time = time.time() - start_time
            return result

        landmarks = np.array([
            [lm.x * w, lm.y * h]
            for lm in mp_results.multi_face_landmarks[0].landmark
        ])

        # Step 2: Extract regions
        left_iris, left_center, left_radius = self._extract_iris_region(
            image, landmarks, self.LEFT_IRIS, self.LEFT_EYE
        )
        right_iris, right_center, right_radius = self._extract_iris_region(
            image, landmarks, self.RIGHT_IRIS, self.RIGHT_EYE
        )

        face_region = self._extract_face_region(image, landmarks)

        result.left_iris = left_iris
        result.right_iris = right_iris
        result.face_region = face_region

        # =====================================================
        # A. IRIS ANALYSIS
        # =====================================================
        if left_iris is not None and right_iris is not None:
            iris_results = self._analyze_iris(left_iris, right_iris)
            result.gradient_iou = iris_results['gradient_iou']
            result.left_pupil_iou = iris_results['left_pupil_iou']
            result.right_pupil_iou = iris_results['right_pupil_iou']
            result.pupil_asymmetry = iris_results['pupil_asymmetry']
            result.min_pupil_iou = min(iris_results['left_pupil_iou'], iris_results['right_pupil_iou'])
            result.avg_pupil_iou = (iris_results['left_pupil_iou'] + iris_results['right_pupil_iou']) / 2
            result.inconsistencies.extend(iris_results['inconsistencies'])

            # Copy iris visualization images
            result.left_iris_with_contour = iris_results.get('left_iris_with_contour')
            result.right_iris_with_contour = iris_results.get('right_iris_with_contour')
            result.left_gradient = iris_results.get('left_gradient')
            result.right_gradient = iris_results.get('right_gradient')
            result.left_gradient_heatmap = iris_results.get('left_gradient_heatmap')
            result.right_gradient_heatmap = iris_results.get('right_gradient_heatmap')
            result.gradient_difference_map = iris_results.get('gradient_difference_map')
            result.gradient_overlay = iris_results.get('gradient_overlay')

        # =====================================================
        # B. FACE-SWAP DETECTION
        # =====================================================
        if face_region is not None:
            faceswap_results = self._detect_faceswap(image, landmarks, face_region)
            result.blending_score = faceswap_results['blending_score']
            result.skin_tone_score = faceswap_results['skin_tone_score']
            result.lighting_score = faceswap_results['lighting_score']
            result.edge_artifact_score = faceswap_results['edge_artifact_score']
            result.color_mismatch_score = faceswap_results['color_mismatch_score']
            result.inconsistencies.extend(faceswap_results['inconsistencies'])

            # Copy face-swap visualization images
            result.blending_map = faceswap_results.get('blending_map')
            result.blending_map_heatmap = faceswap_results.get('blending_map_heatmap')
            result.edge_map = faceswap_results.get('edge_map')
            result.edge_map_heatmap = faceswap_results.get('edge_map_heatmap')
            result.skin_tone_map = faceswap_results.get('skin_tone_map')
            result.lighting_map = faceswap_results.get('lighting_map')
            result.face_boundary_mask = faceswap_results.get('face_boundary_mask')
            result.color_histogram_comparison = faceswap_results.get('color_histogram_comparison')

        # =====================================================
        # C. AI-GENERATED DETECTION
        # =====================================================
        ai_results = self._detect_ai_generated(image, landmarks, face_region)
        result.frequency_score = ai_results['frequency_score']
        result.noise_pattern_score = ai_results['noise_pattern_score']
        result.texture_score = ai_results['texture_score']
        result.symmetry_score = ai_results['symmetry_score']
        result.reflection_score = ai_results['reflection_score']
        result.inconsistencies.extend(ai_results['inconsistencies'])

        # Copy AI-generated visualization images
        result.frequency_map = ai_results.get('frequency_map')
        result.frequency_map_heatmap = ai_results.get('frequency_map_heatmap')
        result.noise_map = ai_results.get('noise_map')
        result.noise_map_heatmap = ai_results.get('noise_map_heatmap')
        result.texture_map = ai_results.get('texture_map')
        result.symmetry_comparison = ai_results.get('symmetry_comparison')
        result.left_eye_reflection = ai_results.get('left_eye_reflection')
        result.right_eye_reflection = ai_results.get('right_eye_reflection')

        # =====================================================
        # D. FACE-BODY COLOR CONSISTENCY ANALYSIS
        # =====================================================
        face_body_results = self._analyze_face_body_color(image, landmarks)
        result.face_body_color_score = face_body_results['score']
        result.face_region_color = face_body_results.get('face_region')
        result.neck_region_color = face_body_results.get('neck_region')
        result.face_body_comparison = face_body_results.get('comparison_image')
        result.face_body_heatmap = face_body_results.get('heatmap')
        result.color_distribution_chart = face_body_results.get('distribution_chart')
        result.inconsistencies.extend(face_body_results.get('inconsistencies', []))

        # =====================================================
        # E. FINAL CLASSIFICATION
        # =====================================================
        classification = self._classify_image(result)
        result.is_fake = classification['is_fake']
        result.confidence = classification['confidence']
        result.fake_type = classification['fake_type']
        result.detection_details = classification['details']

        # Create annotated image
        result.annotated_image = self._create_annotated_image(
            image, landmarks, result
        )

        result.status = "Success"
        result.processing_time = time.time() - start_time

        return result

    # =========================================================================
    # IRIS ANALYSIS METHODS
    # =========================================================================

    def _extract_iris_region(self, image, landmarks, iris_indices, eye_indices):
        """Extract iris region from image."""
        try:
            h, w = image.shape[:2]
            iris_pts = landmarks[iris_indices]
            eye_pts = landmarks[eye_indices]

            center = np.mean(iris_pts, axis=0)
            radius = np.mean(np.linalg.norm(iris_pts - center, axis=1))

            eye_width = np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])
            eye_height = np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1])

            crop_size = int(max(radius * 4, eye_width * 0.8, eye_height * 1.5))

            x1 = int(max(0, center[0] - crop_size // 2))
            y1 = int(max(0, center[1] - crop_size // 2))
            x2 = int(min(w, center[0] + crop_size // 2))
            y2 = int(min(h, center[1] + crop_size // 2))

            if x2 - x1 < 20 or y2 - y1 < 20:
                return None, None, 0.0

            iris_crop = image[y1:y2, x1:x2]
            iris_crop = cv2.resize(iris_crop, self.iris_size)

            return iris_crop, center, radius
        except:
            return None, None, 0.0

    def _extract_face_region(self, image, landmarks):
        """Extract face region."""
        try:
            face_pts = landmarks[self.FACE_OVAL]
            x_min, y_min = np.min(face_pts, axis=0).astype(int)
            x_max, y_max = np.max(face_pts, axis=0).astype(int)

            h, w = image.shape[:2]
            margin = 20
            x1 = max(0, x_min - margin)
            y1 = max(0, y_min - margin)
            x2 = min(w, x_max + margin)
            y2 = min(h, y_max + margin)

            face_crop = image[y1:y2, x1:x2]
            return cv2.resize(face_crop, (224, 224))
        except:
            return None

    def _analyze_iris(self, left_iris, right_iris):
        """Perform iris analysis with comprehensive visualizations."""
        results = {
            'gradient_iou': 0.0,
            'left_pupil_iou': 0.0,
            'right_pupil_iou': 0.0,
            'pupil_asymmetry': 0.0,
            'inconsistencies': [],
            # Visualization images
            'left_iris_with_contour': None,
            'right_iris_with_contour': None,
            'left_gradient': None,
            'right_gradient': None,
            'left_gradient_heatmap': None,
            'right_gradient_heatmap': None,
            'gradient_difference_map': None,
            'gradient_overlay': None,
            'left_pupil_mask': None,
            'right_pupil_mask': None,
            'pupil_comparison': None
        }

        # Pupil shape analysis with visualizations
        left_pupil_iou, left_pupil_mask, left_contour_img = self._analyze_pupil_shape(left_iris)
        right_pupil_iou, right_pupil_mask, right_contour_img = self._analyze_pupil_shape(right_iris)

        results['left_pupil_iou'] = left_pupil_iou
        results['right_pupil_iou'] = right_pupil_iou
        results['pupil_asymmetry'] = abs(left_pupil_iou - right_pupil_iou)
        results['left_pupil_mask'] = left_pupil_mask
        results['right_pupil_mask'] = right_pupil_mask

        # Create iris with contour visualizations
        results['left_iris_with_contour'] = self._create_iris_with_contour(left_iris, left_pupil_mask)
        results['right_iris_with_contour'] = self._create_iris_with_contour(right_iris, right_pupil_mask)

        # Gradient analysis
        left_gradient = self._compute_gradient_map(left_iris)
        right_gradient = self._compute_gradient_map(right_iris)
        results['gradient_iou'] = self._calculate_gradient_iou(left_gradient, right_gradient)

        results['left_gradient'] = left_gradient
        results['right_gradient'] = right_gradient

        # Create gradient heatmaps
        results['left_gradient_heatmap'] = cv2.applyColorMap(left_gradient, cv2.COLORMAP_JET)
        results['right_gradient_heatmap'] = cv2.applyColorMap(right_gradient, cv2.COLORMAP_JET)

        # Create gradient difference map
        right_gradient_flipped = cv2.flip(right_gradient, 1)
        diff_map = cv2.absdiff(left_gradient, right_gradient_flipped)
        results['gradient_difference_map'] = cv2.applyColorMap(diff_map, cv2.COLORMAP_HOT)

        # Create gradient overlay
        results['gradient_overlay'] = self._create_gradient_overlay(left_iris, left_gradient, right_iris, right_gradient)

        # Create pupil comparison visualization
        results['pupil_comparison'] = self._create_pupil_comparison(
            left_iris, left_pupil_mask, left_pupil_iou,
            right_iris, right_pupil_mask, right_pupil_iou
        )

        # Check for inconsistencies
        if results['pupil_asymmetry'] > 0.15:
            results['inconsistencies'].append("High pupil shape asymmetry between eyes")
        if min(left_pupil_iou, right_pupil_iou) < 0.6:
            results['inconsistencies'].append("Irregular pupil shape detected")
        if results['gradient_iou'] < 0.5:
            results['inconsistencies'].append("Iris texture mismatch between eyes")

        return results

    def _create_iris_with_contour(self, iris_image, pupil_mask):
        """Create visualization of iris with pupil contour overlay."""
        try:
            vis_img = iris_image.copy()
            if len(vis_img.shape) == 2:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)

            # Find contours from pupil mask
            if pupil_mask is not None:
                contours, _ = cv2.findContours(pupil_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)  # Green contour
                    # Fit and draw ellipse
                    largest = max(contours, key=cv2.contourArea)
                    if len(largest) >= 5:
                        ellipse = cv2.fitEllipse(largest)
                        cv2.ellipse(vis_img, ellipse, (255, 0, 0), 1)  # Blue ellipse
            return vis_img
        except:
            return iris_image

    def _create_gradient_overlay(self, left_iris, left_gradient, right_iris, right_gradient):
        """Create combined gradient overlay visualization."""
        try:
            h, w = self.iris_size
            combined = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)

            # Left side
            if len(left_iris.shape) == 2:
                left_rgb = cv2.cvtColor(left_iris, cv2.COLOR_GRAY2RGB)
            else:
                left_rgb = left_iris.copy()
            left_heatmap = cv2.applyColorMap(left_gradient, cv2.COLORMAP_JET)
            left_overlay = cv2.addWeighted(left_rgb, 0.6, left_heatmap, 0.4, 0)
            combined[:, :w] = left_overlay

            # Right side
            if len(right_iris.shape) == 2:
                right_rgb = cv2.cvtColor(right_iris, cv2.COLOR_GRAY2RGB)
            else:
                right_rgb = right_iris.copy()
            right_heatmap = cv2.applyColorMap(right_gradient, cv2.COLORMAP_JET)
            right_overlay = cv2.addWeighted(right_rgb, 0.6, right_heatmap, 0.4, 0)
            combined[:, w+10:] = right_overlay

            # Separator
            combined[:, w:w+10] = [128, 128, 128]

            return combined
        except:
            return None

    def _create_pupil_comparison(self, left_iris, left_mask, left_iou, right_iris, right_mask, right_iou):
        """Create side-by-side pupil comparison visualization."""
        try:
            h, w = self.iris_size
            combined = np.zeros((h + 40, w * 2 + 10, 3), dtype=np.uint8)
            combined.fill(255)

            # Left iris with mask overlay
            if len(left_iris.shape) == 2:
                left_rgb = cv2.cvtColor(left_iris, cv2.COLOR_GRAY2RGB)
            else:
                left_rgb = left_iris.copy()
            if left_mask is not None:
                mask_overlay = np.zeros_like(left_rgb)
                mask_overlay[:, :, 1] = left_mask  # Green channel
                left_rgb = cv2.addWeighted(left_rgb, 0.7, mask_overlay, 0.3, 0)
            combined[:h, :w] = left_rgb

            # Right iris with mask overlay
            if len(right_iris.shape) == 2:
                right_rgb = cv2.cvtColor(right_iris, cv2.COLOR_GRAY2RGB)
            else:
                right_rgb = right_iris.copy()
            if right_mask is not None:
                mask_overlay = np.zeros_like(right_rgb)
                mask_overlay[:, :, 1] = right_mask  # Green channel
                right_rgb = cv2.addWeighted(right_rgb, 0.7, mask_overlay, 0.3, 0)
            combined[:h, w+10:] = right_rgb

            # Add IoU text
            cv2.putText(combined, f"IoU: {left_iou:.3f}", (5, h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(combined, f"IoU: {right_iou:.3f}", (w + 15, h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            return combined
        except:
            return None

    def _analyze_pupil_shape(self, iris_image):
        """Analyze pupil shape using ellipse fitting."""
        try:
            if len(iris_image.shape) == 3:
                gray = cv2.cvtColor(iris_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = iris_image

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.5, cleaned, iris_image

            largest_contour = max(contours, key=cv2.contourArea)

            if len(largest_contour) < 5:
                return 0.5, cleaned, iris_image

            ellipse = cv2.fitEllipse(largest_contour)

            ellipse_mask = np.zeros_like(gray)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)

            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

            intersection = np.logical_and(ellipse_mask > 0, contour_mask > 0).sum()
            union = np.logical_or(ellipse_mask > 0, contour_mask > 0).sum()

            biou = intersection / union if union > 0 else 0.5

            return biou, contour_mask, iris_image
        except:
            return 0.5, np.zeros(self.iris_size, dtype=np.uint8), iris_image

    def _compute_gradient_map(self, iris_image):
        """Compute gradient magnitude map."""
        if len(iris_image.shape) == 3:
            gray = cv2.cvtColor(iris_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = iris_image.astype(np.float32)

        gray = gray / 255.0
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        grad_h = cv2.filter2D(gray, -1, self.SOBEL_H)
        grad_v = cv2.filter2D(gray, -1, self.SOBEL_V)

        gradient = np.sqrt(grad_h ** 2 + grad_v ** 2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)

        return gradient.astype(np.uint8)

    def _calculate_gradient_iou(self, left_gradient, right_gradient):
        """Calculate IoU between gradient maps."""
        right_flipped = cv2.flip(right_gradient, 1)

        threshold = 30
        left_binary = (left_gradient > threshold).astype(np.uint8)
        right_binary = (right_flipped > threshold).astype(np.uint8)

        intersection = np.logical_and(left_binary, right_binary).sum()
        union = np.logical_or(left_binary, right_binary).sum()

        structural_iou = intersection / union if union > 0 else 1.0

        left_norm = (left_gradient - left_gradient.mean()) / (left_gradient.std() + 1e-8)
        right_norm = (right_flipped - right_flipped.mean()) / (right_flipped.std() + 1e-8)
        correlation = np.mean(left_norm * right_norm)
        correlation_sim = (correlation + 1) / 2

        return float(np.clip(0.5 * structural_iou + 0.5 * correlation_sim, 0, 1))

    # =========================================================================
    # FACE-SWAP DETECTION METHODS
    # =========================================================================

    def _detect_faceswap(self, image, landmarks, face_region):
        """Detect face-swap artifacts with comprehensive visualizations."""
        results = {
            'blending_score': 0.0,
            'skin_tone_score': 0.0,
            'lighting_score': 0.0,
            'edge_artifact_score': 0.0,
            'color_mismatch_score': 0.0,
            'blending_map': None,
            'blending_map_heatmap': None,
            'edge_map': None,
            'edge_map_heatmap': None,
            'skin_tone_map': None,
            'lighting_map': None,
            'face_boundary_mask': None,
            'color_histogram_comparison': None,
            'inconsistencies': []
        }

        # 1. Blending Boundary Analysis
        blending_result = self._analyze_blending_boundary(image, landmarks)
        results['blending_score'] = blending_result['score']
        results['blending_map'] = blending_result['map']
        if blending_result['map'] is not None:
            results['blending_map_heatmap'] = cv2.applyColorMap(blending_result['map'], cv2.COLORMAP_JET)
        results['face_boundary_mask'] = blending_result.get('boundary_mask')
        if blending_result['score'] > 0.6:
            results['inconsistencies'].append("Blending artifacts detected at face boundary")

        # 2. Skin Tone Consistency
        skin_result = self._analyze_skin_tone(image, landmarks)
        results['skin_tone_score'] = skin_result['score']
        results['skin_tone_map'] = skin_result.get('map')
        if skin_result['score'] > 0.5:
            results['inconsistencies'].append("Skin tone inconsistency between face regions")

        # 3. Lighting Direction Analysis
        lighting_result = self._analyze_lighting(image, landmarks)
        results['lighting_score'] = lighting_result['score']
        results['lighting_map'] = lighting_result.get('map')
        if lighting_result['score'] > 0.5:
            results['inconsistencies'].append("Inconsistent lighting direction detected")

        # 4. Edge Artifact Detection
        edge_result = self._analyze_edge_artifacts(image, landmarks)
        results['edge_artifact_score'] = edge_result['score']
        results['edge_map'] = edge_result['map']
        if edge_result['map'] is not None:
            results['edge_map_heatmap'] = cv2.applyColorMap(edge_result['map'], cv2.COLORMAP_HOT)
        if edge_result['score'] > 0.5:
            results['inconsistencies'].append("Edge artifacts at face boundaries")

        # 5. Color Histogram Matching
        color_result = self._analyze_color_histogram(image, landmarks)
        results['color_mismatch_score'] = color_result['score']
        results['color_histogram_comparison'] = color_result.get('histogram_image')
        if color_result['score'] > 0.5:
            results['inconsistencies'].append("Color distribution mismatch")

        return results

    def _analyze_blending_boundary(self, image, landmarks):
        """Analyze blending artifacts at face boundary."""
        result = {'score': 0.0, 'map': None, 'boundary_mask': None}

        try:
            h, w = image.shape[:2]

            # Create face mask
            face_pts = landmarks[self.FACE_OVAL].astype(np.int32)
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_pts], 255)

            # Create boundary region (dilate - erode)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(face_mask, kernel, iterations=1)
            eroded = cv2.erode(face_mask, kernel, iterations=1)
            boundary_mask = dilated - eroded

            # Save boundary mask for visualization
            result['boundary_mask'] = boundary_mask

            # Convert to LAB for better color analysis
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            # Analyze gradient at boundary
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gradient = cv2.Laplacian(gray, cv2.CV_64F)
            gradient = np.abs(gradient)

            # Get gradient values at boundary
            boundary_gradient = gradient[boundary_mask > 0]

            if len(boundary_gradient) > 0:
                # Higher gradient at boundary = more blending artifacts
                mean_gradient = np.mean(boundary_gradient)
                std_gradient = np.std(boundary_gradient)

                # Normalize score (empirically determined thresholds)
                score = min(mean_gradient / 30.0, 1.0)
                result['score'] = float(score)

                # Create visualization
                blending_map = np.zeros_like(gray, dtype=np.float32)
                blending_map[boundary_mask > 0] = gradient[boundary_mask > 0]
                blending_map = cv2.normalize(blending_map, None, 0, 255, cv2.NORM_MINMAX)
                result['map'] = blending_map.astype(np.uint8)

        except Exception as e:
            pass

        return result

    def _analyze_skin_tone(self, image, landmarks):
        """Analyze skin tone consistency across face regions."""
        result = {'score': 0.0, 'map': None}

        try:
            h, w = image.shape[:2]

            # Define regions
            regions = {
                'forehead': landmarks[self.FOREHEAD].astype(np.int32),
                'left_cheek': landmarks[self.LEFT_CHEEK].astype(np.int32),
                'right_cheek': landmarks[self.RIGHT_CHEEK].astype(np.int32),
                'nose': landmarks[self.NOSE].astype(np.int32)
            }

            # Convert to YCrCb for skin tone analysis
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

            region_colors = {}
            region_masks = {}
            for name, pts in regions.items():
                mask = np.zeros((h, w), dtype=np.uint8)
                hull = cv2.convexHull(pts)
                cv2.fillPoly(mask, [hull], 255)
                region_masks[name] = mask

                # Get mean color in region
                mean_color = cv2.mean(ycrcb, mask=mask)[:3]
                region_colors[name] = mean_color

            # Compare Cr and Cb channels (skin tone indicators)
            cr_values = [c[1] for c in region_colors.values()]
            cb_values = [c[2] for c in region_colors.values()]

            cr_std = np.std(cr_values)
            cb_std = np.std(cb_values)

            # Higher std = more inconsistent skin tone
            # CONSERVATIVE: Real faces naturally have some skin tone variation
            combined_std = cr_std + cb_std
            if combined_std > 30:  # Very inconsistent skin tone
                score = min((combined_std - 30) / 30.0, 0.8)
            elif combined_std > 20:  # Moderately inconsistent
                score = 0.3
            else:
                score = 0.1  # Normal skin tone variation
            result['score'] = float(score)

            # Create skin tone map visualization
            skin_tone_map = image.copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Different colors for regions
            for i, (name, mask) in enumerate(region_masks.items()):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(skin_tone_map, contours, -1, colors[i % len(colors)], 2)
                # Add region name
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(skin_tone_map, name[:4], (cx-15, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i % len(colors)], 1)
            result['map'] = skin_tone_map

        except:
            pass

        return result

    def _analyze_lighting(self, image, landmarks):
        """Analyze lighting direction consistency."""
        result = {'score': 0.0, 'map': None}

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Compute gradient direction
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Get face regions
            h, w = image.shape[:2]
            left_mask = np.zeros((h, w), dtype=np.uint8)
            right_mask = np.zeros((h, w), dtype=np.uint8)

            left_pts = landmarks[self.LEFT_CHEEK].astype(np.int32)
            right_pts = landmarks[self.RIGHT_CHEEK].astype(np.int32)

            cv2.fillPoly(left_mask, [cv2.convexHull(left_pts)], 255)
            cv2.fillPoly(right_mask, [cv2.convexHull(right_pts)], 255)

            # Calculate dominant gradient direction in each region
            left_grad_x = np.mean(grad_x[left_mask > 0])
            left_grad_y = np.mean(grad_y[left_mask > 0])
            right_grad_x = np.mean(grad_x[right_mask > 0])
            right_grad_y = np.mean(grad_y[right_mask > 0])

            left_angle = np.arctan2(left_grad_y, left_grad_x)
            right_angle = np.arctan2(right_grad_y, right_grad_x)

            # Difference in lighting direction
            angle_diff = abs(left_angle - right_angle)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            # CONSERVATIVE: Only flag if lighting is VERY inconsistent
            # Real faces often have some lighting variation (up to pi/3 is normal)
            if angle_diff > np.pi * 0.6:  # Very inconsistent (>108 degrees)
                score = (angle_diff - np.pi * 0.6) / (np.pi * 0.4)
            elif angle_diff > np.pi * 0.4:  # Moderately inconsistent
                score = 0.3
            else:
                score = 0.1  # Normal lighting variation
            result['score'] = float(min(score, 1.0))

            # Create lighting direction visualization
            lighting_map = image.copy()

            # Calculate gradient magnitude for visualization
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create overlay showing lighting direction
            lighting_heatmap = cv2.applyColorMap(grad_mag_norm, cv2.COLORMAP_BONE)
            lighting_map = cv2.addWeighted(lighting_map, 0.6, lighting_heatmap, 0.4, 0)

            # Draw lighting direction arrows
            left_center = np.mean(left_pts, axis=0).astype(int)
            right_center = np.mean(right_pts, axis=0).astype(int)

            arrow_length = 40
            left_end = (int(left_center[0] + arrow_length * np.cos(left_angle)),
                       int(left_center[1] + arrow_length * np.sin(left_angle)))
            right_end = (int(right_center[0] + arrow_length * np.cos(right_angle)),
                        int(right_center[1] + arrow_length * np.sin(right_angle)))

            cv2.arrowedLine(lighting_map, tuple(left_center), left_end, (0, 255, 0), 2, tipLength=0.3)
            cv2.arrowedLine(lighting_map, tuple(right_center), right_end, (0, 0, 255), 2, tipLength=0.3)

            # Add labels
            cv2.putText(lighting_map, "L", (left_center[0]-10, left_center[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(lighting_map, "R", (right_center[0]-10, right_center[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            result['map'] = lighting_map

        except:
            pass

        return result

    def _analyze_edge_artifacts(self, image, landmarks):
        """Detect edge artifacts at face boundaries."""
        result = {'score': 0.0, 'map': None}

        try:
            h, w = image.shape[:2]

            # Create face boundary
            face_pts = landmarks[self.FACE_OVAL].astype(np.int32)

            # Detect edges
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Create boundary region
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_pts], 255)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dilated = cv2.dilate(face_mask, kernel, iterations=1)
            eroded = cv2.erode(face_mask, kernel, iterations=1)
            boundary_mask = dilated - eroded

            # Count edges at boundary vs inside
            boundary_edges = np.sum(edges[boundary_mask > 0])
            inside_edges = np.sum(edges[eroded > 0])

            boundary_area = np.sum(boundary_mask > 0)
            inside_area = np.sum(eroded > 0)

            if boundary_area > 0 and inside_area > 0:
                boundary_density = boundary_edges / boundary_area
                inside_density = inside_edges / inside_area

                # Higher ratio = more edge artifacts at boundary
                # CONSERVATIVE: Real faces often have strong edges at boundaries (jaw, hairline)
                if inside_density > 0:
                    ratio = boundary_density / inside_density
                    # Only flag if ratio is unusually high (>4.0)
                    if ratio > 4.0:
                        score = min((ratio - 4.0) / 4.0, 0.7)
                    elif ratio > 3.0:
                        score = 0.3
                    else:
                        score = 0.1  # Normal edge distribution
                else:
                    score = 0.2

                result['score'] = float(score)

                # Create edge map
                edge_map = np.zeros_like(gray)
                edge_map[boundary_mask > 0] = edges[boundary_mask > 0]
                result['map'] = edge_map

        except:
            pass

        return result

    def _analyze_color_histogram(self, image, landmarks):
        """Analyze color histogram matching between face and background."""
        result = {'score': 0.0, 'histogram_image': None}

        try:
            h, w = image.shape[:2]

            # Create face mask
            face_pts = landmarks[self.FACE_OVAL].astype(np.int32)
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_pts], 255)

            # Erode to get inner face
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            inner_face = cv2.erode(face_mask, kernel, iterations=1)

            # Get boundary region (neck/hair area near face)
            dilated = cv2.dilate(face_mask, kernel, iterations=1)
            outer_region = dilated - face_mask

            # Calculate histograms in LAB space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            hist_face = []
            hist_outer = []

            for i in range(3):
                h_face = cv2.calcHist([lab], [i], inner_face, [32], [0, 256])
                h_outer = cv2.calcHist([lab], [i], outer_region, [32], [0, 256])

                h_face = h_face / (h_face.sum() + 1e-8)
                h_outer = h_outer / (h_outer.sum() + 1e-8)

                hist_face.append(h_face)
                hist_outer.append(h_outer)

            # Compare histograms
            similarity = 0
            for hf, ho in zip(hist_face, hist_outer):
                sim = cv2.compareHist(hf, ho, cv2.HISTCMP_CORREL)
                similarity += sim

            similarity /= 3

            # Lower similarity = more mismatch = higher score
            score = 1.0 - max(0, similarity)
            result['score'] = float(score)

            # Create histogram comparison visualization
            hist_img = np.zeros((200, 300, 3), dtype=np.uint8)
            hist_img.fill(255)

            # Draw L channel histograms (most informative for skin tone)
            face_hist = hist_face[0].flatten()
            outer_hist = hist_outer[0].flatten()

            max_val = max(face_hist.max(), outer_hist.max())
            if max_val > 0:
                face_hist_norm = face_hist / max_val * 180
                outer_hist_norm = outer_hist / max_val * 180

                bar_width = 300 // 32
                for i in range(32):
                    # Face histogram (blue)
                    pt1 = (i * bar_width, 200)
                    pt2 = (i * bar_width + bar_width - 2, 200 - int(face_hist_norm[i]))
                    cv2.rectangle(hist_img, pt1, pt2, (255, 0, 0), -1)

                    # Outer histogram (red, overlay)
                    pt1 = (i * bar_width, 200)
                    pt2 = (i * bar_width + bar_width - 2, 200 - int(outer_hist_norm[i]))
                    cv2.rectangle(hist_img, pt1, pt2, (0, 0, 255), 1)

            # Add labels
            cv2.putText(hist_img, "Face (blue) vs Boundary (red)", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(hist_img, f"Similarity: {similarity:.3f}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            result['histogram_image'] = hist_img

        except:
            pass

        return result

    # =========================================================================
    # AI-GENERATED DETECTION METHODS
    # =========================================================================

    def _detect_ai_generated(self, image, landmarks, face_region):
        """Detect AI-generated image artifacts with comprehensive visualizations."""
        results = {
            'frequency_score': 0.0,
            'noise_pattern_score': 0.0,
            'texture_score': 0.0,
            'symmetry_score': 0.0,
            'reflection_score': 0.0,
            'frequency_map': None,
            'frequency_map_heatmap': None,
            'noise_map': None,
            'noise_map_heatmap': None,
            'texture_map': None,
            'symmetry_comparison': None,
            'left_eye_reflection': None,
            'right_eye_reflection': None,
            'inconsistencies': []
        }

        # 1. Frequency Domain Analysis (GAN fingerprints)
        freq_result = self._analyze_frequency_domain(image)
        results['frequency_score'] = freq_result['score']
        results['frequency_map'] = freq_result['map']
        if freq_result['map'] is not None:
            results['frequency_map_heatmap'] = cv2.applyColorMap(freq_result['map'], cv2.COLORMAP_MAGMA)
        if freq_result['score'] > 0.5:
            results['inconsistencies'].append("GAN frequency artifacts detected")

        # 2. Noise Pattern Analysis
        noise_result = self._analyze_noise_pattern(image, landmarks)
        results['noise_pattern_score'] = noise_result['score']
        results['noise_map'] = noise_result['map']
        if noise_result['map'] is not None:
            results['noise_map_heatmap'] = cv2.applyColorMap(noise_result['map'], cv2.COLORMAP_VIRIDIS)
        if noise_result['score'] > 0.5:
            results['inconsistencies'].append("Inconsistent noise patterns")

        # 3. Texture Consistency
        texture_result = self._analyze_texture(face_region)
        results['texture_score'] = texture_result['score']
        results['texture_map'] = texture_result.get('map')
        if texture_result['score'] > 0.5:
            results['inconsistencies'].append("Unnatural texture patterns")

        # 4. Facial Symmetry Analysis
        symmetry_result = self._analyze_symmetry(image, landmarks)
        results['symmetry_score'] = symmetry_result['score']
        results['symmetry_comparison'] = symmetry_result.get('comparison_image')
        if symmetry_result['score'] > 0.7:
            results['inconsistencies'].append("Unnatural facial symmetry (AI artifact)")

        # 5. Eye Reflection Consistency
        reflection_result = self._analyze_eye_reflections(image, landmarks)
        results['reflection_score'] = reflection_result['score']
        results['left_eye_reflection'] = reflection_result.get('left_eye')
        results['right_eye_reflection'] = reflection_result.get('right_eye')
        if reflection_result['score'] > 0.5:
            results['inconsistencies'].append("Inconsistent eye reflections")

        return results

    def _analyze_frequency_domain(self, image):
        """Analyze frequency domain for GAN artifacts."""
        result = {'score': 0.0, 'map': None}

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Compute FFT
            f = fft2(gray)
            fshift = fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)

            # Normalize for visualization
            magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            result['map'] = magnitude_norm.astype(np.uint8)

            # Analyze frequency distribution
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2

            # Create radial profile
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center_w)**2 + (y - center_h)**2)

            # Divide into frequency bands
            low_freq = magnitude[(r < min(h, w) * 0.1)]
            mid_freq = magnitude[(r >= min(h, w) * 0.1) & (r < min(h, w) * 0.3)]
            high_freq = magnitude[(r >= min(h, w) * 0.3)]

            # GAN images often have characteristic frequency patterns
            # Check for unusual peaks in specific frequency bands
            low_mean = np.mean(low_freq) if len(low_freq) > 0 else 0
            mid_mean = np.mean(mid_freq) if len(mid_freq) > 0 else 0
            high_mean = np.mean(high_freq) if len(high_freq) > 0 else 0

            # Compute ratio (GAN images often have higher mid-frequency content)
            if low_mean > 0:
                ratio = mid_mean / low_mean
                # CONSERVATIVE: Only flag if ratio is VERY unusual
                # Normal ratios can vary widely (0.1 to 0.6 is common)
                if ratio > 0.8:  # Very high mid-frequency (strong GAN signal)
                    score = min((ratio - 0.8) / 0.4, 0.8)
                elif ratio < 0.05:  # Unnaturally low mid-frequency
                    score = 0.5
                else:
                    score = 0.1  # Default to low score (assume real)
            else:
                score = 0.2

            result['score'] = float(score)

        except:
            pass

        return result

    def _analyze_noise_pattern(self, image, landmarks):
        """Analyze noise patterns for consistency."""
        result = {'score': 0.0, 'map': None}

        try:
            h, w = image.shape[:2]

            # Create face mask
            face_pts = landmarks[self.FACE_OVAL].astype(np.int32)
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_pts], 255)

            # Extract noise using high-pass filter
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Gaussian blur and subtract to get noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray - blurred

            # Get noise in face region vs background
            face_noise = noise[face_mask > 0]
            bg_noise = noise[face_mask == 0]

            if len(face_noise) > 0 and len(bg_noise) > 0:
                face_noise_std = np.std(face_noise)
                bg_noise_std = np.std(bg_noise)

                # Different noise patterns indicate manipulation
                # CONSERVATIVE: Real images can have different noise in face vs background
                if bg_noise_std > 0:
                    ratio = abs(face_noise_std - bg_noise_std) / bg_noise_std
                    # Only flag if ratio is VERY high (>2.0 means very different noise)
                    if ratio > 2.0:
                        score = min((ratio - 2.0) / 2.0, 0.7)
                    elif ratio > 1.5:
                        score = 0.3
                    else:
                        score = 0.1  # Normal noise variation
                else:
                    score = 0.2

                result['score'] = float(score)

                # Create noise map
                noise_map = cv2.normalize(np.abs(noise), None, 0, 255, cv2.NORM_MINMAX)
                result['map'] = noise_map.astype(np.uint8)

        except:
            pass

        return result

    def _analyze_texture(self, face_region):
        """Analyze texture patterns for unnatural smoothness."""
        result = {'score': 0.0, 'map': None}

        if face_region is None:
            return result

        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)

            # Compute Local Binary Pattern-like texture measure
            # Simple texture analysis using gradient variance

            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Divide into patches and analyze texture variance
            patch_size = 32
            variances = []
            variance_map = np.zeros_like(gray, dtype=np.float32)

            h, w = gray.shape
            for i in range(0, h - patch_size, patch_size):
                for j in range(0, w - patch_size, patch_size):
                    patch = gradient_mag[i:i+patch_size, j:j+patch_size]
                    var = np.var(patch)
                    variances.append(var)
                    variance_map[i:i+patch_size, j:j+patch_size] = var

            if variances:
                # Low variance = smooth/artificial texture
                mean_var = np.mean(variances)
                std_var = np.std(variances)

                # AI images often have more uniform texture
                # CONSERVATIVE: Only flag if EXTREMELY smooth (real images often have low variance too)
                if mean_var < 20:  # Only flag if extremely smooth
                    score = 0.6 * (1.0 - (mean_var / 20.0))
                elif mean_var < 50 and std_var / (mean_var + 1e-8) < 0.15:  # Very smooth AND very uniform
                    score = 0.4
                else:
                    score = 0.1  # Default to low score (assume real)

                result['score'] = float(score)

                # Create texture map visualization
                variance_map_norm = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                result['map'] = cv2.applyColorMap(variance_map_norm, cv2.COLORMAP_PLASMA)

        except:
            pass

        return result

    def _analyze_symmetry(self, image, landmarks):
        """Analyze facial symmetry (AI often generates too-symmetric faces)."""
        result = {'score': 0.0, 'comparison_image': None}

        try:
            h, w = image.shape[:2]

            # Get face center
            face_pts = landmarks[self.FACE_OVAL]
            center_x = np.mean(face_pts[:, 0])

            # Split face into left and right halves
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            left_half = gray[:, :int(center_x)]
            right_half = gray[:, int(center_x):]

            # Flip right half for comparison
            right_flipped = cv2.flip(right_half, 1)

            # Resize to same size
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, -min_width:]
            right_flipped = right_flipped[:, :min_width]

            # Calculate similarity
            diff = np.abs(left_half.astype(float) - right_flipped.astype(float))
            similarity = 1.0 - (np.mean(diff) / 255.0)

            # Too high similarity = possibly AI-generated
            # Natural faces have some asymmetry (similarity ~0.85-0.95)
            if similarity > 0.95:
                score = (similarity - 0.95) / 0.05  # Scale to 0-1
            elif similarity > 0.90:
                score = 0.3
            else:
                score = 0.0

            result['score'] = float(score)

            # Create symmetry comparison visualization
            # Side by side: Left | Right Flipped | Difference
            diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)

            # Create combined image
            comparison_h = min(left_half.shape[0], 200)
            comparison_w = min_width

            # Resize all to same height
            left_vis = cv2.resize(left_half, (comparison_w, comparison_h))
            right_vis = cv2.resize(right_flipped, (comparison_w, comparison_h))
            diff_vis = cv2.resize(diff_color, (comparison_w, comparison_h))

            # Convert grayscale to RGB
            left_vis_rgb = cv2.cvtColor(left_vis, cv2.COLOR_GRAY2RGB)
            right_vis_rgb = cv2.cvtColor(right_vis, cv2.COLOR_GRAY2RGB)

            # Combine
            comparison = np.hstack([left_vis_rgb, right_vis_rgb, diff_vis])

            # Add labels
            cv2.putText(comparison, "Left", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(comparison, "Right(flip)", (comparison_w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(comparison, "Difference", (comparison_w * 2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(comparison, f"Sim: {similarity:.3f}", (10, comparison_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            result['comparison_image'] = comparison

        except:
            pass

        return result

    def _analyze_eye_reflections(self, image, landmarks):
        """Analyze eye reflection consistency."""
        result = {'score': 0.0, 'left_eye': None, 'right_eye': None}

        try:
            # Extract eye regions
            left_eye_pts = landmarks[self.LEFT_EYE].astype(np.int32)
            right_eye_pts = landmarks[self.RIGHT_EYE].astype(np.int32)

            h, w = image.shape[:2]

            # Get bounding boxes
            lx1, ly1 = np.min(left_eye_pts, axis=0)
            lx2, ly2 = np.max(left_eye_pts, axis=0)
            rx1, ry1 = np.min(right_eye_pts, axis=0)
            rx2, ry2 = np.max(right_eye_pts, axis=0)

            # Extract eye crops
            margin = 5
            left_eye = image[max(0, ly1-margin):min(h, ly2+margin),
                           max(0, lx1-margin):min(w, lx2+margin)]
            right_eye = image[max(0, ry1-margin):min(h, ry2+margin),
                            max(0, rx1-margin):min(w, rx2+margin)]

            if left_eye.size == 0 or right_eye.size == 0:
                return result

            # Find bright spots (reflections)
            left_gray = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)

            # Threshold to find bright spots
            left_thresh = cv2.threshold(left_gray, 200, 255, cv2.THRESH_BINARY)[1]
            right_thresh = cv2.threshold(right_gray, 200, 255, cv2.THRESH_BINARY)[1]

            # Count reflection pixels
            left_reflections = np.sum(left_thresh > 0)
            right_reflections = np.sum(right_thresh > 0)

            # Calculate reflection difference
            total_reflections = left_reflections + right_reflections
            if total_reflections > 0:
                diff_ratio = abs(left_reflections - right_reflections) / total_reflections

                # Also check position of reflections
                # Get centroid of reflections
                if left_reflections > 10 and right_reflections > 10:
                    left_y, left_x = np.where(left_thresh > 0)
                    right_y, right_x = np.where(right_thresh > 0)

                    left_centroid = (np.mean(left_x) / left_thresh.shape[1],
                                    np.mean(left_y) / left_thresh.shape[0])
                    right_centroid = (np.mean(right_x) / right_thresh.shape[1],
                                     np.mean(right_y) / right_thresh.shape[0])

                    # Position difference (should be similar in real photos)
                    pos_diff = np.sqrt((left_centroid[0] - (1-right_centroid[0]))**2 +
                                      (left_centroid[1] - right_centroid[1])**2)

                    score = min(diff_ratio + pos_diff, 1.0)
                else:
                    score = diff_ratio
            else:
                # No reflections - might be suspicious for studio/professional photos
                score = 0.3

            result['score'] = float(score)

            # Create eye reflection visualizations
            # Left eye with reflection highlight
            left_eye_vis = left_eye.copy()
            left_mask_color = np.zeros_like(left_eye_vis)
            left_mask_color[:, :, 2] = left_thresh  # Red channel for reflections
            left_eye_vis = cv2.addWeighted(left_eye_vis, 0.7, left_mask_color, 0.3, 0)
            cv2.putText(left_eye_vis, f"Ref: {left_reflections}", (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            result['left_eye'] = cv2.resize(left_eye_vis, (100, 60))

            # Right eye with reflection highlight
            right_eye_vis = right_eye.copy()
            right_mask_color = np.zeros_like(right_eye_vis)
            right_mask_color[:, :, 2] = right_thresh  # Red channel for reflections
            right_eye_vis = cv2.addWeighted(right_eye_vis, 0.7, right_mask_color, 0.3, 0)
            cv2.putText(right_eye_vis, f"Ref: {right_reflections}", (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            result['right_eye'] = cv2.resize(right_eye_vis, (100, 60))

        except:
            pass

        return result

    # =========================================================================
    # FACE-BODY COLOR CONSISTENCY ANALYSIS
    # =========================================================================

    def _analyze_face_body_color(self, image, landmarks):
        """
        Analyze color consistency between face and neck/body regions.
        Face-swap often creates color mismatch between the swapped face and original body.

        Returns:
            dict with score (0-1, higher = more mismatch = likely FAKE)
            and visualization images
        """
        result = {
            'score': 0.0,
            'face_region': None,
            'neck_region': None,
            'comparison_image': None,
            'heatmap': None,
            'distribution_chart': None,
            'inconsistencies': []
        }

        try:
            h, w = image.shape[:2]

            # 1. Extract face region (using cheeks and forehead for skin color)
            face_mask = np.zeros((h, w), dtype=np.uint8)

            # Combine cheeks region for face skin sampling
            left_cheek_pts = landmarks[self.LEFT_CHEEK].astype(np.int32)
            right_cheek_pts = landmarks[self.RIGHT_CHEEK].astype(np.int32)
            forehead_pts = landmarks[self.FOREHEAD].astype(np.int32)

            cv2.fillPoly(face_mask, [cv2.convexHull(left_cheek_pts)], 255)
            cv2.fillPoly(face_mask, [cv2.convexHull(right_cheek_pts)], 255)
            cv2.fillPoly(face_mask, [cv2.convexHull(forehead_pts)], 255)

            # 2. Extract neck region (below chin)
            chin_pts = landmarks[self.CHIN].astype(np.int32)
            chin_bottom = np.max(chin_pts[:, 1])  # Y coordinate of chin bottom
            chin_left = np.min(chin_pts[:, 0])
            chin_right = np.max(chin_pts[:, 0])
            chin_center_x = (chin_left + chin_right) // 2

            # Define neck region below the chin
            neck_top = int(chin_bottom)
            neck_bottom = min(int(chin_bottom + (chin_bottom - np.min(landmarks[self.FOREHEAD][:, 1])) * 0.5), h - 1)
            neck_width = int((chin_right - chin_left) * 0.6)
            neck_left = max(0, chin_center_x - neck_width // 2)
            neck_right = min(w - 1, chin_center_x + neck_width // 2)

            # Create neck mask
            neck_mask = np.zeros((h, w), dtype=np.uint8)
            if neck_bottom > neck_top and neck_right > neck_left:
                cv2.rectangle(neck_mask, (neck_left, neck_top), (neck_right, neck_bottom), 255, -1)

            # 3. Check if neck region is visible (has enough pixels)
            neck_area = np.sum(neck_mask > 0)
            face_area = np.sum(face_mask > 0)

            if neck_area < 100 or face_area < 100:
                # Not enough visible neck/face area for analysis
                result['inconsistencies'].append("Insufficient neck region visible for analysis")
                return result

            # 4. Convert to LAB color space for perceptual color comparison
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            # Get face and neck color statistics
            face_pixels = lab[face_mask > 0]
            neck_pixels = lab[neck_mask > 0]

            if len(face_pixels) < 50 or len(neck_pixels) < 50:
                return result

            # Calculate mean and std for each channel
            face_mean = np.mean(face_pixels, axis=0)
            face_std = np.std(face_pixels, axis=0)
            neck_mean = np.mean(neck_pixels, axis=0)
            neck_std = np.std(neck_pixels, axis=0)

            # 5. Calculate color difference metrics
            # Delta E (color difference in LAB space)
            delta_L = abs(face_mean[0] - neck_mean[0])  # Luminance difference
            delta_a = abs(face_mean[1] - neck_mean[1])  # Green-Red difference
            delta_b = abs(face_mean[2] - neck_mean[2])  # Blue-Yellow difference

            # CIE Delta E formula (simplified)
            delta_E = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

            # Also check YCrCb for skin tone
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            face_ycrcb = ycrcb[face_mask > 0]
            neck_ycrcb = ycrcb[neck_mask > 0]

            face_cr = np.mean(face_ycrcb[:, 1])
            face_cb = np.mean(face_ycrcb[:, 2])
            neck_cr = np.mean(neck_ycrcb[:, 1])
            neck_cb = np.mean(neck_ycrcb[:, 2])

            # Skin tone difference (Cr and Cb are key for skin color)
            skin_tone_diff = np.sqrt((face_cr - neck_cr)**2 + (face_cb - neck_cb)**2)

            # 6. Calculate final score
            # Normal variance between face and neck is 5-15 delta E
            # Higher values indicate potential manipulation
            delta_E_score = min(max(0, (delta_E - 10) / 30), 1.0)  # 10-40 range to 0-1
            skin_tone_score = min(max(0, (skin_tone_diff - 5) / 20), 1.0)  # 5-25 range to 0-1

            # Combined score (weight more on skin tone as it's more reliable)
            score = 0.4 * delta_E_score + 0.6 * skin_tone_score
            result['score'] = float(score)

            # 7. Add inconsistencies based on thresholds
            if delta_E > 25:
                result['inconsistencies'].append(f"High color difference between face and neck (Delta E: {delta_E:.1f})")
            if skin_tone_diff > 15:
                result['inconsistencies'].append(f"Skin tone mismatch between face and neck (diff: {skin_tone_diff:.1f})")

            # 8. Create visualization images

            # Face region visualization
            face_region_vis = image.copy()
            face_overlay = np.zeros_like(image)
            face_overlay[face_mask > 0] = [0, 255, 0]  # Green for face
            result['face_region'] = cv2.addWeighted(face_region_vis, 0.7, face_overlay, 0.3, 0)

            # Neck region visualization
            neck_region_vis = image.copy()
            neck_overlay = np.zeros_like(image)
            neck_overlay[neck_mask > 0] = [0, 0, 255]  # Red for neck
            result['neck_region'] = cv2.addWeighted(neck_region_vis, 0.7, neck_overlay, 0.3, 0)

            # Comparison image (face and neck side by side with color info) - LARGER SIZE
            comparison_h = 350
            comparison_w = 700
            comparison = np.ones((comparison_h, comparison_w, 3), dtype=np.uint8) * 255

            # Left side: Face color sample (larger)
            face_color = tuple(int(x) for x in cv2.cvtColor(np.uint8([[face_mean]]), cv2.COLOR_LAB2RGB)[0][0])
            cv2.rectangle(comparison, (20, 50), (320, 220), face_color, -1)
            cv2.rectangle(comparison, (20, 50), (320, 220), (0, 0, 0), 3)
            cv2.putText(comparison, "Face Color", (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(comparison, f"L:{face_mean[0]:.0f} a:{face_mean[1]:.0f} b:{face_mean[2]:.0f}",
                       (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(comparison, f"Cr:{face_cr:.0f} Cb:{face_cb:.0f}",
                       (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Right side: Neck color sample (larger)
            neck_color = tuple(int(x) for x in cv2.cvtColor(np.uint8([[neck_mean]]), cv2.COLOR_LAB2RGB)[0][0])
            cv2.rectangle(comparison, (380, 50), (680, 220), neck_color, -1)
            cv2.rectangle(comparison, (380, 50), (680, 220), (0, 0, 0), 3)
            cv2.putText(comparison, "Neck Color", (460, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(comparison, f"L:{neck_mean[0]:.0f} a:{neck_mean[1]:.0f} b:{neck_mean[2]:.0f}",
                       (390, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(comparison, f"Cr:{neck_cr:.0f} Cb:{neck_cb:.0f}",
                       (390, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Bottom: Scores (larger text)
            cv2.putText(comparison, f"Delta E: {delta_E:.1f}", (20, 320),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(comparison, f"Skin Diff: {skin_tone_diff:.1f}", (250, 320),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            verdict_color = (0, 0, 255) if score > 0.5 else (0, 128, 0)
            verdict = "MISMATCH" if score > 0.5 else "CONSISTENT"
            cv2.putText(comparison, f"Score: {score:.2f} ({verdict})", (480, 320),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, verdict_color, 2)

            result['comparison_image'] = comparison

            # Heatmap showing color difference across the image
            # Create a map of how different each pixel is from neck color
            diff_map = np.zeros((h, w), dtype=np.float32)

            # Calculate difference from neck mean color for each pixel
            lab_float = lab.astype(np.float32)
            diff_L = np.abs(lab_float[:, :, 0] - neck_mean[0])
            diff_a = np.abs(lab_float[:, :, 1] - neck_mean[1])
            diff_b = np.abs(lab_float[:, :, 2] - neck_mean[2])
            diff_map = np.sqrt(diff_L**2 + diff_a**2 + diff_b**2)

            # Mask out non-skin regions (very roughly)
            combined_mask = cv2.bitwise_or(face_mask, neck_mask)
            # Expand slightly
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
            combined_mask = cv2.dilate(combined_mask, kernel)

            diff_map[combined_mask == 0] = 0
            diff_map_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result['heatmap'] = cv2.applyColorMap(diff_map_norm, cv2.COLORMAP_JET)

            # Color distribution chart - LARGER SIZE
            chart_h, chart_w = 400, 500
            chart = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 255

            # Plot Cr-Cb distribution for face and neck
            # Scale Cr-Cb values (typically 0-255) to chart coordinates
            def scale_cr_cb(cr, cb, chart_w, chart_h):
                x = int((cr - 100) / 56 * (chart_w - 80) + 40)  # Cr range ~100-156
                y = int((cb - 100) / 56 * (chart_h - 80) + 40)  # Cb range ~100-156
                return max(40, min(chart_w-40, x)), max(40, min(chart_h-40, y))

            # Draw axes (thicker)
            cv2.line(chart, (40, chart_h-40), (chart_w-40, chart_h-40), (0, 0, 0), 2)  # X-axis
            cv2.line(chart, (40, 40), (40, chart_h-40), (0, 0, 0), 2)  # Y-axis
            cv2.putText(chart, "Cr", (chart_w-35, chart_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(chart, "Cb", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Title
            cv2.putText(chart, "Color Distribution (Cr-Cb)", (120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Plot face points (green) - larger points
            for pixel in face_ycrcb[::max(1, len(face_ycrcb)//80)]:  # Sample 80 points
                x, y = scale_cr_cb(pixel[1], pixel[2], chart_w, chart_h)
                cv2.circle(chart, (x, y), 4, (0, 255, 0), -1)

            # Plot neck points (red) - larger points
            for pixel in neck_ycrcb[::max(1, len(neck_ycrcb)//80)]:  # Sample 80 points
                x, y = scale_cr_cb(pixel[1], pixel[2], chart_w, chart_h)
                cv2.circle(chart, (x, y), 4, (0, 0, 255), -1)

            # Draw mean points larger
            fx, fy = scale_cr_cb(face_cr, face_cb, chart_w, chart_h)
            nx, ny = scale_cr_cb(neck_cr, neck_cb, chart_w, chart_h)
            cv2.circle(chart, (fx, fy), 12, (0, 200, 0), -1)
            cv2.circle(chart, (fx, fy), 12, (0, 0, 0), 3)
            cv2.circle(chart, (nx, ny), 12, (0, 0, 200), -1)
            cv2.circle(chart, (nx, ny), 12, (0, 0, 0), 3)

            # Draw line between means (thicker)
            cv2.line(chart, (fx, fy), (nx, ny), (128, 128, 128), 2, cv2.LINE_AA)

            # Legend (larger, with colored squares)
            cv2.rectangle(chart, (chart_w-120, 45), (chart_w-100, 65), (0, 200, 0), -1)
            cv2.rectangle(chart, (chart_w-120, 45), (chart_w-100, 65), (0, 0, 0), 2)
            cv2.putText(chart, "Face", (chart_w-95, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.rectangle(chart, (chart_w-120, 75), (chart_w-100, 95), (0, 0, 200), -1)
            cv2.rectangle(chart, (chart_w-120, 75), (chart_w-100, 95), (0, 0, 0), 2)
            cv2.putText(chart, "Neck", (chart_w-95, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            result['distribution_chart'] = chart

        except Exception as e:
            result['inconsistencies'].append(f"Face-body analysis error: {str(e)}")

        return result

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    def _classify_image(self, result: AdvancedAnalysisResult):
        
        classification = {
            'is_fake': False,
            'confidence': 0.0,
            'fake_type': 'Real',  # Will be 'Real' or 'Fake' only
            'details': {}
        }

        # =====================================================
        # 1. IRIS ANALYSIS - Check for natural similarity
        # =====================================================
        # Real eyes have natural variations - we should NOT penalize small differences
        iris_is_natural = True  # Assume natural unless proven otherwise
        iris_fake_indicators = 0

        if result.left_pupil_iou > 0 and result.right_pupil_iou > 0:
            min_pupil = min(result.left_pupil_iou, result.right_pupil_iou)
            avg_pupil = (result.left_pupil_iou + result.right_pupil_iou) / 2

            # Only flag as suspicious if VERY abnormal (very lenient thresholds)
            # Natural eyes can have asymmetry up to 0.4
            if result.pupil_asymmetry > 0.45:  # Very high asymmetry
                iris_fake_indicators += 1
            # Very low pupil detection suggests artificial iris
            if min_pupil < 0.25:
                iris_fake_indicators += 1
            # Very different gradient patterns
            if result.gradient_iou < 0.2:
                iris_fake_indicators += 1

            # Need at least 2 indicators to suspect fake iris
            if iris_fake_indicators >= 2:
                iris_is_natural = False

        # Calculate iris naturalness score (higher = more natural)
        iris_natural_score = 1.0 - (iris_fake_indicators / 3.0)

        # =====================================================
        # 2. FACE-SWAP DETECTION - Require MULTIPLE strong signals
        # =====================================================
        faceswap_scores_list = [
            ('blending', result.blending_score),
            ('skin_tone', result.skin_tone_score),
            ('lighting', result.lighting_score),
            ('edge_artifact', result.edge_artifact_score),
            ('color_mismatch', result.color_mismatch_score),
            ('face_body_color', result.face_body_color_score)
        ]

        # Count only VERY HIGH indicators (>0.7)
        faceswap_very_high = sum(1 for _, s in faceswap_scores_list if s > 0.7)
        faceswap_high = sum(1 for _, s in faceswap_scores_list if s > 0.6)

        faceswap_values = [s for _, s in faceswap_scores_list if s > 0]
        faceswap_avg = np.mean(faceswap_values) if faceswap_values else 0

        # Only add bonus if MULTIPLE very high indicators
        faceswap_score = faceswap_avg
        if faceswap_very_high >= 2:
            faceswap_score = min(1.0, faceswap_avg + 0.2)
        elif faceswap_high >= 3:
            faceswap_score = min(1.0, faceswap_avg + 0.1)

        # =====================================================
        # 3. AI-GENERATED DETECTION - Require MULTIPLE strong signals
        # =====================================================
        ai_scores_list = [
            ('frequency', result.frequency_score),
            ('noise_pattern', result.noise_pattern_score),
            ('texture', result.texture_score),
            ('symmetry', result.symmetry_score),
            ('reflection', result.reflection_score)
        ]

        # Count only VERY HIGH indicators (>0.7)
        ai_very_high = sum(1 for _, s in ai_scores_list if s > 0.7)
        ai_high = sum(1 for _, s in ai_scores_list if s > 0.6)

        ai_values = [s for _, s in ai_scores_list if s > 0]
        ai_avg = np.mean(ai_values) if ai_values else 0

        # Only add bonus if MULTIPLE very high indicators
        ai_score = ai_avg
        if ai_very_high >= 2:
            ai_score = min(1.0, ai_avg + 0.2)
        elif result.frequency_score > 0.85:  # Very strong GAN fingerprint
            ai_score = min(1.0, ai_avg + 0.15)

        # Store details
        classification['details'] = {
            'iris_natural_score': iris_natural_score,
            'iris_fake_indicators': iris_fake_indicators,
            'faceswap_score': faceswap_score,
            'ai_score': ai_score,
            'faceswap_very_high': faceswap_very_high,
            'ai_very_high': ai_very_high
        }

        # =====================================================
        # BINARY CLASSIFICATION: REAL vs FAKE
        # =====================================================

        is_fake = False
        fake_probability = 0.0

        # Get the maximum manipulation score
        max_manipulation_score = max(faceswap_score, ai_score)

        # =====================================================
        # DECISION RULES (VERY Conservative - Minimize False Positives)
        # =====================================================
        # The goal is to err on the side of "Real" to avoid false accusations

        # RULE 1: Iris looks artificial AND very high manipulation scores
        if not iris_is_natural and max_manipulation_score > 0.70:
            is_fake = True
            fake_probability = max_manipulation_score

        # RULE 2: Very high manipulation scores (regardless of iris)
        # Requires VERY STRONG evidence: score > 0.75 AND multiple very high indicators
        elif max_manipulation_score > 0.75 and (faceswap_very_high >= 3 or ai_very_high >= 3):
            is_fake = True
            fake_probability = max_manipulation_score

        # RULE 3: Extremely high manipulation score (overwhelming evidence)
        elif max_manipulation_score > 0.85:
            is_fake = True
            fake_probability = max_manipulation_score

        # RULE 4: Default to REAL (most images should fall here)
        else:
            is_fake = False
            # Confidence in "Real" is inverse of manipulation evidence
            fake_probability = max_manipulation_score * 0.3  # Heavy discount

        # Cap probability
        fake_probability = min(1.0, max(0.0, fake_probability))

        # Set classification result (BINARY: Real or Fake)
        if is_fake:
            classification['is_fake'] = True
            classification['confidence'] = fake_probability
            classification['fake_type'] = 'Fake'  # Binary - just "Fake"
        else:
            classification['is_fake'] = False
            classification['confidence'] = 1.0 - fake_probability
            classification['fake_type'] = 'Real'

        return classification

    def _create_annotated_image(self, image, landmarks, result):
        """Create annotated image with detection highlights."""
        annotated = image.copy()

        # Draw face oval
        face_pts = landmarks[self.FACE_OVAL].astype(np.int32)
        color = (255, 0, 0) if result.is_fake else (0, 255, 0)
        cv2.polylines(annotated, [face_pts], True, color, 2)

        # Draw eye regions
        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            pt = landmarks[idx].astype(int)
            cv2.circle(annotated, tuple(pt), 1, (255, 255, 0), -1)

        # Draw iris points
        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
            pt = landmarks[idx].astype(int)
            cv2.circle(annotated, tuple(pt), 2, (0, 0, 255), -1)

        # Add result text
        text = f"{'FAKE' if result.is_fake else 'REAL'}: {result.fake_type}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, color, 2)
        cv2.putText(annotated, f"Confidence: {result.confidence*100:.1f}%",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return annotated

    def close(self):
        """Release resources."""
        self.face_mesh.close()


# =============================================================================
# Utility Functions
# =============================================================================

def analyze_image_advanced(image_path: str) -> AdvancedAnalysisResult:
    """Convenience function to analyze a single image."""
    detector = AdvancedDeepfakeDetector()

    image = cv2.imread(image_path)
    if image is None:
        result = AdvancedAnalysisResult()
        result.status = "Failed to load image"
        return result

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect(image)
    detector.close()

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = analyze_image_advanced(image_path)

        print("\n" + "=" * 60)
        print("ADVANCED DEEPFAKE DETECTION RESULT")
        print("=" * 60)
        print(f"Status: {result.status}")
        print(f"Is Fake: {result.is_fake}")
        print(f"Fake Type: {result.fake_type}")
        print(f"Confidence: {result.confidence * 100:.1f}%")

        print("\n--- Iris Analysis ---")
        print(f"Gradient IoU: {result.gradient_iou:.3f}")
        print(f"Left Pupil IoU: {result.left_pupil_iou:.3f}")
        print(f"Right Pupil IoU: {result.right_pupil_iou:.3f}")
        print(f"Pupil Asymmetry: {result.pupil_asymmetry:.3f}")

        print("\n--- Face-Swap Detection ---")
        print(f"Blending Score: {result.blending_score:.3f}")
        print(f"Skin Tone Score: {result.skin_tone_score:.3f}")
        print(f"Lighting Score: {result.lighting_score:.3f}")
        print(f"Edge Artifact Score: {result.edge_artifact_score:.3f}")
        print(f"Color Mismatch Score: {result.color_mismatch_score:.3f}")

        print("\n--- AI-Generated Detection ---")
        print(f"Frequency Score: {result.frequency_score:.3f}")
        print(f"Noise Pattern Score: {result.noise_pattern_score:.3f}")
        print(f"Texture Score: {result.texture_score:.3f}")
        print(f"Symmetry Score: {result.symmetry_score:.3f}")
        print(f"Reflection Score: {result.reflection_score:.3f}")

        print("\n--- Inconsistencies ---")
        for inc in result.inconsistencies:
            print(f"  - {inc}")
    else:
        print("Usage: python advanced_detector.py <image_path>")
