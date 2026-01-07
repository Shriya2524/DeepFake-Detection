"""
Face detection and extraction utilities.

Uses MediaPipe for face detection and provides face cropping functionality.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# Try to import mediapipe, fall back to OpenCV Haar cascade
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from ..utils.logger import get_logger


logger = get_logger(__name__)


class FaceExtractor:
    """
    Face detection and extraction using MediaPipe or OpenCV.
    
    Example:
        extractor = FaceExtractor(method="mediapipe")
        face_crops = extractor.extract_faces(frame)
    """
    
    def __init__(
        self,
        method: str = "mediapipe",
        confidence_threshold: float = 0.5,
        padding: float = 0.2,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        min_face_size: int = 50
    ):
        """
        Initialize face extractor.
        
        Args:
            method: Detection method ("mediapipe" or "opencv_haar")
            confidence_threshold: Minimum confidence for face detection
            padding: Padding around detected face (percentage)
            target_size: Size to resize face crops to (width, height)
            min_face_size: Minimum face size in pixels
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.padding = padding
        self.target_size = target_size
        self.min_face_size = min_face_size
        
        self._init_detector()
    
    def _init_detector(self) -> None:
        """Initialize the face detector based on method."""
        if self.method == "mediapipe" and HAS_MEDIAPIPE:
            try:
                self.mp_face_detection = mp.solutions.face_detection
                self.detector = self.mp_face_detection.FaceDetection(
                    model_selection=1,  # 1 for full range, 0 for short range
                    min_detection_confidence=self.confidence_threshold
                )
                logger.info("Using MediaPipe face detector")
                return
            except (RuntimeError, Exception) as e:
                logger.warning(f"MediaPipe initialization failed: {e}")
                logger.info("Falling back to OpenCV Haar cascade")
        
        # Fall back to OpenCV Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.method = "opencv_haar"
        logger.info("Using OpenCV Haar cascade face detector")
    
    def detect_faces(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image in RGB format (H, W, C)
            
        Returns:
            List of (x, y, width, height, confidence) tuples
        """
        h, w = image.shape[:2]
        faces = []
        
        if self.method == "mediapipe":
            results = self.detector.process(image)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    confidence = detection.score[0]
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    face_w = int(bbox.width * w)
                    face_h = int(bbox.height * h)
                    
                    # Skip small faces
                    if face_w >= self.min_face_size and face_h >= self.min_face_size:
                        faces.append((x, y, face_w, face_h, confidence))
        else:
            # OpenCV Haar cascade
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detections = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            for (x, y, face_w, face_h) in detections:
                faces.append((x, y, face_w, face_h, 1.0))  # No confidence score
        
        return faces
    
    def extract_faces(
        self,
        image: np.ndarray,
        return_coords: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[Tuple]]]:
        """
        Extract face crops from an image.
        
        Args:
            image: Input image in RGB format
            return_coords: Whether to return face coordinates
            
        Returns:
            List of face crop arrays
            If return_coords: tuple of (crops, coordinates)
        """
        h, w = image.shape[:2]
        faces = self.detect_faces(image)
        
        crops = []
        coords = []
        
        for (x, y, face_w, face_h, conf) in faces:
            # Add padding
            pad_w = int(face_w * self.padding)
            pad_h = int(face_h * self.padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + face_w + pad_w)
            y2 = min(h, y + face_h + pad_h)
            
            # Extract and resize
            crop = image[y1:y2, x1:x2]
            
            if self.target_size:
                crop = cv2.resize(crop, self.target_size)
            
            crops.append(crop)
            coords.append((x1, y1, x2, y2, conf))
        
        if return_coords:
            return crops, coords
        
        return crops
    
    def extract_primary_face(
        self,
        image: np.ndarray,
        return_coords: bool = False
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, Tuple]]]:
        """
        Extract the largest/most prominent face from an image.
        
        Args:
            image: Input image in RGB format
            return_coords: Whether to return face coordinates
            
        Returns:
            Face crop array (or None if no face found)
            If return_coords: tuple of (crop, coordinates)
        """
        crops, coords = self.extract_faces(image, return_coords=True)
        
        if not crops:
            return None
        
        # Find largest face
        areas = [(c[2] - c[0]) * (c[3] - c[1]) for c in coords]
        max_idx = np.argmax(areas)
        
        if return_coords:
            return crops[max_idx], coords[max_idx]
        
        return crops[max_idx]
    
    def close(self) -> None:
        """Release resources."""
        if self.method == "mediapipe" and hasattr(self, 'detector'):
            self.detector.close()


def extract_faces_from_frames(
    frames: np.ndarray,
    method: str = "mediapipe",
    target_size: Tuple[int, int] = (224, 224),
    padding: float = 0.2
) -> Tuple[np.ndarray, List[bool]]:
    """
    Extract faces from a sequence of frames.
    
    Args:
        frames: Array of frames (N, H, W, C)
        method: Face detection method
        target_size: Target face crop size
        padding: Padding around face
        
    Returns:
        Tuple of:
        - Array of face crops (M, target_H, target_W, C)
        - List of booleans indicating which frames had faces
    """
    extractor = FaceExtractor(
        method=method,
        target_size=target_size,
        padding=padding
    )
    
    face_crops = []
    has_face = []
    
    for frame in frames:
        face = extractor.extract_primary_face(frame)
        
        if face is not None:
            face_crops.append(face)
            has_face.append(True)
        else:
            # Use blank frame or skip
            has_face.append(False)
    
    extractor.close()
    
    if face_crops:
        return np.array(face_crops), has_face
    else:
        return np.array([]), has_face


def draw_face_boxes(
    image: np.ndarray,
    coords: List[Tuple],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes around detected faces.
    
    Args:
        image: Input image
        coords: List of (x1, y1, x2, y2, confidence) tuples
        color: Box color (RGB)
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    image = image.copy()
    
    for (x1, y1, x2, y2, conf) in coords:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{conf:.2f}"
        cv2.putText(
            image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness
        )
    
    return image

