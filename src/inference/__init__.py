"""
Inference modules for deepfake detection.
"""

from .pipeline import DeepfakeDetectionPipeline
from .detect_video import detect_video
from .detect_image import detect_image

__all__ = [
    "DeepfakeDetectionPipeline",
    "detect_video",
    "detect_image",
]

