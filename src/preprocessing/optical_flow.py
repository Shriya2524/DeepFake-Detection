"""
Optical flow computation for temporal analysis.

Computes dense optical flow between consecutive frames to capture
motion patterns and temporal inconsistencies.
"""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from ..utils.logger import get_logger


logger = get_logger(__name__)


class OpticalFlowExtractor:
    """
    Optical flow extractor using Farneback or Lucas-Kanade method.
    
    Example:
        extractor = OpticalFlowExtractor(method="farneback")
        flows = extractor.compute_sequence(frames)
    """
    
    def __init__(
        self,
        method: str = "farneback",
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        normalize_output: bool = True
    ):
        """
        Initialize optical flow extractor.
        
        Args:
            method: Flow method ("farneback" or "lucas_kanade")
            pyr_scale: Pyramid scale for Farneback
            levels: Number of pyramid levels
            winsize: Averaging window size
            iterations: Number of iterations at each level
            poly_n: Polynomial neighborhood size
            poly_sigma: Gaussian standard deviation
            normalize_output: Whether to normalize flow values
        """
        self.method = method
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.normalize_output = normalize_output
    
    def compute(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        Compute optical flow between two frames.
        
        Args:
            frame1: First frame (H, W, C) or (H, W)
            frame2: Second frame (H, W, C) or (H, W)
            
        Returns:
            Optical flow array (H, W, 2) with x, y components
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = frame1
        
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = frame2
        
        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.winsize,
                iterations=self.iterations,
                poly_n=self.poly_n,
                poly_sigma=self.poly_sigma,
                flags=0
            )
        else:
            # Lucas-Kanade (sparse flow, less common for this use case)
            # We'll use Farneback as fallback
            logger.warning("Lucas-Kanade selected, using Farneback instead for dense flow")
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
        
        if self.normalize_output:
            flow = self._normalize_flow(flow)
        
        return flow
    
    def _normalize_flow(self, flow: np.ndarray) -> np.ndarray:
        """
        Normalize optical flow to [-1, 1] range.
        
        Args:
            flow: Raw optical flow (H, W, 2)
            
        Returns:
            Normalized flow
        """
        # Get magnitude for normalization
        mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        max_mag = np.percentile(mag, 99) + 1e-6  # Avoid division by zero
        
        normalized = flow / max_mag
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def compute_sequence(
        self,
        frames: np.ndarray
    ) -> np.ndarray:
        """
        Compute optical flow for a sequence of frames.
        
        Args:
            frames: Array of frames (N, H, W, C)
            
        Returns:
            Array of optical flows (N-1, H, W, 2)
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to compute optical flow")
        
        flows = []
        
        for i in range(len(frames) - 1):
            flow = self.compute(frames[i], frames[i + 1])
            flows.append(flow)
        
        return np.array(flows)
    
    def compute_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute magnitude of optical flow.
        
        Args:
            flow: Optical flow (H, W, 2)
            
        Returns:
            Magnitude array (H, W)
        """
        return np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    
    def compute_angle(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute angle/direction of optical flow.
        
        Args:
            flow: Optical flow (H, W, 2)
            
        Returns:
            Angle array (H, W) in radians
        """
        return np.arctan2(flow[:, :, 1], flow[:, :, 0])


def compute_optical_flow(
    frames: np.ndarray,
    method: str = "farneback"
) -> np.ndarray:
    """
    Convenience function to compute optical flow for frame sequence.
    
    Args:
        frames: Array of frames (N, H, W, C)
        method: Flow computation method
        
    Returns:
        Array of flows (N-1, H, W, 2)
    """
    extractor = OpticalFlowExtractor(method=method)
    return extractor.compute_sequence(frames)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to RGB visualization.
    
    Uses HSV color wheel where:
    - Hue represents direction
    - Saturation is constant
    - Value represents magnitude
    
    Args:
        flow: Optical flow (H, W, 2)
        
    Returns:
        RGB visualization (H, W, 3)
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Compute magnitude and angle
    mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    ang = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    
    # Map angle to hue (0-180 in OpenCV)
    hsv[:, :, 0] = (ang + np.pi) * 180 / (2 * np.pi)
    
    # Saturation is constant
    hsv[:, :, 1] = 255
    
    # Value is magnitude (normalized)
    mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = mag_normalized.astype(np.uint8)
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def compute_flow_statistics(flows: np.ndarray) -> dict:
    """
    Compute statistics from optical flow sequence.
    
    Useful for detecting temporal inconsistencies.
    
    Args:
        flows: Sequence of optical flows (N, H, W, 2)
        
    Returns:
        Dictionary of flow statistics
    """
    # Compute magnitudes
    magnitudes = np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2)
    
    # Per-frame statistics
    frame_means = magnitudes.mean(axis=(1, 2))
    frame_stds = magnitudes.std(axis=(1, 2))
    frame_maxs = magnitudes.max(axis=(1, 2))
    
    # Temporal consistency (difference between consecutive flow magnitudes)
    temporal_diff = np.abs(np.diff(frame_means))
    
    return {
        "mean_magnitude": float(magnitudes.mean()),
        "std_magnitude": float(magnitudes.std()),
        "max_magnitude": float(magnitudes.max()),
        "frame_means": frame_means.tolist(),
        "frame_stds": frame_stds.tolist(),
        "temporal_consistency": float(temporal_diff.mean()),
        "temporal_variance": float(temporal_diff.var()),
        "num_flows": len(flows)
    }

