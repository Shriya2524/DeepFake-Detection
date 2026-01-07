"""
Frequency domain analysis for GAN fingerprinting.

Implements DCT, FFT, and high-frequency extraction to detect
GAN-generated artifacts that are typically visible in frequency domain.
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from scipy import fftpack

from ..utils.logger import get_logger


logger = get_logger(__name__)


class FrequencyAnalyzer:
    """
    Frequency domain analyzer for detecting GAN artifacts.
    
    GANs often leave characteristic patterns in the frequency domain,
    particularly in high-frequency components.
    
    Example:
        analyzer = FrequencyAnalyzer()
        dct_features = analyzer.compute_dct(image)
        high_freq = analyzer.extract_high_frequency(image)
    """
    
    def __init__(
        self,
        dct_size: int = 224,
        high_freq_threshold: float = 0.3,
        normalize: bool = True
    ):
        """
        Initialize frequency analyzer.
        
        Args:
            dct_size: Size for DCT computation
            high_freq_threshold: Threshold for high-frequency extraction
                                 (top N% of frequencies)
            normalize: Whether to normalize output
        """
        self.dct_size = dct_size
        self.high_freq_threshold = high_freq_threshold
        self.normalize = normalize
    
    def compute_dct(self, image: np.ndarray) -> np.ndarray:
        """
        Compute 2D DCT (Discrete Cosine Transform) of an image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            DCT coefficients (H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Resize if needed
        if gray.shape[0] != self.dct_size or gray.shape[1] != self.dct_size:
            gray = cv2.resize(gray, (self.dct_size, self.dct_size))
        
        # Convert to float
        gray = gray.astype(np.float32)
        
        # Compute 2D DCT
        dct = fftpack.dct(fftpack.dct(gray.T, norm='ortho').T, norm='ortho')
        
        if self.normalize:
            dct = self._normalize(dct)
        
        return dct
    
    def compute_fft(self, image: np.ndarray) -> np.ndarray:
        """
        Compute 2D FFT magnitude spectrum.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            FFT magnitude spectrum (H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float32)
        
        # Compute FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Get magnitude spectrum (log scale for better visualization)
        magnitude = np.log1p(np.abs(fft_shift))
        
        if self.normalize:
            magnitude = self._normalize(magnitude)
        
        return magnitude
    
    def extract_high_frequency(self, image: np.ndarray) -> np.ndarray:
        """
        Extract high-frequency components using high-pass filter.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            High-frequency components (H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float32)
        
        # Apply Gaussian blur to get low frequencies
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        
        # High frequency = original - low frequency
        high_freq = gray - blur
        
        if self.normalize:
            high_freq = self._normalize(high_freq)
        
        return high_freq
    
    def compute_dct_histogram(
        self,
        image: np.ndarray,
        num_bins: int = 64
    ) -> np.ndarray:
        """
        Compute histogram of DCT coefficients.
        
        Useful as a feature vector for classification.
        
        Args:
            image: Input image
            num_bins: Number of histogram bins
            
        Returns:
            Histogram array (num_bins,)
        """
        dct = self.compute_dct(image)
        
        # Flatten and compute histogram
        hist, _ = np.histogram(dct.flatten(), bins=num_bins, density=True)
        
        return hist
    
    def compute_spectral_features(self, image: np.ndarray) -> dict:
        """
        Compute various spectral features for classification.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of spectral features
        """
        dct = self.compute_dct(image)
        fft_mag = self.compute_fft(image)
        high_freq = self.extract_high_frequency(image)
        
        # DCT statistics
        dct_flat = dct.flatten()
        
        # Divide into frequency bands (low, mid, high)
        h, w = dct.shape
        low_mask = np.zeros((h, w), dtype=bool)
        mid_mask = np.zeros((h, w), dtype=bool)
        high_mask = np.zeros((h, w), dtype=bool)
        
        # Create distance matrix from top-left (DC component)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt(x**2 + y**2) / np.sqrt(h**2 + w**2)
        
        low_mask = dist < 0.3
        mid_mask = (dist >= 0.3) & (dist < 0.6)
        high_mask = dist >= 0.6
        
        return {
            "dct_mean": float(np.mean(np.abs(dct_flat))),
            "dct_std": float(np.std(dct_flat)),
            "dct_energy": float(np.sum(dct**2)),
            "low_freq_energy": float(np.sum(dct[low_mask]**2)),
            "mid_freq_energy": float(np.sum(dct[mid_mask]**2)),
            "high_freq_energy": float(np.sum(dct[high_mask]**2)),
            "high_freq_ratio": float(np.sum(dct[high_mask]**2) / (np.sum(dct**2) + 1e-8)),
            "fft_mean": float(np.mean(fft_mag)),
            "fft_std": float(np.std(fft_mag)),
            "high_freq_mean": float(np.mean(np.abs(high_freq))),
            "high_freq_std": float(np.std(high_freq)),
        }
    
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        arr_min = arr.min()
        arr_max = arr.max()
        
        if arr_max - arr_min > 1e-8:
            return (arr - arr_min) / (arr_max - arr_min)
        else:
            return np.zeros_like(arr)


def compute_dct_features(image: np.ndarray) -> np.ndarray:
    """
    Convenience function to compute DCT features.
    
    Args:
        image: Input image
        
    Returns:
        DCT coefficients
    """
    analyzer = FrequencyAnalyzer()
    return analyzer.compute_dct(image)


def compute_edge_features(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute edge features using Sobel and Laplacian filters.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        
    Returns:
        Tuple of (sobel_x, sobel_y, laplacian)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)
    
    # Sobel filters
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    
    # Normalize to [0, 1]
    def norm(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-8:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)
    
    return norm(sobel_x), norm(sobel_y), norm(laplacian)


def convert_color_spaces(image: np.ndarray) -> dict:
    """
    Convert image to multiple color spaces.
    
    Args:
        image: Input image in RGB format (H, W, 3)
        
    Returns:
        Dictionary with different color space representations
    """
    # Ensure RGB input
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input must be RGB image (H, W, 3)")
    
    # Convert to various color spaces
    ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    return {
        "rgb": image,
        "ycbcr": ycbcr,
        "y": ycbcr[:, :, 0],
        "cb": ycbcr[:, :, 1],
        "cr": ycbcr[:, :, 2],
        "hsv": hsv,
        "h": hsv[:, :, 0],
        "s": hsv[:, :, 1],
        "v": hsv[:, :, 2],
        "lab": lab,
        "l": lab[:, :, 0],
        "a": lab[:, :, 1],
        "b": lab[:, :, 2],
    }


def create_multichannel_input(
    image: np.ndarray,
    include_ycbcr: bool = True,
    include_frequency: bool = True,
    include_edges: bool = True
) -> np.ndarray:
    """
    Create multi-channel input for GAN fingerprint detection.
    
    Combines RGB, color space channels, frequency features, and edge features.
    
    Args:
        image: Input image in RGB format (H, W, 3)
        include_ycbcr: Include Cb and Cr channels
        include_frequency: Include DCT high-frequency channel
        include_edges: Include Sobel and Laplacian channels
        
    Returns:
        Multi-channel array (H, W, C)
    """
    h, w = image.shape[:2]
    channels = [image]  # Start with RGB (3 channels)
    
    if include_ycbcr:
        color_spaces = convert_color_spaces(image)
        # Add Cb and Cr channels (2 channels)
        cb = color_spaces["cb"][:, :, np.newaxis].astype(np.float32) / 255.0
        cr = color_spaces["cr"][:, :, np.newaxis].astype(np.float32) / 255.0
        channels.extend([cb, cr])
    
    if include_frequency:
        # Add DCT/high-frequency channel (1 channel)
        analyzer = FrequencyAnalyzer()
        high_freq = analyzer.extract_high_frequency(image)
        high_freq = cv2.resize(high_freq, (w, h))[:, :, np.newaxis]
        channels.append(high_freq.astype(np.float32))
    
    if include_edges:
        # Add Sobel and Laplacian channels (3 channels)
        sobel_x, sobel_y, laplacian = compute_edge_features(image)
        sobel_x = cv2.resize(sobel_x, (w, h))[:, :, np.newaxis]
        sobel_y = cv2.resize(sobel_y, (w, h))[:, :, np.newaxis]
        laplacian = cv2.resize(laplacian, (w, h))[:, :, np.newaxis]
        channels.extend([
            sobel_x.astype(np.float32),
            sobel_y.astype(np.float32),
            laplacian.astype(np.float32)
        ])
    
    # Normalize RGB to [0, 1] if needed
    if channels[0].max() > 1:
        channels[0] = channels[0].astype(np.float32) / 255.0
    
    return np.concatenate(channels, axis=2)

