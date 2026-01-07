"""
Video reading and frame extraction utilities.

Handles loading videos, sampling frames at configurable FPS,
and basic video preprocessing.
"""

import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..utils.logger import get_logger


logger = get_logger(__name__)


class VideoReader:
    """
    Video reader with configurable frame sampling.
    
    Example:
        reader = VideoReader(target_fps=10, max_frames=32)
        frames = reader.read("video.mp4")
    """
    
    def __init__(
        self,
        target_fps: int = 10,
        max_frames: int = 32,
        min_frames: int = 8,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = False
    ):
        """
        Initialize video reader.
        
        Args:
            target_fps: Target frames per second for sampling
            max_frames: Maximum number of frames to extract
            min_frames: Minimum frames required (raises error if not met)
            resize: Optional (width, height) to resize frames
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.resize = resize
        self.normalize = normalize
    
    def read(
        self,
        video_path: Union[str, Path],
        return_timestamps: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Read and sample frames from a video.
        
        Args:
            video_path: Path to video file
            return_timestamps: Whether to return frame timestamps
            
        Returns:
            Array of frames (N, H, W, C) in RGB format
            If return_timestamps: tuple of (frames, timestamps)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video has fewer than min_frames
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps == 0:
            logger.warning(f"Could not get FPS for {video_path}, using 30")
            video_fps = 30
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(video_fps / self.target_fps))
        
        frames = []
        timestamps = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample at target FPS
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if specified
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                
                # Normalize if specified
                if self.normalize:
                    frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
                timestamps.append(frame_idx / video_fps)
                
                # Stop if we have enough frames
                if len(frames) >= self.max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        # Check minimum frames
        if len(frames) < self.min_frames:
            raise ValueError(
                f"Video has {len(frames)} frames, minimum required: {self.min_frames}"
            )
        
        frames_array = np.array(frames)
        
        if return_timestamps:
            return frames_array, np.array(timestamps)
        
        return frames_array
    
    def read_generator(
        self,
        video_path: Union[str, Path]
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Read frames as a generator (memory efficient for large videos).
        
        Args:
            video_path: Path to video file
            
        Yields:
            Tuple of (frame, timestamp)
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(video_fps / self.target_fps))
        
        frame_idx = 0
        frames_yielded = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret or frames_yielded >= self.max_frames:
                break
            
            if frame_idx % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                
                if self.normalize:
                    frame = frame.astype(np.float32) / 255.0
                
                yield frame, frame_idx / video_fps
                frames_yielded += 1
            
            frame_idx += 1
        
        cap.release()
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(str(video_path))
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": None
        }
        
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        
        cap.release()
        return info


def extract_frames(
    video_path: Union[str, Path],
    target_fps: int = 10,
    max_frames: int = 32,
    resize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Convenience function to extract frames from a video.
    
    Args:
        video_path: Path to video file
        target_fps: Target FPS for sampling
        max_frames: Maximum frames to extract
        resize: Optional resize dimensions
        
    Returns:
        Array of frames (N, H, W, C) in RGB
    """
    reader = VideoReader(
        target_fps=target_fps,
        max_frames=max_frames,
        resize=resize
    )
    return reader.read(video_path)


def sample_frames_uniformly(
    video_path: Union[str, Path],
    num_frames: int = 16,
    resize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Sample frames uniformly across the video duration.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        resize: Optional resize dimensions
        
    Returns:
        Array of uniformly sampled frames
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Calculate uniform sample indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
    
    cap.release()
    
    return np.array(frames)

