"""
End-to-end inference pipeline for deepfake detection.

Combines all preprocessing, models, and post-processing into
a single easy-to-use pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..preprocessing.video_reader import VideoReader
from ..preprocessing.face_extraction import FaceExtractor
from ..preprocessing.optical_flow import OpticalFlowExtractor
from ..preprocessing.frequency_analysis import FrequencyAnalyzer, compute_edge_features
from ..models.spatiotemporal_model import SpatiotemporalModel, create_spatiotemporal_model
from ..models.gan_fingerprint_model import GANFingerprintModel, create_gan_fingerprint_model
from ..models.gan_family_model import GANFamilyClassifier, create_gan_family_model, DEFAULT_GAN_FAMILY_CLASSES
from ..models.fusion_model import FusionModel, create_fusion_model
from ..utils.config import load_config
from ..utils.device import get_device
from ..utils.logger import get_logger


logger = get_logger(__name__)


class DeepfakeDetectionPipeline:
    """
    Complete deepfake detection pipeline.
    
    Handles video and image inputs, preprocessing, model inference,
    and result post-processing.
    
    Example:
        pipeline = DeepfakeDetectionPipeline(config_path="config/default_config.yaml")
        pipeline.load_weights(
            spatiotemporal="checkpoints/spatiotemporal/best_model.pt",
            gan_fingerprint="checkpoints/gan_fingerprint/best_model.pt",
            fusion="checkpoints/fusion/best_model.pt"
        )
        
        result = pipeline.detect_video("path/to/video.mp4")
        print(f"Label: {result['label']}, Probability: {result['probability']:.2f}")
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize detection pipeline.
        
        Args:
            config_path: Path to configuration file
            device: Device to run inference on
        """
        # Load config
        if config_path:
            self.config = load_config(config_path)
        else:
            # Use defaults
            self.config = None
        
        # Setup device
        self.device = get_device(device)
        
        # Initialize preprocessors
        self._init_preprocessors()
        
        # Initialize models (loaded later)
        self.spatiotemporal_model = None
        self.gan_fingerprint_model = None
        self.gan_family_model = None
        self.fusion_model = None
        
        # Inference settings
        self.threshold = 0.5
        
        logger.info(f"Pipeline initialized on {self.device}")
    
    def _init_preprocessors(self) -> None:
        """Initialize preprocessing components."""
        if self.config:
            target_fps = self.config.preprocessing.target_fps
            max_frames = self.config.preprocessing.max_frames
            image_size = self.config.preprocessing.image_size
        else:
            target_fps = 10
            max_frames = 32
            image_size = 224
        
        self.video_reader = VideoReader(
            target_fps=target_fps,
            max_frames=max_frames,
            min_frames=4
        )
        
        self.face_extractor = FaceExtractor(
            target_size=(image_size, image_size),
            padding=0.2
        )
        
        self.flow_extractor = OpticalFlowExtractor()
        self.freq_analyzer = FrequencyAnalyzer()
        
        self.image_size = image_size
        self.max_frames = max_frames
    
    def load_weights(
        self,
        spatiotemporal: Optional[str] = None,
        gan_fingerprint: Optional[str] = None,
        gan_family: Optional[str] = None,
        fusion: Optional[str] = None
    ) -> None:
        """
        Load model weights.
        
        Args:
            spatiotemporal: Path to spatiotemporal model weights
            gan_fingerprint: Path to GAN fingerprint model weights
            gan_family: Path to GAN family classifier weights
            fusion: Path to fusion model weights
        """
        # Load spatiotemporal branch if provided
        if spatiotemporal is not None:
            self.spatiotemporal_model = (
                create_spatiotemporal_model(self.config)
                if self.config else SpatiotemporalModel()
            )
            state_dict = torch.load(spatiotemporal, map_location=self.device, weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.spatiotemporal_model.load_state_dict(state_dict)
            self.spatiotemporal_model = self.spatiotemporal_model.to(self.device).eval()
            logger.info(f"Loaded spatiotemporal weights: {spatiotemporal}")
        elif self.spatiotemporal_model is not None:
            # Ensure already-loaded model is on the right device
            self.spatiotemporal_model = self.spatiotemporal_model.to(self.device).eval()
        
        # Load GAN fingerprint branch (required for both image and video)
        if gan_fingerprint is not None:
            self.gan_fingerprint_model = (
                create_gan_fingerprint_model(self.config)
                if self.config else GANFingerprintModel()
            )
            state_dict = torch.load(gan_fingerprint, map_location=self.device, weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.gan_fingerprint_model.load_state_dict(state_dict)
            self.gan_fingerprint_model = self.gan_fingerprint_model.to(self.device).eval()
            logger.info(f"Loaded GAN fingerprint weights: {gan_fingerprint}")
        elif self.gan_fingerprint_model is not None:
            self.gan_fingerprint_model = self.gan_fingerprint_model.to(self.device).eval()
        
        # Load GAN family classifier if provided
        if gan_family is not None:
            # Try to load class names from checkpoint if available
            checkpoint = torch.load(gan_family, map_location=self.device, weights_only=False)
            class_names = None
            if isinstance(checkpoint, dict):
                class_names = checkpoint.get('class_names', None)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            else:
                state_dict = checkpoint
            
            self.gan_family_model = (
                create_gan_family_model(self.config, class_names=class_names)
                if self.config else GANFamilyClassifier(class_names=class_names)
            )
            self.gan_family_model.load_state_dict(state_dict)
            self.gan_family_model = self.gan_family_model.to(self.device).eval()
            logger.info(f"Loaded GAN family classifier weights: {gan_family}")
            logger.info(f"GAN family classes: {self.gan_family_model.get_class_names()}")
        elif self.gan_family_model is not None:
            self.gan_family_model = self.gan_family_model.to(self.device).eval()
        
        # Load fusion head if provided
        if fusion is not None:
            self.fusion_model = (
                create_fusion_model(self.config)
                if self.config else FusionModel()
            )
            state_dict = torch.load(fusion, map_location=self.device, weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.fusion_model.load_state_dict(state_dict)
            self.fusion_model = self.fusion_model.to(self.device).eval()
            logger.info(f"Loaded fusion weights: {fusion}")
        elif self.fusion_model is not None:
            self.fusion_model = self.fusion_model.to(self.device).eval()
    
    def preprocess_video(
        self,
        video_path: Union[str, Path]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess video for inference.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with preprocessed tensors
        """
        # Read frames
        frames = self.video_reader.read(video_path)
        
        # Extract faces
        face_frames = []
        for frame in frames:
            face = self.face_extractor.extract_primary_face(frame)
            if face is not None:
                face_frames.append(face)
            else:
                # Center crop fallback
                h, w = frame.shape[:2]
                size = min(h, w)
                y, x = (h - size) // 2, (w - size) // 2
                crop = frame[y:y+size, x:x+size]
                crop = cv2.resize(crop, (self.image_size, self.image_size))
                face_frames.append(crop)
        
        face_frames = np.array(face_frames)
        
        # Compute optical flow
        flow = self.flow_extractor.compute_sequence(face_frames)
        
        # Create multi-channel frames for GAN fingerprint
        multichannel_frames = []
        for frame in face_frames:
            mc = self._create_multichannel(frame)
            multichannel_frames.append(mc)
        multichannel_frames = np.stack(multichannel_frames)
        
        # Normalize RGB frames
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        normalized_frames = face_frames.astype(np.float32) / 255.0
        normalized_frames = (normalized_frames - mean) / std
        
        # Pad to max_frames
        T = len(face_frames)
        if T < self.max_frames:
            pad = self.max_frames - T
            normalized_frames = np.pad(
                normalized_frames,
                ((0, pad), (0, 0), (0, 0), (0, 0)),
                mode='constant'
            )
            multichannel_frames = np.pad(
                multichannel_frames,
                ((0, pad), (0, 0), (0, 0), (0, 0)),
                mode='constant'
            )
            flow = np.pad(
                flow,
                ((0, pad), (0, 0), (0, 0), (0, 0)),
                mode='constant'
            )
            mask = np.concatenate([np.ones(T), np.zeros(pad)])
        else:
            normalized_frames = normalized_frames[:self.max_frames]
            multichannel_frames = multichannel_frames[:self.max_frames]
            flow = flow[:self.max_frames - 1]
            mask = np.ones(self.max_frames)
        
        # Convert to tensors
        frames_tensor = torch.from_numpy(normalized_frames).permute(0, 3, 1, 2).float()
        mc_tensor = torch.from_numpy(multichannel_frames).permute(0, 3, 1, 2).float()
        flow_tensor = torch.from_numpy(flow).permute(0, 3, 1, 2).float()
        mask_tensor = torch.from_numpy(mask).bool()
        
        return {
            'frames': frames_tensor.unsqueeze(0),
            'multichannel': mc_tensor.unsqueeze(0),
            'flow': flow_tensor.unsqueeze(0),
            'mask': mask_tensor.unsqueeze(0),
            'original_frames': face_frames
        }
    
    def preprocess_image(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess image for inference.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Dictionary with preprocessed tensors
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image
        
        # Extract face
        face = self.face_extractor.extract_primary_face(img)
        if face is None:
            # Center crop fallback
            h, w = img.shape[:2]
            size = min(h, w)
            y, x = (h - size) // 2, (w - size) // 2
            face = img[y:y+size, x:x+size]
            face = cv2.resize(face, (self.image_size, self.image_size))
        
        # Create multi-channel input
        multichannel = self._create_multichannel(face)
        
        # Normalize RGB
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = face.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std
        
        # Convert to tensors
        image_tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        mc_tensor = torch.from_numpy(multichannel).permute(2, 0, 1).float()
        
        return {
            'image': image_tensor.unsqueeze(0),
            'multichannel': mc_tensor.unsqueeze(0),
            'original_image': face
        }
    
    def _create_multichannel(self, image: np.ndarray) -> np.ndarray:
        """Create multi-channel input for GAN fingerprint model."""
        # RGB
        rgb = image.astype(np.float32) / 255.0
        
        # YCbCr
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        cb = ycbcr[:, :, 1:2].astype(np.float32) / 255.0
        cr = ycbcr[:, :, 2:3].astype(np.float32) / 255.0
        
        # DCT high-frequency
        high_freq = self.freq_analyzer.extract_high_frequency(image)
        high_freq = high_freq[:, :, np.newaxis]
        
        # Edge features
        sobel_x, sobel_y, laplacian = compute_edge_features(image)
        sobel_x = sobel_x[:, :, np.newaxis]
        sobel_y = sobel_y[:, :, np.newaxis]
        laplacian = laplacian[:, :, np.newaxis]
        
        # Concatenate all channels
        multichannel = np.concatenate([
            rgb, cb, cr, high_freq, sobel_x, sobel_y, laplacian
        ], axis=2)
        
        return multichannel.astype(np.float32)
    
    @torch.no_grad()
    def detect_video(
        self,
        video_path: Union[str, Path],
        return_frame_scores: bool = False,
        return_frames: bool = False
    ) -> Dict:
        """
        Detect deepfake in a video.
        
        Args:
            video_path: Path to video file
            return_frame_scores: Whether to return per-frame scores
            
        Returns:
            Dictionary with detection results
        """
        if self.gan_fingerprint_model is None and self.spatiotemporal_model is None:
            raise RuntimeError(
                "Models not loaded. Call load_weights() with at least GAN fingerprint weights."
            )
        
        # Preprocess
        data = self.preprocess_video(video_path)
        
        frames = data['frames'].to(self.device)
        mc_frames = data['multichannel'].to(self.device)
        flow = data['flow'].to(self.device)
        mask = data['mask'].to(self.device)
        
        B, T, C, H, W = frames.shape
        
        use_spatiotemporal = self.spatiotemporal_model is not None
        use_fusion = use_spatiotemporal and self.fusion_model is not None
        
        video_fake_prob = None
        attention_scores = None
        
        if use_spatiotemporal:
            spatio_out = self.spatiotemporal_model(frames, flow, mask)
            video_fake_prob = spatio_out['probs'][0, 1].item()
            attention_scores = spatio_out.get('frame_scores')
        
        if self.gan_fingerprint_model is None:
            raise RuntimeError("GAN fingerprint model not loaded.")
        
        # GAN fingerprint prediction (per frame)
        mc_flat = mc_frames.view(B * T, -1, H, W)
        gan_out = self.gan_fingerprint_model(mc_flat)
        frame_probs = gan_out['probs'].view(B, T, -1)
        frame_fake_probs = frame_probs[0, :, 1]  # (T,)
        
        # GAN family classification if available
        gan_family = None
        gan_family_confidence = None
        family_probs = None
        
        if self.gan_family_model is not None:
            # Run per-frame family classification
            family_out = self.gan_family_model(mc_flat)
            family_frame_probs = family_out['probs'].view(B, T, -1)  # (B, T, num_classes)
            
            # Aggregate per-frame family probs over valid frames (mean)
            valid_mask = mask[0].float()  # (T,)
            valid_count = valid_mask.sum()
            
            if valid_count > 0:
                # Weight by valid mask and average
                weighted_probs = family_frame_probs[0] * valid_mask.unsqueeze(-1)  # (T, num_classes)
                mean_family_probs = weighted_probs.sum(dim=0) / valid_count  # (num_classes,)
            else:
                mean_family_probs = family_frame_probs[0].mean(dim=0)  # (num_classes,)
            
            # Get top family prediction
            top_idx = torch.argmax(mean_family_probs).item()
            class_names = self.gan_family_model.get_class_names()
            gan_family = class_names[top_idx]
            gan_family_confidence = mean_family_probs[top_idx].item() * 100  # As percentage
            
            # Build family_probs dict
            family_probs = {
                name: mean_family_probs[i].item() * 100
                for i, name in enumerate(class_names)
            }
        
        # Mask invalid frames
        valid_mask = mask[0].float()
        valid_count = valid_mask.sum()
        if valid_count > 0:
            masked_probs = frame_fake_probs * valid_mask
            mean_prob_tensor = masked_probs.sum() / valid_count
            var_tensor = ((frame_fake_probs - mean_prob_tensor) ** 2 * valid_mask).sum() / valid_count
        else:
            masked_probs = frame_fake_probs
            mean_prob_tensor = frame_fake_probs.mean()
            var_tensor = torch.tensor(0.0, device=self.device)
        
        std_prob = torch.sqrt(var_tensor)
        max_prob = frame_fake_probs[mask[0]].max().item() if mask[0].any() else frame_fake_probs.max().item()
        min_prob = frame_fake_probs[mask[0]].min().item() if mask[0].any() else frame_fake_probs.min().item()
        mean_prob = mean_prob_tensor.item()
        
        # Combine scores
        if use_fusion:
            # Create fusion features tensor
            fusion_features = torch.stack([
                torch.tensor(video_fake_prob, device=self.device),
                mean_prob_tensor,
                std_prob,
                torch.tensor(max_prob, device=self.device),
                torch.tensor(min_prob, device=self.device)
            ], dim=0).unsqueeze(0)  # (1, 5)
            
            fusion_out = self.fusion_model.forward_features(fusion_features)
            final_prob = fusion_out['probs'][0, 1].item()
            score_source = "fusion"
        elif use_spatiotemporal:
            # Average temporal and frame-level scores when fusion head is unavailable
            final_prob = float((video_fake_prob + mean_prob) / 2.0)
            score_source = "temporal+gan-avg"
        else:
            final_prob = mean_prob
            score_source = "gan-only"
        
        # Determine label
        label = "DEEPFAKE" if final_prob > self.threshold else "REAL"
        confidence = abs(final_prob - 0.5) * 2
        
        result = {
            'label': label,
            'probability': final_prob,
            'confidence': confidence,
            'video_score': video_fake_prob if video_fake_prob is not None else mean_prob,
            'mean_frame_score': mean_prob,
            'std_frame_score': std_prob.item(),
            'max_frame_score': max_prob,
            'min_frame_score': min_prob,
            'threshold': self.threshold,
            'score_source': score_source,
            'gan_family': gan_family,
            'gan_family_confidence': gan_family_confidence,
            'family_probs': family_probs
        }
        
        if return_frame_scores:
            result['frame_scores'] = frame_fake_probs.cpu().numpy()
            if attention_scores is not None:
                result['attention_scores'] = attention_scores[0].cpu().numpy()
            if return_frames and 'original_frames' in data:
                result['frames'] = data['original_frames']
        
        return result
    
    @torch.no_grad()
    def detect_image(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> Dict:
        """
        Detect deepfake in an image.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Dictionary with detection results
        """
        if self.gan_fingerprint_model is None:
            raise RuntimeError("Models not loaded. Call load_weights() first.")
        
        # Preprocess
        data = self.preprocess_image(image)
        mc_input = data['multichannel'].to(self.device)
        
        # GAN fingerprint prediction
        gan_out = self.gan_fingerprint_model(mc_input)
        fake_prob = gan_out['probs'][0, 1].item()
        
        # GAN family classification if available
        gan_family = None
        gan_family_confidence = None
        family_probs = None
        
        if self.gan_family_model is not None:
            family_out = self.gan_family_model(mc_input)
            probs = family_out['probs'][0]  # (num_classes,)
            
            # Get top family prediction
            top_idx = torch.argmax(probs).item()
            class_names = self.gan_family_model.get_class_names()
            gan_family = class_names[top_idx]
            gan_family_confidence = probs[top_idx].item() * 100  # As percentage
            
            # Build family_probs dict
            family_probs = {
                name: probs[i].item() * 100
                for i, name in enumerate(class_names)
            }
        
        # Determine label
        label = "DEEPFAKE" if fake_prob > self.threshold else "REAL"
        confidence = abs(fake_prob - 0.5) * 2
        
        return {
            'label': label,
            'probability': fake_prob,
            'confidence': confidence,
            'threshold': self.threshold,
            'gan_score': fake_prob,
            'score_source': "gan-only",
            'gan_family': gan_family,
            'gan_family_confidence': gan_family_confidence,
            'family_probs': family_probs
        }
    
    def set_threshold(self, threshold: float) -> None:
        """Set detection threshold."""
        self.threshold = threshold

