"""
Tests for model modules.
"""

import numpy as np
import pytest
import torch


class TestBackbones:
    """Tests for backbone models."""
    
    def test_get_backbone_resnet18(self):
        """Test ResNet18 backbone creation."""
        from src.models.backbones import get_backbone
        
        backbone = get_backbone(
            backbone_type="resnet18",
            pretrained=False,
            in_channels=3
        )
        
        assert backbone is not None
        assert backbone.get_feature_dim() == 512
    
    def test_get_backbone_with_custom_channels(self):
        """Test backbone with custom input channels."""
        from src.models.backbones import get_backbone
        
        backbone = get_backbone(
            backbone_type="resnet18",
            pretrained=False,
            in_channels=9
        )
        
        # Test forward pass
        x = torch.randn(2, 9, 224, 224)
        features = backbone(x)
        
        assert features.shape == (2, 512)
    
    def test_optical_flow_encoder(self):
        """Test optical flow encoder."""
        from src.models.backbones import OpticalFlowEncoder
        
        encoder = OpticalFlowEncoder(in_channels=2, feature_dim=128)
        
        x = torch.randn(2, 2, 224, 224)
        features = encoder(x)
        
        assert features.shape == (2, 128)


class TestSpatiotemporalModel:
    """Tests for spatiotemporal model."""
    
    def test_model_creation(self):
        """Test model initialization."""
        from src.models.spatiotemporal_model import SpatiotemporalModel
        
        model = SpatiotemporalModel(
            backbone="resnet18",
            backbone_pretrained=False,
            hidden_size=128,
            num_layers=1,
            use_optical_flow=False
        )
        
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass without optical flow."""
        from src.models.spatiotemporal_model import SpatiotemporalModel
        
        model = SpatiotemporalModel(
            backbone="resnet18",
            backbone_pretrained=False,
            hidden_size=64,
            num_layers=1,
            bidirectional=False,
            use_optical_flow=False
        )
        model.eval()
        
        # Batch of 2 videos, 8 frames each
        frames = torch.randn(2, 8, 3, 224, 224)
        
        with torch.no_grad():
            output = model(frames)
        
        assert 'logits' in output
        assert 'probs' in output
        assert output['logits'].shape == (2, 2)
        assert output['probs'].shape == (2, 2)
    
    def test_forward_pass_with_flow(self):
        """Test forward pass with optical flow."""
        from src.models.spatiotemporal_model import SpatiotemporalModel
        
        model = SpatiotemporalModel(
            backbone="resnet18",
            backbone_pretrained=False,
            hidden_size=64,
            num_layers=1,
            bidirectional=False,
            use_optical_flow=True,
            flow_feature_dim=32
        )
        model.eval()
        
        frames = torch.randn(2, 8, 3, 224, 224)
        flow = torch.randn(2, 7, 2, 224, 224)  # N-1 flows
        
        with torch.no_grad():
            output = model(frames, flow)
        
        assert output['logits'].shape == (2, 2)
    
    def test_create_from_config(self):
        """Test creating model from config."""
        from src.models.spatiotemporal_model import create_spatiotemporal_model
        
        config = {
            'spatiotemporal_model': {
                'backbone': 'resnet18',
                'backbone_pretrained': False,
                'hidden_size': 64,
                'num_layers': 1,
                'use_optical_flow': False
            }
        }
        
        model = create_spatiotemporal_model(config)
        
        assert model is not None


class TestGANFingerprintModel:
    """Tests for GAN fingerprint model."""
    
    def test_model_creation(self):
        """Test model initialization."""
        from src.models.gan_fingerprint_model import GANFingerprintModel
        
        model = GANFingerprintModel(
            backbone="resnet18",
            backbone_pretrained=False,
            use_frequency_branch=False,
            use_edge_branch=False
        )
        
        assert model is not None
    
    def test_forward_pass_multichannel(self):
        """Test forward pass with multi-channel input."""
        from src.models.gan_fingerprint_model import GANFingerprintModel
        
        model = GANFingerprintModel(
            backbone="resnet18",
            backbone_pretrained=False,
            use_frequency_branch=True,
            use_edge_branch=True
        )
        model.eval()
        
        # 9-channel input: RGB(3) + CbCr(2) + DCT(1) + Edges(3)
        x = torch.randn(2, 9, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert 'logits' in output
        assert 'probs' in output
        assert output['logits'].shape == (2, 2)
    
    def test_forward_rgb_only(self):
        """Test RGB-only forward pass."""
        from src.models.gan_fingerprint_model import GANFingerprintModel
        
        model = GANFingerprintModel(
            backbone="resnet18",
            backbone_pretrained=False,
            use_frequency_branch=False,
            use_edge_branch=False
        )
        model.eval()
        
        rgb = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model.forward_rgb_only(rgb)
        
        assert output['logits'].shape == (2, 2)
    
    def test_simple_model(self):
        """Test simplified model."""
        from src.models.gan_fingerprint_model import GANFingerprintModelSimple
        
        model = GANFingerprintModelSimple(
            backbone="resnet18",
            backbone_pretrained=False
        )
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output['logits'].shape == (2, 2)


class TestFusionModel:
    """Tests for fusion model."""
    
    def test_model_creation(self):
        """Test model initialization."""
        from src.models.fusion_model import FusionModel
        
        model = FusionModel(
            input_features=5,
            hidden_dims=[32, 16]
        )
        
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        from src.models.fusion_model import FusionModel
        
        model = FusionModel(
            input_features=5,
            hidden_dims=[16, 8]
        )
        model.eval()
        
        video_prob = torch.rand(4)
        image_probs = torch.rand(4, 16)  # 4 videos, 16 frames each
        
        with torch.no_grad():
            output = model(video_prob, image_probs)
        
        assert output['logits'].shape == (4, 2)
        assert output['fusion_features'].shape == (4, 5)
    
    def test_forward_features(self):
        """Test forward with pre-computed features."""
        from src.models.fusion_model import FusionModel
        
        model = FusionModel()
        model.eval()
        
        features = torch.rand(4, 5)
        
        with torch.no_grad():
            output = model.forward_features(features)
        
        assert output['logits'].shape == (4, 2)


class TestDeepfakeDetector:
    """Tests for complete detector."""
    
    def test_detector_creation(self):
        """Test creating complete detector."""
        from src.models.fusion_model import DeepfakeDetector
        from src.models.spatiotemporal_model import SpatiotemporalModel
        from src.models.gan_fingerprint_model import GANFingerprintModel
        from src.models.fusion_model import FusionModel
        
        spatio = SpatiotemporalModel(
            backbone="resnet18",
            backbone_pretrained=False,
            hidden_size=64,
            num_layers=1,
            use_optical_flow=False
        )
        
        gan = GANFingerprintModel(
            backbone="resnet18",
            backbone_pretrained=False,
            use_frequency_branch=False,
            use_edge_branch=False
        )
        
        fusion = FusionModel()
        
        detector = DeepfakeDetector(
            spatiotemporal_model=spatio,
            gan_fingerprint_model=gan,
            fusion_model=fusion
        )
        
        assert detector is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

