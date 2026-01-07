"""
Tests for preprocessing modules.
"""

import numpy as np
import pytest
import torch


class TestVideoReader:
    """Tests for VideoReader class."""
    
    def test_video_reader_init(self):
        """Test VideoReader initialization."""
        from src.preprocessing.video_reader import VideoReader
        
        reader = VideoReader(target_fps=10, max_frames=32)
        
        assert reader.target_fps == 10
        assert reader.max_frames == 32
        assert reader.min_frames == 8
    
    def test_sample_frames_uniformly(self):
        """Test uniform frame sampling function exists."""
        from src.preprocessing.video_reader import sample_frames_uniformly
        
        assert callable(sample_frames_uniformly)


class TestFaceExtraction:
    """Tests for face extraction module."""
    
    def test_face_extractor_init(self):
        """Test FaceExtractor initialization."""
        from src.preprocessing.face_extraction import FaceExtractor
        
        extractor = FaceExtractor(
            target_size=(224, 224),
            padding=0.2
        )
        
        assert extractor.target_size == (224, 224)
        assert extractor.padding == 0.2
    
    def test_detect_faces_on_blank_image(self):
        """Test face detection on blank image."""
        from src.preprocessing.face_extraction import FaceExtractor
        
        extractor = FaceExtractor()
        
        # Create blank image
        blank_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        faces = extractor.detect_faces(blank_image)
        
        assert isinstance(faces, list)
        # Blank image should have no faces
        assert len(faces) == 0
    
    def test_extract_faces_from_frames(self):
        """Test batch face extraction function exists."""
        from src.preprocessing.face_extraction import extract_faces_from_frames
        
        assert callable(extract_faces_from_frames)


class TestOpticalFlow:
    """Tests for optical flow module."""
    
    def test_optical_flow_extractor_init(self):
        """Test OpticalFlowExtractor initialization."""
        from src.preprocessing.optical_flow import OpticalFlowExtractor
        
        extractor = OpticalFlowExtractor(method="farneback")
        
        assert extractor.method == "farneback"
    
    def test_compute_flow_between_frames(self):
        """Test computing optical flow between two frames."""
        from src.preprocessing.optical_flow import OpticalFlowExtractor
        
        extractor = OpticalFlowExtractor()
        
        # Create two dummy frames
        frame1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        flow = extractor.compute(frame1, frame2)
        
        assert flow.shape == (64, 64, 2)
        assert flow.dtype == np.float32 or flow.dtype == np.float64
    
    def test_compute_flow_sequence(self):
        """Test computing optical flow for a sequence."""
        from src.preprocessing.optical_flow import OpticalFlowExtractor
        
        extractor = OpticalFlowExtractor()
        
        # Create sequence of frames
        frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        
        flows = extractor.compute_sequence(frames)
        
        assert flows.shape == (4, 64, 64, 2)  # N-1 flows for N frames
    
    def test_flow_to_rgb_visualization(self):
        """Test optical flow to RGB conversion."""
        from src.preprocessing.optical_flow import flow_to_rgb
        
        flow = np.random.randn(64, 64, 2).astype(np.float32)
        
        rgb = flow_to_rgb(flow)
        
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8


class TestFrequencyAnalysis:
    """Tests for frequency analysis module."""
    
    def test_frequency_analyzer_init(self):
        """Test FrequencyAnalyzer initialization."""
        from src.preprocessing.frequency_analysis import FrequencyAnalyzer
        
        analyzer = FrequencyAnalyzer(dct_size=224)
        
        assert analyzer.dct_size == 224
    
    def test_compute_dct(self):
        """Test DCT computation."""
        from src.preprocessing.frequency_analysis import FrequencyAnalyzer
        
        analyzer = FrequencyAnalyzer(dct_size=64)
        
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        dct = analyzer.compute_dct(image)
        
        assert dct.shape == (64, 64)
    
    def test_extract_high_frequency(self):
        """Test high-frequency extraction."""
        from src.preprocessing.frequency_analysis import FrequencyAnalyzer
        
        analyzer = FrequencyAnalyzer()
        
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        high_freq = analyzer.extract_high_frequency(image)
        
        assert high_freq.shape == (64, 64)
    
    def test_compute_edge_features(self):
        """Test edge feature computation."""
        from src.preprocessing.frequency_analysis import compute_edge_features
        
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        sobel_x, sobel_y, laplacian = compute_edge_features(image)
        
        assert sobel_x.shape == (64, 64)
        assert sobel_y.shape == (64, 64)
        assert laplacian.shape == (64, 64)
    
    def test_create_multichannel_input(self):
        """Test multi-channel input creation."""
        from src.preprocessing.frequency_analysis import create_multichannel_input
        
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        multichannel = create_multichannel_input(
            image,
            include_ycbcr=True,
            include_frequency=True,
            include_edges=True
        )
        
        # RGB(3) + CbCr(2) + DCT(1) + Edges(3) = 9 channels
        assert multichannel.shape[2] == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

