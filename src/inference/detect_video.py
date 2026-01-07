"""
CLI tool for video deepfake detection.

Usage:
    python -m src.inference.detect_video --video path/to/video.mp4
"""

import argparse
import json
from pathlib import Path

from .pipeline import DeepfakeDetectionPipeline
from ..utils.logger import setup_logger


def detect_video(
    video_path: str,
    config_path: str = "config/default_config.yaml",
    spatiotemporal_weights: str = None,
    gan_fingerprint_weights: str = None,
    fusion_weights: str = None,
    threshold: float = 0.5,
    output_path: str = None,
    device: str = "auto"
) -> dict:
    """
    Detect deepfake in a video.
    
    Args:
        video_path: Path to video file
        config_path: Path to configuration file
        spatiotemporal_weights: Path to spatiotemporal model weights
        gan_fingerprint_weights: Path to GAN fingerprint model weights
        fusion_weights: Path to fusion model weights
        threshold: Detection threshold
        output_path: Path to save results JSON
        device: Device to use
        
    Returns:
        Detection results dictionary
    """
    # Initialize pipeline
    pipeline = DeepfakeDetectionPipeline(
        config_path=config_path,
        device=device
    )
    
    # Load weights
    pipeline.load_weights(
        spatiotemporal=spatiotemporal_weights,
        gan_fingerprint=gan_fingerprint_weights,
        fusion=fusion_weights
    )
    
    # Set threshold
    pipeline.set_threshold(threshold)
    
    # Run detection
    result = pipeline.detect_video(video_path, return_frame_scores=True)
    
    # Add video path to result
    result['video_path'] = str(video_path)
    
    # Convert numpy arrays to lists for JSON serialization
    if 'frame_scores' in result:
        result['frame_scores'] = result['frame_scores'].tolist()
    if 'attention_scores' in result:
        result['attention_scores'] = result['attention_scores'].tolist()
    
    # Save results if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect deepfake in a video"
    )
    parser.add_argument(
        "--video",
        "--video-path",
        dest="video_path",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--spatiotemporal-weights",
        type=str,
        default=None,
        help="Path to spatiotemporal model weights"
    )
    parser.add_argument(
        "--gan-weights",
        type=str,
        default=None,
        help="Path to GAN fingerprint model weights"
    )
    parser.add_argument(
        "--fusion-weights",
        type=str,
        default=None,
        help="Path to fusion model weights"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("detect_video")
    
    logger.info(f"Detecting deepfake in: {args.video_path}")
    
    # Run detection
    result = detect_video(
        video_path=args.video_path,
        config_path=args.config,
        spatiotemporal_weights=args.spatiotemporal_weights,
        gan_fingerprint_weights=args.gan_weights,
        fusion_weights=args.fusion_weights,
        threshold=args.threshold,
        output_path=args.output,
        device=args.device
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("DEEPFAKE DETECTION RESULTS")
    print("=" * 50)
    print(f"Video: {args.video_path}")
    print(f"Label: {result['label']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Video Score: {result['video_score']:.4f}")
    print(f"Mean Frame Score: {result['mean_frame_score']:.4f}")
    print("=" * 50)
    
    if args.output:
        print(f"\nResults saved to: {args.output}")
    
    return result


if __name__ == "__main__":
    main()

