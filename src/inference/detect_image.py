"""
CLI tool for image deepfake detection.

Usage:
    python -m src.inference.detect_image --image path/to/image.jpg
"""

import argparse
import json
from pathlib import Path

from .pipeline import DeepfakeDetectionPipeline
from ..utils.logger import setup_logger


def detect_image(
    image_path: str,
    config_path: str = "config/default_config.yaml",
    gan_fingerprint_weights: str = None,
    threshold: float = 0.5,
    output_path: str = None,
    device: str = "auto"
) -> dict:
    """
    Detect deepfake in an image.
    
    Args:
        image_path: Path to image file
        config_path: Path to configuration file
        gan_fingerprint_weights: Path to GAN fingerprint model weights
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
    
    # Load weights (only GAN fingerprint needed for images)
    pipeline.load_weights(
        gan_fingerprint=gan_fingerprint_weights
    )
    
    # Set threshold
    pipeline.set_threshold(threshold)
    
    # Run detection
    result = pipeline.detect_image(image_path)
    
    # Add image path to result
    result['image_path'] = str(image_path)
    
    # Save results if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect deepfake in an image"
    )
    parser.add_argument(
        "--image",
        "--image-path",
        dest="image_path",
        type=str,
        required=True,
        help="Path to image file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--gan-weights",
        type=str,
        default=None,
        help="Path to GAN fingerprint model weights"
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
    logger = setup_logger("detect_image")
    
    logger.info(f"Detecting deepfake in: {args.image_path}")
    
    # Run detection
    result = detect_image(
        image_path=args.image_path,
        config_path=args.config,
        gan_fingerprint_weights=args.gan_weights,
        threshold=args.threshold,
        output_path=args.output,
        device=args.device
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("DEEPFAKE DETECTION RESULTS")
    print("=" * 50)
    print(f"Image: {args.image_path}")
    print(f"Label: {result['label']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("=" * 50)
    
    if args.output:
        print(f"\nResults saved to: {args.output}")
    
    return result


if __name__ == "__main__":
    main()

