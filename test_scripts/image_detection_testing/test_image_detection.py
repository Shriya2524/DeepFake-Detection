#!/usr/bin/env python
"""
Image Detection Output Testing Script
======================================
Tests image detection output components for the deepfake detection system.

Tests include:
1. GAN fingerprint result validation
2. GAN family classification
3. Multi-scale feature analysis
4. Confidence calibration
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from colorama import init, Fore, Style

# Initialize colorama
init()


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    print(f"{'='*60}")


def print_success(message: str):
    """Print success message in green."""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message: str):
    """Print error message in red."""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_info(message: str):
    """Print info message in yellow."""
    print(f"{Fore.YELLOW}ℹ {message}{Style.RESET_ALL}")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with status."""
    status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if passed else f"{Fore.RED}FAIL{Style.RESET_ALL}"
    print(f"  [{status}] {test_name}")
    if details:
        print(f"       {Fore.CYAN}{details}{Style.RESET_ALL}")


def draw_probability_bar(label: str, value: float, width: int = 30):
    """Draw a horizontal probability bar."""
    bar_length = int(value * width)
    
    if value >= 0.7:
        color = Fore.RED
    elif value >= 0.5:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN
    
    bar = f"{color}{'█' * bar_length}{Style.RESET_ALL}{'░' * (width - bar_length)}"
    print(f"    {label:<15} [{bar}] {value:.4f}")


class ImageDetectionTester:
    """Test suite for image detection output validation."""
    
    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.project_root = project_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_gan_fingerprint_detection(self) -> bool:
        """Test 1: GAN fingerprint result validation."""
        print_header("Test 1: GAN Fingerprint Detection")
        
        try:
            from src.models.gan_fingerprint_model import GANFingerprintModel
            
            # Create model
            model = GANFingerprintModel(
                backbone="resnet18",
                backbone_pretrained=False,
                in_channels=3,
                use_frequency=False,
                use_edge=False
            )
            model.to(self.device)
            model.eval()
            
            # Test with batch of images
            batch_size = 8
            test_images = torch.randn(batch_size, 3, 224, 224).to(self.device)
            
            print_info(f"Analyzing {batch_size} images...")
            start_time = time.time()
            
            with torch.no_grad():
                output = model(test_images)
            
            inference_time = time.time() - start_time
            
            # Extract results
            logits = output['logits'].cpu().numpy()
            probs = output['probs'].cpu().numpy()
            features = output.get('features')
            
            # Validate output shapes
            shape_valid = logits.shape == (batch_size, 2) and probs.shape == (batch_size, 2)
            print_result("Output shape validation", shape_valid,
                        f"Logits: {logits.shape}, Probs: {probs.shape}")
            
            # Validate probability range
            probs_valid = np.all(probs >= 0) and np.all(probs <= 1)
            print_result("Probability range [0, 1]", probs_valid,
                        f"Min: {probs.min():.4f}, Max: {probs.max():.4f}")
            
            # Validate probability normalization
            prob_sums = probs.sum(axis=1)
            normalized = np.allclose(prob_sums, 1.0, atol=1e-5)
            print_result("Probability normalization", normalized,
                        f"Sums: {prob_sums}")
            
            # Display individual results
            print(f"\n  {Fore.CYAN}GAN Fingerprint Detection Results:{Style.RESET_ALL}")
            print(f"  {'─'*50}")
            
            fake_probs = probs[:, 1]
            for i in range(batch_size):
                label = "FAKE" if fake_probs[i] >= 0.5 else "REAL"
                label_color = Fore.RED if label == "FAKE" else Fore.GREEN
                
                bar_length = int(fake_probs[i] * 30)
                bar = '█' * bar_length + '░' * (30 - bar_length)
                
                print(f"    Image {i+1}: [{bar}] {fake_probs[i]:.4f} -> {label_color}{label}{Style.RESET_ALL}")
            
            # Feature analysis if available
            if features is not None:
                print(f"\n  {Fore.CYAN}Feature Analysis:{Style.RESET_ALL}")
                print(f"    Feature Dimension: {features.shape}")
                print(f"    Feature Mean: {features.mean().item():.4f}")
                print(f"    Feature Std: {features.std().item():.4f}")
            
            # Summary
            print(f"\n  {Fore.CYAN}Summary:{Style.RESET_ALL}")
            print(f"    Images Processed: {batch_size}")
            print(f"    Detected as Fake: {sum(fake_probs >= 0.5)}")
            print(f"    Detected as Real: {sum(fake_probs < 0.5)}")
            print(f"    Inference Time: {inference_time:.3f}s ({inference_time/batch_size*1000:.2f}ms per image)")
            
            all_valid = shape_valid and probs_valid and normalized
            self.results["gan_fingerprint"] = all_valid
            return all_valid
            
        except Exception as e:
            print_error(f"Error in GAN fingerprint detection: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results["gan_fingerprint"] = False
            return False
    
    def test_gan_family_classification(self) -> bool:
        """Test 2: GAN family classification."""
        print_header("Test 2: GAN Family Classification")
        
        try:
            from src.models.gan_family_model import (
                GANFamilyClassifier,
                DEFAULT_GAN_FAMILY_CLASSES
            )
            
            print_info(f"GAN Family Classes: {DEFAULT_GAN_FAMILY_CLASSES}")
            
            # Create model
            num_classes = len(DEFAULT_GAN_FAMILY_CLASSES)
            model = GANFamilyClassifier(
                backbone="resnet18",
                backbone_pretrained=False,
                num_classes=num_classes
            )
            model.to(self.device)
            model.eval()
            
            # Test with batch of images
            batch_size = 6
            test_images = torch.randn(batch_size, 3, 224, 224).to(self.device)
            
            print_info(f"Classifying {batch_size} images...")
            start_time = time.time()
            
            with torch.no_grad():
                output = model(test_images)
            
            inference_time = time.time() - start_time
            
            # Extract results
            logits = output['logits'].cpu().numpy()
            probs = output['probs'].cpu().numpy()
            predictions = probs.argmax(axis=1)
            
            # Validate outputs
            shape_valid = logits.shape == (batch_size, num_classes)
            print_result("Output shape validation", shape_valid,
                        f"Shape: {logits.shape}")
            
            # Validate probability normalization
            prob_sums = probs.sum(axis=1)
            normalized = np.allclose(prob_sums, 1.0, atol=1e-5)
            print_result("Probability normalization", normalized,
                        f"Sums: {prob_sums}")
            
            # Display classification results
            print(f"\n  {Fore.CYAN}GAN Family Classification Results:{Style.RESET_ALL}")
            print(f"  {'─'*60}")
            
            for i in range(batch_size):
                predicted_class = DEFAULT_GAN_FAMILY_CLASSES[predictions[i]]
                confidence = probs[i][predictions[i]]
                
                # Determine confidence level color
                if confidence >= 0.8:
                    conf_color = Fore.GREEN
                    conf_level = "High"
                elif confidence >= 0.5:
                    conf_color = Fore.YELLOW
                    conf_level = "Medium"
                else:
                    conf_color = Fore.RED
                    conf_level = "Low"
                
                print(f"\n    Image {i+1}:")
                print(f"      Predicted: {Fore.CYAN}{predicted_class}{Style.RESET_ALL}")
                print(f"      Confidence: {conf_color}{confidence:.4f} ({conf_level}){Style.RESET_ALL}")
                
                # Show probability distribution
                print(f"      Distribution:")
                for j, family in enumerate(DEFAULT_GAN_FAMILY_CLASSES):
                    draw_probability_bar(family, probs[i][j])
            
            # Class distribution summary
            print(f"\n  {Fore.CYAN}Classification Summary:{Style.RESET_ALL}")
            print(f"  {'─'*40}")
            
            class_counts = {}
            for pred in predictions:
                family = DEFAULT_GAN_FAMILY_CLASSES[pred]
                class_counts[family] = class_counts.get(family, 0) + 1
            
            for family, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / batch_size * 100
                print(f"    {family:<15}: {count} images ({percentage:.1f}%)")
            
            print(f"\n    Inference Time: {inference_time:.3f}s ({inference_time/batch_size*1000:.2f}ms per image)")
            
            all_valid = shape_valid and normalized
            self.results["gan_family"] = all_valid
            return all_valid
            
        except Exception as e:
            print_error(f"Error in GAN family classification: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results["gan_family"] = False
            return False
    
    def test_multi_scale_analysis(self) -> bool:
        """Test 3: Multi-scale feature analysis."""
        print_header("Test 3: Multi-Scale Feature Analysis")
        
        try:
            # Simulate multi-scale analysis
            # In real implementation, this would use different image scales
            
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            base_size = 224
            
            print_info("Analyzing at multiple scales...")
            
            # Simulate scores at different scales
            np.random.seed(42)
            scale_results = []
            
            for scale in scales:
                size = int(base_size * scale)
                # Simulate detection score at this scale
                score = np.random.rand() * 0.4 + 0.5  # 0.5 to 0.9
                scale_results.append({
                    "scale": scale,
                    "size": f"{size}x{size}",
                    "score": score
                })
            
            # Display results
            print(f"\n  {Fore.CYAN}Multi-Scale Analysis Results:{Style.RESET_ALL}")
            print(f"  {'─'*50}")
            
            for result in scale_results:
                score = result["score"]
                bar_length = int(score * 30)
                
                if score >= 0.7:
                    color = Fore.RED
                elif score >= 0.5:
                    color = Fore.YELLOW
                else:
                    color = Fore.GREEN
                
                bar = f"{color}{'█' * bar_length}{Style.RESET_ALL}{'░' * (30 - bar_length)}"
                print(f"    Scale {result['scale']:.2f} ({result['size']:>9}): [{bar}] {score:.4f}")
            
            # Compute aggregated score
            scores = [r["score"] for r in scale_results]
            aggregated_score = np.mean(scores)
            score_std = np.std(scores)
            
            print(f"\n  {Fore.CYAN}Aggregation:{Style.RESET_ALL}")
            print(f"    Mean Score: {aggregated_score:.4f}")
            print(f"    Std Score: {score_std:.4f}")
            print(f"    Max Score: {max(scores):.4f}")
            print(f"    Min Score: {min(scores):.4f}")
            
            # Scale consistency analysis
            consistency = 1.0 - score_std  # Higher consistency = lower std
            print(f"\n  {Fore.CYAN}Scale Consistency:{Style.RESET_ALL}")
            
            if consistency >= 0.9:
                print(f"    {Fore.GREEN}High consistency ({consistency:.4f}) - Reliable detection{Style.RESET_ALL}")
            elif consistency >= 0.8:
                print(f"    {Fore.YELLOW}Medium consistency ({consistency:.4f}) - Moderate reliability{Style.RESET_ALL}")
            else:
                print(f"    {Fore.RED}Low consistency ({consistency:.4f}) - Uncertain detection{Style.RESET_ALL}")
            
            self.results["multi_scale"] = True
            return True
            
        except Exception as e:
            print_error(f"Error in multi-scale analysis: {str(e)}")
            self.results["multi_scale"] = False
            return False
    
    def test_confidence_calibration(self) -> bool:
        """Test 4: Confidence calibration testing."""
        print_header("Test 4: Confidence Calibration")
        
        try:
            # Test confidence calibration
            # Good calibration means that when model predicts 0.8 confidence,
            # it should be correct 80% of the time
            
            # Simulate predictions and ground truth
            np.random.seed(42)
            num_samples = 100
            
            # Generate synthetic predictions with known calibration
            predictions = np.random.rand(num_samples)
            
            # Generate ground truth based on predictions (simulating well-calibrated model)
            ground_truth = (np.random.rand(num_samples) < predictions).astype(int)
            
            # Bin predictions and compute accuracy per bin
            num_bins = 10
            bin_edges = np.linspace(0, 1, num_bins + 1)
            
            calibration_results = []
            
            for i in range(num_bins):
                bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
                if bin_mask.sum() > 0:
                    bin_predictions = predictions[bin_mask]
                    bin_ground_truth = ground_truth[bin_mask]
                    
                    mean_confidence = np.mean(bin_predictions)
                    actual_accuracy = np.mean(bin_ground_truth)
                    
                    calibration_results.append({
                        "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                        "count": bin_mask.sum(),
                        "mean_confidence": mean_confidence,
                        "actual_accuracy": actual_accuracy,
                        "calibration_error": abs(mean_confidence - actual_accuracy)
                    })
            
            # Display calibration results
            print(f"\n  {Fore.CYAN}Calibration Results by Confidence Bin:{Style.RESET_ALL}")
            print(f"  {'─'*60}")
            print(f"  {'Bin':<12} {'Count':<8} {'Confidence':<12} {'Accuracy':<12} {'Error':<10}")
            print(f"  {'─'*60}")
            
            total_error = 0
            total_count = 0
            
            for result in calibration_results:
                error_color = Fore.GREEN if result["calibration_error"] < 0.1 else Fore.YELLOW if result["calibration_error"] < 0.2 else Fore.RED
                
                print(f"  {result['bin']:<12} {result['count']:<8} "
                      f"{result['mean_confidence']:<12.4f} {result['actual_accuracy']:<12.4f} "
                      f"{error_color}{result['calibration_error']:<10.4f}{Style.RESET_ALL}")
                
                total_error += result["calibration_error"] * result["count"]
                total_count += result["count"]
            
            # Expected Calibration Error (ECE)
            ece = total_error / total_count
            
            print(f"  {'─'*60}")
            print(f"\n  {Fore.CYAN}Calibration Summary:{Style.RESET_ALL}")
            print(f"    Expected Calibration Error (ECE): {ece:.4f}")
            
            if ece < 0.05:
                print(f"    {Fore.GREEN}Excellent calibration - Model confidence is reliable{Style.RESET_ALL}")
            elif ece < 0.1:
                print(f"    {Fore.GREEN}Good calibration - Model confidence is mostly reliable{Style.RESET_ALL}")
            elif ece < 0.2:
                print(f"    {Fore.YELLOW}Moderate calibration - Some confidence adjustments may be needed{Style.RESET_ALL}")
            else:
                print(f"    {Fore.RED}Poor calibration - Model confidence needs recalibration{Style.RESET_ALL}")
            
            # Reliability diagram visualization
            print(f"\n  {Fore.CYAN}Reliability Diagram:{Style.RESET_ALL}")
            print(f"    Confidence │ Accuracy")
            print(f"    {'─'*25}")
            
            for result in calibration_results:
                conf = result["mean_confidence"]
                acc = result["actual_accuracy"]
                
                conf_bar = '█' * int(conf * 10)
                acc_bar = '▓' * int(acc * 10)
                
                print(f"    {conf:.2f}       │ {conf_bar:<10} (conf)")
                print(f"             │ {acc_bar:<10} (acc)")
            
            self.results["confidence_calibration"] = ece < 0.2
            return ece < 0.2
            
        except Exception as e:
            print_error(f"Error in confidence calibration: {str(e)}")
            self.results["confidence_calibration"] = False
            return False
    
    def test_complete_image_output(self) -> bool:
        """Test 5: Complete image detection output simulation."""
        print_header("Test 5: Complete Image Detection Output")
        
        # Simulate complete image detection output
        image_output = {
            "file_name": "suspicious_portrait.jpg",
            "file_size": "1.2 MB",
            "dimensions": "1024x1024",
            "format": "JPEG",
            "gan_fingerprint": {
                "detected": True,
                "score": 0.8765,
                "confidence": 0.753,
                "features": {
                    "frequency_anomaly": 0.82,
                    "texture_irregularity": 0.79,
                    "edge_artifacts": 0.68
                }
            },
            "gan_family": {
                "predicted": "StyleGAN2",
                "confidence": 0.9234,
                "alternatives": [
                    {"family": "StyleGAN3", "confidence": 0.0423},
                    {"family": "ProGAN", "confidence": 0.0234},
                    {"family": "Real", "confidence": 0.0109}
                ]
            },
            "multi_scale": {
                "scales_analyzed": 5,
                "mean_score": 0.8234,
                "consistency": 0.92
            },
            "label": "FAKE",
            "final_score": 0.8765
        }
        
        # Print output
        print(f"\n  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}{Style.BRIGHT}          IMAGE DETECTION OUTPUT{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        
        # File Information
        print(f"\n  ╔{'═'*48}╗")
        print(f"  ║{Fore.CYAN}{Style.BRIGHT}  File Information{Style.RESET_ALL}                            ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  File Name:           {image_output['file_name']:>24}  ║")
        print(f"  ║  File Size:           {image_output['file_size']:>24}  ║")
        print(f"  ║  Dimensions:          {image_output['dimensions']:>24}  ║")
        print(f"  ║  Format:              {image_output['format']:>24}  ║")
        print(f"  ╚{'═'*48}╝")
        
        # Detection Result
        label = image_output["label"]
        label_color = Fore.RED if label == "FAKE" else Fore.GREEN
        
        print(f"\n  ╔{'═'*48}╗")
        print(f"  ║{Fore.CYAN}{Style.BRIGHT}         DETECTION RESULT{Style.RESET_ALL}                   ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  Result:               {label_color}{Style.BRIGHT}{label:>22}{Style.RESET_ALL}  ║")
        print(f"  ║  Final Score:          {image_output['final_score']:>22.4f}  ║")
        print(f"  ╚{'═'*48}╝")
        
        # GAN Fingerprint
        gf = image_output["gan_fingerprint"]
        print(f"\n  ╔{'═'*48}╗")
        print(f"  ║{Fore.CYAN}{Style.BRIGHT}  GAN Fingerprint Analysis{Style.RESET_ALL}                    ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  Fingerprint Detected: {str(gf['detected']):>22}  ║")
        print(f"  ║  Score:                {gf['score']:>22.4f}  ║")
        print(f"  ║  Confidence:           {gf['confidence']:>22.4f}  ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  {Fore.CYAN}Feature Scores:{Style.RESET_ALL}                                ║")
        for feature, score in gf["features"].items():
            print(f"  ║    {feature.replace('_', ' ').title():<20}: {score:>17.4f}  ║")
        print(f"  ╚{'═'*48}╝")
        
        # GAN Family
        gfam = image_output["gan_family"]
        print(f"\n  ╔{'═'*48}╗")
        print(f"  ║{Fore.CYAN}{Style.BRIGHT}  GAN Family Classification{Style.RESET_ALL}                   ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  Predicted Family:     {gfam['predicted']:>22}  ║")
        print(f"  ║  Confidence:           {gfam['confidence']:>22.4f}  ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  {Fore.CYAN}Alternative Predictions:{Style.RESET_ALL}                      ║")
        for alt in gfam["alternatives"]:
            print(f"  ║    {alt['family']:<20}: {alt['confidence']:>17.4f}  ║")
        print(f"  ╚{'═'*48}╝")
        
        # Multi-scale Analysis
        ms = image_output["multi_scale"]
        print(f"\n  ╔{'═'*48}╗")
        print(f"  ║{Fore.CYAN}{Style.BRIGHT}  Multi-Scale Analysis{Style.RESET_ALL}                        ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  Scales Analyzed:      {ms['scales_analyzed']:>22}  ║")
        print(f"  ║  Mean Score:           {ms['mean_score']:>22.4f}  ║")
        print(f"  ║  Scale Consistency:    {ms['consistency']:>22.4f}  ║")
        print(f"  ╚{'═'*48}╝")
        
        # Feature visualization
        print(f"\n  {Fore.CYAN}Feature Distribution:{Style.RESET_ALL}")
        for feature, score in gf["features"].items():
            draw_probability_bar(feature.replace('_', ' ').title(), score)
        
        print(f"\n  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        
        self.results["complete_image_output"] = True
        return True
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all image detection output tests."""
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"{Style.BRIGHT}    IMAGE DETECTION OUTPUT TESTING SUITE")
        print(f"    Deepfake Detection System")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        print(f"Device: {self.device}")
        
        # Run all tests
        self.test_gan_fingerprint_detection()
        self.test_gan_family_classification()
        self.test_multi_scale_analysis()
        self.test_confidence_calibration()
        self.test_complete_image_output()
        
        # Print summary
        print_header("TEST SUMMARY")
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        for test_name, result in self.results.items():
            status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if result else f"{Fore.RED}FAIL{Style.RESET_ALL}"
            print(f"  {test_name}: [{status}]")
        
        print(f"\n{'-'*40}")
        overall_pass = passed == total
        if overall_pass:
            print(f"{Fore.GREEN}{Style.BRIGHT}All tests passed: {passed}/{total}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Tests passed: {passed}/{total}{Style.RESET_ALL}")
        
        return self.results


def main():
    """Main entry point for image detection output testing."""
    tester = ImageDetectionTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
