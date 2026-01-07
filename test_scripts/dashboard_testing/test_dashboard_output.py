#!/usr/bin/env python
"""
Dashboard Output Testing Script
===============================
Tests dashboard output components for the deepfake detection system.

Tests include:
1. Deepfake or Real label output
2. Confidence Score calculation
3. GAN Signal Strength measurement
4. GAN Family Prediction validation
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import json
from dataclasses import dataclass

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


def display_dashboard_box(title: str, content: Dict[str, Any]):
    """Display a dashboard-style output box."""
    print(f"\n  ╔{'═'*48}╗")
    print(f"  ║{Fore.CYAN}{Style.BRIGHT}  {title:^44}{Style.RESET_ALL}  ║")
    print(f"  ╠{'═'*48}╣")
    for key, value in content.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        key_display = f"  {key}:"
        print(f"  ║{key_display:<20}{formatted_value:>26}  ║")
    print(f"  ╚{'═'*48}╝")


@dataclass
class DashboardOutput:
    """Container for dashboard output data."""
    label: str  # "REAL" or "FAKE"
    probability: float  # Raw probability (0-1)
    confidence: float  # Confidence score (0-1)
    gan_signal_strength: float  # GAN fingerprint score
    gan_family: str  # Predicted GAN family
    gan_family_confidence: float  # Confidence in GAN family prediction


class DashboardTester:
    """Test suite for dashboard output validation."""
    
    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.project_root = project_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _compute_label(self, probability: float, threshold: float = 0.5) -> str:
        """Compute the label based on probability and threshold."""
        return "FAKE" if probability >= threshold else "REAL"
    
    def _compute_confidence(self, probability: float) -> float:
        """Compute confidence score from probability."""
        # Confidence is how far from 0.5 the probability is
        return abs(probability - 0.5) * 2
    
    def test_label_output(self) -> bool:
        """Test 1: Deepfake or Real label output."""
        print_header("Test 1: Deepfake or Real Label Output")
        
        # Test cases with expected labels
        test_cases = [
            (0.9, 0.5, "FAKE"),
            (0.1, 0.5, "REAL"),
            (0.5, 0.5, "FAKE"),  # Tie goes to FAKE
            (0.51, 0.5, "FAKE"),
            (0.49, 0.5, "REAL"),
            (0.7, 0.6, "FAKE"),
            (0.5, 0.6, "REAL"),  # With higher threshold
        ]
        
        all_passed = True
        for probability, threshold, expected_label in test_cases:
            computed_label = self._compute_label(probability, threshold)
            passed = computed_label == expected_label
            if not passed:
                all_passed = False
            print_result(
                f"Prob={probability:.2f}, Thresh={threshold:.2f}",
                passed,
                f"Expected: {expected_label}, Got: {computed_label}"
            )
        
        # Display sample dashboard output
        sample_output = DashboardOutput(
            label="FAKE",
            probability=0.8765,
            confidence=0.753,
            gan_signal_strength=0.8234,
            gan_family="StyleGAN2",
            gan_family_confidence=0.9123
        )
        
        display_dashboard_box("Label Output Example", {
            "Detection Result": sample_output.label,
            "Probability": sample_output.probability,
            "Threshold": 0.5,
            "Status": "✓ Above Threshold" if sample_output.label == "FAKE" else "✓ Below Threshold"
        })
        
        self.results["label_output"] = all_passed
        return all_passed
    
    def test_confidence_score(self) -> bool:
        """Test 2: Confidence Score calculation."""
        print_header("Test 2: Confidence Score Calculation")
        
        # Test cases for confidence calculation
        test_cases = [
            (0.9, 0.8),   # High probability -> High confidence
            (0.1, 0.8),   # Low probability -> High confidence
            (0.5, 0.0),   # 0.5 probability -> Zero confidence
            (1.0, 1.0),   # Maximum probability -> Maximum confidence
            (0.0, 1.0),   # Minimum probability -> Maximum confidence
            (0.75, 0.5),  # 0.75 -> 0.5 confidence
        ]
        
        all_passed = True
        for probability, expected_confidence in test_cases:
            computed_confidence = self._compute_confidence(probability)
            passed = abs(computed_confidence - expected_confidence) < 0.001
            if not passed:
                all_passed = False
            print_result(
                f"Probability={probability:.2f}",
                passed,
                f"Expected: {expected_confidence:.2f}, Got: {computed_confidence:.2f}"
            )
        
        # Display confidence interpretation
        print(f"\n  {Fore.CYAN}Confidence Interpretation:{Style.RESET_ALL}")
        print(f"    0.00 - 0.30: Low confidence")
        print(f"    0.30 - 0.60: Medium confidence")
        print(f"    0.60 - 0.80: High confidence")
        print(f"    0.80 - 1.00: Very high confidence")
        
        display_dashboard_box("Confidence Example", {
            "Raw Probability": 0.9234,
            "Confidence Score": self._compute_confidence(0.9234),
            "Confidence Level": "Very High",
            "Reliability": "Highly Reliable"
        })
        
        self.results["confidence_score"] = all_passed
        return all_passed
    
    def test_gan_signal_strength(self) -> bool:
        """Test 3: GAN Signal Strength measurement."""
        print_header("Test 3: GAN Signal Strength Measurement")
        
        try:
            from src.models.gan_fingerprint_model import GANFingerprintModel
            
            # Create model for testing
            model = GANFingerprintModel(
                backbone="resnet18",
                backbone_pretrained=False,
                in_channels=3,
                use_frequency=False,
                use_edge=False
            )
            model.to(self.device)
            model.eval()
            
            # Test with synthetic data
            batch_size = 4
            test_images = torch.randn(batch_size, 3, 224, 224).to(self.device)
            
            print_info("Running GAN fingerprint detection...")
            start_time = time.time()
            with torch.no_grad():
                output = model(test_images)
            inference_time = time.time() - start_time
            
            # Extract GAN signal strength (fake probability)
            gan_signals = output['probs'][:, 1].cpu().numpy()
            
            # Validate output range
            signals_valid = all(0 <= s <= 1 for s in gan_signals)
            print_result("Signal range [0, 1]", signals_valid,
                        f"Signals: {gan_signals}")
            
            # Display signal strength interpretation
            print(f"\n  {Fore.CYAN}GAN Signal Strength Interpretation:{Style.RESET_ALL}")
            print(f"    0.00 - 0.30: Weak signal (likely real)")
            print(f"    0.30 - 0.50: Moderate signal (uncertain)")
            print(f"    0.50 - 0.70: Strong signal (likely fake)")
            print(f"    0.70 - 1.00: Very strong signal (highly likely fake)")
            
            # Sample outputs
            for i, signal in enumerate(gan_signals):
                if signal < 0.3:
                    strength = "Weak"
                    interpretation = "Likely Real"
                elif signal < 0.5:
                    strength = "Moderate"
                    interpretation = "Uncertain"
                elif signal < 0.7:
                    strength = "Strong"
                    interpretation = "Likely Fake"
                else:
                    strength = "Very Strong"
                    interpretation = "Highly Likely Fake"
                
                print(f"    Sample {i+1}: {signal:.4f} - {strength} ({interpretation})")
            
            display_dashboard_box("GAN Signal Strength", {
                "Average Signal": float(np.mean(gan_signals)),
                "Max Signal": float(np.max(gan_signals)),
                "Min Signal": float(np.min(gan_signals)),
                "Inference Time": f"{inference_time:.3f}s"
            })
            
            self.results["gan_signal_strength"] = signals_valid
            return signals_valid
            
        except Exception as e:
            print_error(f"Error testing GAN signal strength: {str(e)}")
            self.results["gan_signal_strength"] = False
            return False
    
    def test_gan_family_prediction(self) -> bool:
        """Test 4: GAN Family Prediction validation."""
        print_header("Test 4: GAN Family Prediction")
        
        try:
            from src.models.gan_family_model import (
                GANFamilyClassifier, 
                create_gan_family_model,
                DEFAULT_GAN_FAMILY_CLASSES
            )
            
            print_info(f"Available GAN Families: {DEFAULT_GAN_FAMILY_CLASSES}")
            
            # Create model
            model = GANFamilyClassifier(
                backbone="resnet18",
                backbone_pretrained=False,
                num_classes=len(DEFAULT_GAN_FAMILY_CLASSES)
            )
            model.to(self.device)
            model.eval()
            
            # Test with synthetic data
            batch_size = 4
            test_images = torch.randn(batch_size, 3, 224, 224).to(self.device)
            
            print_info("Running GAN family classification...")
            start_time = time.time()
            with torch.no_grad():
                output = model(test_images)
            inference_time = time.time() - start_time
            
            # Extract predictions
            probs = output['probs'].cpu().numpy()
            predictions = probs.argmax(axis=1)
            
            # Validate output
            probs_valid = np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
            print_result("Probability sum = 1.0", probs_valid,
                        f"Sums: {probs.sum(axis=1)}")
            
            # Display predictions for each sample
            print(f"\n  {Fore.CYAN}GAN Family Predictions:{Style.RESET_ALL}")
            for i in range(batch_size):
                predicted_class = DEFAULT_GAN_FAMILY_CLASSES[predictions[i]]
                confidence = probs[i][predictions[i]]
                
                # Get top 3 predictions
                top3_indices = probs[i].argsort()[-3:][::-1]
                top3 = [(DEFAULT_GAN_FAMILY_CLASSES[idx], probs[i][idx]) for idx in top3_indices]
                
                print(f"    Sample {i+1}: {predicted_class} (confidence: {confidence:.4f})")
                print(f"             Top 3: {', '.join([f'{c}:{p:.2f}' for c, p in top3])}")
            
            # Display family probabilities for first sample
            family_probs = {
                DEFAULT_GAN_FAMILY_CLASSES[i]: float(probs[0][i])
                for i in range(len(DEFAULT_GAN_FAMILY_CLASSES))
            }
            
            display_dashboard_box("GAN Family Classification", {
                "Predicted Family": DEFAULT_GAN_FAMILY_CLASSES[predictions[0]],
                "Confidence": float(probs[0][predictions[0]]),
                "Inference Time": f"{inference_time:.3f}s"
            })
            
            print(f"\n  {Fore.CYAN}All Family Probabilities:{Style.RESET_ALL}")
            for family, prob in sorted(family_probs.items(), key=lambda x: x[1], reverse=True):
                bar_length = int(prob * 30)
                bar = '█' * bar_length + '░' * (30 - bar_length)
                print(f"    {family:<15} [{bar}] {prob:.4f}")
            
            self.results["gan_family_prediction"] = probs_valid
            return probs_valid
            
        except Exception as e:
            print_error(f"Error testing GAN family prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results["gan_family_prediction"] = False
            return False
    
    def test_complete_dashboard_output(self) -> bool:
        """Test 5: Complete dashboard output simulation."""
        print_header("Test 5: Complete Dashboard Output Simulation")
        
        # Simulate a complete detection result
        detection_result = {
            "file_name": "suspicious_video.mp4",
            "file_type": "video",
            "frames_analyzed": 150,
            "label": "FAKE",
            "probability": 0.9234,
            "confidence": 0.8468,
            "threshold": 0.5,
            "gan_signal_strength": 0.8567,
            "spatiotemporal_score": 0.9012,
            "fusion_score": 0.9234,
            "gan_family": "StyleGAN2",
            "gan_family_confidence": 0.9123,
            "family_probabilities": {
                "StyleGAN2": 0.9123,
                "StyleGAN3": 0.0534,
                "ProGAN": 0.0234,
                "Real": 0.0109
            }
        }
        
        print(f"\n  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}{Style.BRIGHT}           DEEPFAKE DETECTION DASHBOARD{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        
        # File Information
        display_dashboard_box("File Information", {
            "File Name": detection_result["file_name"],
            "File Type": detection_result["file_type"],
            "Frames Analyzed": detection_result["frames_analyzed"]
        })
        
        # Detection Result
        label_color = Fore.RED if detection_result["label"] == "FAKE" else Fore.GREEN
        print(f"\n  ╔{'═'*48}╗")
        print(f"  ║{Fore.CYAN}{Style.BRIGHT}         DETECTION RESULT{Style.RESET_ALL}                   ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  Result:               {label_color}{Style.BRIGHT}{detection_result['label']:>22}{Style.RESET_ALL}  ║")
        print(f"  ║  Probability:          {detection_result['probability']:>22.4f}  ║")
        print(f"  ║  Confidence:           {detection_result['confidence']:>22.4f}  ║")
        print(f"  ║  Threshold:            {detection_result['threshold']:>22.2f}  ║")
        print(f"  ╚{'═'*48}╝")
        
        # Detection Scores
        display_dashboard_box("Detection Scores", {
            "GAN Signal Strength": detection_result["gan_signal_strength"],
            "Spatiotemporal Score": detection_result["spatiotemporal_score"],
            "Fusion Score": detection_result["fusion_score"]
        })
        
        # GAN Family
        display_dashboard_box("GAN Family Analysis", {
            "Predicted Family": detection_result["gan_family"],
            "Family Confidence": detection_result["gan_family_confidence"]
        })
        
        # Family probabilities bar chart
        print(f"\n  {Fore.CYAN}GAN Family Probabilities:{Style.RESET_ALL}")
        for family, prob in sorted(
            detection_result["family_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"    {family:<12} [{bar}] {prob*100:.2f}%")
        
        print(f"\n  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        
        self.results["complete_dashboard"] = True
        return True
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all dashboard output tests."""
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"{Style.BRIGHT}    DASHBOARD OUTPUT TESTING SUITE")
        print(f"    Deepfake Detection System")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        print(f"Device: {self.device}")
        
        # Run all tests
        self.test_label_output()
        self.test_confidence_score()
        self.test_gan_signal_strength()
        self.test_gan_family_prediction()
        self.test_complete_dashboard_output()
        
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
    """Main entry point for dashboard output testing."""
    tester = DashboardTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
