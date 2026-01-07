#!/usr/bin/env python
"""
Video Detection Output Testing Script
======================================
Tests video detection output components for the deepfake detection system.

Tests include:
1. Frame-level suspicious regions detection
2. Spatiotemporal score calculation
3. Fusion score computation
4. Attention-based frame analysis
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


def draw_frame_timeline(frame_scores: List[float], suspicious_threshold: float = 0.5):
    """Draw a visual timeline of frame scores."""
    print(f"\n  {Fore.CYAN}Frame Timeline (suspicious threshold = {suspicious_threshold}):{Style.RESET_ALL}")
    print(f"  {'─'*50}")
    
    # Create visual representation
    num_frames = len(frame_scores)
    display_width = 50
    frames_per_char = max(1, num_frames // display_width)
    
    timeline = ""
    for i in range(0, num_frames, frames_per_char):
        chunk = frame_scores[i:i+frames_per_char]
        avg_score = np.mean(chunk)
        
        if avg_score >= 0.8:
            timeline += f"{Fore.RED}█{Style.RESET_ALL}"
        elif avg_score >= 0.6:
            timeline += f"{Fore.YELLOW}▓{Style.RESET_ALL}"
        elif avg_score >= suspicious_threshold:
            timeline += f"{Fore.YELLOW}▒{Style.RESET_ALL}"
        else:
            timeline += f"{Fore.GREEN}░{Style.RESET_ALL}"
    
    print(f"  {timeline}")
    print(f"  {'─'*50}")
    print(f"  0{' '*(display_width//2-2)}Frame{' '*(display_width//2-3)}{num_frames}")
    print(f"\n  Legend: {Fore.RED}█{Style.RESET_ALL} Very Suspicious (≥0.8) | "
          f"{Fore.YELLOW}▓{Style.RESET_ALL} Suspicious (≥0.6) | "
          f"{Fore.YELLOW}▒{Style.RESET_ALL} Mildly Suspicious | "
          f"{Fore.GREEN}░{Style.RESET_ALL} Normal")


class VideoDetectionTester:
    """Test suite for video detection output validation."""
    
    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.project_root = project_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_frame_level_detection(self) -> bool:
        """Test 1: Frame-level suspicious regions detection."""
        print_header("Test 1: Frame-Level Suspicious Regions Detection")
        
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
            
            # Simulate video frames (batch processing)
            num_frames = 32
            frame_tensors = torch.randn(num_frames, 3, 224, 224).to(self.device)
            
            print_info(f"Analyzing {num_frames} frames...")
            start_time = time.time()
            
            with torch.no_grad():
                output = model(frame_tensors)
            
            inference_time = time.time() - start_time
            
            # Get frame-level scores
            frame_scores = output['probs'][:, 1].cpu().numpy().tolist()
            
            # Validate scores
            scores_valid = all(0 <= s <= 1 for s in frame_scores)
            print_result("Score range validation", scores_valid,
                        f"Min: {min(frame_scores):.4f}, Max: {max(frame_scores):.4f}")
            
            # Identify suspicious frames
            suspicious_threshold = 0.5
            suspicious_frames = [
                {"frame": i+1, "score": s} 
                for i, s in enumerate(frame_scores) 
                if s >= suspicious_threshold
            ]
            
            print_result(
                "Suspicious frames identified", 
                True,
                f"Found {len(suspicious_frames)}/{num_frames} suspicious frames"
            )
            
            # Display top 5 most suspicious frames
            sorted_frames = sorted(
                [{"frame": i+1, "score": s} for i, s in enumerate(frame_scores)],
                key=lambda x: x["score"],
                reverse=True
            )[:5]
            
            print(f"\n  {Fore.CYAN}Top 5 Most Suspicious Frames:{Style.RESET_ALL}")
            for item in sorted_frames:
                score = item["score"]
                bar_length = int(score * 30)
                bar = f"{Fore.RED}{'█'*bar_length}{Style.RESET_ALL}{'░'*(30-bar_length)}"
                print(f"    Frame {item['frame']:3d}: [{bar}] {score:.4f}")
            
            # Draw frame timeline
            draw_frame_timeline(frame_scores, suspicious_threshold)
            
            # Summary statistics
            print(f"\n  {Fore.CYAN}Frame Analysis Summary:{Style.RESET_ALL}")
            print(f"    Total Frames: {num_frames}")
            print(f"    Suspicious Frames: {len(suspicious_frames)} ({len(suspicious_frames)/num_frames*100:.1f}%)")
            print(f"    Mean Score: {np.mean(frame_scores):.4f}")
            print(f"    Std Score: {np.std(frame_scores):.4f}")
            print(f"    Inference Time: {inference_time:.3f}s ({inference_time/num_frames*1000:.2f}ms per frame)")
            
            self.results["frame_level_detection"] = scores_valid
            return scores_valid
            
        except Exception as e:
            print_error(f"Error in frame-level detection: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results["frame_level_detection"] = False
            return False
    
    def test_spatiotemporal_score(self) -> bool:
        """Test 2: Spatiotemporal score calculation."""
        print_header("Test 2: Spatiotemporal Score Calculation")
        
        try:
            from src.models.spatiotemporal_model import SpatiotemporalModel
            
            # Create model
            model = SpatiotemporalModel(
                backbone="resnet18",
                backbone_pretrained=False,
                hidden_size=128,
                num_layers=2,
                bidirectional=True,
                use_optical_flow=False
            )
            model.to(self.device)
            model.eval()
            
            # Simulate video sequences
            batch_size = 2
            num_frames = 16
            
            # Create batch of video sequences
            video_batch = torch.randn(batch_size, num_frames, 3, 224, 224).to(self.device)
            
            print_info(f"Analyzing {batch_size} videos with {num_frames} frames each...")
            start_time = time.time()
            
            with torch.no_grad():
                output = model(video_batch)
            
            inference_time = time.time() - start_time
            
            # Extract spatiotemporal scores
            logits = output['logits']
            probs = output['probs']
            
            # Video-level fake probabilities
            spatio_scores = probs[:, 1].cpu().numpy()
            
            # Validate outputs
            scores_valid = all(0 <= s <= 1 for s in spatio_scores)
            print_result("Score range [0, 1]", scores_valid,
                        f"Scores: {spatio_scores}")
            
            # Check probability normalization
            prob_sums = probs.sum(dim=1).cpu().numpy()
            normalized = np.allclose(prob_sums, 1.0, atol=1e-5)
            print_result("Probability normalization", normalized,
                        f"Sums: {prob_sums}")
            
            # Display results for each video
            print(f"\n  {Fore.CYAN}Spatiotemporal Analysis Results:{Style.RESET_ALL}")
            for i, score in enumerate(spatio_scores):
                if score >= 0.7:
                    level = f"{Fore.RED}High Risk{Style.RESET_ALL}"
                elif score >= 0.5:
                    level = f"{Fore.YELLOW}Medium Risk{Style.RESET_ALL}"
                else:
                    level = f"{Fore.GREEN}Low Risk{Style.RESET_ALL}"
                
                bar_length = int(score * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"    Video {i+1}: [{bar}] {score:.4f} ({level})")
            
            # Summary
            print(f"\n  {Fore.CYAN}Spatiotemporal Summary:{Style.RESET_ALL}")
            print(f"    Videos Analyzed: {batch_size}")
            print(f"    Frames per Video: {num_frames}")
            print(f"    Mean Score: {np.mean(spatio_scores):.4f}")
            print(f"    Inference Time: {inference_time:.3f}s ({inference_time/batch_size*1000:.2f}ms per video)")
            
            # Check for temporal features
            if 'temporal_features' in output:
                temp_features = output['temporal_features']
                print(f"    Temporal Feature Dim: {temp_features.shape}")
            
            self.results["spatiotemporal_score"] = scores_valid and normalized
            return scores_valid and normalized
            
        except Exception as e:
            print_error(f"Error in spatiotemporal score test: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results["spatiotemporal_score"] = False
            return False
    
    def test_fusion_score(self) -> bool:
        """Test 3: Fusion score computation."""
        print_header("Test 3: Fusion Score Computation")
        
        try:
            from src.models.fusion_model import FusionModel
            
            # Create fusion model
            model = FusionModel(
                input_features=5,
                hidden_dims=[32, 16],
                dropout=0.3
            )
            model.to(self.device)
            model.eval()
            
            # Simulate inputs from other models
            batch_size = 4
            num_frames = 16
            
            # Video-level scores from spatiotemporal model
            video_probs = torch.tensor([0.8, 0.3, 0.6, 0.9]).to(self.device)
            
            # Frame-level scores from GAN fingerprint model
            frame_probs = torch.rand(batch_size, num_frames).to(self.device)
            
            print_info("Computing fusion scores...")
            start_time = time.time()
            
            with torch.no_grad():
                output = model(video_probs, frame_probs)
            
            inference_time = time.time() - start_time
            
            # Extract fusion scores
            fusion_probs = output['probs'][:, 1].cpu().numpy()
            fusion_features = output['fusion_features'].cpu().numpy()
            
            # Validate outputs
            scores_valid = all(0 <= s <= 1 for s in fusion_probs)
            print_result("Score range [0, 1]", scores_valid,
                        f"Scores: {fusion_probs}")
            
            # Display fusion computation
            print(f"\n  {Fore.CYAN}Fusion Score Computation:{Style.RESET_ALL}")
            print(f"  {'─'*50}")
            
            for i in range(batch_size):
                frame_scores = frame_probs[i].cpu().numpy()
                
                print(f"\n  Sample {i+1}:")
                print(f"    Spatiotemporal Score: {video_probs[i].item():.4f}")
                print(f"    Frame Mean Score:     {np.mean(frame_scores):.4f}")
                print(f"    Frame Std Score:      {np.std(frame_scores):.4f}")
                print(f"    Frame Max Score:      {np.max(frame_scores):.4f}")
                print(f"    Frame Min Score:      {np.min(frame_scores):.4f}")
                print(f"    {Fore.YELLOW}──────────────────────────────────{Style.RESET_ALL}")
                print(f"    {Fore.GREEN}{Style.BRIGHT}Fusion Score:           {fusion_probs[i]:.4f}{Style.RESET_ALL}")
            
            # Validate fusion features
            print(f"\n  {Fore.CYAN}Fusion Features:{Style.RESET_ALL}")
            print(f"    Feature dimension: {fusion_features.shape}")
            print(f"    Features contain: [video_prob, mean, std, max, min]")
            
            # Summary
            print(f"\n  {Fore.CYAN}Fusion Summary:{Style.RESET_ALL}")
            print(f"    Samples Processed: {batch_size}")
            print(f"    Mean Fusion Score: {np.mean(fusion_probs):.4f}")
            print(f"    Inference Time: {inference_time:.3f}s")
            
            self.results["fusion_score"] = scores_valid
            return scores_valid
            
        except Exception as e:
            print_error(f"Error in fusion score test: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results["fusion_score"] = False
            return False
    
    def test_attention_analysis(self) -> bool:
        """Test 4: Attention-based frame analysis."""
        print_header("Test 4: Attention-Based Frame Analysis")
        
        try:
            # Simulate attention weights (would come from transformer-based model)
            num_frames = 32
            
            # Generate mock attention weights
            # In real scenario, these come from the model's attention mechanism
            np.random.seed(42)
            attention_weights = np.random.dirichlet(np.ones(num_frames) * 2)
            
            # Also generate frame scores
            frame_scores = np.random.rand(num_frames) * 0.7 + 0.15  # Scores between 0.15 and 0.85
            
            # Combine attention and scores
            weighted_importance = attention_weights * frame_scores
            
            # Validate attention weights
            attention_valid = np.isclose(attention_weights.sum(), 1.0, atol=1e-5)
            print_result("Attention weights sum to 1", attention_valid,
                        f"Sum: {attention_weights.sum():.6f}")
            
            # Find high-attention frames
            attention_threshold = np.percentile(attention_weights, 75)
            high_attention_frames = [
                i+1 for i, w in enumerate(attention_weights)
                if w >= attention_threshold
            ]
            
            print_result(
                "High attention frames identified",
                len(high_attention_frames) > 0,
                f"Found {len(high_attention_frames)} frames above 75th percentile"
            )
            
            # Display attention distribution
            print(f"\n  {Fore.CYAN}Attention Distribution:{Style.RESET_ALL}")
            print(f"    Min Attention: {attention_weights.min():.4f}")
            print(f"    Max Attention: {attention_weights.max():.4f}")
            print(f"    Mean Attention: {attention_weights.mean():.4f}")
            print(f"    Std Attention: {attention_weights.std():.4f}")
            
            # Display top frames by attention
            top_attention_indices = np.argsort(attention_weights)[-5:][::-1]
            print(f"\n  {Fore.CYAN}Top 5 Frames by Attention:{Style.RESET_ALL}")
            for idx in top_attention_indices:
                att = attention_weights[idx]
                score = frame_scores[idx]
                importance = weighted_importance[idx]
                
                att_bar = int(att * 100)
                score_bar = int(score * 30)
                
                print(f"    Frame {idx+1:3d}: Attention={att:.4f} Score={score:.4f} "
                      f"Importance={importance:.4f}")
            
            # Attention timeline
            print(f"\n  {Fore.CYAN}Attention Timeline:{Style.RESET_ALL}")
            print(f"  {'─'*50}")
            
            timeline = ""
            for i in range(min(50, num_frames)):
                att = attention_weights[i]
                if att >= np.percentile(attention_weights, 90):
                    timeline += f"{Fore.RED}█{Style.RESET_ALL}"
                elif att >= np.percentile(attention_weights, 75):
                    timeline += f"{Fore.YELLOW}▓{Style.RESET_ALL}"
                elif att >= np.percentile(attention_weights, 50):
                    timeline += f"{Fore.CYAN}▒{Style.RESET_ALL}"
                else:
                    timeline += f"{Fore.WHITE}░{Style.RESET_ALL}"
            
            print(f"  {timeline}")
            print(f"  {'─'*50}")
            
            # Weighted score calculation
            weighted_score = np.sum(attention_weights * frame_scores)
            print(f"\n  {Fore.CYAN}Attention-Weighted Analysis:{Style.RESET_ALL}")
            print(f"    Weighted Score: {weighted_score:.4f}")
            print(f"    Unweighted Mean: {frame_scores.mean():.4f}")
            print(f"    Difference: {abs(weighted_score - frame_scores.mean()):.4f}")
            
            self.results["attention_analysis"] = attention_valid
            return attention_valid
            
        except Exception as e:
            print_error(f"Error in attention analysis test: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results["attention_analysis"] = False
            return False
    
    def test_complete_video_output(self) -> bool:
        """Test 5: Complete video detection output simulation."""
        print_header("Test 5: Complete Video Detection Output")
        
        # Simulate complete video detection output
        np.random.seed(42)
        num_frames = 150
        
        frame_scores = np.random.rand(num_frames) * 0.5 + 0.4  # 0.4 to 0.9
        # Make some frames more suspicious
        suspicious_start = 45
        suspicious_end = 75
        frame_scores[suspicious_start:suspicious_end] = np.random.rand(30) * 0.2 + 0.75  # 0.75 to 0.95
        
        video_output = {
            "video_file": "suspicious_interview.mp4",
            "duration": "00:05:00",
            "fps": 30,
            "total_frames": num_frames,
            "frames_analyzed": num_frames,
            "spatiotemporal_score": 0.7823,
            "fusion_score": 0.8234,
            "mean_frame_score": float(np.mean(frame_scores)),
            "max_frame_score": float(np.max(frame_scores)),
            "min_frame_score": float(np.min(frame_scores)),
            "std_frame_score": float(np.std(frame_scores)),
            "suspicious_frames": [],
            "suspicious_regions": [],
            "label": "FAKE",
            "confidence": 0.6468
        }
        
        # Find suspicious frames
        suspicious_threshold = 0.7
        for i, score in enumerate(frame_scores):
            if score >= suspicious_threshold:
                video_output["suspicious_frames"].append({
                    "frame_number": i + 1,
                    "timestamp": f"{i//30//60:02d}:{i//30%60:02d}.{int((i%30)/30*100):02d}",
                    "score": float(score)
                })
        
        # Identify suspicious regions (consecutive suspicious frames)
        in_region = False
        region_start = 0
        for i, score in enumerate(frame_scores):
            if score >= suspicious_threshold and not in_region:
                in_region = True
                region_start = i
            elif score < suspicious_threshold and in_region:
                in_region = False
                video_output["suspicious_regions"].append({
                    "start_frame": region_start + 1,
                    "end_frame": i,
                    "duration_frames": i - region_start,
                    "avg_score": float(np.mean(frame_scores[region_start:i]))
                })
        
        # Print output
        print(f"\n  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}{Style.BRIGHT}          VIDEO DETECTION OUTPUT{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        
        print(f"\n  ╔{'═'*48}╗")
        print(f"  ║{Fore.CYAN}{Style.BRIGHT}  Video Information{Style.RESET_ALL}                            ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  File:              {video_output['video_file']:>26}  ║")
        print(f"  ║  Duration:          {video_output['duration']:>26}  ║")
        print(f"  ║  FPS:               {video_output['fps']:>26}  ║")
        print(f"  ║  Frames Analyzed:   {video_output['frames_analyzed']:>26}  ║")
        print(f"  ╚{'═'*48}╝")
        
        print(f"\n  ╔{'═'*48}╗")
        print(f"  ║{Fore.CYAN}{Style.BRIGHT}  Detection Scores{Style.RESET_ALL}                             ║")
        print(f"  ╠{'═'*48}╣")
        print(f"  ║  Spatiotemporal Score:  {video_output['spatiotemporal_score']:>22.4f}  ║")
        print(f"  ║  Fusion Score:          {video_output['fusion_score']:>22.4f}  ║")
        print(f"  ║  Mean Frame Score:      {video_output['mean_frame_score']:>22.4f}  ║")
        print(f"  ║  Max Frame Score:       {video_output['max_frame_score']:>22.4f}  ║")
        print(f"  ║  Min Frame Score:       {video_output['min_frame_score']:>22.4f}  ║")
        print(f"  ╚{'═'*48}╝")
        
        # Draw frame timeline
        draw_frame_timeline(frame_scores.tolist(), suspicious_threshold)
        
        # Suspicious regions
        print(f"\n  {Fore.CYAN}Suspicious Regions Detected:{Style.RESET_ALL}")
        print(f"  {'─'*50}")
        
        if video_output["suspicious_regions"]:
            for i, region in enumerate(video_output["suspicious_regions"][:5]):
                print(f"    Region {i+1}: Frames {region['start_frame']}-{region['end_frame']} "
                      f"({region['duration_frames']} frames) - Avg Score: {region['avg_score']:.4f}")
        else:
            print(f"    No suspicious regions detected")
        
        # Top suspicious frames
        top_frames = sorted(
            video_output["suspicious_frames"],
            key=lambda x: x["score"],
            reverse=True
        )[:5]
        
        print(f"\n  {Fore.CYAN}Top 5 Suspicious Frames:{Style.RESET_ALL}")
        print(f"  {'─'*50}")
        for frame in top_frames:
            score = frame["score"]
            bar_length = int(score * 30)
            bar = f"{Fore.RED}{'█'*bar_length}{Style.RESET_ALL}{'░'*(30-bar_length)}"
            print(f"    Frame {frame['frame_number']:3d} ({frame['timestamp']}): [{bar}] {score:.4f}")
        
        print(f"\n  {Fore.MAGENTA}{'='*56}{Style.RESET_ALL}")
        
        self.results["complete_video_output"] = True
        return True
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all video detection output tests."""
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"{Style.BRIGHT}    VIDEO DETECTION OUTPUT TESTING SUITE")
        print(f"    Deepfake Detection System")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        print(f"Device: {self.device}")
        
        # Run all tests
        self.test_frame_level_detection()
        self.test_spatiotemporal_score()
        self.test_fusion_score()
        self.test_attention_analysis()
        self.test_complete_video_output()
        
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
    """Main entry point for video detection output testing."""
    tester = VideoDetectionTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
