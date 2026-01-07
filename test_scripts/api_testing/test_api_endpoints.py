#!/usr/bin/env python
"""
API Endpoints Testing Script
============================
Tests all API endpoints for the deepfake detection system.

Endpoints tested:
1. GET /health - Health check
2. GET /models - List available models
3. POST /models/switch - Switch active model
4. POST /detect/image - Image detection
5. POST /detect/video - Video detection
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from colorama import init, Fore, Style

# Initialize colorama
init()

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


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


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON data."""
    formatted = json.dumps(data, indent=indent, default=str)
    for line in formatted.split('\n'):
        print(f"       {Fore.WHITE}{line}{Style.RESET_ALL}")


@dataclass
class TestResult:
    """Container for test results."""
    test_name: str
    passed: bool
    status_code: Optional[int]
    response_time: float
    response_data: Optional[Dict]
    error: Optional[str]


class APIEndpointTester:
    """Test suite for API endpoints."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.results: Dict[str, TestResult] = {}
        self.project_root = project_root
        
        # Find sample files for testing
        self.sample_image = self._find_sample_image()
        self.sample_video = self._find_sample_video()
    
    def _find_sample_image(self) -> Optional[Path]:
        """Find a sample image for testing."""
        search_dirs = [
            self.project_root / "stylegan" / "Final Dataset" / "Fake",
            self.project_root / "stylegan" / "Final Dataset" / "Real",
            self.project_root / "data" / "faceforensics" / "fake",
            self.project_root / "data" / "faceforensics" / "real",
        ]
        
        for dir_path in search_dirs:
            if dir_path.exists():
                for ext in ['.jpg', '.jpeg', '.png']:
                    images = list(dir_path.glob(f"*{ext}"))
                    if images:
                        return images[0]
        return None
    
    def _find_sample_video(self) -> Optional[Path]:
        """Find a sample video for testing."""
        search_dirs = [
            self.project_root / "data" / "manipulated_sequences" / "Deepfakes",
            self.project_root / "data" / "original_sequences" / "youtube",
            self.project_root / "data",
        ]
        
        for dir_path in search_dirs:
            if dir_path.exists():
                for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    videos = list(dir_path.rglob(f"*{ext}"))
                    if videos:
                        return videos[0]
        return None
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        timeout: int = 60
    ) -> TestResult:
        """Make an HTTP request and return result."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = requests.get(url, timeout=timeout)
            elif method.upper() == "POST":
                if files:
                    response = requests.post(url, files=files, data=data, timeout=timeout)
                elif json_data:
                    response = requests.post(url, json=json_data, timeout=timeout)
                else:
                    response = requests.post(url, data=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            elapsed = time.time() - start_time
            
            try:
                response_data = response.json()
            except:
                response_data = {"raw": response.text[:500]}
            
            return TestResult(
                test_name=f"{method} {endpoint}",
                passed=response.status_code in [200, 201],
                status_code=response.status_code,
                response_time=elapsed,
                response_data=response_data,
                error=None
            )
            
        except requests.exceptions.ConnectionError:
            return TestResult(
                test_name=f"{method} {endpoint}",
                passed=False,
                status_code=None,
                response_time=0,
                response_data=None,
                error="Connection refused - Is the API server running?"
            )
        except requests.exceptions.Timeout:
            return TestResult(
                test_name=f"{method} {endpoint}",
                passed=False,
                status_code=None,
                response_time=timeout,
                response_data=None,
                error=f"Request timed out after {timeout}s"
            )
        except Exception as e:
            return TestResult(
                test_name=f"{method} {endpoint}",
                passed=False,
                status_code=None,
                response_time=0,
                response_data=None,
                error=str(e)
            )
    
    def test_health_endpoint(self) -> TestResult:
        """Test 1: GET /health endpoint."""
        print_header("Test 1: GET /health - Health Check")
        
        result = self._make_request("GET", "/health")
        
        if result.error:
            print_error(result.error)
            print_info("Make sure the API server is running: python -m src.api.main")
        else:
            print_result(
                "Health Check",
                result.passed,
                f"Status: {result.status_code}, Time: {result.response_time:.3f}s"
            )
            
            if result.response_data:
                print(f"\n       Response:")
                print_json(result.response_data)
                
                # Validate response structure
                expected_keys = ["status", "model", "loaded"]
                has_all_keys = all(k in result.response_data for k in expected_keys)
                print_result(
                    "Response structure",
                    has_all_keys,
                    f"Expected keys: {expected_keys}"
                )
        
        self.results["health"] = result
        return result
    
    def test_models_endpoint(self) -> TestResult:
        """Test 2: GET /models endpoint."""
        print_header("Test 2: GET /models - List Available Models")
        
        result = self._make_request("GET", "/models")
        
        if result.error:
            print_error(result.error)
        else:
            print_result(
                "List Models",
                result.passed,
                f"Status: {result.status_code}, Time: {result.response_time:.3f}s"
            )
            
            if result.response_data and isinstance(result.response_data, list):
                print(f"\n       Found {len(result.response_data)} models:")
                for model in result.response_data:
                    active_status = " (ACTIVE)" if model.get('active') else ""
                    available = "✓" if model.get('available') else "✗"
                    print(f"         [{available}] {model.get('name', 'Unknown')}{active_status}")
                    print(f"             {Fore.CYAN}{model.get('description', 'No description')}{Style.RESET_ALL}")
        
        self.results["models"] = result
        return result
    
    def test_model_switch_endpoint(self) -> TestResult:
        """Test 3: POST /models/switch endpoint."""
        print_header("Test 3: POST /models/switch - Switch Model")
        
        # Try switching to different models
        test_models = [
            "StyleGAN (Original)",
            "Full Pipeline (Recommended)",
        ]
        
        for model_name in test_models:
            result = self._make_request(
                "POST", 
                "/models/switch",
                data={"model_name": model_name}
            )
            
            if result.error:
                print_error(result.error)
                break
            else:
                print_result(
                    f"Switch to '{model_name}'",
                    result.passed,
                    f"Status: {result.status_code}, Time: {result.response_time:.3f}s"
                )
                if result.response_data:
                    print(f"       Response: {result.response_data}")
        
        # Test invalid model
        invalid_result = self._make_request(
            "POST",
            "/models/switch",
            data={"model_name": "NonexistentModel123"}
        )
        
        # Should fail (400 status)
        invalid_test_passed = invalid_result.status_code == 400
        print_result(
            "Invalid model rejection",
            invalid_test_passed,
            f"Status: {invalid_result.status_code} (expected 400)"
        )
        
        self.results["model_switch"] = result
        return result
    
    def test_detect_image_endpoint(self) -> TestResult:
        """Test 4: POST /detect/image endpoint."""
        print_header("Test 4: POST /detect/image - Image Detection")
        
        if not self.sample_image:
            print_info("No sample image found for testing")
            result = TestResult(
                test_name="POST /detect/image",
                passed=False,
                status_code=None,
                response_time=0,
                response_data=None,
                error="No sample image found"
            )
            self.results["detect_image"] = result
            return result
        
        print_info(f"Using sample image: {self.sample_image.name}")
        
        with open(self.sample_image, 'rb') as f:
            result = self._make_request(
                "POST",
                "/detect/image",
                files={"file": (self.sample_image.name, f, "image/jpeg")},
                data={"threshold": "0.5", "model_name": "Full Pipeline (Recommended)"}
            )
        
        if result.error:
            print_error(result.error)
        else:
            print_result(
                "Image Detection",
                result.passed,
                f"Status: {result.status_code}, Time: {result.response_time:.3f}s"
            )
            
            if result.response_data and result.passed:
                print(f"\n       {Fore.GREEN}Detection Result:{Style.RESET_ALL}")
                print(f"         Label: {result.response_data.get('label')}")
                print(f"         Probability: {result.response_data.get('probability', 0):.4f}")
                print(f"         Confidence: {result.response_data.get('confidence', 0):.4f}")
                print(f"         Model: {result.response_data.get('model')}")
                
                details = result.response_data.get('details', {})
                print(f"         GAN Score: {details.get('gan_score', 'N/A')}")
                print(f"         GAN Family: {details.get('gan_family', 'N/A')}")
                
                # Validate response structure
                required_fields = ["label", "probability", "confidence", "model"]
                has_all_fields = all(f in result.response_data for f in required_fields)
                print_result(
                    "Response structure validation",
                    has_all_fields,
                    f"Required fields: {required_fields}"
                )
        
        self.results["detect_image"] = result
        return result
    
    def test_detect_video_endpoint(self) -> TestResult:
        """Test 5: POST /detect/video endpoint."""
        print_header("Test 5: POST /detect/video - Video Detection")
        
        if not self.sample_video:
            print_info("No sample video found for testing")
            result = TestResult(
                test_name="POST /detect/video",
                passed=False,
                status_code=None,
                response_time=0,
                response_data=None,
                error="No sample video found"
            )
            self.results["detect_video"] = result
            return result
        
        print_info(f"Using sample video: {self.sample_video.name}")
        print_info("This may take a while depending on video length...")
        
        with open(self.sample_video, 'rb') as f:
            result = self._make_request(
                "POST",
                "/detect/video",
                files={"file": (self.sample_video.name, f, "video/mp4")},
                data={"threshold": "0.5", "model_name": "Full Pipeline (Recommended)"},
                timeout=300  # 5 minute timeout for video processing
            )
        
        if result.error:
            print_error(result.error)
        else:
            print_result(
                "Video Detection",
                result.passed,
                f"Status: {result.status_code}, Time: {result.response_time:.3f}s"
            )
            
            if result.response_data and result.passed:
                print(f"\n       {Fore.GREEN}Detection Result:{Style.RESET_ALL}")
                print(f"         Label: {result.response_data.get('label')}")
                print(f"         Probability: {result.response_data.get('probability', 0):.4f}")
                print(f"         Confidence: {result.response_data.get('confidence', 0):.4f}")
                print(f"         Model: {result.response_data.get('model')}")
                
                details = result.response_data.get('details', {})
                print(f"         Video Score: {details.get('video_score', 'N/A')}")
                print(f"         Mean Frame Score: {details.get('mean_frame_score', 'N/A')}")
                print(f"         Score Source: {details.get('score_source', 'N/A')}")
                
                # Frame scores summary
                frame_scores = result.response_data.get('frame_scores')
                if frame_scores:
                    print(f"         Frame Scores: {len(frame_scores)} frames analyzed")
                    print(f"         Max Frame Score: {max(frame_scores):.4f}")
                    print(f"         Min Frame Score: {min(frame_scores):.4f}")
                
                # Validate response structure
                required_fields = ["label", "probability", "confidence", "model", "details"]
                has_all_fields = all(f in result.response_data for f in required_fields)
                print_result(
                    "Response structure validation",
                    has_all_fields,
                    f"Required fields: {required_fields}"
                )
        
        self.results["detect_video"] = result
        return result
    
    def test_error_handling(self) -> TestResult:
        """Test 6: Error handling for invalid requests."""
        print_header("Test 6: Error Handling")
        
        # Test missing file
        result = self._make_request("POST", "/detect/image", data={})
        missing_file_handled = result.status_code == 400
        print_result(
            "Missing file error",
            missing_file_handled,
            f"Status: {result.status_code} (expected 400)"
        )
        
        # Test invalid threshold
        if self.sample_image:
            with open(self.sample_image, 'rb') as f:
                result = self._make_request(
                    "POST",
                    "/detect/image",
                    files={"file": (self.sample_image.name, f, "image/jpeg")},
                    data={"threshold": "invalid"}
                )
            invalid_threshold_handled = result.status_code == 400
            print_result(
                "Invalid threshold error",
                invalid_threshold_handled,
                f"Status: {result.status_code} (expected 400)"
            )
        else:
            print_info("Skipping invalid threshold test (no sample image)")
            invalid_threshold_handled = True
        
        # Test invalid endpoint
        result = self._make_request("GET", "/nonexistent")
        invalid_endpoint_handled = result.status_code in [404, 405]
        print_result(
            "Invalid endpoint error",
            invalid_endpoint_handled,
            f"Status: {result.status_code} (expected 404/405)"
        )
        
        all_passed = missing_file_handled and invalid_threshold_handled and invalid_endpoint_handled
        self.results["error_handling"] = TestResult(
            test_name="Error Handling",
            passed=all_passed,
            status_code=None,
            response_time=0,
            response_data=None,
            error=None
        )
        return self.results["error_handling"]
    
    def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all API endpoint tests."""
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"{Style.BRIGHT}    API ENDPOINTS TESTING SUITE")
        print(f"    Deepfake Detection System")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        print(f"API Base URL: {self.base_url}")
        print(f"Sample Image: {self.sample_image.name if self.sample_image else 'Not found'}")
        print(f"Sample Video: {self.sample_video.name if self.sample_video else 'Not found'}")
        
        # Run all tests
        self.test_health_endpoint()
        self.test_models_endpoint()
        self.test_model_switch_endpoint()
        self.test_detect_image_endpoint()
        self.test_detect_video_endpoint()
        self.test_error_handling()
        
        # Print summary
        print_header("TEST SUMMARY")
        
        passed = sum(1 for r in self.results.values() if r.passed)
        total = len(self.results)
        
        for name, result in self.results.items():
            status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if result.passed else f"{Fore.RED}FAIL{Style.RESET_ALL}"
            time_str = f"({result.response_time:.2f}s)" if result.response_time else ""
            print(f"  {name}: [{status}] {time_str}")
            if result.error:
                print(f"       Error: {Fore.RED}{result.error}{Style.RESET_ALL}")
        
        print(f"\n{'-'*40}")
        overall_pass = passed == total
        if overall_pass:
            print(f"{Fore.GREEN}{Style.BRIGHT}All tests passed: {passed}/{total}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Tests passed: {passed}/{total}{Style.RESET_ALL}")
        
        return self.results


def main():
    """Main entry point for API testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test API endpoints")
    parser.add_argument("--url", default=API_BASE_URL, help="API base URL")
    args = parser.parse_args()
    
    tester = APIEndpointTester(base_url=args.url)
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(r.passed for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
