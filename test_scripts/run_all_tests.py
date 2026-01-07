#!/usr/bin/env python
"""
Master Test Runner Script
=========================
Runs all test scripts for the deepfake detection system.

Usage:
    python run_all_tests.py            # Run all tests
    python run_all_tests.py database   # Run only database tests
    python run_all_tests.py api        # Run only API tests
    python run_all_tests.py --list     # List available tests
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from colorama import init, Fore, Style

# Initialize colorama
init()


# Test configurations
TEST_SCRIPTS = {
    "database": {
        "name": "Database Testing",
        "script": "database_testing/test_database.py",
        "description": "Dataset validation and input-output verification"
    },
    "api": {
        "name": "API Endpoints Testing",
        "script": "api_testing/test_api_endpoints.py",
        "description": "Tests for /detect/video, /detect/image, /models, /health endpoints"
    },
    "models": {
        "name": "Model Testing",
        "script": "model_testing/test_models.py",
        "description": "Spatiotemporal and GAN Fingerprint model accuracy tests"
    },
    "frontend": {
        "name": "Frontend & PDF Testing",
        "script": "frontend_testing/test_frontend.py",
        "description": "Frontend components and PDF report generation tests"
    },
    "dashboard": {
        "name": "Dashboard Output Testing",
        "script": "dashboard_testing/test_dashboard_output.py",
        "description": "Deepfake/Real label, confidence, GAN signal, family prediction"
    },
    "video": {
        "name": "Video Detection Testing",
        "script": "video_detection_testing/test_video_detection.py",
        "description": "Frame-level regions, spatiotemporal score, fusion score"
    },
    "image": {
        "name": "Image Detection Testing",
        "script": "image_detection_testing/test_image_detection.py",
        "description": "GAN fingerprint result and family classification"
    },
    "accuracy": {
        "name": "Accuracy Graph Testing",
        "script": "accuracy_graphs/test_accuracy_graphs.py",
        "description": "GAN, spatiotemporal, and fusion model accuracy graphs"
    }
}


def print_banner():
    """Print the test runner banner."""
    banner = f"""
{Fore.MAGENTA}╔══════════════════════════════════════════════════════════════╗
║                                                                ║
║  {Style.BRIGHT}DEEPFAKE DETECTION SYSTEM - TEST SUITE{Style.RESET_ALL}{Fore.MAGENTA}                       ║
║                                                                ║
║  Tests for models, API, frontend, and output validation        ║
║                                                                ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{Fore.CYAN}{'═'*60}")
    print(f"{Style.BRIGHT}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═'*60}{Style.RESET_ALL}")


def run_test(test_key: str, test_config: dict) -> Tuple[bool, float, str]:
    """
    Run a single test script.
    
    Returns:
        Tuple of (passed, duration, output)
    """
    script_path = Path(__file__).parent / test_config["script"]
    
    if not script_path.exists():
        return False, 0, f"Script not found: {script_path}"
    
    print(f"\n{Fore.YELLOW}Running {test_config['name']}...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Script: {script_path}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{test_config['description']}{Style.RESET_ALL}")
    print(f"{'-'*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"{Fore.RED}{result.stderr}{Style.RESET_ALL}")
        
        passed = result.returncode == 0
        return passed, duration, result.stdout + result.stderr
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return False, duration, "Test timed out after 5 minutes"
    except Exception as e:
        duration = time.time() - start_time
        return False, duration, str(e)


def list_tests():
    """List all available tests."""
    print_header("Available Tests")
    
    for key, config in TEST_SCRIPTS.items():
        print(f"\n  {Fore.GREEN}{key}{Style.RESET_ALL}")
        print(f"    Name: {config['name']}")
        print(f"    Description: {config['description']}")
        print(f"    Script: {config['script']}")


def run_all_tests(selected_tests: List[str] = None) -> Dict[str, Tuple[bool, float]]:
    """
    Run all or selected tests.
    
    Returns:
        Dictionary of test results
    """
    print_banner()
    
    results = {}
    
    tests_to_run = selected_tests if selected_tests else list(TEST_SCRIPTS.keys())
    
    print_header(f"Running {len(tests_to_run)} Test Suite(s)")
    
    total_start = time.time()
    
    for test_key in tests_to_run:
        if test_key not in TEST_SCRIPTS:
            print(f"{Fore.RED}Unknown test: {test_key}{Style.RESET_ALL}")
            continue
        
        config = TEST_SCRIPTS[test_key]
        passed, duration, _ = run_test(test_key, config)
        results[test_key] = (passed, duration)
    
    total_duration = time.time() - total_start
    
    # Print summary
    print_header("TEST RESULTS SUMMARY")
    
    passed_count = sum(1 for passed, _ in results.values() if passed)
    total_count = len(results)
    
    print(f"\n  {'Test':<25} {'Status':>10} {'Duration':>12}")
    print(f"  {'-'*50}")
    
    for test_key, (passed, duration) in results.items():
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if passed else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        print(f"  {TEST_SCRIPTS[test_key]['name']:<25} [{status}]    {duration:>8.2f}s")
    
    print(f"  {'-'*50}")
    print(f"  {'Total':<25} {passed_count}/{total_count}    {total_duration:>8.2f}s")
    
    # Final result
    if passed_count == total_count:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ All tests passed!{Style.RESET_ALL}")
    else:
        failed_count = total_count - passed_count
        print(f"\n{Fore.RED}{Style.BRIGHT}✗ {failed_count} test(s) failed{Style.RESET_ALL}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run deepfake detection system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all_tests.py                  # Run all tests
    python run_all_tests.py database api     # Run database and API tests
    python run_all_tests.py --list           # List available tests
        """
    )
    
    parser.add_argument(
        "tests",
        nargs="*",
        help="Specific tests to run (default: all)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available tests"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
        sys.exit(0)
    
    # Run tests
    selected_tests = args.tests if args.tests else None
    results = run_all_tests(selected_tests)
    
    # Exit with appropriate code
    all_passed = all(passed for passed, _ in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
