#!/usr/bin/env python3
"""
Deepfake Detection System - Setup Script
=========================================

This script automatically sets up all dependencies for the project with
comprehensive fallback logic to handle installation failures gracefully.

Usage:
    python setup.py              # Full setup
    python setup.py --backend    # Backend only
    python setup.py --frontend   # Frontend only
    python setup.py --check      # Check installation status only

Created: December 10, 2025
"""

import subprocess
import sys
import os
import platform
import shutil
import argparse
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def run_command(command, capture_output=False, check=False):
    """Run a shell command with error handling."""
    try:
        if capture_output:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(command, shell=True, check=check)
            return result.returncode == 0, "", ""
    except subprocess.CalledProcessError as e:
        return False, "", str(e)
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Check if Python version is compatible."""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False

def check_node_version():
    """Check if Node.js is installed."""
    print_info("Checking Node.js...")
    success, stdout, _ = run_command("node --version", capture_output=True)
    if success and stdout.strip():
        print_success(f"Node.js {stdout.strip()} ✓")
        return True
    else:
        print_warning("Node.js not found. Frontend setup will be skipped.")
        print_info("Install Node.js from https://nodejs.org/")
        return False

def check_npm():
    """Check if npm is installed."""
    print_info("Checking npm...")
    success, stdout, _ = run_command("npm --version", capture_output=True)
    if success and stdout.strip():
        print_success(f"npm {stdout.strip()} ✓")
        return True
    else:
        print_warning("npm not found.")
        return False

def install_pip_package(package, fallback_packages=None, extra_index=None):
    """
    Install a pip package with fallback options.
    
    Args:
        package: Primary package to install
        fallback_packages: List of fallback packages if primary fails
        extra_index: Alternative pip index URL
    """
    print_info(f"Installing {package}...")
    
    # Try primary installation
    cmd = f"pip install {package}"
    if extra_index:
        cmd = f"pip install {package} --index-url {extra_index}"
    
    success, _, stderr = run_command(cmd, capture_output=True)
    
    if success:
        print_success(f"{package} installed successfully")
        return True
    
    # Try fallback packages
    if fallback_packages:
        for fallback in fallback_packages:
            print_warning(f"{package} failed, trying fallback: {fallback}")
            success, _, _ = run_command(f"pip install {fallback}", capture_output=True)
            if success:
                print_success(f"{fallback} installed as fallback")
                return True
    
    print_error(f"Failed to install {package}")
    if stderr:
        print_info(f"Error: {stderr[:200]}...")
    return False

def install_python_dependencies():
    """Install all Python dependencies with fallback logic."""
    print_header("Installing Python Dependencies")
    
    # Core packages with fallbacks
    packages = [
        # PyTorch - try CUDA first, then CPU
        {
            "name": "torch torchvision",
            "fallback": ["torch torchvision --index-url https://download.pytorch.org/whl/cpu"],
            "required": True
        },
        # TensorFlow - try GPU first, then CPU
        {
            "name": "tensorflow",
            "fallback": ["tensorflow-cpu", "tensorflow==2.13.0"],
            "required": True
        },
        # Image processing
        {
            "name": "opencv-python",
            "fallback": ["opencv-python-headless"],
            "required": True
        },
        {
            "name": "Pillow",
            "fallback": ["Pillow==9.5.0"],
            "required": True
        },
        {
            "name": "scikit-image",
            "fallback": ["scikit-image==0.21.0"],
            "required": True
        },
        # Face detection
        {
            "name": "mediapipe",
            "fallback": ["mediapipe==0.10.0"],
            "required": False  # Not critical, has fallback in code
        },
        # Scientific computing
        {
            "name": "numpy",
            "fallback": ["numpy==1.24.0"],
            "required": True
        },
        {
            "name": "scipy",
            "fallback": ["scipy==1.11.0"],
            "required": True
        },
        {
            "name": "scikit-learn",
            "fallback": [],
            "required": True
        },
        # Web framework
        {
            "name": "flask flask-cors flask-jwt-extended flask-sqlalchemy",
            "fallback": [],
            "required": True
        },
        # Additional tools
        {
            "name": "tqdm",
            "fallback": [],
            "required": False
        },
        {
            "name": "PyYAML",
            "fallback": [],
            "required": False
        },
        {
            "name": "imageio imageio-ffmpeg",
            "fallback": ["imageio"],
            "required": False
        },
    ]
    
    failed_critical = []
    failed_optional = []
    
    for pkg_info in packages:
        name = pkg_info["name"]
        fallbacks = pkg_info.get("fallback", [])
        required = pkg_info.get("required", True)
        
        success = install_pip_package(name, fallbacks)
        
        if not success:
            if required:
                failed_critical.append(name)
            else:
                failed_optional.append(name)
    
    # Summary
    print("\n" + "-"*50)
    if failed_critical:
        print_error(f"Critical packages failed: {', '.join(failed_critical)}")
        print_info("Try running: pip install <package-name>")
    else:
        print_success("All critical packages installed!")
    
    if failed_optional:
        print_warning(f"Optional packages failed: {', '.join(failed_optional)}")
        print_info("These are not critical, system will work without them.")
    
    return len(failed_critical) == 0

def install_from_requirements():
    """Try to install from requirements.txt as a batch."""
    print_info("Attempting batch install from requirements.txt...")
    
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print_warning("requirements.txt not found, using individual package installation.")
        return False
    
    success, _, stderr = run_command("pip install -r requirements.txt", capture_output=True)
    
    if success:
        print_success("All requirements installed from requirements.txt")
        return True
    else:
        print_warning("Batch install failed, falling back to individual packages.")
        return False

def install_frontend_dependencies():
    """Install frontend Node.js dependencies."""
    print_header("Installing Frontend Dependencies")
    
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print_error("frontend/ directory not found!")
        return False
    
    # Check if package.json exists
    package_json = frontend_path / "package.json"
    if not package_json.exists():
        print_error("frontend/package.json not found!")
        return False
    
    # Change to frontend directory
    original_dir = os.getcwd()
    os.chdir(frontend_path)
    
    try:
        # Try npm install
        print_info("Running npm install...")
        success, _, stderr = run_command("npm install", capture_output=True)
        
        if success:
            print_success("Frontend dependencies installed!")
            os.chdir(original_dir)
            return True
        
        # Fallback 1: Try with --legacy-peer-deps
        print_warning("npm install failed, trying with --legacy-peer-deps...")
        success, _, _ = run_command("npm install --legacy-peer-deps", capture_output=True)
        
        if success:
            print_success("Frontend dependencies installed with legacy peer deps!")
            os.chdir(original_dir)
            return True
        
        # Fallback 2: Try with --force
        print_warning("Still failing, trying with --force...")
        success, _, _ = run_command("npm install --force", capture_output=True)
        
        if success:
            print_success("Frontend dependencies installed with force!")
            os.chdir(original_dir)
            return True
        
        # Fallback 3: Clear cache and retry
        print_warning("Still failing, clearing npm cache...")
        run_command("npm cache clean --force", capture_output=True)
        success, _, _ = run_command("npm install", capture_output=True)
        
        if success:
            print_success("Frontend dependencies installed after cache clear!")
            os.chdir(original_dir)
            return True
        
        print_error("Failed to install frontend dependencies.")
        print_info("Try manually: cd frontend && npm install")
        
    finally:
        os.chdir(original_dir)
    
    return False

def verify_model_files():
    """Check if all model files are present."""
    print_header("Verifying Model Files")
    
    model_files = [
        "checkpoints/spatiotemporal/best_model.pt",
        "checkpoints/gan_fingerprint/best_model.pt",
        "checkpoints/fusion/best_model.pt",
        "src/api/ai_generated/image_classifier_model.keras",
        "src/api/ai_generated/multi_check_detector.py",
    ]
    
    missing = []
    for model_file in model_files:
        path = Path(model_file)
        if path.exists():
            print_success(f"Found: {model_file}")
        else:
            print_warning(f"Missing: {model_file}")
            missing.append(model_file)
    
    if missing:
        print_warning(f"\n{len(missing)} model file(s) missing.")
        print_info("Some detection features may not work without these files.")
        return False
    else:
        print_success("\nAll model files present!")
        return True

def test_imports():
    """Test if critical imports work."""
    print_header("Testing Imports")
    
    imports = [
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("flask", "Flask"),
        ("PIL", "Pillow"),
    ]
    
    failed = []
    for module, name in imports:
        try:
            __import__(module)
            print_success(f"{name} imports correctly")
        except ImportError as e:
            print_error(f"{name} import failed: {e}")
            failed.append(name)
    
    if failed:
        print_warning(f"\n{len(failed)} import(s) failed: {', '.join(failed)}")
        return False
    else:
        print_success("\nAll critical imports working!")
        return True

def create_env_file():
    """Create .env file with default configuration."""
    print_info("Creating .env configuration file...")
    
    env_content = """# Deepfake Detection System Configuration
# Created by setup.py

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=0
FLASK_PORT=8000

# JWT Secret (change in production!)
JWT_SECRET_KEY=your-secret-key-change-in-production

# Database
DATABASE_URL=sqlite:///deepfake_detection.db

# Model Configuration
USE_GPU=auto
MAX_BATCH_SIZE=32

# Logging
LOG_LEVEL=INFO
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        print_success(".env file created")
    else:
        print_info(".env file already exists, skipping")

def print_final_instructions():
    """Print final setup instructions."""
    print_header("Setup Complete!")
    
    print(f"""
{Colors.GREEN}Your Deepfake Detection System is ready!{Colors.END}

{Colors.BOLD}To run the application:{Colors.END}

  1. Start the backend (Terminal 1):
     {Colors.BLUE}python -m src.api.main{Colors.END}

  2. Start the frontend (Terminal 2):
     {Colors.BLUE}cd frontend && npm run dev{Colors.END}

  3. Open in browser:
     {Colors.BLUE}http://localhost:3000{Colors.END}

{Colors.BOLD}Quick test:{Colors.END}
  {Colors.BLUE}curl http://localhost:8000/health{Colors.END}

{Colors.BOLD}For more details, see:{Colors.END}
  - COMMANDS.txt - All available commands
  - README.md - Project documentation
  - COMPLETE_IMPLEMENTATION_GUIDE.txt - Full system guide
""")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Deepfake Detection System")
    parser.add_argument("--backend", action="store_true", help="Install backend only")
    parser.add_argument("--frontend", action="store_true", help="Install frontend only")
    parser.add_argument("--check", action="store_true", help="Check installation status only")
    args = parser.parse_args()
    
    print_header("Deepfake Detection System Setup")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("")
    
    # Check mode
    if args.check:
        check_python_version()
        check_node_version()
        check_npm()
        verify_model_files()
        test_imports()
        return
    
    # Full setup or selective
    install_backend = not args.frontend  # Install backend unless --frontend only
    install_frontend_flag = not args.backend  # Install frontend unless --backend only
    
    success = True
    
    # Step 1: Check Python
    if not check_python_version():
        print_error("Please install Python 3.8 or higher.")
        sys.exit(1)
    
    # Step 2: Backend setup
    if install_backend:
        # Try batch install first
        if not install_from_requirements():
            # Fall back to individual package installation
            if not install_python_dependencies():
                success = False
                print_warning("Some backend packages failed to install.")
        
        # Test imports
        test_imports()
    
    # Step 3: Frontend setup
    if install_frontend_flag:
        if check_node_version() and check_npm():
            if not install_frontend_dependencies():
                success = False
                print_warning("Frontend setup had issues.")
        else:
            print_warning("Skipping frontend setup (Node.js not found).")
    
    # Step 4: Verify models
    verify_model_files()
    
    # Step 5: Create config
    create_env_file()
    
    # Final instructions
    print_final_instructions()
    
    if not success:
        print_warning("\nSetup completed with some issues.")
        print_info("Check the errors above and try manual installation if needed.")
        sys.exit(1)
    else:
        print_success("\nSetup completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
