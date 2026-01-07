"""
Production server using Waitress WSGI server.

Usage:
    python run_production.py
"""

from waitress import serve
from src.api.main import app

if __name__ == '__main__':
    print("=" * 70)
    print("Starting Production Deepfake Detection API Server")
    print("=" * 70)
    print(f"Server running on: http://0.0.0.0:8000")
    print(f"Local URL: http://127.0.0.1:8000")
    print(f"Press CTRL+C to quit")
    print("=" * 70)
    
    # Serve with Waitress
    serve(app, host='0.0.0.0', port=8000, threads=4)
