#!/usr/bin/env python
"""Quick test script for GAN family detection API."""

import requests
import json

IMAGE_PATH = r"stylegan\Final Dataset\Fake\0VLCASRXV2.jpg"
API_URL = "http://localhost:8000/detect/image"

def test_image_detection():
    print("Testing /detect/image endpoint...")
    with open(IMAGE_PATH, 'rb') as f:
        response = requests.post(
            API_URL,
            files={'file': f},
            data={'threshold': 0.5, 'model_name': 'Full Pipeline (Recommended)'}
        )
    
    result = response.json()
    print(json.dumps(result, indent=2))
    
    # Validate GAN family data is present
    details = result.get('details', {})
    print("\n--- GAN Family Detection Validation ---")
    print(f"gan_family: {details.get('gan_family')}")
    print(f"gan_family_confidence: {details.get('gan_family_confidence')}")
    print(f"family_probs: {details.get('family_probs')}")
    
    if details.get('family_probs'):
        print("\n✅ GAN Family detection is working!")
    else:
        print("\n❌ GAN Family detection not returning expected data")

if __name__ == "__main__":
    test_image_detection()
