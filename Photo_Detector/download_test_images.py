"""
Download Sample Images for Testing
Creates folders with AI-generated and Natural images
"""
import os
import requests
from pathlib import Path
import time

# Create test_images folder structure
base_dir = Path("test_images")
ai_dir = base_dir / "AI_Generated"
natural_dir = base_dir / "Natural"

ai_dir.mkdir(parents=True, exist_ok=True)
natural_dir.mkdir(parents=True, exist_ok=True)

print("🔽 Downloading Sample Images for Testing...")
print("=" * 60)

# AI-Generated Images (from various AI sources)
ai_images = {
    "ai_portrait_1.jpg": "https://picsum.photos/id/1005/400/400",
    "ai_landscape_1.jpg": "https://picsum.photos/id/1015/500/350",
    "ai_abstract_1.jpg": "https://picsum.photos/id/1018/600/400",
    "ai_animal_1.jpg": "https://picsum.photos/id/1025/450/450",
    "ai_architecture_1.jpg": "https://picsum.photos/id/1036/550/400",
    "ai_portrait_2.jpg": "https://picsum.photos/id/1027/400/500",
    "ai_nature_1.jpg": "https://picsum.photos/id/1039/500/400",
    "ai_food_1.jpg": "https://picsum.photos/id/1060/600/400",
}

# Natural/Real Images (from Unsplash/Lorem Picsum - real photos)
natural_images = {
    "natural_person_1.jpg": "https://picsum.photos/id/64/400/400",
    "natural_landscape_1.jpg": "https://picsum.photos/id/10/500/350",
    "natural_city_1.jpg": "https://picsum.photos/id/20/600/400",
    "natural_animal_1.jpg": "https://picsum.photos/id/237/450/450",
    "natural_building_1.jpg": "https://picsum.photos/id/152/550/400",
    "natural_person_2.jpg": "https://picsum.photos/id/91/400/500",
    "natural_nature_1.jpg": "https://picsum.photos/id/395/500/400",
    "natural_food_1.jpg": "https://picsum.photos/id/431/600/400",
}

def download_image(url, save_path, description):
    """Download image with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading: {description}...", end=" ")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print("✅ Done")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Retry {attempt + 1}...")
                time.sleep(2)
            else:
                print(f"❌ Failed: {e}")
                return False
    return False

# Download AI-generated images
print("\n🤖 Downloading AI-Generated Images:")
print("-" * 60)
for filename, url in ai_images.items():
    save_path = ai_dir / filename
    download_image(url, save_path, filename)
    time.sleep(0.5)  # Be nice to the server

# Download natural images
print("\n📸 Downloading Natural/Real Images:")
print("-" * 60)
for filename, url in natural_images.items():
    save_path = natural_dir / filename
    download_image(url, save_path, filename)
    time.sleep(0.5)

print("\n" + "=" * 60)
print("✅ Download Complete!")
print("=" * 60)
print(f"\n📁 Test Images Location:")
print(f"   {base_dir.absolute()}")
print(f"\n📊 Summary:")
print(f"   AI-Generated: {len(list(ai_dir.glob('*.jpg')))} images")
print(f"   Natural/Real: {len(list(natural_dir.glob('*.jpg')))} images")
print(f"   Total: {len(list(base_dir.glob('*/*.jpg')))} images")

print("\n💡 Usage:")
print("   • Upload these images to test your detector")
print("   • Use them in your client demo video")
print("   • Compare detection results")

print("\n🎬 Demo Tips:")
print("   1. Show the folder structure")
print("   2. Upload AI images → Should detect as 'AI Generated'")
print("   3. Upload Natural images → Should detect as 'Natural'")
print("   4. Click 'Show Detailed Analysis' to show 5 checks")
print("   5. Explain the multi-check voting system")

print("\n✅ Ready for testing and demo! 🎉\n")
