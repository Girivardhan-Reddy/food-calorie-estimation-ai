import os
import random
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------
SOURCE_DIR = "data"             # original dataset
OUTPUT_DIR = "split_data"       # FINAL OUTPUT FOLDER

TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

IMAGE_SIZE = (224, 224)

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42
# ----------------------------------------

random.seed(RANDOM_SEED)

def create_dir(path):
    os.makedirs(path, exist_ok=True)

# Create base directories
for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    create_dir(os.path.join(OUTPUT_DIR, split_dir))

# Collect classes
classes = [
    d for d in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, d))
]

print(f"📁 Found {len(classes)} classes")

# Process each class
for class_name in tqdm(classes, desc="🍔 Processing Classes"):
    class_path = os.path.join(SOURCE_DIR, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    splits = {
        TRAIN_DIR: images[:train_end],
        VAL_DIR: images[train_end:val_end],
        TEST_DIR: images[val_end:]
    }

    # Create class directories
    for split_dir in splits:
        create_dir(os.path.join(OUTPUT_DIR, split_dir, class_name))

    # Resize + save with progress bar
    for split_dir, imgs in splits.items():
        for img_name in tqdm(
            imgs,
            desc=f"🖼️ {class_name} → {split_dir}",
            leave=False
        ):
            src_img_path = os.path.join(class_path, img_name)
            dst_img_path = os.path.join(
                OUTPUT_DIR, split_dir, class_name, img_name
            )

            try:
                with Image.open(src_img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
                    img.save(dst_img_path, quality=95)
            except Exception as e:
                print(f"⚠️ Skipped {src_img_path}: {e}")

print("\n🎉 Images resized, split, and saved in 'split_data/' successfully!")
