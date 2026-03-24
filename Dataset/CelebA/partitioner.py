# python3 partitioner.py [...] -> partitioned dataset roots
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMG_DIR = Path("Dataset/CelebA/img_align_celeba")
IDENTITY_FILE = Path("Dataset/CelebA/identity_CelebA.txt")
OUTPUT_DIR = Path("Dataset/CelebA")

MIN_IMAGES_PER_IDENTITY = 15     # only keep identities with at least this many images
MAX_IDENTITIES = 500             # set to None to keep all eligible identities
MAX_IMAGES_PER_IDENTITY = 20     # set to None to keep all images for each identity
TRAIN_RATIO = 0.8
SEED = 42


def read_identity_file(identity_file: Path):
    """
    Reads identity_CelebA.txt and returns:
        {identity_id: [img1.jpg, img2.jpg, ...]}
    """
    identity_to_images = defaultdict(list)

    with open(identity_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_name, identity_id = parts
            identity_to_images[identity_id].append(img_name)

    return identity_to_images


def filter_identities(identity_to_images, min_images, max_identities=None, max_images_per_identity=None):
    """
    Filters identities by image count and optionally limits identities/images.
    Returns a new dict.
    """
    filtered = {
        identity: images
        for identity, images in identity_to_images.items()
        if len(images) >= min_images
    }

    identities = sorted(filtered.keys())

    if max_identities is not None:
        identities = identities[:max_identities]

    result = {}
    for identity in identities:
        images = sorted(filtered[identity])

        if max_images_per_identity is not None:
            images = images[:max_images_per_identity]

        result[identity] = images

    return result


def make_split_dirs(base_dir: Path):
    for split in ["train", "val"]:
        (base_dir / split).mkdir(parents=True, exist_ok=True)


def copy_images_by_identity(identity_to_images, img_dir: Path, output_dir: Path, train_ratio=0.8, seed=42):
    """
    Creates:
        output_dir/
            train/id_xxxxx/
            val/id_xxxxx/
    """
    random.seed(seed)
    make_split_dirs(output_dir)

    total_train = 0
    total_val = 0

    for identity, images in identity_to_images.items():
        images = images.copy()
        random.shuffle(images)

        split_idx = max(1, int(len(images) * train_ratio))
        if len(images) > 1 and split_idx == len(images):
            split_idx -= 1

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        identity_folder = f"id_{int(identity):05d}"

        train_identity_dir = output_dir / "train" / identity_folder
        val_identity_dir = output_dir / "val" / identity_folder

        train_identity_dir.mkdir(parents=True, exist_ok=True)
        val_identity_dir.mkdir(parents=True, exist_ok=True)

        for img_name in train_imgs:
            src = img_dir / img_name
            dst = train_identity_dir / img_name
            if src.exists():
                shutil.copy2(src, dst)
                total_train += 1

        for img_name in val_imgs:
            src = img_dir / img_name
            dst = val_identity_dir / img_name
            if src.exists():
                shutil.copy2(src, dst)
                total_val += 1

    print(f"Done.")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")
    print(f"Identities:   {len(identity_to_images)}")
    print(f"Saved to:     {output_dir.resolve()}")


def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    if not IDENTITY_FILE.exists():
        raise FileNotFoundError(f"Identity file not found: {IDENTITY_FILE}")

    if OUTPUT_DIR.exists():
        print(f"Output directory already exists: {OUTPUT_DIR.resolve()}")
        print("Delete it first if you want a fresh rebuild.")
        return

    print("Reading identity file...")
    identity_to_images = read_identity_file(IDENTITY_FILE)

    print(f"Total raw identities: {len(identity_to_images)}")

    print("Filtering identities...")
    filtered = filter_identities(
        identity_to_images,
        min_images=MIN_IMAGES_PER_IDENTITY,
        max_identities=MAX_IDENTITIES,
        max_images_per_identity=MAX_IMAGES_PER_IDENTITY,
    )

    kept_images = sum(len(v) for v in filtered.values())
    print(f"Kept identities: {len(filtered)}")
    print(f"Kept images:     {kept_images}")

    print("Copying images into ImageFolder structure...")
    copy_images_by_identity(
        filtered,
        img_dir=IMG_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
