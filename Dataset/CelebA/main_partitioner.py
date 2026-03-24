# python3 main_partitioner.py [...] -> partitioned dataset roots
import shutil
from collections import defaultdict
from pathlib import Path


IMG_DIR = Path("Dataset/CelebA/img_align_celeba")
IDENTITY_FILE = Path("Dataset/CelebA/identity_CelebA.txt")
OUTPUT_DIR = Path("Dataset/CelebA")
PARTITION_NAME = "main"

MIN_IMAGES_PER_IDENTITY = 1     # only keep identities with at least this many images
MAX_IDENTITIES = None             # set to None to keep all eligible identities
MAX_IMAGES_PER_IDENTITY = None     # set to None to keep all images for each identity


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


def make_main_dir(base_dir: Path, partition_name: str):
    (base_dir / partition_name).mkdir(parents=True, exist_ok=True)


def copy_images_by_identity(identity_to_images, img_dir: Path, output_dir: Path, partition_name: str):
    """
    Creates:
        output_dir/
            main/id_xxxxx/
    """
    make_main_dir(output_dir, partition_name)

    total_main = 0

    for identity, images in identity_to_images.items():
        identity_folder = f"id_{int(identity):05d}"

        identity_dir = output_dir / partition_name / identity_folder
        identity_dir.mkdir(parents=True, exist_ok=True)

        for img_name in images:
            src = img_dir / img_name
            dst = identity_dir / img_name
            if src.exists():
                shutil.copy2(src, dst)
                total_main += 1

    print(f"Done.")
    print(f"Main images:  {total_main}")
    print(f"Identities:   {len(identity_to_images)}")
    print(f"Saved to:     {output_dir.resolve()}")


def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    if not IDENTITY_FILE.exists():
        raise FileNotFoundError(f"Identity file not found: {IDENTITY_FILE}")

    main_output_dir = OUTPUT_DIR / PARTITION_NAME

    if main_output_dir.exists():
        print(f"Output directory already exists: {main_output_dir.resolve()}")
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

    print("Copying images into main partition...")
    copy_images_by_identity(
        filtered,
        img_dir=IMG_DIR,
        output_dir=OUTPUT_DIR,
        partition_name=PARTITION_NAME,
    )


if __name__ == "__main__":
    main()
