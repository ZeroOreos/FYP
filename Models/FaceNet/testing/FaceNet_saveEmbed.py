# python3 FaceNet_saveEmbed.py <root_dir> [...] -> embedding output
import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceFolderDataset(Dataset):
    """
    Expects:
        root/
            identity_1/
                img1.jpg
                img2.jpg
            identity_2/
                img3.jpg
                ...

    Returns:
        image_tensor, identity_name, image_path
    """

    def __init__(self, root_dir: str, image_size: int = 160):
        self.root_dir = Path(root_dir)
        self.samples: List[Tuple[str, str]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root_dir}")

        for identity_dir in sorted(self.root_dir.iterdir()):
            if not identity_dir.is_dir():
                continue

            identity_name = identity_dir.name
            for img_path in sorted(identity_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((str(img_path), identity_name))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {self.root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, identity_name = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        return img_tensor, identity_name, img_path


def collate_keep_meta(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    identities = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    return images, identities, paths


def save_outputs(
    output_npz: str,
    output_csv: str,
    paths: List[str],
    identities: List[str],
    embeddings: np.ndarray
):
    np.savez_compressed(
        output_npz,
        embeddings=embeddings,
        labels=np.array(identities),
        paths=np.array(paths)
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "identity"])
        for p, identity in zip(paths, identities):
            writer.writerow([p, identity])


def main():
    parser = argparse.ArgumentParser(description="Extract FaceNet embeddings from aligned face folders.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder of aligned CelebA identities")
    parser.add_argument("--output_dir", type=str, default="facenet_embeddings",
                        help="Folder to save embeddings and metadata")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--pretrained", type=str, default="vggface2",
                        choices=["vggface2", "casia-webface"],
                        help="Pretrained FaceNet weights")
    parser.add_argument("--device", type=str, default=None,
                        help='Force device: "cpu" or "cuda". Default auto-detect')
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = FaceFolderDataset(args.data_root, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_keep_meta
    )

    print(f"[INFO] Found {len(dataset)} images.")

    model = InceptionResnetV1(pretrained=args.pretrained).eval().to(device)

    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch_idx, (images, identities, paths) in enumerate(dataloader):
            images = images.to(device)

            emb = model(images)  # shape: [B, 512]
            emb = emb.cpu().numpy()

            all_embeddings.append(emb)
            all_labels.extend(identities)
            all_paths.extend(paths)

            print(f"[INFO] Processed batch {batch_idx + 1}/{len(dataloader)}")

    all_embeddings = np.vstack(all_embeddings)

    os.makedirs(args.output_dir, exist_ok=True)
    output_npz = os.path.join(args.output_dir, "facenet_embeddings.npz")
    output_csv = os.path.join(args.output_dir, "facenet_metadata.csv")

    save_outputs(
        output_npz=output_npz,
        output_csv=output_csv,
        paths=all_paths,
        identities=all_labels,
        embeddings=all_embeddings
    )

    print(f"[DONE] Saved embeddings to: {output_npz}")
    print(f"[DONE] Saved metadata to:   {output_csv}")
    print(f"[DONE] Embedding shape: {all_embeddings.shape}")


if __name__ == "__main__":
    main()
