import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_train_transform(img_size=112):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_val_transform(img_size=112):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def build_class_to_idx(train_dir):
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    ids = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if len(ids) == 0:
        raise RuntimeError(f"No identity folders found in {train_dir}")

    return {name: i for i, name in enumerate(ids)}


class FaceFolderDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")

        for identity_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            name = identity_dir.name
            if name not in class_to_idx:
                continue

            label = class_to_idx[name]
            for img_path in sorted(identity_dir.iterdir()):
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(img_path), int(label)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
