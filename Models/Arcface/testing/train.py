# python3 train.py [...] -> model / checkpoint output
import os
import random
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Arcface.testing.model import ArcFaceModel, ArcMarginProduct
from Utility.runtime import resolve_torch_device


TRAIN_DIR = "/content/drive/MyDrive/FYP/dataset/celeba_arcface/train"
VAL_DIR = "/content/drive/MyDrive/FYP/dataset/celeba_arcface/val"
SAVE_PATH = "arcface_resnet18_best.pth"

IMG_SIZE = 160
BATCH_SIZE = 64
NUM_EPOCHS = 25
FREEZE_EPOCHS = 6
EMBEDDING_DIM = 256
NUM_WORKERS = 2

BACKBONE_LR = 5e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 5e-4

MARGIN_S = 30.0
MARGIN_M = 0.50

SEED = 42
DEVICE = resolve_torch_device()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


def build_class_to_idx(root_dir):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    class_names = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    class_names = sorted(class_names)
    if len(class_names) == 0:
        raise RuntimeError(f"No class folders found in: {root_dir}")

    return {cls_name: idx for idx, cls_name in enumerate(class_names)}


class FaceFolderDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        for cls_name in sorted(os.listdir(root_dir)):
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue

            if cls_name not in class_to_idx:
                continue

            label = class_to_idx[cls_name]
            for fname in sorted(os.listdir(cls_path)):
                fpath = os.path.join(cls_path, fname)
                if os.path.isfile(fpath) and is_image_file(fname):
                    self.samples.append((fpath, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def freeze_backbone(model):
    for p in model.backbone.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model):
    for p in model.backbone.backbone.parameters():
        p.requires_grad = True


def get_dataloaders(train_dir, val_dir):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    class_to_idx = build_class_to_idx(train_dir)

    train_ds = FaceFolderDataset(train_dir, class_to_idx, transform=train_transform)
    val_ds = FaceFolderDataset(val_dir, class_to_idx, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    return train_ds, val_ds, train_loader, val_loader, class_to_idx


def make_optimizer(model, margin_layer):
    return torch.optim.AdamW([
        {"params": model.backbone.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.backbone.embedding.parameters(), "lr": HEAD_LR},
        {"params": model.backbone.bn.parameters(), "lr": HEAD_LR},
        {"params": margin_layer.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.size(0)


def run_train_epoch(model, margin_layer, loader, criterion, optimizer, device):
    model.train()
    margin_layer.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        embeddings = model(images)
        logits = margin_layer(embeddings, labels)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct, count = accuracy_from_logits(logits, labels)
        total_correct += correct
        total_samples += count

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def run_val_epoch(model, margin_layer, loader, criterion, device):
    model.eval()
    margin_layer.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(images)
        logits = margin_layer(embeddings, labels)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct, count = accuracy_from_logits(logits, labels)
        total_correct += correct
        total_samples += count

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def print_dataset_stats(train_ds, val_ds, class_to_idx):
    train_labels = [label for _, label in train_ds.samples]
    val_labels = [label for _, label in val_ds.samples]

    print(f"Device: {DEVICE}")
    print(f"Identities: {len(class_to_idx)}")
    print(f"Train images: {len(train_ds)}")
    print(f"Val images: {len(val_ds)}")
    print(f"Sample train labels: {train_labels[:10]}")
    print(f"Min label: {min(train_labels)}")
    print(f"Max label: {max(train_labels)}")

    train_count = Counter(train_labels)
    counts = list(train_count.values())
    print(f"Train images/class -> min: {min(counts)}, max: {max(counts)}, avg: {sum(counts)/len(counts):.2f}")


def main():
    set_seed(SEED)

    train_ds, val_ds, train_loader, val_loader, class_to_idx = get_dataloaders(TRAIN_DIR, VAL_DIR)
    print_dataset_stats(train_ds, val_ds, class_to_idx)

    num_classes = len(class_to_idx)

    model = ArcFaceModel(
        embedding_dim=EMBEDDING_DIM,
        pretrained=True
    ).to(DEVICE)

    margin_layer = ArcMarginProduct(
        in_features=EMBEDDING_DIM,
        out_features=num_classes,
        s=MARGIN_S,
        m=MARGIN_M
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, margin_layer)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=24, gamma=0.5)

    freeze_backbone(model)
    print("[INFO] Backbone frozen")

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        if epoch == FREEZE_EPOCHS:
            unfreeze_backbone(model)
            print("[INFO] Backbone unfrozen")

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = run_train_epoch(
            model, margin_layer, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = run_val_epoch(
            model, margin_layer, val_loader, criterion, DEVICE
        )

        lrs = [pg["lr"] for pg in optimizer.param_groups]

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")
        print(f"LRs: {lrs}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "margin_state_dict": margin_layer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_to_idx": class_to_idx,
                "embedding_dim": EMBEDDING_DIM,
                "img_size": IMG_SIZE,
            }, SAVE_PATH)
            print(f"[INFO] Saved best model to {SAVE_PATH}")

        scheduler.step()

    print(f"Training done. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
