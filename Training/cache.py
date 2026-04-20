from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms


def load_cached_adversarial_batch(
    rel_paths: list[str],
    cache_roots: list[str],
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, int]:
    if not cache_roots:
        return None, 0

    resize_to_tensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    tensors = []
    hits = 0
    for rel_path in rel_paths:
        cached_path = None
        for root in cache_roots:
            candidate = (Path(root).resolve() / rel_path).resolve()
            if candidate.exists() and candidate.is_file():
                cached_path = candidate
                break
        if cached_path is None:
            return None, hits
        image = Image.open(cached_path).convert("RGB")
        tensors.append(resize_to_tensor(image))
        hits += 1

    if not tensors:
        return None, 0
    return torch.stack(tensors).to(device), hits

