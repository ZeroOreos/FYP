from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def embedding_consistency_loss(clean_embeddings: torch.Tensor, adv_embeddings: torch.Tensor) -> torch.Tensor:
    clean_norm = F.normalize(clean_embeddings, dim=1)
    adv_norm = F.normalize(adv_embeddings, dim=1)
    cosine = F.cosine_similarity(clean_norm, adv_norm, dim=1)
    return (1.0 - cosine).mean()

