from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HardPair:
    anchor_index: int
    partner_index: int
    label_a: int
    label_b: int
    pair_type: str
    hardness: float


@dataclass
class HardPairMiningResult:
    strategy: str
    pair_count: int = 0
    positive_pairs: int = 0
    negative_pairs: int = 0
    hard_sample_count: int = 0
    mean_hardness: float = 0.0
    selected_pairs: list[HardPair] = field(default_factory=list)
    sample_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))


def _normalize_weight_map(
    weights: dict[str, float] | None,
    available_names: set[str],
) -> dict[str, float]:
    cleaned = {
        name: float(weight)
        for name, weight in (weights or {"target": 1.0}).items()
        if name in available_names and float(weight) > 0
    }
    if not cleaned:
        cleaned = {"target": 1.0}
    total = sum(cleaned.values())
    return {name: weight / total for name, weight in cleaned.items()}


def _pairwise_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings.detach(), dim=1)
    return normalized @ normalized.t()


def _top_pair_indices(mask: torch.Tensor, scores: torch.Tensor, limit: int, largest: bool) -> torch.Tensor:
    if limit <= 0:
        return torch.empty(0, device=scores.device, dtype=torch.long)
    fill_value = float("-inf") if largest else float("inf")
    flat_scores = scores.masked_fill(~mask, fill_value).reshape(-1)
    if largest:
        candidate_mask = torch.isfinite(flat_scores)
    else:
        candidate_mask = torch.isfinite(flat_scores) & (flat_scores < float("inf"))
    if not torch.any(candidate_mask):
        return torch.empty(0, device=scores.device, dtype=torch.long)
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    candidate_scores = flat_scores[candidate_indices]
    topk = min(limit, candidate_indices.numel())
    top_indices = torch.topk(candidate_scores, k=topk, largest=largest).indices
    return candidate_indices[top_indices]


def mine_hard_pairs(
    *,
    target_embeddings: torch.Tensor,
    labels: torch.Tensor,
    surrogate_embeddings: dict[str, torch.Tensor] | None,
    strategy: str,
    hard_pair_fraction: float,
    min_pairs: int,
    max_pairs: int,
    surrogate_weights: dict[str, float] | None = None,
) -> HardPairMiningResult:
    strategy_name = strategy.strip().lower()
    if strategy_name in {"", "none", "off"}:
        return HardPairMiningResult(strategy=strategy_name)

    available_embeddings = {"target": target_embeddings}
    if surrogate_embeddings:
        available_embeddings.update(surrogate_embeddings)
    weights = _normalize_weight_map(surrogate_weights, set(available_embeddings))

    combined_similarity = None
    for name, weight in weights.items():
        similarity = _pairwise_similarity(available_embeddings[name]) * float(weight)
        combined_similarity = similarity if combined_similarity is None else (combined_similarity + similarity)
    if combined_similarity is None:
        return HardPairMiningResult(strategy=strategy_name)

    labels = labels.detach().to(torch.long)
    upper_triangle = torch.triu(torch.ones_like(combined_similarity, dtype=torch.bool), diagonal=1)
    same_identity = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = upper_triangle & same_identity
    negative_mask = upper_triangle & (~same_identity)

    positive_pairs = int(positive_mask.sum().item())
    negative_pairs = int(negative_mask.sum().item())
    total_candidates = positive_pairs + negative_pairs
    if total_candidates == 0:
        return HardPairMiningResult(
            strategy=strategy_name,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
        )

    pair_budget = int(round(total_candidates * max(0.0, hard_pair_fraction)))
    pair_budget = max(min_pairs, pair_budget)
    pair_budget = min(max_pairs, pair_budget)
    if pair_budget <= 0:
        return HardPairMiningResult(
            strategy=strategy_name,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
        )

    positive_budget = min(positive_pairs, max(1, pair_budget // 2)) if positive_pairs else 0
    negative_budget = min(negative_pairs, max(1, pair_budget - positive_budget)) if negative_pairs else 0
    if positive_budget + negative_budget > pair_budget:
        negative_budget = max(0, pair_budget - positive_budget)

    selected: list[HardPair] = []
    sample_indices: set[int] = set()
    side = combined_similarity.size(1)
    selection_order = [(index, "positive") for index in _top_pair_indices(positive_mask, combined_similarity, positive_budget, largest=False)]
    selection_order += [(index, "negative") for index in _top_pair_indices(negative_mask, combined_similarity, negative_budget, largest=True)]

    for flat_index, pair_type in selection_order:
        anchor = int(torch.div(flat_index, side, rounding_mode="floor").item())
        partner = int((flat_index % side).item())
        similarity = float(combined_similarity[anchor, partner].item())
        hardness = (1.0 - similarity) if pair_type == "positive" else similarity
        selected.append(
            HardPair(
                anchor_index=anchor,
                partner_index=partner,
                label_a=int(labels[anchor].item()),
                label_b=int(labels[partner].item()),
                pair_type=pair_type,
                hardness=hardness,
            )
        )
        sample_indices.add(anchor)
        sample_indices.add(partner)

    index_tensor = torch.tensor(sorted(sample_indices), dtype=torch.long, device=labels.device) if sample_indices else torch.empty(0, dtype=torch.long, device=labels.device)
    mean_hardness = sum(pair.hardness for pair in selected) / len(selected) if selected else 0.0
    return HardPairMiningResult(
        strategy=strategy_name,
        pair_count=len(selected),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        hard_sample_count=int(index_tensor.numel()),
        mean_hardness=float(mean_hardness),
        selected_pairs=selected,
        sample_indices=index_tensor,
    )
