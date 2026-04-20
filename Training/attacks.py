from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from Training.cache import load_cached_adversarial_batch
from Training.config import AttackPolicy
from Training.recognizers import SurrogateWrapper, TrainableRecognizer


@dataclass
class AttackResult:
    images: torch.Tensor
    policy_name: str
    family: str
    cached_hits: int = 0


@dataclass
class AttackPolicySampler:
    policies: list[AttackPolicy]
    strategy: str
    rng: random.Random
    _policy_index: int = 0
    _family_index: int = 0

    def _enabled(self) -> list[AttackPolicy]:
        enabled = enabled_attack_policies(self.policies)
        if not enabled:
            raise RuntimeError("No enabled attack policies configured.")
        return enabled

    def choose(self) -> AttackPolicy:
        enabled = self._enabled()
        strategy = self.strategy.strip().lower()
        if strategy == "weighted_random":
            weights = [policy.weight for policy in enabled]
            return self.rng.choices(enabled, weights=weights, k=1)[0]
        if strategy == "round_robin":
            policy = enabled[self._policy_index % len(enabled)]
            self._policy_index += 1
            return policy
        if strategy == "family_round_robin":
            families = sorted({policy.family for policy in enabled})
            family = families[self._family_index % len(families)]
            family_policies = [policy for policy in enabled if policy.family == family]
            self._family_index += 1
            if len(family_policies) == 1:
                return family_policies[0]
            weights = [policy.weight for policy in family_policies]
            return self.rng.choices(family_policies, weights=weights, k=1)[0]
        raise ValueError(
            f"Unsupported attack sampling strategy '{self.strategy}'. "
            "Use weighted_random, round_robin, or family_round_robin."
        )


def build_attack_policy_sampler(
    policies: list[AttackPolicy],
    strategy: str,
    rng: random.Random,
) -> AttackPolicySampler:
    return AttackPolicySampler(policies=policies, strategy=strategy, rng=rng)


def is_surrogate_attack_policy(policy: AttackPolicy) -> bool:
    family = policy.family.strip().lower()
    return family.startswith("surrogate") or policy.kind == "cached"


def is_primary_attack_policy(policy: AttackPolicy) -> bool:
    return not is_surrogate_attack_policy(policy)


def enabled_attack_policies(policies: list[AttackPolicy]) -> list[AttackPolicy]:
    return [policy for policy in policies if policy.enabled and policy.weight > 0]


def _surrogate_embedding_loss(
    *,
    policy: AttackPolicy,
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    target_model: TrainableRecognizer,
    labels: torch.Tensor,
    surrogates: dict[str, SurrogateWrapper],
) -> torch.Tensor:
    weights = policy.surrogate_weights or {"target": 1.0}
    total = torch.zeros((), device=adv_images.device)
    weight_sum = 0.0

    for name, weight in weights.items():
        if weight <= 0:
            continue
        if name == "target":
            logits, _ = target_model.forward_logits(adv_images, labels)
            total = total + float(weight) * F.cross_entropy(logits, labels)
            weight_sum += float(weight)
            continue

        surrogate = surrogates.get(name)
        if surrogate is None:
            continue
        clean_embeddings = surrogate.embed(clean_images).detach()
        adv_embeddings = surrogate.embed(adv_images)
        cosine = F.cosine_similarity(
            F.normalize(adv_embeddings, dim=1),
            F.normalize(clean_embeddings, dim=1),
            dim=1,
        )
        total = total + float(weight) * (1.0 - cosine).mean()
        weight_sum += float(weight)

    if weight_sum == 0:
        raise RuntimeError(f"Attack policy '{policy.name}' did not resolve any usable surrogate weights.")
    return total / weight_sum


def _cw_margin_loss(
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    target_model: TrainableRecognizer,
) -> torch.Tensor:
    logits, _ = target_model.forward_logits(adv_images, labels)
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked_logits = logits.clone()
    masked_logits.scatter_(1, labels.view(-1, 1), -1e9)
    max_other = masked_logits.max(dim=1).values
    return (max_other - true_logits).mean()


def _affine_jitter(
    images: torch.Tensor,
    *,
    scale: float,
    translate_x: float,
    translate_y: float,
    flip: bool,
) -> torch.Tensor:
    _, _, height, width = images.shape
    scaled_h = max(1, int(round(height * scale)))
    scaled_w = max(1, int(round(width * scale)))
    transformed = F.interpolate(images, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)

    if scaled_h >= height:
        top = max(0, (scaled_h - height) // 2)
        transformed = transformed[:, :, top:top + height, :]
    else:
        pad_top = max(0, (height - scaled_h) // 2)
        pad_bottom = max(0, height - scaled_h - pad_top)
        transformed = F.pad(transformed, (0, 0, pad_top, pad_bottom))

    if scaled_w >= width:
        left = max(0, (scaled_w - width) // 2)
        transformed = transformed[:, :, :, left:left + width]
    else:
        pad_left = max(0, (width - scaled_w) // 2)
        pad_right = max(0, width - scaled_w - pad_left)
        transformed = F.pad(transformed, (pad_left, pad_right, 0, 0))

    shift_x = int(round(translate_x * width))
    shift_y = int(round(translate_y * height))
    if shift_x != 0 or shift_y != 0:
        transformed = torch.roll(transformed, shifts=(shift_y, shift_x), dims=(2, 3))
    if flip:
        transformed = torch.flip(transformed, dims=[3])
    return transformed


def _sample_feature_augmented_views(images: torch.Tensor, *, views: int) -> list[torch.Tensor]:
    augmented: list[torch.Tensor] = []
    for _ in range(max(1, views)):
        scale = float(1.0 + torch.empty((), device=images.device).uniform_(-0.08, 0.08).item())
        translate_x = float(torch.empty((), device=images.device).uniform_(-0.04, 0.04).item())
        translate_y = float(torch.empty((), device=images.device).uniform_(-0.04, 0.04).item())
        flip = bool(torch.rand((), device=images.device).item() > 0.5)
        view = _affine_jitter(
            images,
            scale=scale,
            translate_x=translate_x,
            translate_y=translate_y,
            flip=flip,
        )
        if bool(torch.rand((), device=images.device).item() > 0.5):
            view = F.avg_pool2d(view, kernel_size=3, stride=1, padding=1)
        augmented.append(view)
    return augmented


def _bpfa_loss(
    *,
    policy: AttackPolicy,
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    target_model: TrainableRecognizer,
    labels: torch.Tensor,
    surrogates: dict[str, SurrogateWrapper],
) -> torch.Tensor:
    ce_loss = _surrogate_embedding_loss(
        policy=policy,
        adv_images=adv_images,
        clean_images=clean_images,
        target_model=target_model,
        labels=labels,
        surrogates=surrogates,
    )
    augmented_adv = _sample_feature_augmented_views(adv_images, views=3)
    augmented_clean = _sample_feature_augmented_views(clean_images, views=3)

    transfer_loss = torch.zeros((), device=adv_images.device)
    for adv_view, clean_view in zip(augmented_adv, augmented_clean):
        clean_embeddings = target_model.forward_embeddings(clean_view).detach()
        adv_logits, adv_embeddings = target_model.forward_logits(adv_view, labels)
        cosine = F.cosine_similarity(
            F.normalize(adv_embeddings, dim=1),
            F.normalize(clean_embeddings, dim=1),
            dim=1,
        )
        transfer_loss = transfer_loss + 0.5 * F.cross_entropy(adv_logits, labels) + 0.5 * (1.0 - cosine).mean()
    transfer_loss = transfer_loss / float(max(1, len(augmented_adv)))
    return 0.4 * ce_loss + 0.6 * transfer_loss


def _dfanet_loss(
    *,
    policy: AttackPolicy,
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    target_model: TrainableRecognizer,
    labels: torch.Tensor,
    surrogates: dict[str, SurrogateWrapper],
) -> torch.Tensor:
    logits, adv_embeddings = target_model.forward_logits(adv_images, labels)
    clean_embeddings = target_model.forward_embeddings(clean_images).detach()

    feature_loss = torch.zeros((), device=adv_images.device)
    for _ in range(3):
        mask = (torch.rand_like(adv_embeddings) > 0.2).to(adv_embeddings.dtype)
        masked_adv = F.normalize(adv_embeddings * mask, dim=1)
        masked_clean = F.normalize(clean_embeddings * mask, dim=1)
        feature_loss = feature_loss + (1.0 - F.cosine_similarity(masked_adv, masked_clean, dim=1)).mean()
    feature_loss = feature_loss / 3.0

    ensemble_loss = _surrogate_embedding_loss(
        policy=policy,
        adv_images=adv_images,
        clean_images=clean_images,
        target_model=target_model,
        labels=labels,
        surrogates=surrogates,
    )
    return 0.35 * F.cross_entropy(logits, labels) + 0.45 * feature_loss + 0.20 * ensemble_loss


def _pgd_like_attack(
    *,
    policy: AttackPolicy,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_model: TrainableRecognizer,
    surrogates: dict[str, SurrogateWrapper],
    objective: str,
) -> torch.Tensor:
    clean_images = images.detach()
    best_adv = clean_images.clone().detach()
    best_loss = None

    for restart_index in range(max(1, policy.restarts)):
        if policy.random_start:
            noise = torch.empty_like(clean_images).uniform_(-policy.eps, policy.eps)
            adv = torch.clamp(clean_images + noise, min=0.0, max=1.0).detach()
        else:
            adv = clean_images.clone().detach()

        for _ in range(max(1, policy.steps)):
            adv.requires_grad_(True)
            if objective == "pgd":
                loss = _surrogate_embedding_loss(
                    policy=policy,
                    adv_images=adv,
                    clean_images=clean_images,
                    target_model=target_model,
                    labels=labels,
                    surrogates=surrogates,
                )
            elif objective == "bpfa":
                loss = _bpfa_loss(
                    policy=policy,
                    adv_images=adv,
                    clean_images=clean_images,
                    target_model=target_model,
                    labels=labels,
                    surrogates=surrogates,
                )
            elif objective == "dfanet":
                loss = _dfanet_loss(
                    policy=policy,
                    adv_images=adv,
                    clean_images=clean_images,
                    target_model=target_model,
                    labels=labels,
                    surrogates=surrogates,
                )
            elif objective == "cw":
                loss = _cw_margin_loss(adv, labels, target_model)
            else:
                raise ValueError(f"Unsupported PGD-like attack objective '{objective}'.")
            gradient = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + policy.alpha * gradient.sign()
            delta = torch.clamp(adv - clean_images, min=-policy.eps, max=policy.eps)
            adv = torch.clamp(clean_images + delta, min=0.0, max=1.0).detach()

        with torch.no_grad():
            if objective == "pgd":
                restart_loss = _surrogate_embedding_loss(
                    policy=policy,
                    adv_images=adv,
                    clean_images=clean_images,
                    target_model=target_model,
                    labels=labels,
                    surrogates=surrogates,
                )
            elif objective == "bpfa":
                restart_loss = _bpfa_loss(
                    policy=policy,
                    adv_images=adv,
                    clean_images=clean_images,
                    target_model=target_model,
                    labels=labels,
                    surrogates=surrogates,
                )
            elif objective == "dfanet":
                restart_loss = _dfanet_loss(
                    policy=policy,
                    adv_images=adv,
                    clean_images=clean_images,
                    target_model=target_model,
                    labels=labels,
                    surrogates=surrogates,
                )
            elif objective == "cw":
                restart_loss = _cw_margin_loss(adv, labels, target_model)
            else:
                raise ValueError(f"Unsupported PGD-like attack objective '{objective}'.")
            if best_loss is None or restart_loss.item() > best_loss:
                best_loss = restart_loss.item()
                best_adv = adv.detach()

        if restart_index + 1 >= max(1, policy.restarts):
            break

    return best_adv


def generate_attack_batch(
    *,
    policy: AttackPolicy,
    images: torch.Tensor,
    labels: torch.Tensor,
    rel_paths: list[str],
    image_size: int,
    target_model: TrainableRecognizer,
    surrogates: dict[str, SurrogateWrapper],
    device: torch.device,
) -> AttackResult:
    if policy.kind == "cached":
        cached_batch, hits = load_cached_adversarial_batch(
            rel_paths=rel_paths,
            cache_roots=policy.cache_roots,
            image_size=image_size,
            device=device,
        )
        if cached_batch is None:
            return AttackResult(images=images.detach(), policy_name=policy.name, family=policy.family, cached_hits=hits)
        return AttackResult(images=cached_batch, policy_name=policy.name, family=policy.family, cached_hits=hits)

    policy_name = policy.name.lower()
    if policy_name == "pgd":
        adv = _pgd_like_attack(
            policy=policy,
            images=images,
            labels=labels,
            target_model=target_model,
            surrogates=surrogates,
            objective="pgd",
        )
        return AttackResult(images=adv, policy_name=policy.name, family=policy.family)
    if policy_name in {"bpfa", "cw"}:
        adv = _pgd_like_attack(
            policy=policy,
            images=images,
            labels=labels,
            target_model=target_model,
            surrogates=surrogates,
            objective="bpfa" if policy_name == "bpfa" else "cw",
        )
        return AttackResult(images=adv, policy_name=policy.name, family=policy.family)
    if policy_name in {"dfanet", "feature_space"}:
        adv = _pgd_like_attack(
            policy=policy,
            images=images,
            labels=labels,
            target_model=target_model,
            surrogates=surrogates,
            objective="dfanet",
        )
        return AttackResult(images=adv, policy_name=policy.name, family=policy.family)
    raise ValueError(f"Unsupported attack policy name: {policy.name}")
