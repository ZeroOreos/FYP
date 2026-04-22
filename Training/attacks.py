from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field

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
class AttackCache:
    clean_target_embeddings: torch.Tensor | None = None
    clean_surrogate_embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    bpfa_view_specs: list["FeatureAugmentSpec"] | None = None
    bpfa_clean_view_embeddings: list[torch.Tensor] | None = None


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


def _autocast_disabled(device: torch.device):
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def _target_training_outputs(
    target_model: TrainableRecognizer,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]]:
    if hasattr(target_model, "forward_attack_outputs"):
        outputs = target_model.forward_attack_outputs(images, labels)
    else:
        outputs = target_model(images, labels)
    if not isinstance(outputs, dict):
        raise RuntimeError("Target model forward() must return a training output dictionary for attacks.")
    return outputs


def _surrogate_embedding_loss(
    *,
    policy: AttackPolicy,
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    target_model: TrainableRecognizer,
    labels: torch.Tensor,
    surrogates: dict[str, SurrogateWrapper],
    cache: AttackCache | None = None,
    target_outputs: dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]] | None = None,
) -> torch.Tensor:
    weights = policy.surrogate_weights or {"target": 1.0}
    total = torch.zeros((), device=adv_images.device)
    weight_sum = 0.0

    for name, weight in weights.items():
        if weight <= 0:
            continue
        if name == "target":
            resolved_outputs = target_outputs
            if resolved_outputs is None:
                resolved_outputs = _target_training_outputs(target_model, adv_images, labels)
            logits = resolved_outputs["logits"]
            loss_labels = resolved_outputs.get("loss_labels", labels)
            total = total + float(weight) * F.cross_entropy(logits, loss_labels)
            weight_sum += float(weight)
            continue

        surrogate = surrogates.get(name)
        if surrogate is None:
            continue
        if cache is not None and name in cache.clean_surrogate_embeddings:
            clean_embeddings = cache.clean_surrogate_embeddings[name]
        else:
            with torch.no_grad():
                clean_embeddings = surrogate.embed(clean_images).detach()
            if cache is not None:
                cache.clean_surrogate_embeddings[name] = clean_embeddings
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
    outputs = _target_training_outputs(target_model, adv_images, labels)
    logits = outputs["logits"]
    loss_labels = outputs.get("loss_labels", labels)
    true_logits = logits.gather(1, loss_labels.view(-1, 1)).squeeze(1)
    masked_logits = logits.clone()
    masked_logits.scatter_(1, loss_labels.view(-1, 1), -1e9)
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


@dataclass(frozen=True)
class FeatureAugmentSpec:
    scale: float
    translate_x: float
    translate_y: float
    flip: bool
    blur: bool


def _sample_feature_augment_specs(images: torch.Tensor, *, views: int) -> list[FeatureAugmentSpec]:
    sampled: list[FeatureAugmentSpec] = []
    for _ in range(max(1, views)):
        sampled.append(
            FeatureAugmentSpec(
                scale=float(1.0 + torch.empty((), device=images.device).uniform_(-0.08, 0.08).item()),
                translate_x=float(torch.empty((), device=images.device).uniform_(-0.04, 0.04).item()),
                translate_y=float(torch.empty((), device=images.device).uniform_(-0.04, 0.04).item()),
                flip=bool(torch.rand((), device=images.device).item() > 0.5),
                blur=bool(torch.rand((), device=images.device).item() > 0.5),
            )
        )
    return sampled


def _apply_feature_augment_specs(images: torch.Tensor, specs: list[FeatureAugmentSpec]) -> list[torch.Tensor]:
    augmented: list[torch.Tensor] = []
    for spec in specs:
        view = _affine_jitter(
            images,
            scale=spec.scale,
            translate_x=spec.translate_x,
            translate_y=spec.translate_y,
            flip=spec.flip,
        )
        if spec.blur:
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
    cache: AttackCache | None = None,
) -> torch.Tensor:
    base_adv_outputs = _target_training_outputs(target_model, adv_images, labels)
    ce_loss = _surrogate_embedding_loss(
        policy=policy,
        adv_images=adv_images,
        clean_images=clean_images,
        target_model=target_model,
        labels=labels,
        surrogates=surrogates,
        cache=cache,
        target_outputs=base_adv_outputs,
    )
    if cache is not None and cache.bpfa_view_specs is not None:
        view_specs = cache.bpfa_view_specs
    else:
        view_specs = _sample_feature_augment_specs(clean_images, views=3)
        if cache is not None:
            cache.bpfa_view_specs = view_specs
    augmented_adv = _apply_feature_augment_specs(adv_images, view_specs)

    transfer_loss = torch.zeros((), device=adv_images.device)
    if cache is not None and cache.bpfa_clean_view_embeddings is not None:
        clean_view_embeddings = cache.bpfa_clean_view_embeddings
    else:
        augmented_clean = _apply_feature_augment_specs(clean_images, view_specs)
        clean_view_embeddings = []
        with torch.no_grad():
            for clean_view in augmented_clean:
                clean_view_embeddings.append(target_model.forward_embeddings(clean_view).detach())
        if cache is not None:
            cache.bpfa_clean_view_embeddings = clean_view_embeddings

    for adv_view, clean_embeddings in zip(augmented_adv, clean_view_embeddings):
        adv_outputs = _target_training_outputs(target_model, adv_view, labels)
        adv_logits = adv_outputs["logits"]
        adv_loss_labels = adv_outputs.get("loss_labels", labels)
        adv_embeddings = adv_outputs["embeddings"]
        cosine = F.cosine_similarity(
            F.normalize(adv_embeddings, dim=1),
            F.normalize(clean_embeddings, dim=1),
            dim=1,
        )
        transfer_loss = transfer_loss + 0.5 * F.cross_entropy(adv_logits, adv_loss_labels) + 0.5 * (1.0 - cosine).mean()
    transfer_loss = transfer_loss / float(max(1, len(augmented_adv)))
    return 0.4 * ce_loss + 0.6 * transfer_loss


def _restart_objective_score(
    *,
    policy: AttackPolicy,
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    target_model: TrainableRecognizer,
    labels: torch.Tensor,
    surrogates: dict[str, SurrogateWrapper],
    cache: AttackCache | None,
    objective: str,
) -> torch.Tensor:
    if objective == "bpfa":
        return _bpfa_loss(
            policy=policy,
            adv_images=adv_images,
            clean_images=clean_images,
            target_model=target_model,
            labels=labels,
            surrogates=surrogates,
            cache=cache,
        )
    if objective == "pgd":
        return _surrogate_embedding_loss(
            policy=policy,
            adv_images=adv_images,
            clean_images=clean_images,
            target_model=target_model,
            labels=labels,
            surrogates=surrogates,
            cache=cache,
        )
    if objective == "dfanet":
        return _dfanet_loss(
            policy=policy,
            adv_images=adv_images,
            clean_images=clean_images,
            target_model=target_model,
            labels=labels,
            surrogates=surrogates,
            cache=cache,
        )
    if objective == "cw":
        return _cw_margin_loss(adv_images, labels, target_model)
    raise ValueError(f"Unsupported PGD-like attack objective '{objective}'.")


def _dfanet_loss(
    *,
    policy: AttackPolicy,
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    target_model: TrainableRecognizer,
    labels: torch.Tensor,
    surrogates: dict[str, SurrogateWrapper],
    cache: AttackCache | None = None,
) -> torch.Tensor:
    outputs = _target_training_outputs(target_model, adv_images, labels)
    logits = outputs["logits"]
    adv_loss_labels = outputs.get("loss_labels", labels)
    adv_embeddings = outputs["embeddings"]
    if cache is not None and cache.clean_target_embeddings is not None:
        clean_embeddings = cache.clean_target_embeddings
    else:
        with torch.no_grad():
            clean_embeddings = target_model.forward_embeddings(clean_images).detach()
        if cache is not None:
            cache.clean_target_embeddings = clean_embeddings

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
        cache=cache,
        target_outputs=outputs,
    )
    return 0.35 * F.cross_entropy(logits, adv_loss_labels) + 0.45 * feature_loss + 0.20 * ensemble_loss


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
    cache = AttackCache()

    with _autocast_disabled(clean_images.device):
        attack_clean_images = clean_images.float()
        attack_labels = labels
        for restart_index in range(max(1, policy.restarts)):
            if policy.random_start:
                noise = torch.empty_like(attack_clean_images).uniform_(-policy.eps, policy.eps)
                adv = torch.clamp(attack_clean_images + noise, min=0.0, max=1.0).detach()
            else:
                adv = attack_clean_images.clone().detach()

            for _ in range(max(1, policy.steps)):
                adv.requires_grad_(True)
                if objective == "pgd":
                    loss = _surrogate_embedding_loss(
                        policy=policy,
                        adv_images=adv,
                        clean_images=attack_clean_images,
                        target_model=target_model,
                        labels=attack_labels,
                        surrogates=surrogates,
                        cache=cache,
                    )
                elif objective == "bpfa":
                    loss = _bpfa_loss(
                        policy=policy,
                        adv_images=adv,
                        clean_images=attack_clean_images,
                        target_model=target_model,
                        labels=attack_labels,
                        surrogates=surrogates,
                        cache=cache,
                    )
                elif objective == "dfanet":
                    loss = _dfanet_loss(
                        policy=policy,
                        adv_images=adv,
                        clean_images=attack_clean_images,
                        target_model=target_model,
                        labels=attack_labels,
                        surrogates=surrogates,
                        cache=cache,
                    )
                elif objective == "cw":
                    loss = _cw_margin_loss(adv, attack_labels, target_model)
                else:
                    raise ValueError(f"Unsupported PGD-like attack objective '{objective}'.")
                gradient = torch.autograd.grad(loss, adv)[0]
                adv = adv.detach() + policy.alpha * gradient.sign()
                delta = torch.clamp(adv - attack_clean_images, min=-policy.eps, max=policy.eps)
                adv = torch.clamp(attack_clean_images + delta, min=0.0, max=1.0).detach()

            with torch.no_grad():
                restart_loss = _restart_objective_score(
                    policy=policy,
                    adv_images=adv,
                    clean_images=attack_clean_images,
                    target_model=target_model,
                    labels=attack_labels,
                    surrogates=surrogates,
                    cache=cache,
                    objective=objective,
                )
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
    attack_chunk_size: int | None = None,
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

    chunk_size = None if attack_chunk_size is None else max(1, int(attack_chunk_size))
    if chunk_size is not None and images.shape[0] > chunk_size:
        adv_chunks: list[torch.Tensor] = []
        total_hits = 0
        start = 0
        total = int(images.shape[0])
        while start < total:
            remaining = total - start
            current_chunk_size = min(chunk_size, remaining)
            # Avoid a trailing singleton chunk for training-time BatchNorm paths.
            if remaining > chunk_size and (remaining - current_chunk_size) == 1 and current_chunk_size > 2:
                current_chunk_size -= 1
            end = start + current_chunk_size
            chunk_result = generate_attack_batch(
                policy=policy,
                images=images[start:end],
                labels=labels[start:end],
                rel_paths=rel_paths[start:end],
                image_size=image_size,
                target_model=target_model,
                surrogates=surrogates,
                device=device,
                attack_chunk_size=None,
            )
            adv_chunks.append(chunk_result.images)
            total_hits += int(chunk_result.cached_hits)
            start = end
        return AttackResult(
            images=torch.cat(adv_chunks, dim=0),
            policy_name=policy.name,
            family=policy.family,
            cached_hits=total_hits,
        )

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
