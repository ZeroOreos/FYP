from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from Training.arcface import ArcFaceCeilingSpec


@dataclass
class AttackPolicy:
    name: str
    family: str
    kind: str
    weight: float = 1.0
    enabled: bool = True
    eps: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 4
    random_start: bool = False
    restarts: int = 1
    surrogate_weights: dict[str, float] = field(default_factory=dict)
    cache_roots: list[str] = field(default_factory=list)
    refresh_every_epochs: int | None = None


@dataclass
class AttackEnsemble:
    primary_attackers: list[AttackPolicy] = field(default_factory=list)
    surrogate_attackers: list[AttackPolicy] = field(default_factory=list)

    def all_attackers(self) -> list[AttackPolicy]:
        return [*self.primary_attackers, *self.surrogate_attackers]

    def enabled_primary_attackers(self) -> list[AttackPolicy]:
        return [policy for policy in self.primary_attackers if policy.enabled and policy.weight > 0]

    def enabled_surrogate_attackers(self) -> list[AttackPolicy]:
        return [policy for policy in self.surrogate_attackers if policy.enabled and policy.weight > 0]

    def enabled_attackers(self) -> list[AttackPolicy]:
        return [policy for policy in self.all_attackers() if policy.enabled and policy.weight > 0]


@dataclass
class RecognizerPolicy:
    name: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class RecognizerEnsemble:
    recognizers: list[RecognizerPolicy] = field(default_factory=list)

    def all_recognizers(self) -> list[RecognizerPolicy]:
        return list(self.recognizers)

    def enabled_recognizers(self) -> list[RecognizerPolicy]:
        return [policy for policy in self.recognizers if policy.enabled and policy.weight > 0]


@dataclass
class EnsembleTrainingConfig:
    train_dir: str
    val_dir: str
    output_dir: str
    resume_from: str | None = None
    dataset_fraction: float = 1.0
    dataset_subset_seed: int = 42
    dataset_min_images_per_identity: int = 2
    target_model: str = "arcface"
    recognizers_mode: str = "serial_target"
    runtime_profile: str = "custom"
    target_backbone: str = "resnet18"
    surrogate_models: list[str] = field(default_factory=lambda: ["target"])
    recognizer_ensemble: list[RecognizerPolicy] = field(default_factory=list)
    recognizer_clean_weight: float = 0.0
    recognizer_adv_weight: float = 0.0
    recognizer_start_epoch: int = 1
    recognizer_weight_strategy: str = "static"
    attack_sampling_strategy: str = "weighted_random"
    batch_size: int = 32
    clean_warmup_batch_size: int | None = None
    shallow_adv_batch_size: int | None = None
    full_adv_batch_size: int | None = None
    attack_chunk_size: int | None = None
    gradient_accumulation_steps: int = 1
    epochs: int = 10
    learning_rate: float = 0.1
    auto_scale_learning_rate: bool = False
    reference_batch_size: int | None = None
    lr_scale_mode: str = "linear"
    weight_decay: float = 5e-4
    embedding_dim: int = 256
    image_size: int = 112
    num_workers: int = 2
    persistent_workers: bool = True
    prefetch_factor: int = 4
    pin_memory: bool = True
    clean_weight: float = 1.0
    adv_weight: float = 1.0
    consistency_weight: float = 0.1
    clean_fraction: float = 0.5
    lr_warmup_epochs: int = 0
    attack_start_epoch: int = 1
    checkpoint_every: int = 1
    log_every_batches: int = 10
    device: str = "auto"
    seed: int = 42
    eval_attack_name: str | None = None
    evaluate_all_attacks: bool = True
    clean_eval_every_epochs: int = 1
    robust_eval_every_epochs: int = 1
    full_robust_eval_every_epochs: int = 1
    verification_eval_every_epochs: int = 1
    verification_eval_offset: int = 0
    clean_only: bool = False
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "auto"
    use_gradient_checkpointing: bool = True
    use_distributed: bool = True
    distributed_backend: str = "nccl"
    distributed_timeout_seconds: int = 7200
    use_sync_batchnorm: bool = True
    ddp_no_sync_accumulation: bool = True
    ddp_broadcast_buffers: bool = False
    ddp_gradient_as_bucket_view: bool = True
    ddp_static_graph: bool = False
    ddp_bucket_cap_mb: int | None = None
    optimizer_name: str = "sgd"
    optimizer_foreach: bool | None = True
    optimizer_fused: bool | None = None
    momentum: float = 0.9
    lr_milestones: list[int] = field(default_factory=lambda: [8, 12, 16])
    lr_gamma: float = 0.1
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5
    use_partial_fc: bool = True
    partial_fc_negative_sample_rate: float = 0.3
    recognizer_partial_fc_negative_sample_rate: float | None = None
    sub_center_count: int = 1
    dropout_p: float = 0.4
    alignment_detector: str = "RetinaFace-class"
    alignment_landmarks: int = 5
    normalized_crop_size: int = 112
    dataset_name: str = "WebFace42M"
    joint_pool_member_weights: dict[str, float] = field(
        default_factory=lambda: {
            "arcface": 0.4,
            "cosface": 0.3,
            "curricularface": 0.3,
        }
    )
    val_pairs_path: str | None = None
    test_dir: str | None = None
    test_pairs_path: str | None = None
    pairing_strategy: str = "ensemble_hard"
    pairing_surrogate_weights: dict[str, float] = field(
        default_factory=lambda: {
            "target": 1.0,
        }
    )
    hard_pair_fraction: float = 0.20
    hard_pair_min_pairs: int = 4
    hard_pair_max_pairs: int = 32
    hard_pair_weight: float = 0.15
    curriculum_enabled: bool = True
    curriculum_warmup_epochs: int = 3
    curriculum_clean_fraction_end: float = 0.25
    curriculum_hard_pair_fraction_end: float = 0.35
    curriculum_hard_pair_weight_end: float = 0.30
    training_depth_mode: str = "three_stage"
    clean_warmup_epochs: int = 3
    shallow_adv_epochs: int = 13
    shallow_clean_fraction: float = 0.75
    shallow_attack_eps_scale: float = 0.5
    shallow_attack_alpha_scale: float = 0.5
    shallow_attack_step_scale: float = 0.5
    shallow_attack_restart_cap: int = 1
    shallow_attack_names: list[str] = field(default_factory=lambda: ["pgd", "bpfa"])
    enable_tf32: bool = True
    float32_matmul_precision: str = "high"
    use_channels_last: bool = True
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "max-autotune-no-cudagraphs"
    primary_attackers: list[AttackPolicy] = field(default_factory=list)
    surrogate_attackers: list[AttackPolicy] = field(default_factory=list)
    eval_attackers: list[AttackPolicy] = field(default_factory=list)

    def resolved_train_dir(self) -> Path:
        return Path(self.train_dir).resolve()

    def resolved_val_dir(self) -> Path:
        return Path(self.val_dir).resolve()

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).resolve()

    def resolved_resume_from(self) -> Path | None:
        if self.resume_from is None:
            return None
        raw_resume = str(self.resume_from).strip()
        if raw_resume.lower() in {"", "none", "null"}:
            return None
        return Path(raw_resume).resolve()

    def resolved_test_dir(self) -> Path | None:
        if self.test_dir is None:
            return None
        return Path(self.test_dir).resolve()

    def attack_ensemble(self) -> AttackEnsemble:
        return AttackEnsemble(
            primary_attackers=list(self.primary_attackers),
            surrogate_attackers=list(self.surrogate_attackers),
        )

    def recognizer_ensemble_config(self) -> RecognizerEnsemble:
        return RecognizerEnsemble(recognizers=list(self.recognizer_ensemble))

    def normalized_recognizers_mode(self) -> str:
        return self.recognizers_mode.strip().lower()

    def uses_joint_train(self) -> bool:
        return self.normalized_recognizers_mode() == "joint_train"

    def normalized_runtime_profile(self) -> str:
        return self.runtime_profile.strip().lower()

    def all_attackers(self) -> list[AttackPolicy]:
        return self.attack_ensemble().all_attackers()

    def enabled_primary_attackers(self) -> list[AttackPolicy]:
        return self.attack_ensemble().enabled_primary_attackers()

    def enabled_surrogate_attackers(self) -> list[AttackPolicy]:
        return self.attack_ensemble().enabled_surrogate_attackers()

    def enabled_attackers(self) -> list[AttackPolicy]:
        return self.attack_ensemble().enabled_attackers()

    def all_eval_attackers(self) -> list[AttackPolicy]:
        return list(self.eval_attackers) if self.eval_attackers else self.all_attackers()

    def enabled_eval_attackers(self) -> list[AttackPolicy]:
        policies = self.eval_attackers if self.eval_attackers else self.all_attackers()
        return [policy for policy in policies if policy.enabled and policy.weight > 0]

    def all_recognizers(self) -> list[RecognizerPolicy]:
        return self.recognizer_ensemble_config().all_recognizers()

    def enabled_recognizers(self) -> list[RecognizerPolicy]:
        return self.recognizer_ensemble_config().enabled_recognizers()


def default_mode_b_config(train_dir: Path, val_dir: Path, output_dir: Path) -> EnsembleTrainingConfig:
    ceiling = ArcFaceCeilingSpec()
    return EnsembleTrainingConfig(
        train_dir=str(train_dir.resolve()),
        val_dir=str(val_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        target_backbone="resnet18",
        embedding_dim=256,
        learning_rate=ceiling.lr_start,
        weight_decay=ceiling.weight_decay,
        momentum=ceiling.momentum,
        arcface_scale=ceiling.arcface_scale,
        arcface_margin=ceiling.arcface_margin,
        use_partial_fc=True,
        partial_fc_negative_sample_rate=ceiling.partial_fc_negative_sample_rate,
        sub_center_count=ceiling.sub_center_count,
        alignment_detector=ceiling.detector_name,
        alignment_landmarks=ceiling.alignment_points,
        normalized_crop_size=ceiling.normalized_crop_size,
        dataset_name=ceiling.dataset_name,
        pairing_strategy="ensemble_hard",
        hard_pair_fraction=0.20,
        hard_pair_min_pairs=4,
        hard_pair_max_pairs=32,
        hard_pair_weight=0.15,
        curriculum_enabled=True,
        curriculum_warmup_epochs=3,
        curriculum_clean_fraction_end=0.25,
        curriculum_hard_pair_fraction_end=0.35,
        curriculum_hard_pair_weight_end=0.30,
        surrogate_models=["target"],
        attack_sampling_strategy="weighted_random",
        evaluate_all_attacks=True,
        primary_attackers=[
            AttackPolicy(
                name="pgd",
                family="primary_white_box",
                kind="online",
                weight=0.30,
                eps=8.0 / 255.0,
                alpha=2.0 / 255.0,
                steps=6,
                random_start=True,
                restarts=2,
                surrogate_weights={"target": 1.0},
            ),
            AttackPolicy(
                name="bpfa",
                family="primary_transfer",
                kind="online",
                weight=0.25,
                eps=8.0 / 255.0,
                alpha=1.0 / 255.0,
                steps=8,
                random_start=True,
                restarts=2,
                surrogate_weights={"target": 1.0},
            ),
            AttackPolicy(
                name="dfanet",
                family="primary_feature",
                kind="online",
                weight=0.25,
                eps=8.0 / 255.0,
                alpha=2.0 / 255.0,
                steps=6,
                random_start=True,
                restarts=2,
                surrogate_weights={"target": 1.0},
            ),
        ],
        surrogate_attackers=[
            AttackPolicy(
                name="advfacegan",
                family="surrogate_cached",
                kind="cached",
                enabled=False,
                weight=0.10,
            ),
            AttackPolicy(
                name="adv_makeup",
                family="surrogate_cached",
                kind="cached",
                enabled=False,
                weight=0.05,
            ),
            AttackPolicy(
                name="greedy_dim",
                family="surrogate_cached",
                kind="cached",
                enabled=False,
                weight=0.05,
            ),
        ],
    )


def published_ceiling_placeholder_config(train_dir: Path, val_dir: Path, output_dir: Path) -> EnsembleTrainingConfig:
    config = default_mode_b_config(train_dir, val_dir, output_dir)
    config.dataset_name = "WebFace42M"
    config.alignment_detector = "RetinaFace-class"
    config.alignment_landmarks = 5
    config.normalized_crop_size = 112
    config.target_backbone = "resnet18"
    config.embedding_dim = 256
    config.use_gradient_checkpointing = True
    config.use_mixed_precision = True
    config.use_distributed = True
    config.use_sync_batchnorm = True
    config.optimizer_name = "sgd"
    config.learning_rate = 0.1
    config.momentum = 0.9
    config.weight_decay = 5e-4
    config.lr_milestones = [8, 12, 16]
    config.lr_gamma = 0.1
    config.arcface_scale = 64.0
    config.arcface_margin = 0.5
    config.use_partial_fc = True
    config.partial_fc_negative_sample_rate = 0.3
    config.sub_center_count = ArcFaceCeilingSpec().sub_center_count
    config.pairing_strategy = "ensemble_hard"
    config.hard_pair_fraction = 0.20
    config.hard_pair_weight = 0.15
    config.curriculum_enabled = True
    config.curriculum_warmup_epochs = 3
    config.curriculum_clean_fraction_end = 0.25
    config.curriculum_hard_pair_fraction_end = 0.35
    config.curriculum_hard_pair_weight_end = 0.30
    return config


def _policy_from_dict(data: dict) -> AttackPolicy:
    return AttackPolicy(**data)


def _recognizer_policy_from_dict(data: dict) -> RecognizerPolicy:
    return RecognizerPolicy(**data)


def _split_legacy_attack_policies(policies: list[AttackPolicy]) -> tuple[list[AttackPolicy], list[AttackPolicy]]:
    primary: list[AttackPolicy] = []
    surrogate: list[AttackPolicy] = []
    for policy in policies:
        family = policy.family.strip().lower()
        if family.startswith("surrogate") or policy.kind == "cached":
            surrogate.append(policy)
        else:
            primary.append(policy)
    return primary, surrogate


def _remap_dataset_path(raw_path: str, *, dataset_name: str, env_var: str) -> str:
    env_root = os.environ.get(env_var)
    if not env_root:
        return raw_path
    normalized = raw_path.replace("\\", "/")
    marker = f"/Dataset/{dataset_name}/"
    if marker not in normalized:
        return os.path.expandvars(raw_path)
    suffix = normalized.split(marker, 1)[1]
    return str((Path(env_root).expanduser() / suffix).resolve())


def _remap_pairs_path(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/")
    marker = "/Dataset/pairs/"
    if marker not in normalized:
        return raw_path

    dataset_root = os.environ.get("FYP_WEBFACE4M_ROOT") or os.environ.get("FYP_WEBFACE42M_ROOT")
    if not dataset_root:
        return raw_path

    pairs_root = Path(dataset_root).expanduser().resolve().parent / "pairs"
    suffix = normalized.split(marker, 1)[1]
    return str((pairs_root / suffix).resolve())


def _remap_output_path(raw_path: str) -> str:
    env_root = os.environ.get("FYP_OUTPUT_ROOT")
    if not env_root:
        return raw_path
    normalized = raw_path.replace("\\", "/")
    marker = "/TrainingRuns/"
    if marker not in normalized:
        return os.path.expandvars(raw_path)
    suffix = normalized.split(marker, 1)[1]
    return str((Path(env_root).expanduser() / suffix).resolve())


def _portable_config_paths(config_data: dict) -> dict:
    adjusted = dict(config_data)
    for key in ("train_dir", "val_dir", "test_dir", "val_pairs_path", "test_pairs_path", "output_dir", "resume_from"):
        value = adjusted.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        expanded = os.path.expandvars(value)
        if key == "output_dir":
            adjusted[key] = _remap_output_path(expanded)
            continue
        if key == "resume_from":
            adjusted[key] = _remap_output_path(expanded)
            continue
        if key in {"val_pairs_path", "test_pairs_path"}:
            adjusted[key] = _remap_pairs_path(expanded)
            continue
        remapped = _remap_dataset_path(expanded, dataset_name="WebFace4M", env_var="FYP_WEBFACE4M_ROOT")
        remapped = _remap_dataset_path(remapped, dataset_name="WebFace42M", env_var="FYP_WEBFACE42M_ROOT")
        adjusted[key] = remapped
    return adjusted


def config_from_dict(data: dict) -> EnsembleTrainingConfig:
    raw_primary = data.get("primary_attackers")
    raw_surrogate = data.get("surrogate_attackers")
    if raw_primary is None and raw_surrogate is None:
        legacy = [_policy_from_dict(item) for item in data.get("attacks", [])]
        primary_attackers, surrogate_attackers = _split_legacy_attack_policies(legacy)
    else:
        primary_attackers = [_policy_from_dict(item) for item in (raw_primary or [])]
        surrogate_attackers = [_policy_from_dict(item) for item in (raw_surrogate or [])]
    eval_attackers = [_policy_from_dict(item) for item in data.get("eval_attackers", [])]
    config_data = {
        **data,
        "primary_attackers": primary_attackers,
        "surrogate_attackers": surrogate_attackers,
        "eval_attackers": eval_attackers,
        "recognizer_ensemble": [_recognizer_policy_from_dict(item) for item in data.get("recognizer_ensemble", [])],
    }
    config_data = _portable_config_paths(config_data)
    config_data.pop("attacks", None)
    raw_mode = str(config_data.get("recognizers_mode", "")).strip().lower()
    raw_target_model = str(config_data.get("target_model", "arcface")).strip().lower()
    if not raw_mode:
        config_data["recognizers_mode"] = "joint_train" if raw_target_model in {"joint_pool", "jointpool", "pool"} else "serial_target"
    elif raw_mode in {"serial_target", "joint_train"}:
        config_data["recognizers_mode"] = raw_mode
    else:
        raise ValueError(
            f"Unsupported recognizers_mode '{config_data.get('recognizers_mode')}'. "
            "Supported values are 'serial_target' and 'joint_train'."
        )
    if config_data["recognizers_mode"] == "serial_target" and raw_target_model in {"joint_pool", "jointpool", "pool"}:
        raise ValueError(
            "target_model='joint_pool' requires recognizers_mode='joint_train'. "
            "Use a single target recognizer for the serial_target mainline."
        )
    raw_profile = str(config_data.get("runtime_profile", "")).strip().lower()
    if not raw_profile:
        config_data["runtime_profile"] = "custom"
    elif raw_profile in {"custom", "benchmark", "paper_full"}:
        config_data["runtime_profile"] = raw_profile
    else:
        raise ValueError(
            f"Unsupported runtime_profile '{config_data.get('runtime_profile')}'. "
            "Supported values are 'custom', 'benchmark', and 'paper_full'."
        )
    raw_lr_scale_mode = str(config_data.get("lr_scale_mode", "linear")).strip().lower()
    if raw_lr_scale_mode in {"linear", "sqrt"}:
        config_data["lr_scale_mode"] = raw_lr_scale_mode
    else:
        raise ValueError(
            f"Unsupported lr_scale_mode '{config_data.get('lr_scale_mode')}'. "
            "Supported values are 'linear' and 'sqrt'."
        )
    raw_recognizer_weight_strategy = str(config_data.get("recognizer_weight_strategy", "static")).strip().lower()
    if raw_recognizer_weight_strategy in {"static", "loss_proportional"}:
        config_data["recognizer_weight_strategy"] = raw_recognizer_weight_strategy
    else:
        raise ValueError(
            f"Unsupported recognizer_weight_strategy '{config_data.get('recognizer_weight_strategy')}'. "
            "Supported values are 'static' and 'loss_proportional'."
        )
    for key in ("primary_attackers", "surrogate_attackers", "eval_attackers"):
        policies = config_data.get(key) or []
        for policy in policies:
            if isinstance(policy, dict):
                policy.pop("bpfa_train_views", None)
                policy.pop("bpfa_eval_views", None)
                policy.pop("bpfa_view_forward_group_size", None)
                policy.pop("bpfa_restart_score_mode", None)
    return EnsembleTrainingConfig(**config_data)


def load_config(path: Path) -> EnsembleTrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return config_from_dict(data)


def save_config_snapshot(config: EnsembleTrainingConfig, path: Path) -> None:
    payload = asdict(config)
    payload.pop("attacks", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
