from __future__ import annotations

import io
import json
import math
import random
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label_name: str
    label_idx: int
    rel_path: str


@dataclass(frozen=True)
class WebDatasetSample:
    shard_path: Path
    key: str
    label_name: str
    label_idx: int
    rel_path: str


@dataclass(frozen=True)
class DatasetSubsetPlan:
    selected_labels: tuple[str, ...]
    max_samples_per_identity: dict[str, int] | None = None
    selected_rel_paths: dict[str, tuple[str, ...]] | None = None


class _TransformMixin:
    def __init__(self, image_size: int) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def _load_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))


class FaceTrainDataset(_TransformMixin, Dataset):
    def __init__(
        self,
        root_dir: Path,
        class_to_idx: dict[str, int],
        image_size: int,
        subset_plan: DatasetSubsetPlan | None = None,
    ) -> None:
        super().__init__(image_size=image_size)
        self.root_dir = root_dir.resolve()
        self.class_to_idx = class_to_idx
        self.subset_plan = subset_plan
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[ImageSample]:
        samples: list[ImageSample] = []
        allowed = set(self.subset_plan.selected_labels) if self.subset_plan is not None else None
        quotas = self.subset_plan.max_samples_per_identity if self.subset_plan is not None else None
        allowed_rel_paths = (
            {label: set(paths) for label, paths in (self.subset_plan.selected_rel_paths or {}).items()}
            if self.subset_plan is not None
            else {}
        )
        for identity_dir in sorted(self.root_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            label_name = identity_dir.name
            if allowed is not None and label_name not in allowed:
                continue
            if label_name not in self.class_to_idx:
                continue
            remaining = None if quotas is None else quotas.get(label_name)
            if remaining is not None and remaining <= 0:
                continue
            for img_path in sorted(identity_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
                    continue
                rel_path = img_path.relative_to(self.root_dir).as_posix()
                label_allowed_rel_paths = allowed_rel_paths.get(label_name)
                if label_allowed_rel_paths and rel_path not in label_allowed_rel_paths:
                    continue
                samples.append(
                    ImageSample(
                        path=img_path.resolve(),
                        label_name=label_name,
                        label_idx=self.class_to_idx[label_name],
                        rel_path=rel_path,
                    )
                )
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
        if not samples:
            raise RuntimeError(f"No valid images found in dataset: {self.root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        tensor = self._load_tensor(image)
        return tensor, sample.label_idx, str(sample.path), sample.rel_path


class WebFaceManifestDataset(_TransformMixin, Dataset):
    def __init__(
        self,
        manifest_path: Path,
        class_to_idx: dict[str, int],
        image_size: int,
        subset_plan: DatasetSubsetPlan | None = None,
    ) -> None:
        super().__init__(image_size=image_size)
        self.manifest_path = manifest_path.resolve()
        self.class_to_idx = class_to_idx
        self.subset_plan = subset_plan
        self._tar_cache: OrderedDict[Path, tarfile.TarFile] = OrderedDict()
        self._tar_cache_size = 4
        self.samples = self._load_manifest()

    def _load_manifest(self) -> list[WebDatasetSample]:
        samples: list[WebDatasetSample] = []
        allowed = set(self.subset_plan.selected_labels) if self.subset_plan is not None else None
        quotas = dict(self.subset_plan.max_samples_per_identity or {}) if self.subset_plan is not None else {}
        allowed_rel_paths = (
            {label: set(paths) for label, paths in (self.subset_plan.selected_rel_paths or {}).items()}
            if self.subset_plan is not None
            else {}
        )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                label_name = str(item["label_name"])
                rel_path = str(item["rel_path"])
                if allowed is not None and label_name not in allowed:
                    continue
                label_allowed_rel_paths = allowed_rel_paths.get(label_name)
                if label_allowed_rel_paths and rel_path not in label_allowed_rel_paths:
                    continue
                remaining = quotas.get(label_name)
                if remaining is not None and remaining <= 0:
                    continue
                if label_name not in self.class_to_idx:
                    continue
                samples.append(
                    WebDatasetSample(
                        shard_path=Path(item["shard_path"]).resolve(),
                        key=str(item["key"]),
                        label_name=label_name,
                        label_idx=self.class_to_idx[label_name],
                        rel_path=rel_path,
                    )
                )
                if remaining is not None:
                    quotas[label_name] = remaining - 1
        if not samples:
            raise RuntimeError(f"No usable samples found in manifest: {self.manifest_path}")
        return samples

    def _get_tar(self, shard_path: Path) -> tarfile.TarFile:
        shard_path = shard_path.resolve()
        cached = self._tar_cache.pop(shard_path, None)
        if cached is None:
            cached = tarfile.open(shard_path, mode="r:gz")
        self._tar_cache[shard_path] = cached
        while len(self._tar_cache) > self._tar_cache_size:
            _, old_tar = self._tar_cache.popitem(last=False)
            old_tar.close()
        return cached

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        tar_handle = self._get_tar(sample.shard_path)
        member = tar_handle.getmember(f"{sample.key}.jpg")
        file_obj = tar_handle.extractfile(member)
        if file_obj is None:
            raise RuntimeError(f"Failed to read image bytes for {sample.key} in {sample.shard_path}")
        image = Image.open(io.BytesIO(file_obj.read())).convert("RGB")
        tensor = self._load_tensor(image)
        source = f"{sample.shard_path}::{sample.key}.jpg"
        return tensor, sample.label_idx, source, sample.rel_path

    def __del__(self) -> None:
        for tar_handle in getattr(self, "_tar_cache", {}).values():
            try:
                tar_handle.close()
            except Exception:
                pass


class ShardAwareSampler(Sampler[int]):
    def __init__(self, dataset: WebFaceManifestDataset, *, seed: int = 42) -> None:
        self.dataset = dataset
        self.seed = seed
        grouped: dict[Path, list[int]] = {}
        for index, sample in enumerate(dataset.samples):
            grouped.setdefault(sample.shard_path, []).append(index)
        self._groups = [indices for _, indices in sorted(grouped.items(), key=lambda item: str(item[0]))]
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        groups = [list(indices) for indices in self._groups]
        rng.shuffle(groups)
        ordered: list[int] = []
        for indices in groups:
            rng.shuffle(indices)
            ordered.extend(indices)
        return iter(ordered)

    def __len__(self) -> int:
        return len(self.dataset)


def _manifest_sidecar(manifest_path: Path, filename: str) -> Path:
    return manifest_path.resolve().parent / filename


def _load_class_to_idx_from_manifest(manifest_path: Path) -> dict[str, int]:
    mapping_path = _manifest_sidecar(manifest_path, "class_to_idx.json")
    with mapping_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): int(value) for key, value in data.items()}


def _is_manifest_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".jsonl"


def _collect_manifest_rel_paths(manifest_path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            label_name = str(item["label_name"])
            grouped.setdefault(label_name, []).append(str(item["rel_path"]))
    return {label: sorted(paths) for label, paths in grouped.items()}


def _collect_root_rel_paths(root_dir: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for identity_dir in sorted(root_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        images = [
            path.relative_to(root_dir).as_posix()
            for path in sorted(identity_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_EXTS
        ]
        if images:
            grouped[identity_dir.name] = images
    return grouped


def _collect_label_rel_paths(data_path: Path) -> dict[str, list[str]]:
    resolved = data_path.resolve()
    if _is_manifest_path(resolved):
        return _collect_manifest_rel_paths(resolved)
    return _collect_root_rel_paths(resolved)


def _identity_sort_key(item: tuple[str, list[str]]) -> tuple[int, str]:
    label, rel_paths = item
    return (-len(rel_paths), label)


def _make_subset_plan(
    grouped_rel_paths: dict[str, list[str]],
    *,
    fraction: float,
    subset_seed: int,
    min_images_per_identity: int,
) -> DatasetSubsetPlan:
    ordered_items = sorted(grouped_rel_paths.items(), key=_identity_sort_key)
    if not ordered_items:
        raise RuntimeError("No usable identity groups found while building dataset subset.")
    if fraction >= 1.0:
        return DatasetSubsetPlan(selected_labels=tuple(label for label, _ in ordered_items))

    clamped_fraction = max(0.0, fraction)
    total_images = sum(len(paths) for _, paths in ordered_items)
    target_images = max(1, min(total_images, int(math.ceil(total_images * clamped_fraction))))
    eligible = [(label, paths) for label, paths in ordered_items if len(paths) >= max(1, min_images_per_identity)]
    candidate_items = eligible or ordered_items

    best_k = 1
    best_floor = 1
    for k in range(len(candidate_items), 0, -1):
        kth_count = len(candidate_items[k - 1][1])
        base_floor = max(1, min_images_per_identity if eligible else 1)
        per_identity_floor = max(base_floor, target_images // k)
        if per_identity_floor <= kth_count and (per_identity_floor * k) <= target_images:
            best_k = k
            best_floor = per_identity_floor
            break

    selected_items = candidate_items[:best_k]
    quotas = {label: best_floor for label, _ in selected_items}
    remaining_budget = max(0, target_images - (best_floor * len(selected_items)))

    if remaining_budget > 0:
        capacities = [(label, len(paths) - best_floor) for label, paths in selected_items]
        while remaining_budget > 0:
            progressed = False
            for label, capacity in capacities:
                if capacity > quotas[label] - best_floor:
                    quotas[label] += 1
                    remaining_budget -= 1
                    progressed = True
                    if remaining_budget <= 0:
                        break
            if not progressed:
                break

    rng = random.Random(subset_seed)
    selected_rel_paths: dict[str, tuple[str, ...]] = {}
    for label, rel_paths in selected_items:
        if quotas[label] >= len(rel_paths):
            selected_rel_paths[label] = tuple(rel_paths)
            continue
        indices = list(range(len(rel_paths)))
        rng.shuffle(indices)
        chosen = sorted(indices[: quotas[label]])
        selected_rel_paths[label] = tuple(rel_paths[index] for index in chosen)
        quotas[label] = len(chosen)

    return DatasetSubsetPlan(
        selected_labels=tuple(label for label, _ in selected_items),
        max_samples_per_identity=quotas,
        selected_rel_paths=selected_rel_paths,
    )


def build_class_to_idx(
    root_dir: Path,
    *,
    subset_plan: DatasetSubsetPlan | None = None,
) -> dict[str, int]:
    resolved = root_dir.resolve()
    if subset_plan is not None:
        classes = list(subset_plan.selected_labels)
    elif _is_manifest_path(resolved):
        classes = sorted(_load_class_to_idx_from_manifest(resolved))
    else:
        classes = [entry.name for entry in sorted(resolved.iterdir()) if entry.is_dir()]
    if not classes:
        raise RuntimeError(f"No identity directories found in: {resolved}")
    return {name: idx for idx, name in enumerate(classes)}


def _build_dataset(
    data_path: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    subset_plan: DatasetSubsetPlan | None = None,
) -> Dataset:
    resolved = data_path.resolve()
    if _is_manifest_path(resolved):
        return WebFaceManifestDataset(resolved, class_to_idx, image_size, subset_plan=subset_plan)
    return FaceTrainDataset(resolved, class_to_idx, image_size, subset_plan=subset_plan)


def build_dataloaders(
    train_dir: Path,
    val_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
    dataset_fraction: float = 1.0,
    dataset_subset_seed: int = 42,
    dataset_min_images_per_identity: int = 2,
) -> tuple[Dataset, Dataset, DataLoader, DataLoader, dict[str, int]]:
    train_grouped = _collect_label_rel_paths(train_dir)
    val_grouped = _collect_label_rel_paths(val_dir)
    overlap_labels = set(train_grouped) & set(val_grouped)
    subset_source = (
        {label: train_grouped[label] for label in sorted(overlap_labels)}
        if overlap_labels
        else train_grouped
    )

    train_subset_plan = _make_subset_plan(
        subset_source,
        fraction=dataset_fraction,
        subset_seed=dataset_subset_seed,
        min_images_per_identity=dataset_min_images_per_identity,
    )
    val_subset_plan = DatasetSubsetPlan(selected_labels=train_subset_plan.selected_labels)
    class_to_idx = build_class_to_idx(train_dir, subset_plan=train_subset_plan)
    train_ds = _build_dataset(train_dir, class_to_idx, image_size, subset_plan=train_subset_plan)
    val_ds = _build_dataset(val_dir, class_to_idx, image_size, subset_plan=val_subset_plan)

    train_sampler = None
    train_shuffle = True
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            shuffle=True,
            drop_last=True,
            seed=dataset_subset_seed,
        )
        train_shuffle = False
    elif isinstance(train_ds, WebFaceManifestDataset):
        train_sampler = ShardAwareSampler(train_ds, seed=dataset_subset_seed)
        train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_ds, val_ds, train_loader, val_loader, class_to_idx


def build_eval_loader(
    data_dir: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[Dataset, DataLoader]:
    dataset = _build_dataset(
        data_dir,
        class_to_idx,
        image_size,
        subset_plan=DatasetSubsetPlan(selected_labels=tuple(class_to_idx)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader
