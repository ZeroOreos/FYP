from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


DATASET_DIRNAME = "Dataset"
PAIRS_DIRNAME = "pairs"


@dataclass(frozen=True)
class DatasetContext:
    dataset_dir: Path
    dataset_root: Path
    dataset_relative_parts: tuple[str, ...]
    base_root_name: str
    base_dataset_relative_parts: tuple[str, ...]
    variant_name: str
    base_dataset_name: str
    variant_type: str
    transform_chain: str
    num_transforms: int

    @property
    def pair_file_stem(self) -> str:
        return "_".join(self.base_dataset_relative_parts)

    @property
    def pair_filename(self) -> str:
        return f"{self.pair_file_stem}_pairs.npz"

    @property
    def pair_metadata_filename(self) -> str:
        return f"{self.pair_file_stem}_pairs.json"


def _find_dataset_root(dataset_dir: Path) -> tuple[Path, tuple[str, ...]]:
    resolved = dataset_dir.resolve()
    parts = list(resolved.parts)

    if DATASET_DIRNAME in parts:
        idx = parts.index(DATASET_DIRNAME)
        return Path(*parts[: idx + 1]), tuple(parts[idx + 1 :])

    fallback_relative = tuple(resolved.parts[-2:])
    return resolved.parent, fallback_relative


def _parse_base_root_name(variant_name: str) -> str:
    if not variant_name:
        raise ValueError("Variant name cannot be empty.")
    if "_" not in variant_name:
        return variant_name
    return variant_name.rsplit("_", 1)[-1]


def _parse_transform_tokens(variant_name: str, base_root_name: str) -> List[str]:
    if variant_name == base_root_name:
        return []

    suffix = f"__{base_root_name}"
    if variant_name.endswith(suffix):
        prefix = variant_name[: -len(suffix)]
        return [token for token in prefix.split("__") if token]

    suffix = f"_{base_root_name}"
    if variant_name.endswith(suffix):
        prefix = variant_name[: -len(suffix)]
        return [token for token in prefix.split("__") if token] if prefix else []

    if variant_name.endswith(base_root_name):
        prefix = variant_name[: -len(base_root_name)].rstrip("_")
        return [token for token in prefix.split("__") if token] if prefix else []

    return [variant_name]


def determine_variant_type(variant_name: str, base_root_name: str, transform_tokens: List[str]) -> str:
    if variant_name == base_root_name:
        return "clean"
    if len(transform_tokens) <= 1:
        return "single"
    return "hybrid"


def resolve_dataset_context(dataset_dir: Path) -> DatasetContext:
    dataset_root, relative_parts = _find_dataset_root(dataset_dir)
    if len(relative_parts) < 2:
        raise ValueError(
            f"Dataset directory must include at least <dataset>/<variant>: {dataset_dir}"
        )

    base_dataset_name = relative_parts[0]
    variant_name = relative_parts[-1]
    base_root_name = _parse_base_root_name(variant_name)
    transform_tokens = _parse_transform_tokens(variant_name, base_root_name)
    variant_type = determine_variant_type(variant_name, base_root_name, transform_tokens)
    transform_chain = "__".join(transform_tokens) if transform_tokens else "clean"

    return DatasetContext(
        dataset_dir=dataset_dir.resolve(),
        dataset_root=dataset_root,
        dataset_relative_parts=relative_parts,
        base_root_name=base_root_name,
        base_dataset_relative_parts=relative_parts[:-1] + (base_root_name,),
        variant_name=variant_name,
        base_dataset_name=base_dataset_name,
        variant_type=variant_type,
        transform_chain=transform_chain,
        num_transforms=len(transform_tokens),
    )


def derive_pairs_output_path(dataset_dir: Path) -> Path:
    context = resolve_dataset_context(dataset_dir)
    return context.dataset_root / PAIRS_DIRNAME / context.pair_filename
