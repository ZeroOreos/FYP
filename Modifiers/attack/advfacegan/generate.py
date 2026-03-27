#!/usr/bin/env python3
# python3 generate.py <dataset_dir> <pair_input> <output_dir> <records_out> -> AdvFaceGAN-style dual-identity probes

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from Modifiers.attack.shared import build_base_arg_parser, build_output_path, finalize_generator_args
from Modifiers.attack.shared import load_image_tensor, load_pair_rows, make_record, print_run_header
from Modifiers.attack.shared import require_existing_paths, save_tensor_image, set_global_seed, write_summary
from Utility.runtime import resolve_torch_device

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError as exc:  # pragma: no cover - dependency is optional at import time
    InceptionResnetV1 = None
    FACENET_IMPORT_ERROR = exc
else:
    FACENET_IMPORT_ERROR = None


DEFAULT_IMAGE_SIZE = 112
DEFAULT_MAX_PERTURBATION = 0.12
DEFAULT_STEPS = 120
DEFAULT_LR = 0.02
DEFAULT_SOURCE_WEIGHT = 1.0
DEFAULT_TARGET_WEIGHT = 1.35
DEFAULT_SOURCE_PIXEL_WEIGHT = 12.0
DEFAULT_TARGET_PIXEL_WEIGHT = 1.5
DEFAULT_TV_WEIGHT = 0.002
DEFAULT_BALANCE_WEIGHT = 0.1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AdvFaceGANGenerator(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 64, num_blocks: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(*[ResidualBlock(base_channels * 4) for _ in range(num_blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Tanh(),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([source, target], dim=1)
        features = self.encoder(stacked)
        features = self.residual(features)
        return self.decoder(features)


class FaceNetEmbedder(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        if InceptionResnetV1 is None:
            raise ImportError(
                "facenet_pytorch is needed for AdvFaceGAN native generation"
            ) from FACENET_IMPORT_ERROR
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(image_tensor)
        return F.normalize(embeddings, p=2, dim=1)


def parse_args() -> Any:
    parser = build_base_arg_parser(
        "Generate AdvFaceGAN attack probes.",
        default_image_size=DEFAULT_IMAGE_SIZE,
        include_seed=True,
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-perturbation", type=float, default=DEFAULT_MAX_PERTURBATION)
    parser.add_argument("--source-weight", type=float, default=DEFAULT_SOURCE_WEIGHT)
    parser.add_argument("--target-weight", type=float, default=DEFAULT_TARGET_WEIGHT)
    parser.add_argument("--source-pixel-weight", type=float, default=DEFAULT_SOURCE_PIXEL_WEIGHT)
    parser.add_argument("--target-pixel-weight", type=float, default=DEFAULT_TARGET_PIXEL_WEIGHT)
    parser.add_argument("--tv-weight", type=float, default=DEFAULT_TV_WEIGHT)
    parser.add_argument("--balance-weight", type=float, default=DEFAULT_BALANCE_WEIGHT)
    parser.add_argument("--generator-checkpoint", type=Path, default=None)
    return finalize_generator_args(parser.parse_args(), optional_path_fields=("generator_checkpoint",))


def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    return torch.sum(F.normalize(emb_a, dim=1) * F.normalize(emb_b, dim=1), dim=1)


def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    horizontal = torch.mean(torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]))
    return horizontal + vertical


def optimize_dual_identity_image(
    source: torch.Tensor,
    target: torch.Tensor,
    embedder: nn.Module,
    *,
    steps: int,
    lr: float,
    max_perturbation: float,
    source_weight: float,
    target_weight: float,
    source_pixel_weight: float,
    target_pixel_weight: float,
    tv_weight: float,
    balance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        source_embedding = embedder(source)
        target_embedding = embedder(target)

    delta_param = torch.zeros_like(source, requires_grad=True)
    optimizer = torch.optim.Adam([delta_param], lr=lr)

    best_adv = source.detach().clone()
    best_stats = {
        "objective": math.inf,
        "source_similarity": float(cosine_similarity(source_embedding, source_embedding).item()),
        "target_similarity": float(cosine_similarity(source_embedding, target_embedding).item()),
        "perturbation_linf": 0.0,
    }

    for _ in range(max(1, steps)):
        delta = torch.tanh(delta_param) * max_perturbation
        adv = torch.clamp(source + delta, min=-1.0, max=1.0)
        adv_embedding = embedder(adv)

        source_similarity = cosine_similarity(adv_embedding, source_embedding)
        target_similarity = cosine_similarity(adv_embedding, target_embedding)
        source_identity_loss = 1.0 - source_similarity.mean()
        target_identity_loss = 1.0 - target_similarity.mean()
        source_pixel_loss = F.l1_loss(adv, source)
        target_pixel_loss = F.l1_loss(adv, target)
        smoothness_loss = total_variation_loss(delta)
        balance_loss = torch.mean(torch.abs(target_similarity - source_similarity))

        loss = (
            source_weight * source_identity_loss
            + target_weight * target_identity_loss
            + source_pixel_weight * source_pixel_loss
            + target_pixel_weight * target_pixel_loss
            + tv_weight * smoothness_loss
            + balance_weight * balance_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        objective = float(loss.item())
        if objective < best_stats["objective"]:
            best_adv = adv.detach().clone()
            best_stats = {
                "objective": objective,
                "source_similarity": float(source_similarity.mean().item()),
                "target_similarity": float(target_similarity.mean().item()),
                "perturbation_linf": float(delta.detach().abs().amax().item()),
            }

    return best_adv, best_stats


def generate_with_checkpoint(
    source: torch.Tensor,
    target: torch.Tensor,
    checkpoint_path: Path,
    device: torch.device,
    max_perturbation: float,
    embedder: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("generator", checkpoint)

    generator = AdvFaceGANGenerator().to(device)
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()

    with torch.no_grad():
        perturbation = torch.tanh(generator(source, target)) * max_perturbation
        adv = torch.clamp(source + perturbation, min=-1.0, max=1.0)
        source_embedding = embedder(source)
        target_embedding = embedder(target)
        adv_embedding = embedder(adv)
        stats = {
            "objective": 0.0,
            "source_similarity": float(cosine_similarity(adv_embedding, source_embedding).mean().item()),
            "target_similarity": float(cosine_similarity(adv_embedding, target_embedding).mean().item()),
            "perturbation_linf": float(perturbation.abs().amax().item()),
        }
    return adv, stats


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed), include_torch=True)

    dataset_dir = args.dataset_dir
    pair_input = args.pair_input
    output_dir = args.output_dir
    records_out = args.records_out
    checkpoint_path = args.generator_checkpoint
    device = torch.device(resolve_torch_device(args.device))

    require_existing_paths(dataset_dir, pair_input, checkpoint_path)
    rows = load_pair_rows(pair_input)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_out.parent.mkdir(parents=True, exist_ok=True)

    embedder = FaceNetEmbedder(device)
    print_run_header(
        method_name="advfacegan_native",
        dataset_dir=dataset_dir,
        pair_input=pair_input,
        output_dir=output_dir,
        num_pairs=len(rows),
        extra_lines=(
            f"device: {device}",
            (
                f"generator checkpoint: {checkpoint_path}"
                if checkpoint_path is not None
                else "using direct dual-identity optimization"
            ),
        ),
    )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_path = Path(row["attacker_image"]).resolve()
        target_path = Path(row["victim_image"]).resolve()
        output_path = build_output_path(output_dir, row)
        print(f"[PAIR] {index + 1}/{len(rows)} -> {output_path}")

        if output_path.exists() and not args.overwrite:
            records.append(make_record(
                index=index,
                row=row,
                output_path=output_path,
                status="skipped",
                message="output already exists",
                source_path=source_path,
                target_path=target_path,
            ))
            continue

        try:
            source = load_image_tensor(source_path, image_size=int(args.image_size), device=device)
            target = load_image_tensor(target_path, image_size=int(args.image_size), device=device)

            if checkpoint_path is not None:
                adv, stats = generate_with_checkpoint(
                    source=source,
                    target=target,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    max_perturbation=float(args.max_perturbation),
                    embedder=embedder,
                )
            else:
                adv, stats = optimize_dual_identity_image(
                    source=source,
                    target=target,
                    embedder=embedder,
                    steps=int(args.steps),
                    lr=float(args.lr),
                    max_perturbation=float(args.max_perturbation),
                    source_weight=float(args.source_weight),
                    target_weight=float(args.target_weight),
                    source_pixel_weight=float(args.source_pixel_weight),
                    target_pixel_weight=float(args.target_pixel_weight),
                    tv_weight=float(args.tv_weight),
                    balance_weight=float(args.balance_weight),
                )

            save_tensor_image(adv, output_path, jpeg_quality=int(args.jpeg_quality))
            records.append(make_record(
                index=index,
                row=row,
                output_path=output_path,
                status="ok",
                message="generated",
                source_path=source_path,
                target_path=target_path,
                source_similarity=stats["source_similarity"],
                target_similarity=stats["target_similarity"],
                perturbation_linf=stats["perturbation_linf"],
            ))
        except Exception as exc:  # pragma: no cover - per-sample failures are part of runtime behavior
            records.append(make_record(
                index=index,
                row=row,
                output_path=output_path,
                status="failed",
                message=str(exc),
                source_path=source_path,
                target_path=target_path,
            ))
            print(f"[WARN] Pair failed {index}: {exc}")

    write_summary(
        method_name="advfacegan_native",
        dataset_dir=dataset_dir,
        pair_input=pair_input,
        output_dir=output_dir,
        records_out=records_out,
        args=args,
        records=records,
        extra_summary={
            "device": str(device),
            "generator_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        },
    )
    print(f"[INFO] records written: {records_out}")


if __name__ == "__main__":
    main()
