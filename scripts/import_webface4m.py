#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


DEFAULT_TARGET = Path("Dataset/WebFace4M")
BASE_URL = "https://huggingface.co/datasets/gaunernst/webface4m-wds-gz/resolve/main"
NUM_SHARDS = 121


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WebFace4M WebDataset shards with progress.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=NUM_SHARDS - 1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _download(url: str, destination: Path) -> None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as response:
        total_bytes = int(response.headers.get("Content-Length", "0"))
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    start_time = time.time()
    downloaded = 0

    def report(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal downloaded
        downloaded = block_count * block_size
        total = total_size if total_size > 0 else total_bytes
        if total <= 0:
            return
        progress = min(100.0, 100.0 * downloaded / total)
        rate = downloaded / max(1e-6, time.time() - start_time) / (1024 * 1024)
        print(
            f"[DOWNLOAD] {destination.name} {progress:6.2f}% "
            f"{downloaded / (1024 * 1024):8.1f}/{total / (1024 * 1024):8.1f} MB "
            f"{rate:6.2f} MB/s"
        )

    urllib.request.urlretrieve(url, tmp_path, reporthook=report)
    tmp_path.replace(destination)


def main() -> None:
    args = parse_args()
    target_root = args.target_root.resolve()
    raw_root = target_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": "WebFace4M",
        "format": "webdataset-tar-gz",
        "source": BASE_URL,
        "num_shards_total": NUM_SHARDS,
        "downloaded_shards": [],
    }
    for shard in range(args.start_shard, args.end_shard + 1):
        name = f"webface4m-{shard:04d}.tar.gz"
        destination = raw_root / name
        if destination.exists() and not args.overwrite:
            print(f"[SKIP] {name}")
            manifest["downloaded_shards"].append(name)
            continue
        url = f"{BASE_URL}/{name}"
        _download(url, destination)
        manifest["downloaded_shards"].append(name)
    manifest_path = target_root / "import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = target_root / "README.md"
    readme.write_text(
        "# WebFace4M Local Target\n\n"
        "This folder stores the local WebFace4M download in WebDataset shard form.\n\n"
        "- Source: gaunernst/webface4m-wds-gz on Hugging Face\n"
        "- Format: `.tar.gz` WebDataset shards\n"
        "- Native manifest-backed training is supported through `Training/dataset.py`\n"
        "- After download, run `scripts/prepare_webface4m_manifests.py` to build `train.jsonl` and `val.jsonl`\n",
        encoding="utf-8",
    )
    print(f"[INFO] Downloaded shards: {len(manifest['downloaded_shards'])}")
    print(f"[INFO] Target root:       {target_root}")


if __name__ == "__main__":
    main()
