"""
Utility for downloading the OSV5M dataset locally.

The default path expects the Hugging Face repo layout:
<root>/
  train/*.tar
  val/*.tar
  metadata/train.jsonl
  metadata/val.jsonl

Adjust patterns via CLI flags if your layout differs.
"""
import argparse
import os
from typing import Iterable, List

from huggingface_hub import snapshot_download


def _build_patterns(splits: Iterable[str], include_images: bool, include_metadata: bool) -> List[str]:
    patterns: List[str] = []
    for split in splits:
        if include_images:
            patterns.append(f"{split}/*.tar")
        if include_metadata:
            patterns.append(f"metadata/{split}*")
    return patterns


def download_osv5m(
    target_dir: str,
    repo_id: str = "mmint/osv5m",
    revision: str | None = None,
    splits: Iterable[str] = ("train", "val"),
    include_images: bool = True,
    include_metadata: bool = True,
    max_workers: int = 8,
) -> str:
    os.makedirs(target_dir, exist_ok=True)
    allow_patterns = _build_patterns(splits, include_images, include_metadata) or None
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target_dir,
        allow_patterns=allow_patterns,
        resume_download=True,
        max_workers=max_workers,
    )
    return snapshot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OSV5M dataset locally.")
    parser.add_argument("--target_dir", type=str, required=True, help="Destination directory for the dataset.")
    parser.add_argument("--repo_id", type=str, default="mmint/osv5m", help="Hugging Face dataset repo id.")
    parser.add_argument("--revision", type=str, default=None, help="Optional commit hash or tag.")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Splits to download.")
    parser.add_argument("--images", action="store_true", default=True, help="Download image shards.")
    parser.add_argument("--no-images", dest="images", action="store_false", help="Skip image shards.")
    parser.add_argument("--metadata", action="store_true", default=True, help="Download metadata files.")
    parser.add_argument("--no-metadata", dest="metadata", action="store_false", help="Skip metadata files.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_path = download_osv5m(
        target_dir=args.target_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        splits=args.splits,
        include_images=args.images,
        include_metadata=args.metadata,
        max_workers=args.workers,
    )
    print(f"Dataset materialized at {snapshot_path}")


if __name__ == "__main__":
    main()
