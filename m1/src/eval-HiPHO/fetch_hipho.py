#!/usr/bin/env python3
""""""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


REPO_RAW = "https://raw.githubusercontent.com/SciYu/HiPhO/main/data"

PAPER_EXAMS = [
    "APhO_2025.json",
    "EuPhO_2024.json",
    "EuPhO_2025.json",
    "F=MA_2024.json",
    "F=MA_2025.json",
    "IPhO_2024.json",
    "IPhO_2025.json",
    "NBPhO_2024.json",
    "NBPhO_2025.json",
    "PanMechanics_2024.json",
    "PanMechanics_2025.json",
    "PanPhO_2024.json",
    "PanPhO_2025.json",
]

EXTRA_CURRENT_EXAMS = [
    "CPhO_2025.json",
]


def download(url: str, path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("m1/src/eval-HiPHO/data"))
    parser.add_argument("--include-current-extra", action="store_true", help="Also fetch exams added after the paper, e.g. CPhO_2025.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    exams = list(PAPER_EXAMS)
    if args.include_current_extra:
        exams.extend(EXTRA_CURRENT_EXAMS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    referenced_images: set[str] = set()
    for exam in exams:
        out_path = args.output_dir / exam
        download(f"{REPO_RAW}/{exam}", out_path, overwrite=args.overwrite)
        data = json.loads(out_path.read_text())
        for item in data:
            for image_path in item.get("image_question", []) or []:
                referenced_images.add(image_path)

    manifest = {
        "source": "https://github.com/SciYu/HiPhO",
        "paper_exams": PAPER_EXAMS,
        "include_current_extra": args.include_current_extra,
        "extra_current_exams": EXTRA_CURRENT_EXAMS if args.include_current_extra else [],
        "text_only": True,
        "num_referenced_images_not_downloaded": len(referenced_images),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {len(exams)} exam JSON files to {args.output_dir}")
    if referenced_images:
        print(f"Skipped {len(referenced_images)} referenced images; this evaluator is text-only.")


if __name__ == "__main__":
    main()
