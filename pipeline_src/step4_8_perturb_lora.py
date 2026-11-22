""""""

import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .utils import (
    BASE_DIR,
    log,
    run_cmd,
)
def _ix_to_tag(ix: float) -> str:
    return f"{ix:g}x"


def _parse_ix_values(ix_values: List[str]) -> List[str]:
    cleaned = sorted(set(str(v) for v in ix_values))
    if not cleaned:
        raise ValueError("No ix values provided for Step 4.8")
    for v in cleaned:
        if v not in ["infl", "half-infl"] and float(v) < 1.0:
            raise ValueError(f"Invalid ix value (<1.0): {v}")
    if "half-infl" in cleaned and "infl" not in cleaned:
        raise ValueError("Cannot specify 'half-infl' without 'infl'.")
    return cleaned


def _discover_over_checkpoints(base_dir: Path, ix_values: List[str] = None) -> List[Path]:
    if ix_values is not None and any(t in ["infl", "half-infl"] for t in ix_values):
        tags = None
    else:
        tags = {_ix_to_tag(float(v)) for v in ix_values} if ix_values is not None else None
    out: List[Path] = []
    if not base_dir.exists():
        return out
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("original_"):
            continue
        if "_over_" not in p.name:
            continue
        suffix = p.name.split("_over_", 1)[1]
        if (tags is None or suffix in tags) and _is_valid_adapter_checkpoint(p):
            out.append(p)
    return out


def _is_valid_adapter_checkpoint(path: Path) -> bool:
    if not (path / "adapter_config.json").exists():
        return False
    if (path / "adapter_model.safetensors").exists():
        return True
    if (path / "adapter_model.bin").exists():
        return True
    return False


def _write_sampled_subset(source_path: Path, target_path: Path, seed: int, max_samples: int = 500) -> int:
    rows = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line)

    if not rows:
        raise RuntimeError(f"No rows found in {source_path}")

    sample_size = min(max_samples, len(rows))
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(rows)), k=sample_size))

    with open(target_path, "w", encoding="utf-8") as f:
        for idx in selected_indices:
            f.write(rows[idx])

    return sample_size


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    log("=" * 60)
    log("STEP 4.8: Perturb LoRA (Over Only)")
    log("=" * 60)

    lora_adapter_path = workdir / "step4_7_lora" / "adapter"
    if not lora_adapter_path.exists():
        legacy_link = workdir / "step4_7_lora_adapter"
        if legacy_link.exists():
            lora_adapter_path = legacy_link.resolve() if legacy_link.is_symlink() else legacy_link

    if not lora_adapter_path.exists():
        raise FileNotFoundError(
            "Missing Step 4.7 LoRA adapter path. "
            f"Checked: {workdir / 'step4_7_lora' / 'adapter'}"
        )

    source_dataset_file = workdir / "step4_continued_1k.jsonl"
    if not source_dataset_file.exists():
        raise FileNotFoundError(f"Missing Step 4.8 source dataset: {source_dataset_file}")

    sampled_subset_file = workdir / "step4_8_sampled_subset.jsonl"
    sampled_count = _write_sampled_subset(
        source_dataset_file,
        sampled_subset_file,
        seed=int(args.seed),
        max_samples=int(args.partial_x),
    )
    log(
        f"Step 4.8 using {sampled_count} sampled examples "
        f"(partial_x={int(args.partial_x)}) from {source_dataset_file}"
    )

    ix_mode = getattr(args, "lora_perturb_ix_values", [2.0, 3.0])
    if ix_mode is None:
        ix_values = None
    else:
        ix_values = _parse_ix_values(ix_mode)

    out_dir = workdir / "step4_8_perturbed_lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = workdir / "step4_8_metadata.json"

    existing = _discover_over_checkpoints(out_dir, ix_values)
    skipped = bool(existing and metadata_file.exists())
    if skipped:
        log("[SKIP] Step 4.8 (LoRA perturbation): over checkpoints already exist")
    else:
        max_length = 1000

        cmd = [
            sys.executable,
            str(BASE_DIR / "m1" / "src" / "perturb" / "perturb-model-lora.py"),
            "--lora_model", str(lora_adapter_path),
            "--dataset", str(sampled_subset_file),
            "--output_dir", str(out_dir),
            "--max_samples", "-1",
            "--max_length", str(max_length),
            "--source-eval", "none",
            "--skip-original-direction",
            "--save_ckpt",
            "--save-as-adapter",
        ]
        for v in ix_values:
            cmd.extend(["--ix", str(v)])

        run_cmd(cmd)

    over_ckpts = _discover_over_checkpoints(out_dir, ix_values)
    if not over_ckpts:
        raise RuntimeError(f"No valid *_over_* checkpoints found in {out_dir}")

    metadata = {
        "seed": int(args.seed),
        "sampled_subset_file": str(sampled_subset_file),
        "sampled_count": int(sampled_count),
        "ix_values": ix_values if ix_values is not None else "inflection",
        "ix_mode": "inflection" if ix_values is None else "numeric",
        "lora_adapter_path": str(lora_adapter_path),
        "dataset_for_eval": str(source_dataset_file),
        "output_dir": str(out_dir),
        "checkpoint_format": "adapter",
        "over_checkpoints": [str(p) for p in over_ckpts],
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_dir, skipped
