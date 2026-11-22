""""""

import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

from .utils import (
    BASE_DIR,
    check_output_exists,
    log,
    run_cmd,
    DEFAULTS,
)


def _discover_difficulty_ckpt_name(step2_loss_dir: Path) -> str:
    """"""
    if (step2_loss_dir / "checkpoint-inflection").exists():
        return "checkpoint-inflection"

    if step2_loss_dir.exists():
        ckpt_dirs = sorted(
            [p for p in step2_loss_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: p.name,
        )
        if ckpt_dirs:
            return ckpt_dirs[0].name

    raise FileNotFoundError(
        f"Cannot determine a difficulty checkpoint: no checkpoint-* under {step2_loss_dir}."
    )


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    """"""
    output_file = workdir / "step3_difficult_samples.npy"

    if check_output_exists(output_file, "Step 3 (Difficulty filtering)"):
        return output_file, True

    log("=" * 60)
    log("STEP 3: Difficulty Filtering (High-Loss Samples)")
    log("=" * 60)

    loss_dir = workdir / "step2_loss_100tokens"
    difficulty_ckpt = _discover_difficulty_ckpt_name(loss_dir)

    cmd = [
        sys.executable,
        str(BASE_DIR / "train.py"),
        "--config_file", str(BASE_DIR / "configs" / "default-s2l-args.yml"),
        "--full_data_path", config["prompt_source"],
        "--ref_model_path", str(loss_dir),
        "--result_dir_name", str(workdir / "step3_difficulty"),
        "--sampling_method", "high_loss_clusters",
        "--n_components", str(DEFAULTS["difficulty_n_components"]),
        "--all_samples",
        "--loss_file_name", f"loss-100token-r-losses.pt",
        "--use_ckpts", difficulty_ckpt,
    ]

    if args.difficulty_filter_without_source:
        cmd.append("--ignore_sources")

    run_cmd(cmd)

    source_npy = workdir / "step3_difficulty" / "data" / "labeled_idx.npy"
    if source_npy.exists():
        shutil.copy(source_npy, output_file)

    return output_file, False
