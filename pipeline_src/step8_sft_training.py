""""""

import re
import shutil
import socket
import sys
from pathlib import Path
from typing import Dict, Tuple

from .utils import (
    BASE_DIR,
    SFT_CACHE_BASE,
    HOSTNAME,
    check_output_exists,
    log,
    run_cmd,
    STEP7_INDEXED_TRACES,
    STEP7_SELECTED_TRACES,
)


def prepare_sft_symlink(workdir: Path, run_name: str) -> Path:
    """"""
    link_path = workdir / "step8_sft_model"
    target_dir = SFT_CACHE_BASE / "curation-sft" / run_name

    if link_path.is_symlink():
        current_target = link_path.resolve()

        if current_target == target_dir.resolve():
            log(f"[SKIP] SFT link exists and points to correct host: {current_target}")
            return link_path

        match = re.match(r".*local-cache-(.+?)/.*", str(current_target))
        if match:
            host_in_target = match.group(1)
            if host_in_target not in HOSTNAME:
                raise RuntimeError(f"SFT link points to another host's cache: {current_target}. Current host: {HOSTNAME}")
            else:
                return link_path

        raise RuntimeError(f"SFT link points to unexpected location: {current_target}. Expected {target_dir}")

    elif link_path.exists():
        raise RuntimeError(f"Path {link_path} exists but is not a symlink!")

    if not target_dir.exists():
        log(f"Creating SFT cache directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        log(f"Reusing existing SFT cache directory: {target_dir}")

    link_path.symlink_to(target_dir)
    log(f"Created symlink: {link_path} -> {target_dir}")

    return link_path


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    """"""
    output_link = prepare_sft_symlink(workdir, args.run_name)
    cache_output = output_link.resolve()
    completion_marker = cache_output / "train_results.json"

    if check_output_exists(completion_marker, "Step 8 (SFT training)"):
        return output_link, True

    selected_file = workdir / STEP7_SELECTED_TRACES
    indexed_file = workdir / STEP7_INDEXED_TRACES
    subset_file = workdir / "step6_labeled_idx.npy"

    max_repetition = getattr(args, "max_repetition", 1)
    is_direct = bool(getattr(args, "direct_file", None) or getattr(args, "direct_index", None))
    expected_selected = (args.resample or max_repetition > 1) and not is_direct

    if expected_selected:
        source_file = selected_file
        use_subset = False
        log(f"Config expects self-contained trace file: {STEP7_SELECTED_TRACES}")
    else:
        source_file = indexed_file
        use_subset = True
        log(f"Config expects indexed trace file: {STEP7_INDEXED_TRACES}")

    if not source_file.exists():
        raise FileNotFoundError(
            f"Expected trace file {source_file.name} (based on config: resample={args.resample}, "
            f"max_repetition={max_repetition}) was not found in {workdir}"
        )

    if use_subset and subset_file.exists():
        import numpy as np
        indices = np.load(subset_file)
        with open(source_file) as f:
            line_count = sum(1 for _ in f)
        if indices.max() >= line_count:
            raise ValueError(
                f"step6_labeled_idx.npy contains index {indices.max()} but "
                f"{STEP7_INDEXED_TRACES} only has {line_count} lines. "
                f"The indices are global (into the full dataset) but step7 appears to "
                f"contain only a subset. Delete {STEP7_INDEXED_TRACES} and re-run."
            )

    log("=" * 60)
    log("STEP 8: SFT Training")
    log("=" * 60)

    cmd = [
        "python",
        str(BASE_DIR / "m1" / "llama_factory_s2l.py"),
        "--dataset_type", config["dataset_type"],
        "--seed", str(args.seed),
        "--output_dir", str(output_link),
        "--source_data_file", str(source_file),
    ]

    if getattr(args, "original_model_name", None):
        cmd.extend(["--model_name_or_path", str(args.original_model_name)])

    if args.step8_cutoff_len is not None:
        cmd.extend(["--cutoff_len", str(int(args.step8_cutoff_len))])

    if use_subset:
        cmd.extend(["--subset_file", str(subset_file)])

    try:
        run_cmd(cmd)
    except Exception as e:
        log(f"SFT Training failed: {e}")
        if output_link.is_symlink() or output_link.exists():
            log(f"Unlinking {output_link} due to failure")
            output_link.unlink()
        raise e

    return output_link, False
