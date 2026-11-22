""""""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .loss_plan import LossJob
from .utils import BASE_DIR, log, run_cmd


def _overlay_metadata_for_compare(value) -> Optional[Dict]:
    if not value:
        return None
    if not isinstance(value, dict):
        return {"value": value}
    fingerprint = value.get("fingerprint")
    return {
        "base_model": value.get("base_model"),
        "perturbation_surface": value.get("perturbation_surface"),
        "perturbation_surface_version": value.get("perturbation_surface_version"),
        "seed": value.get("seed"),
        "scale": value.get("scale"),
        "fingerprint_digest": (
            fingerprint.get("digest") if isinstance(fingerprint, dict) else value.get("fingerprint_digest")
        ),
        "fingerprint_algorithm": (
            fingerprint.get("algorithm") if isinstance(fingerprint, dict) else value.get("fingerprint_algorithm")
        ),
        "fingerprint_samples_per_tensor": (
            fingerprint.get("samples_per_tensor")
            if isinstance(fingerprint, dict)
            else value.get("fingerprint_samples_per_tensor")
        ),
    }


def _job_to_metadata(job: LossJob) -> Dict:
    return {
        "name": job.name,
        "checkpoint": job.checkpoint,
        "max_tokens": job.max_tokens,
        "is_adapter": job.is_adapter,
        "loss_file": job.loss_file,
        "perturbation_overlay": job.perturbation_overlay,
    }


def _load_existing_job_metadata(metadata_file: Path) -> Dict[str, Dict]:
    if not metadata_file.exists():
        return {}
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    out = {}
    for row in jobs:
        if isinstance(row, dict) and row.get("name"):
            out[str(row["name"])] = row
    return out


def _job_metadata_matches(job: LossJob, row: Optional[Dict]) -> bool:
    if row is None:
        return False
    try:
        if int(row.get("max_tokens")) != int(job.max_tokens):
            return False
    except (TypeError, ValueError):
        return False
    if bool(row.get("is_adapter")) != bool(job.is_adapter):
        return False
    if str(row.get("loss_file")) != str(job.loss_file):
        return False
    if str(row.get("checkpoint")) != str(job.checkpoint):
        return False
    return _overlay_metadata_for_compare(row.get("perturbation_overlay")) == _overlay_metadata_for_compare(
        job.perturbation_overlay
    )


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        return f"{int(mins)}m{secs:.0f}s"
    hrs, mins = divmod(mins, 60)
    return f"{int(hrs)}h{int(mins)}m{secs:.0f}s"


def run(
    jobs: List[LossJob],
    output_dir: Path,
    dataset_path: Path,
    subset_indices_path: Optional[Path],
    full_dataset_size: int,
    *,
    reasoning_marker: Optional[str] = None,
) -> Tuple[Path, bool]:
    """"""
    log("=" * 60)
    log("STEP 5: Compute Diversity Losses")
    log("=" * 60)
    step_start = time.time()
    metadata_file = output_dir.parent / "step5_metadata.json"
    existing_jobs = _load_existing_job_metadata(metadata_file)

    has_prior_metadata = bool(existing_jobs)
    log(f"Plan: {len(jobs)} job(s), output_dir={output_dir}")
    pending_jobs: List[LossJob] = []
    for job in jobs:
        loss_file_path = output_dir / job.name / job.loss_file
        if has_prior_metadata:
            metadata_matches = _job_metadata_matches(job, existing_jobs.get(job.name))
        else:
            metadata_matches = True
        status = "CACHED" if loss_file_path.exists() and metadata_matches else "PENDING"
        if loss_file_path.exists() and not metadata_matches:
            log(f"  [STALE] {job.name}: loss exists but Step 5 metadata does not match; recomputing")
            if loss_file_path.is_symlink() or loss_file_path.is_file():
                loss_file_path.unlink()
        log(f"  [{status}] {job.name} (adapter={job.is_adapter}, "
            f"tokens={job.max_tokens}, ckpt={job.checkpoint})")
        if status == "PENDING":
            pending_jobs.append(job)

    if not pending_jobs:
        log(f"[SKIP] Step 5: All {len(jobs)} loss files already exist")
        return output_dir, True

    log(f"Step 5 will compute {len(pending_jobs)} pending job(s); "
        f"{len(jobs) - len(pending_jobs)} already cached")

    output_dir.mkdir(parents=True, exist_ok=True)

    loss_input_path = dataset_path
    subset_indices = None

    if subset_indices_path and subset_indices_path.exists():
        subset_indices = np.load(subset_indices_path)
        log(f"Using difficulty subset: {len(subset_indices)} / {full_dataset_size} samples")

        subset_input_path = dataset_path.parent / "difficulty_selected_continued.jsonl"
        if not subset_input_path.exists():
            src_size = Path(dataset_path).stat().st_size if Path(dataset_path).exists() else 0
            log(f"Creating subset dataset: {subset_input_path}")
            log(f"  Source: {dataset_path} ({_fmt_bytes(src_size)}, "
                f"{full_dataset_size} lines)")
            log(f"  Target: {len(subset_indices)} selected lines")
            subset_list = sorted(int(i) for i in subset_indices.tolist())
            subset_len = len(subset_list)

            write_start = time.time()
            target = subset_len

            with open(dataset_path, "r") as fin:
                lines = fin.readlines()

            scanned = len(lines)
            if full_dataset_size and scanned != full_dataset_size:
                log(f"WARNING: Expected {full_dataset_size} lines, read {scanned} lines from {dataset_path}")
            else:
                log(f"Read {scanned} lines from {dataset_path}")
            if subset_list and (subset_list[0] < 0 or subset_list[-1] >= scanned):
                raise ValueError(
                    f"Subset indices out of range for {dataset_path}: "
                    f"min={subset_list[0]}, max={subset_list[-1]}, scanned={scanned}"
                )

            selected_lines = [lines[i] for i in subset_list]
            written = len(selected_lines)

            with open(subset_input_path, "w") as fout:
                fout.writelines(selected_lines)

            out_size = subset_input_path.stat().st_size
            total = time.time() - write_start
            scan_rate = scanned / total if total > 0 else 0.0
            write_rate = written / total if total > 0 else 0.0
            log(f"  Subset built: scanned {scanned}, wrote {written}/{target} "
                f"({_fmt_bytes(out_size)}), took {_fmt_duration(total)} "
                f"(scan={scan_rate:.0f} lines/s, write={write_rate:.0f} selected lines/s)")
        else:
            existing_size = subset_input_path.stat().st_size
            log(f"Using existing subset dataset: {subset_input_path} "
                f"({_fmt_bytes(existing_size)})")

        loss_input_path = subset_input_path

    pending_seen = 0
    total_pending = len(pending_jobs)
    for job in jobs:
        loss_file_path = output_dir / job.name / job.loss_file

        if loss_file_path.exists():
            log(f"[SKIP] {job.name}: Loss file exists")
            continue

        pending_seen += 1
        log(f"[{pending_seen}/{total_pending}] Computing loss for {job.name} "
            f"({job.max_tokens} tokens, adapter={job.is_adapter}, "
            f"model={job.checkpoint})...")
        job_start = time.time()

        ckpt_out_dir = output_dir / job.name
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_loss_cmd(
            model_path=job.checkpoint,
            dataset_path=str(loss_input_path),
            max_tokens=job.max_tokens,
            output_dir=ckpt_out_dir,
            loss_file=job.loss_file,
            reasoning_marker=reasoning_marker,
            perturbation_overlay=job.perturbation_overlay,
        )
        run_cmd(cmd)

        if subset_indices is not None:
            _inflate_subset(loss_file_path, subset_indices, full_dataset_size)

        log(f"[{pending_seen}/{total_pending}] Done {job.name} in "
            f"{_fmt_duration(time.time() - job_start)}")

    metadata = {
        "loss_method": "hf",
        "jobs": [_job_to_metadata(j) for j in jobs],
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log(f"STEP 5 complete in {_fmt_duration(time.time() - step_start)} "
        f"({total_pending} computed, {len(jobs) - total_pending} cached)")
    return output_dir, False


def _build_loss_cmd(
    *,
    model_path: str,
    dataset_path: str,
    max_tokens: int,
    output_dir: Path,
    loss_file: str,
    reasoning_marker: Optional[str],
    perturbation_overlay: Optional[Dict],
) -> list:
    """"""
    loss_path = output_dir / loss_file
    cmd = [
        sys.executable,
        str(BASE_DIR / "m1" / "src" / "process_loss" / "get_mean_loss_optimized.py"),
        "--model_path", model_path,
        "--dataset_path", dataset_path,
        "--max_tokens", str(max_tokens),
        "--output_name", str(loss_path),
    ]
    if perturbation_overlay:
        descriptor_path = perturbation_overlay.get("descriptor_path")
        if not descriptor_path:
            raise ValueError("perturbation_overlay jobs require descriptor_path")
        cmd.extend(["--perturbation_overlay_descriptor", str(descriptor_path)])
    if reasoning_marker is not None:
        cmd.extend(["--reasoning_marker", reasoning_marker])
    return cmd


def _inflate_subset(
    loss_file_path: Path,
    subset_indices: np.ndarray,
    full_size: int,
) -> None:
    """"""
    if not loss_file_path.exists():
        return

    losses = torch.load(loss_file_path, map_location="cpu")
    if isinstance(losses, torch.Tensor) and losses.dim() > 1:
        losses = losses.squeeze()

    if isinstance(losses, torch.Tensor) and losses.numel() == len(subset_indices) and full_size > len(subset_indices):
        log(f"Inflating {loss_file_path.name}: {losses.numel()} → {full_size}")
        inflated = torch.full((full_size,), -1.0, dtype=losses.dtype)
        inflated[subset_indices] = losses
        torch.save(inflated, loss_file_path)
