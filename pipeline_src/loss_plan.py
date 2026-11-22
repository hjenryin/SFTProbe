""""""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from .utils import log


@dataclass(frozen=True)
class LossJob:
    """"""
    name: str
    checkpoint: str
    max_tokens: int
    is_adapter: bool
    loss_file: str
    perturbation_overlay: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SelectionSpec:
    """"""
    budget: int
    clustering_losses: List[str]
    budget_losses: List[str]
    clustering_loss_file: str
    budget_loss_file: str
    budget_loss_tokens: int
    softmax_t: float = 1.0
    n_sample_per_cluster: int = 4
    sort_score_path: Optional[str] = None
    sort_order: str = "descending"
    sampling_method: str = "high_loss"
    standardize_kmeans: bool = True


def build_loss_plan(
    workdir: Path,
    budget: int,
    difficulty_tokens: Optional[int] = None,
    clustering_tokens: Optional[int] = None,
    sampling_method: str = "high_loss",
) -> Tuple[List[LossJob], SelectionSpec]:
    """"""
    jobs: List[LossJob] = []
    clustering_names: List[str] = []
    budget_names: List[str] = []

    difficulty_loss_pattern = re.compile(
        r"^.*?(\d+)token.*\.pt$"
    )
    clustering_loss_pattern = re.compile(
        r"^.*?(\d+)token.*\.pt$"
    )

    step2_dir = workdir / "step2_loss_100tokens"
    if not step2_dir.is_dir():
        raise FileNotFoundError(
            f"Step 2 output not found: {step2_dir}. "
            "Cannot determine difficulty checkpoints for budget allocation."
        )

    discovered_budget_file: Optional[str] = None
    discovered_budget_tokens: Optional[int] = None

    for ckpt_dir in sorted(step2_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        matched_candidates: List[Tuple[int, str]] = []
        for pt_file in ckpt_dir.glob("*.pt"):
            m = difficulty_loss_pattern.match(pt_file.name)
            if not m:
                continue
            token_count = int(m.group(1))
            if difficulty_tokens is None or token_count == difficulty_tokens:
                matched_candidates.append((token_count, pt_file.name))

        if not matched_candidates:
            expected_msg = (
                f" for {difficulty_tokens} tokens" if difficulty_tokens is not None else ""
            )
            log(
                f"WARNING: Skipping {ckpt_dir.name} — no matching difficulty loss file{expected_msg}"
            )
            continue

        if len(matched_candidates) > 1:
            raise RuntimeError(
                f"Found multiple candidate difficulty loss files in {ckpt_dir.name} "
                f"matching token {difficulty_tokens}. Files found: {[c[1] for c in matched_candidates]}"
            )

        matched_tokens, matched_loss_file = matched_candidates[0]

        if discovered_budget_file is None:
            discovered_budget_file = matched_loss_file
            discovered_budget_tokens = matched_tokens
            log(f"Using difficulty loss filename: {discovered_budget_file}")
            log(f"Using difficulty loss token length: {discovered_budget_tokens}")
        elif matched_loss_file != discovered_budget_file:
            log(
                f"WARNING: Skipping {ckpt_dir.name} — found {matched_loss_file}, "
                f"but using {discovered_budget_file} for consistency"
            )
            continue
        elif matched_tokens != discovered_budget_tokens:
            log(
                f"WARNING: Skipping {ckpt_dir.name} — found {matched_tokens} tokens, "
                f"but using {discovered_budget_tokens} for consistency"
            )
            continue

        budget_names.append(ckpt_dir.name)
        log(f"Discovered difficulty checkpoint: {ckpt_dir.name}")

    if not budget_names:
        raise RuntimeError(
            f"No difficulty checkpoints found in {step2_dir}. "
            "Expected at least one directory with a loss file."
        )

    if discovered_budget_file is None:
        raise RuntimeError("Internal error: discovered budget loss filename is undefined")
    if discovered_budget_tokens is None:
        raise RuntimeError("Internal error: discovered difficulty token length is undefined")

    step48_dir = workdir / "step4_8_perturbed_lora"
    if step48_dir.is_symlink():
        step48_dir = step48_dir.resolve()

    if not step48_dir.is_dir():
        raise FileNotFoundError(
            f"Step 4.8 output not found: {step48_dir}. "
            "Cannot determine diversity checkpoints for clustering."
        )

    for ckpt_dir in sorted(step48_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        if ckpt_dir.name.startswith("original_") or ckpt_dir.name.endswith("_base"):
            continue
        clustering_names.append(ckpt_dir.name)
        log(f"Discovered diversity checkpoint: {ckpt_dir.name}")

    if not clustering_names:
        raise RuntimeError(
            f"No diversity checkpoints found in {step48_dir}. "
            "Expected directories like dir0_over_2x, dir1_over_3x, etc."
        )

    discovered_clustering_file: Optional[str] = None
    discovered_clustering_tokens: Optional[int] = None

    step5_dir = workdir / "step5_loss_1k_tokens"
    if step5_dir.is_dir():
        for ckpt_name in clustering_names:
            ckpt_step5_dir = step5_dir / ckpt_name
            if not ckpt_step5_dir.is_dir():
                continue

            matched_candidates: List[Tuple[int, str]] = []
            for pt_file in ckpt_step5_dir.glob("*.pt"):
                m = clustering_loss_pattern.match(pt_file.name)
                if not m:
                    continue
                token_count = int(m.group(1))
                if clustering_tokens is None or token_count == clustering_tokens:
                    matched_candidates.append((token_count, pt_file.name))

            if not matched_candidates:
                continue

            if len(matched_candidates) > 1:
                raise RuntimeError(
                    f"Found multiple candidate clustering loss files in {ckpt_name} "
                    f"matching token {clustering_tokens}. Files found: {[c[1] for c in matched_candidates]}"
                )

            matched_tokens, matched_loss_file = matched_candidates[0]

            if discovered_clustering_file is None:
                discovered_clustering_file = matched_loss_file
                discovered_clustering_tokens = matched_tokens
                log(f"Using clustering loss filename: {discovered_clustering_file}")
                log(f"Using clustering loss token length: {discovered_clustering_tokens}")
            elif matched_loss_file != discovered_clustering_file:
                log(
                    f"WARNING: Ignoring {ckpt_name} step5 loss file {matched_loss_file} "
                    f"(using {discovered_clustering_file} for consistency)"
                )
            elif matched_tokens != discovered_clustering_tokens:
                log(
                    f"WARNING: Ignoring {ckpt_name} step5 loss tokens {matched_tokens} "
                    f"(using {discovered_clustering_tokens} for consistency)"
                )

    if discovered_clustering_file is None or discovered_clustering_tokens is None:
        if clustering_tokens is None:
            raise RuntimeError(
                "Could not discover clustering loss token length from step5 outputs. "
                "Provide clustering_tokens when building loss plan."
            )
        discovered_clustering_tokens = int(clustering_tokens)
        discovered_clustering_file = f"loss-{discovered_clustering_tokens}token-r-losses.pt"
        log(f"Using configured clustering loss filename: {discovered_clustering_file}")
        log(f"Using configured clustering loss token length: {discovered_clustering_tokens}")

    for ckpt_name in clustering_names:
        jobs.append(LossJob(
            name=ckpt_name,
            checkpoint=str(step48_dir / ckpt_name),
            max_tokens=discovered_clustering_tokens,
            is_adapter=True,
            loss_file=discovered_clustering_file,
        ))

    log(f"Loss plan: {len(budget_names)} difficulty, {len(clustering_names)} diversity, budget={budget}")

    spec = SelectionSpec(
        budget=budget,
        clustering_losses=clustering_names,
        budget_losses=budget_names,
        clustering_loss_file=discovered_clustering_file,
        budget_loss_file=discovered_budget_file,
        budget_loss_tokens=discovered_budget_tokens,
        sampling_method=sampling_method,
    )
    return jobs, spec
