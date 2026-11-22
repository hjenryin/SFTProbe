""""""

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .loss_plan import SelectionSpec
from .utils import BASE_DIR, check_output_exists, log, run_cmd


def run(
    spec: SelectionSpec,
    step2_loss_dir: Path,
    step5_loss_dir: Path,
    prompt_source: str,
    subset_indices_path: Path,
    workdir: Path,
    *,
    resample: bool = False,
    max_repetition: int = 1,
) -> Tuple[Path, bool]:
    """"""
    if resample:
        output_file = workdir / "step6_diversity" / "data" / "cluster_orderings.json"
    else:
        output_file = workdir / "step6_labeled_idx.npy"

    if not resample and check_output_exists(output_file, "Step 6 (Diversity selection)"):
        return output_file, True

    log("=" * 60)
    log(f"STEP 6: {'Cluster Ordering (Resample)' if resample else 'Diversity Selection'}")
    if max_repetition > 1:
        log(f"  max_repetition={max_repetition}")
    log("=" * 60)

    combined_dir = workdir / "step6_combined_losses"
    _create_combined_loss_dir(combined_dir, step2_loss_dir, step5_loss_dir, spec)

    if max_repetition <= 1 or resample:
        _run_train_py(spec, combined_dir, prompt_source, subset_indices_path,
                      workdir, resample=resample)

        if not resample:
            source_npy = workdir / "step6_diversity" / "data" / "labeled_idx.npy"
            if source_npy.exists():
                shutil.copy(source_npy, output_file)

        return output_file, False

    all_selected = _run_multi_round_selection(
        spec=spec,
        combined_dir=combined_dir,
        prompt_source=prompt_source,
        subset_indices_path=subset_indices_path,
        workdir=workdir,
        max_repetition=max_repetition,
    )

    np.save(output_file, np.array(all_selected, dtype=np.int64))
    log(f"Step 6: saved {len(all_selected)} indices (with repetitions) to {output_file}")

    return output_file, False


def _run_multi_round_selection(
    *,
    spec: SelectionSpec,
    combined_dir: Path,
    prompt_source: str,
    subset_indices_path: Path,
    workdir: Path,
    max_repetition: int,
) -> List[int]:
    """"""
    data_dir = workdir / "step6_diversity" / "data"
    orderings_file = data_dir / "cluster_orderings.json"
    budgets_file = data_dir / "source_budgets.json"
    weights_file = data_dir / "source_weights.json"

    if not orderings_file.exists():
        log("Step 6 multi-rep: computing cluster orderings via train.py --resample")
        step6_div = workdir / "step6_diversity"
        if step6_div.exists():
            shutil.rmtree(step6_div)
        _run_train_py(spec, combined_dir, prompt_source, subset_indices_path,
                      workdir, resample=True)
    else:
        log(f"Step 6 multi-rep: cluster orderings already exist at {orderings_file}")

    with open(orderings_file) as f:
        cluster_orderings = json.load(f)
    with open(budgets_file) as f:
        source_budgets = json.load(f)

    if weights_file.exists():
        with open(weights_file) as f:
            source_weights = json.load(f)
    else:
        total_original = sum(source_budgets.values())
        if total_original > 0:
            source_weights = {src: b / total_original for src, b in source_budgets.items()}
        else:
            source_weights = {src: 1.0 / len(source_budgets) for src in source_budgets}

    total_budget = spec.budget
    log(f"Step 6 multi-rep: total_budget={total_budget}, max_repetition={max_repetition}")
    log(f"  Sources: {list(cluster_orderings.keys())}")
    log(f"  Source budgets: {source_budgets}")

    source_unique_counts = {}
    for src, clusters in cluster_orderings.items():
        unique_indices = set()
        for ordering in clusters.values():
            unique_indices.update(int(idx) for idx in ordering)
        source_unique_counts[src] = len(unique_indices)

    from budget_utils import budget_allocation

    all_selected: List[int] = []
    usage_counts: Dict[int, int] = {}

    for rep in range(1, max_repetition + 1):
        remaining_budget = total_budget - len(all_selected)
        if remaining_budget <= 0:
            break

        log(f"\n  --- Repetition round {rep}/{max_repetition} (remaining budget: {remaining_budget}) ---")

        source_sizes = {}
        for src, clusters in cluster_orderings.items():
            size = 0
            for ordering in clusters.values():
                for idx in ordering:
                    if usage_counts.get(int(idx), 0) < rep:
                        size += 1
            source_sizes[src] = size

        source_alloc = budget_allocation(source_weights, source_sizes, remaining_budget)
        log(f"  Source allocation this round: {source_alloc}")

        round_selected: List[int] = []
        for src, clusters in cluster_orderings.items():
            src_budget = source_alloc.get(src, 0)
            if src_budget <= 0:
                continue

            available_in_src = source_sizes[src]

            if src_budget >= available_in_src:
                log(f"  Source '{src}': budget={src_budget} >= available={available_in_src}, selecting all")
                for cid, ordering in clusters.items():
                    for idx in ordering:
                        idx = int(idx)
                        if usage_counts.get(idx, 0) < rep:
                            round_selected.append(idx)
                            usage_counts[idx] = usage_counts.get(idx, 0) + 1
            else:
                n_clusters = len(clusters)
                if n_clusters == 0:
                    continue

                cluster_weights = {cid: 1.0 / n_clusters for cid in clusters}
                cluster_sizes = {}
                for cid, ordering in clusters.items():
                    cluster_sizes[cid] = sum(
                        1 for idx in ordering if usage_counts.get(int(idx), 0) < rep
                    )

                cluster_alloc = budget_allocation(cluster_weights, cluster_sizes, src_budget)
                log(f"  Source '{src}': budget={src_budget} < available={available_in_src}, cluster selection")

                for cid, ordering in clusters.items():
                    alloc = cluster_alloc.get(cid, 0)
                    picked = 0
                    for idx in ordering:
                        if picked >= alloc:
                            break
                        idx = int(idx)
                        if usage_counts.get(idx, 0) < rep:
                            round_selected.append(idx)
                            usage_counts[idx] = usage_counts.get(idx, 0) + 1
                            picked += 1

        all_selected.extend(round_selected)
        log(f"  Round {rep} selected {len(round_selected)} indices (total so far: {len(all_selected)})")

    if len(all_selected) < total_budget:
        log(
            f"WARNING: After {max_repetition} repetition rounds, only {len(all_selected)}/{total_budget} "
            f"indices selected. All sources exhausted their repetition capacity."
        )

    rep_counts: Dict[int, int] = {}
    for count in usage_counts.values():
        rep_counts[count] = rep_counts.get(count, 0) + 1
    log(f"Step 6 multi-rep complete: repetition distribution: {dict(sorted(rep_counts.items()))}")

    return all_selected


def _run_train_py(
    spec: SelectionSpec,
    combined_dir: Path,
    prompt_source: str,
    subset_indices_path: Path,
    workdir: Path,
    *,
    resample: bool = False,
) -> None:
    """"""
    cmd = [
        sys.executable,
        str(BASE_DIR / "train.py"),
        "--config_file", str(BASE_DIR / "configs" / "default-s2l-args.yml"),
        "--full_data_path", prompt_source,
        "--ref_model_path", str(combined_dir),
        "--result_dir_name", str(workdir / "step6_diversity"),
        "--init_label_num", str(spec.budget),
        "--use_ckpts", ",".join(spec.clustering_losses),
        "--difficulty_use_ckpts", ",".join(spec.budget_losses),
        "--loss_file_name", spec.clustering_loss_file,
        "--difficulty_loss_file_name", spec.budget_loss_file,
        "--softmax_t", str(spec.softmax_t),
        "--softmax_score", "loss",
        "--n_sample_per_cluster", str(spec.n_sample_per_cluster),
        "--subset_indices_path", str(subset_indices_path),
    ]

    if spec.sort_score_path is not None:
        cmd += ["--sort_score_path", spec.sort_score_path]
        sm = "ascending_loss" if spec.sort_order == "ascending" else "descending_loss"
        cmd += ["--sampling_method", sm]
    else:
        cmd += ["--sampling_method", spec.sampling_method]

    if resample:
        cmd.append("--resample")

    if not spec.standardize_kmeans:
        cmd.append("--no-standardize-kmeans")

    run_cmd(cmd)


def _create_combined_loss_dir(
    combined_dir: Path,
    step2_loss_dir: Path,
    step5_loss_dir: Path,
    spec: SelectionSpec,
) -> None:
    """"""
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True)

    for name in spec.budget_losses:
        src = step2_loss_dir / name
        dst = combined_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.absolute())
            log(f"  Linked difficulty: {name} -> {src}")

    for name in spec.clustering_losses:
        src = step5_loss_dir / name
        dst = combined_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.absolute())
            log(f"  Linked diversity: {name} -> {src}")
