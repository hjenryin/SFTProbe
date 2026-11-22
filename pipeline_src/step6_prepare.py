""""""

from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np

from .loss_plan import SelectionSpec, build_loss_plan
from .step4_3_perturb_statedict import STEP4_3_SORT_EVAL_MAX_TOKENS
from .step6_preprocess import SORT_SCORES
from .utils import DEFAULTS, get_num_token_for_difficulty


def prepare_selection_spec(config: Dict, workdir: Path, args) -> SelectionSpec:
    """"""
    _, spec = build_loss_plan(
        workdir,
        args.num_samples,
        difficulty_tokens=get_num_token_for_difficulty(args),
        clustering_tokens=int(DEFAULTS["clustering_tokens"]),
    )

    if not args.standardize_kmeans:
        spec = replace(spec, standardize_kmeans=False)

    sort_method = args.cluster_sort if args.cluster_sort else "high_loss"

    if sort_method == "random":
        return replace(spec, sampling_method="random")

    if sort_method not in SORT_SCORES:
        raise ValueError(f"Unknown sort method {sort_method}")

    if sort_method == "checkpoint_diff":
        sort_ckpts = getattr(args, "sort_ckpts", None)
        if not sort_ckpts or len(sort_ckpts.split(",")) != 2:
            raise ValueError(
                "--sort-ckpts must be provided with exactly 2 paths when "
                "using --cluster-sort checkpoint_diff"
            )

    subset_indices = np.load(workdir / "step3_difficult_samples.npy")

    step1_path = workdir / "step1_generation_100tokens.jsonl"
    with open(step1_path, "r") as f:
        full_size = sum(1 for _ in f)

    scorer_class = SORT_SCORES[sort_method]
    statedict_sort = sort_method == "learn-diff-statedict-base"
    loss_glob = (
        f"*{STEP4_3_SORT_EVAL_MAX_TOKENS}token*.pt"
        if statedict_sort
        else "*1000token*.pt"
    )
    scorer = scorer_class(
        step5_dir=workdir / "step5_loss_1k_tokens",
        step2_dir=workdir / "step2_loss_100tokens",
        loss_pattern=loss_glob,
        subset_indices=subset_indices,
        full_size=full_size,
        difficulty_loss_pattern=spec.budget_loss_file,
        clustering_losses=spec.clustering_losses,
    )
    sort_path = scorer.run(workdir / "step6_sort_score.pt")
    return replace(
        spec,
        sort_score_path=str(sort_path),
        sort_order="descending",
    )
