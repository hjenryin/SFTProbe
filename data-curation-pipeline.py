#!/usr/bin/env python3
""""""

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_src.utils import (
    BASE_DIR,
    WORKDIR_BASE,
    HOSTNAME,
    DEFAULTS,

    StepArgs,
    get_num_token_for_difficulty,
    log,
)
from pipeline_src.configs import TASK_CONFIGS
from pipeline_src.skip_link import SkipAndLink
from pipeline_src.step4_3_perturb_statedict import (
    STEP4_3_SORT_EVAL_MAX_TOKENS,
    STEP4_3_SORT_LOSS_FILE,
)


ARG_REGISTRY = {
    "difficulty_filter_without_source": {
        "default": False, "action": "store_true",
        "flags": ["--difficulty-filter-without-source"],
        "help": "Ignore source field and treat all data as a single source for Step 3 difficulty clustering.",
        "run_name_fmt": "diffnosrc",
    },
    "num_samples": {
        "default": 1000, "type": int,
        "help": "Number of samples for diversity sampling (default: 1000)",
        "run_name_fmt": "n{v}",
    },
    "seed": {
        "default": 42, "type": int,
        "help": "Random seed",
        "run_name_fmt": "s{v}",
    },
    "num_token_for_difficulty": {
        "default": 100, "type": int,
        "help": "Difficulty token window. Step 1 generates this value + 20, "
                "Step 2 evaluates this value (default: 100).",
        "run_name_fmt": "dtok{v}",
    },
    "original_model_name": {
        "default": "Qwen/Qwen2.5-7B-Instruct", "type": str,
        "help": "Original base model used for Step 2 perturbation and Step 4.7 LoRA training.",
        "run_name_fmt": lambda v: f"stu-{v.split('/')[0]}",
    },
    "step2_inflection_ytol": {
        "default": 1.0, "type": float,
        "help": "Step 2 inflection search tolerance on |second_derivative| (default: 1)",
    },
    "step2_inflection_max_iter": {
        "default": 8, "type": int,
        "help": "Maximum Step 2 inflection refinement iterations after initial scales (default: 8)",
    },
    "step2_inflection_curvature_ratio": {
        "default": 0, "type": float,
        "flags": ["--step2-inflection-curvature-ratio", "--step2_inflection_curvature_ratio"],
        "help": "Step 2 inflection search curvature threshold ratio (default: 0.1)",
        "run_name_fmt": "s2curv{v}",
    },
    "step2_inflection_lower_bound": {
        "default": 0.0, "type": float,
        "flags": ["--step2-inflection-lower-bound", "--step2_inflection_lower_bound"],
        "help": "Step 2 inflection search lower perturbation scale bound (default: 0)",
        "run_name_fmt": "s2lo{v}",
    },
    "step2_inflection_upper_bound": {
        "default": 0.01, "type": float,
        "flags": ["--step2-inflection-upper-bound", "--step2_inflection_upper_bound"],
        "help": "Step 2 inflection search upper perturbation scale bound (default: 0.01)",
        "run_name_fmt": "s2hi{v}",
    },
    "step4_3_inflection_ytol": {
        "default": 1.0, "type": float,
        "flags": ["--step4_3-inflection-ytol", "--step4_3_inflection_ytol"],
        "help": "Step 4.3 inflection search tolerance on |second_derivative| (default: 1).",
    },
    "step4_3_inflection_max_iter": {
        "default": 8, "type": int,
        "flags": ["--step4_3-inflection-max-iter", "--step4_3_inflection_max_iter"],
        "help": "Maximum Step 4.3 inflection refinement iterations after initial scales (default: 8).",
    },
    "step4_3_inflection_curvature_ratio": {
        "default": 0.1, "type": float,
        "flags": ["--step4_3-inflection-curvature-ratio", "--step4_3_inflection_curvature_ratio"],
        "help": "Step 4.3 inflection search curvature threshold ratio (default: 0.1).",
        "run_name_fmt": "s43curv{v}",
    },
    "step4_3_inflection_lower_bound": {
        "default": 0.0003, "type": float,
        "flags": ["--step4_3-inflection-lower-bound", "--step4_3_inflection_lower_bound"],
        "help": "Step 4.3 inflection search lower perturbation scale bound (default: 0.0003). "
                "State-dict perturbation has a much smaller useful scale range than "
                "named_parameters_and_buffers, so the lower bound starts above zero to skip the "
                "trivially-small regime.",
        "run_name_fmt": "s43lo{v}",
    },
    "step4_3_inflection_upper_bound": {
        "default": 0.01, "type": float,
        "flags": ["--step4_3-inflection-upper-bound", "--step4_3_inflection_upper_bound"],
        "help": "Step 4.3 inflection search upper perturbation scale bound (default: 0.01).",
        "run_name_fmt": "s43hi{v}",
    },
    "step4_3_inflection_extend_b_if_no_inflection": {
        "default": True,
        "action": "store_false",
        "dest": "step4_3_inflection_extend_b_if_no_inflection",
        "flags": ["--no-step4-3-inflection-extend-b", "--no_step4_3_inflection_extend_b"],
        "help": "Step 4.3: disable extending the upper search bound when the spline fit has "
                "no curvature-threshold crossing (default: extension enabled, same idea as "
                "Step 4.8 LoRA inflection).",
    },
    "step4_3_inflection_b_extend_factor": {
        "default": 1.5, "type": float,
        "flags": ["--step4-3-inflection-b-extend-factor", "--step4_3_inflection_b_extend_factor"],
        "help": "Step 4.3: multiply the upper bound by this factor on each extension (default: 1.5).",
    },
    "step4_3_inflection_max_b_extensions": {
        "default": 3, "type": int,
        "flags": ["--step4-3-inflection-max-b-extensions", "--step4_3_inflection_max_b_extensions"],
        "help": "Step 4.3: max right-bound extensions when no crossing (default: 3).",
    },
    "step2_perturbation_surface": {
        "default": "named_parameters_and_buffers", "type": str,
        "choices": ["state_dict", "named_parameters_and_buffers"],
        "flags": ["--step2-perturbation-surface", "--step2_perturbation_surface"],
        "help": "Step 2 perturbation tensor surface. Default named_parameters_and_buffers perturbs "
                "non-persistent buffers and only writes loss tensors, not a model checkpoint unless "
                "forced; state_dict can save a perturbed checkpoint.",
        "run_name_fmt": "s2surf_{v}",
    },
    "step2_force_save_perturbed_checkpoint": {
        "default": False, "action": "store_true",
        "flags": ["--step2-force-save-perturbed-checkpoint"],
        "help": "Force Step 2 to save a perturbed checkpoint even when the perturbation "
                "surface includes named buffers. Non-persistent buffers still cannot be "
                "serialized by save_pretrained, so this is for downstream reuse paths that "
                "need a loadable checkpoint directory.",
        "run_name_fmt": "s2forcesave",
    },
    "step2_final_eval_from_saved_checkpoint": {
        "default": False, "action": "store_true",
        "flags": ["--step2-final-eval-from-saved-checkpoint"],
        "help": "Step 2: run the final full-dataset mean-loss job by loading the saved "
                "checkpoint directory. Default is base model + internal overlay "
                "descriptor for consistency with the inflection search.",
        "run_name_fmt": "s2evalsaved",
    },
    "partial_x": {
        "default": None, "type": int,
        "flags": ["--partial_x", "--partial-x"],
        "help": (
            "Step 4.5/4.8: number of examples for LoRA training / perturbation sampling. "
            "If unset, defaults to lora_bs * lora_train_steps (one epoch at the configured step count)."
        ),
        "run_name_fmt": "x{v}",
        "run_name_test": lambda v: v is not None,
    },
    "partial_x_sample": {
        "default": "random", "type": str,
        "choices": ["random", "difficulty_source"],
        "flags": ["--partial-x-sample", "--partial_x_sample"],
        "help": (
            "Step 4.5: how to choose the partial_x LoRA training examples from Step 3 "
            "difficult samples. 'random' (default) draws uniformly. 'difficulty_source' "
            "allocates per-source quotas with the same softmax(mean difficulty loss) "
            "budget rule used later in Step 6 (T=1, score=loss, domain→source hierarchy), "
            "then samples within each source — so the LoRA mix matches the eventual SFT "
            "source distribution."
        ),
        "run_name_fmt": "px_{v}",
        "run_name_test": lambda v: v != "random",
    },
    "partial_y": {
        "default": 32768, "type": int,
        "help": "Step 4.5: additional continuation tokens for sampled examples (default: 32768).",
        "run_name_fmt": "y{v}",
    },
    "step7_max_tokens": {
        "default": 16384, "type": int,
        "help": (
            "Step 7: max new continuation tokens to generate after Step 4 "
            "(default: 16384; m192k/m38k/m38k-dup4/m38k-dup8/m38k-dup16/m192k-dup16 default to 6400 when this flag is omitted)."
        ),
        "run_name_fmt": "s7tok{v}",
    },
    "step8_cutoff_len": {
        "default": None, "type": int,
        "flags": ["--step8_cutoff_len", "--step8-cutoff-len"],
        "help": (
            "Step 8: LLaMA-Factory cutoff_len override for SFT training. "
            "If unset, use the task YAML default (currently 8192 for m23k/llama, "
            "32768 for math/openthoughts-math)."
        ),
        "run_name_fmt": "s8cut{v}",
        "run_name_test": lambda v: v is not None,
    },
    "lora_train_steps": {
        "default": 200, "type": int,
        "help": "Step 4.7: LoRA max training steps (default: 200).",
        "run_name_fmt": "lsteps{v}",
    },
    "lora_train_rank": {
        "default": 4, "type": int,
        "help": "Step 4.7: override LoRA rank (default: use yaml, typically 4).",
        "run_name_fmt": "lrank{v}",
    },
    "lora_bs": {
        "default": 4, "type": int,
        "help": "Step 4.7: target global batch size for LoRA training. "
                "Sets per_device_train_batch_size dynamically based on visible GPU count and "
                "forces gradient_accumulation_steps=1.",
        "run_name_fmt": "lbs{v}",
    },
    "lora_perturb_ix": {
        "default": "infl,half-infl", "type": str,
        "help": "Step 4.8: comma-separated over-loss multipliers for LoRA perturbation (default: 'infl,half-infl').",
        "run_name_fmt": "lix_{v}",
        "run_name_test": lambda v: v != "half-infl,infl",
    },
    "lora_perturb_mode": {
        "default": "inflection", "type": str,
        "choices": ["legacy", "inflection"],
        "flags": ["--lora-perturb-mode"],
        "dest": "lora_perturb_mode",
        "help": (
            "Step 4.8 implementation to use. "
            "'legacy': shells out to m1/src/perturb/perturb-model-lora.py. "
            "'inflection': in-process InflectionPointFinder with direction-level threaded parallelism "
            "across available GPUs. Checkpoint layout is identical between modes, so resumes across "
            "modes are blocked via skip-link (the mode is part of the Step 4.8 resume key)."
        ),
    },
    "lora_perturb_inflection_ytol": {
        "default": 0.2, "type": float,
        "help": "Step 4.8 (inflection mode): relative loss tolerance for InflectionPointFinder (default: 0.2).",
    },
    "lora_perturb_inflection_max_iter": {
        "default": 10, "type": int,
        "help": "Step 4.8 (inflection mode): max refinement iterations per direction (default: 10).",
    },
    "lora_perturb_inflection_curvature_ratio": {
        "default": -0.05, "type": float,
        "help": (
            "Step 4.8 (inflection mode): curvature threshold ratio used by InflectionPointFinder "
            "(default: 0.1)."
        ),
    },
    "lora_perturb_post_min_decrease_tol": {
        "default": 1e-4, "type": float,
        "help": "Step 4.8 (inflection mode): absolute loss floor for the post-minimum "
                "monotone check. Used together with --lora_perturb_post_min_decrease_frac: a "
                "point past the empirical minimum is only treated as an overshoot-into-random "
                "violation when the drop clears both the relative-fraction check AND exceeds "
                "this absolute amount (guards floating-point ties at the running max; "
                "default: 1e-4).",
    },
    "lora_perturb_post_min_decrease_frac": {
        "default": 0.9, "type": float,
        "help": "Step 4.8 (inflection mode): relative threshold in (0, 1) for the "
                "post-minimum monotone check. A loss point past the empirical minimum is "
                "flagged as 'overshoot into the random regime' (and the search right bound "
                "is shrunk) only when f(x) < frac * max_{m<=j<i} f(scales[j]), i.e. it drops "
                "below this fraction of the running post-min max. Smaller values are more "
                "aggressive about truncating history; values closer to 1 keep noise-level "
                "dips on the monotone-up right wall of the U (default: 0.9, i.e. require at "
                "least a 10%% drop off the running post-min max).",
    },
    "lora_perturb_eval_batch_size": {
        "default": 16, "type": int,
        "help": "Step 4.8 (inflection mode): max_eval_batch_size passed to LLMEvaluator (default: 16).",
    },
    "lora_perturb_gpu_threshold": {
        "default": 10.0, "type": float,
        "help": "Step 4.8 (inflection mode): gpu_utils.get_available_gpus memory-utilization "
                "threshold in percent (default: 10.0).",
    },
    "lora_perturb_seed_count": {
        "default": 5, "type": int,
        "help": "Step 4.8 (inflection mode): number of equally-spaced seed scales per "
                "direction (default: 5). Points are linearly spaced in [1, 25*3**dir_idx]; "
                "the prior geometric ladder under-sampled the large-scale regime where the "
                "inflection actually lives.",
    },
    "lora_perturb_b_extend_factor": {
        "default": 1.5, "type": float,
        "help": "Step 4.8 (inflection mode): multiplicative factor applied to the right "
                "bound on each extension (default: 1.5, must be > 1.0).",
    },
    "lora_perturb_max_b_extensions": {
        "default": 3, "type": int,
        "help": "Step 4.8 (inflection mode): hard cap on the number of right-bound "
                "extensions per direction. With default factor 1.5 and cap 3, the bound "
                "can grow to at most ~3.4x its initial value (default: 3).",
    },
    "step9_eval_temperature": {
        "default": 0.0,
        "type": float,
        "flags": ["--step9_eval_temperature"],
        "help": (
            "For medical tasks (eval_type=m23k, e.g. m23k/m192k/m38k/m38k-dup4/m38k-dup8/m38k-dup16/m192k-dup16): sampling temperature for "
            "step 9 Huatuo eval (SGLang). "
            "Default 0."
        ),
        "run_name_fmt": "s9t{v}",
    },
    "max_repetition": {
        "default": 1, "type": int,
        "flags": ["--max-repetition", "--max_repetition"],
        "help": (
            "Maximum times a prompt can appear in the final dataset with different "
            "generated responses. Only activates when a source/cluster has fewer "
            "unique prompts than its budget allocation. When > 1, sources exhaust "
            "all repetition rounds before surplus budget is redistributed. "
            "Requires temperature > 0 for generation diversity. Default 1 (no repetition)."
        ),
        "run_name_fmt": "rep{v}",
    },
    "cluster_sort": {
        "default": "learn-diff-base", "type": str,
        "choices": ["high_loss", "avg_max_scale", "avg_max_scale_minus_base",
                    "learn-diff-base", "learn-diff-statedict-base",
                    "diff_perturb_1k", "random", "checkpoint_diff"],
        "flags": ["--cluster-sort"],
        "help": "Intra-cluster sorting strategy for Step 6 sample selection. "
                "All non-random strategies produce a per-sample score and sort descending "
                "(highest score selected first). "
                ""
                "'high_loss': use the last difficulty checkpoint's loss from "
                "step2_loss_100tokens/ (sorted alphabetically; typically checkpoint-inflection). "
                "Score = loss(last_difficulty_ckpt). No extra Step 5 jobs required. "
                ""
                "'avg_max_scale': group Step 4.8 LoRA perturbation checkpoints in "
                "step5_loss_1k_tokens/ by direction prefix (dir0, dir1, ...), pick the "
                "highest-scale checkpoint per direction (e.g. dir0_over_3x over dir0_over_2x), "
                "load each checkpoint's 1000-token loss, and average across directions. "
                "Score = mean_dir(loss(max_scale_ckpt)). No extra Step 5 jobs required. "
                ""
                "'avg_max_scale_minus_base': same as avg_max_scale but subtracts the base "
                "model's 1000-token loss (perturbation lift). "
                "Score = mean_dir(loss(max_scale_ckpt)) - loss(base_unperturbed). "
                "Adds an extra Step 5 job evaluating --original_model_name. "
                ""
                "'learn-diff-base' (default): learnability lift — 1000-token loss of the "
                "Step 2 difficulty-inflection checkpoint minus 1000-token loss of the base "
                "pretrained model. Score = loss(difficulty_inflection) - loss(base_unperturbed). "
                "Adds two extra Step 5 jobs: one for "
                "workdir/step2_loss_100tokens/checkpoint-inflection (or the base model with "
                "a virtual perturbation overlay when --step2-perturbation-surface is not "
                "state_dict) and one for --original_model_name. "
                "Requires --step2-perturbation-surface state_dict or "
                "--step2-force-save-perturbed-checkpoint or a perturbation_overlay.json "
                "descriptor produced by Step 2. "
                ""
                "'diff_perturb_1k': 1000-token loss of the Step 2 difficulty-inflection "
                "checkpoint without subtracting the base model. "
                "Score = loss(difficulty_inflection). "
                "Adds one extra Step 5 job for the difficulty-inflection checkpoint; same "
                "state_dict / overlay requirements as learn-diff-base. "
                ""
                "'learn-diff-statedict-base': like learn-diff-base, but the perturbed "
                "checkpoint is produced by a separate Step 4.3 that runs a state-dict-pinned "
                "inflection search on the step-3-filtered Step 4 continuations using its own "
                "step4_3_inflection_* knobs (independent of step2_inflection_* and of "
                "--step2-perturbation-surface). Inflection search and the two sort-only Step 5 "
                "jobs use 100 response tokens per sample; Step 5 LoRA diversity jobs stay at "
                "1000 tokens. "
                "Score = loss(step4_3_statedict_inflection) - loss(base_unperturbed). "
                "Adds Step 4.3 plus two extra Step 5 jobs "
                "(statedict_difficulty_inflection, base_unperturbed). "
                ""
                "'random': randomly select within each cluster (no score computation). "
                ""
                "'checkpoint_diff': loss difference of two user-provided checkpoints. "
                "Score = loss(sort_ckpt_0) - loss(sort_ckpt_1). "
                "Requires --sort-ckpts with exactly two comma-separated checkpoint paths.",
        "run_name_fmt": "srt_{v}",
    },
}


def _pipeline_timing_gpu_stats_suffix() -> str:
    """"""
    cvd_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd_raw is not None and str(cvd_raw).strip() != "":
        n_cuda_visible = len([x for x in str(cvd_raw).split(",") if x.strip()])
        cvd_display = str(cvd_raw).replace("\n", " ")
    else:
        n_cuda_visible = None
        cvd_display = "unset"

    cuda_vis_str = "na" if n_cuda_visible is None else str(n_cuda_visible)
    try:
        from gpu_utils import get_num_gpus, get_physical_num_gpus

        n_gpu_utils = get_num_gpus()
        n_physical = get_physical_num_gpus()
    except Exception as exc:
        return (
            f", CUDA_VISIBLE_DEVICES={cvd_display!r}, n_gpus_cuda_visible={cuda_vis_str}, "
            f"n_gpus_gpu_utils=?, n_gpus_physical=? (gpu_utils_error={exc!r})"
        )

    return (
        f", CUDA_VISIBLE_DEVICES={cvd_display!r}, n_gpus_cuda_visible={cuda_vis_str}, "
        f"n_gpus_gpu_utils={n_gpu_utils}, n_gpus_physical={n_physical}"
    )


def run_timed_step(step_name: str, func, *args):
    """"""
    start_time = datetime.now()
    log(f"Starting {step_name} at {start_time}")

    workdir = args[1]
    timing_file = workdir / "pipeline_timeing.txt"

    try:
        result = func(*args)
        skipped = False
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], bool):
            result, skipped = result
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        hostname = socket.gethostname()
        gpu_sfx = _pipeline_timing_gpu_stats_suffix()
        with open(timing_file, "a") as f:
            f.write(
                f"{step_name}: Host={hostname}, Start={start_time}, End={end_time}, "
                f"Duration={duration}{gpu_sfx} (FAILED)\n"
            )
        raise e

    if skipped:
        return result

    end_time = datetime.now()
    duration = end_time - start_time

    hostname = socket.gethostname()
    gpu_sfx = _pipeline_timing_gpu_stats_suffix()
    with open(timing_file, "a") as f:
        f.write(
            f"{step_name}: Host={hostname}, Start={start_time}, End={end_time}, "
            f"Duration={duration}{gpu_sfx}\n"
        )

    log(f"Finished {step_name} at {end_time}. Duration: {duration}")
    return result


def _parse_ix_values(ix_spec: str) -> List[str]:
    """"""
    values: List[str] = []
    for tok in str(ix_spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ["infl", "half-infl"]:
            values.append(tok)
        else:
            v = float(tok)
            if v < 1.0:
                raise ValueError(f"Invalid ix value (<1.0): {v}")
            values.append(f"{v:g}")

    if "half-infl" in values and "infl" not in values:
        raise ValueError("Cannot specify 'half-infl' without 'infl'.")

    if not values:
        raise ValueError("--lora_perturb_ix must contain at least one value")
    return sorted(set(values))

def _extract_host_from_cache_path(path_str: str) -> Optional[str]:
    match = re.match(r".*local-cache-(.+?)/.*", path_str)
    if match:
        return match.group(1)
    return None


def _assert_step4_links_on_current_host(workdir: Path) -> None:
    """"""
    for link_name in ["step4_8_perturbed_lora"]:
        link_path = workdir / link_name
        if not link_path.is_symlink():
            continue

        target = str(link_path.readlink())
        host_in_target = _extract_host_from_cache_path(target)
        if host_in_target and host_in_target not in HOSTNAME:
            raise RuntimeError(
                f"{link_name} points to another host's cache: {target}. "
                f"Current host: {HOSTNAME}. "
                "Step 5 is incomplete, so partial resume across hosts is blocked."
            )


def _run_combined_4_7_4_8_5_group(
    inputs: Dict,
    config: Dict,
    workdir: Path,
    args,
    skip_linker: SkipAndLink,
    group_specs: List,
) -> None:
    """"""
    from pipeline_src.step_registry import params_from_inputs

    specs_by_name = {s.name: s for s in group_specs}
    s47 = specs_by_name["step4_7"]
    s48 = specs_by_name["step4_8"]
    s5  = specs_by_name["step5"]

    if not skip_linker.register_step(
        s47.name, params_from_inputs(s47, inputs), list(s47.outputs)
    ):
        _assert_step4_links_on_current_host(workdir)
        run_timed_step(s47.label, s47.run, config, workdir,
                       StepArgs(args, s47.compat_keys, s47.name))

    if not skip_linker.register_step(
        s48.name, params_from_inputs(s48, inputs), list(s48.outputs)
    ):
        _assert_step4_links_on_current_host(workdir)
        run_timed_step(s48.label, s48.run, config, workdir,
                       StepArgs(args, s48.compat_keys, s48.name))

    if not skip_linker.register_step(
        s5.name, params_from_inputs(s5, inputs), list(s5.outputs)
    ):
        _assert_step4_links_on_current_host(workdir)

        run_timed_step(s5.label, s5.run, config, workdir,
                       StepArgs(args, s5.compat_keys, s5.name))


def _run_step5_discovery(config: Dict, workdir: Path, args) -> None:
    """"""
    from pipeline_src.loss_plan import build_loss_plan, LossJob
    from pipeline_src.reuse_losses import link_reusable_step5_jobs
    from pipeline_src.step5_compute_losses import run as step5_new_run

    jobs, spec = build_loss_plan(
        workdir,
        args.num_samples,
        difficulty_tokens=get_num_token_for_difficulty(args),
        clustering_tokens=int(DEFAULTS["clustering_tokens"]),
    )

    if getattr(args, "sort_ckpts", None):
        ckpt_paths = [Path(p.strip()) for p in args.sort_ckpts.split(',')]
        for i, ckpt_path in enumerate(ckpt_paths):
            is_adapter = (ckpt_path / "adapter_config.json").exists()
            jobs.append(LossJob(
                name=f"sort_ckpt_{i}_{ckpt_path.name}",
                checkpoint=str(ckpt_path),
                max_tokens=1000,
                is_adapter=is_adapter,
                loss_file=spec.clustering_loss_file,
            ))

    if getattr(args, "cluster_sort", None) == "avg_max_scale_minus_base":
        jobs.append(LossJob(
            name="base_unperturbed",
            checkpoint=args.original_model_name,
            max_tokens=1000,
            is_adapter=False,
            loss_file=spec.clustering_loss_file,
        ))

    sort_mode = args.cluster_sort
    step2_surface = args.step2_perturbation_surface
    step2_force_save = bool(args.step2_force_save_perturbed_checkpoint)
    requires_saved_step2_model = sort_mode in {"learn-diff-base", "diff_perturb_1k"}
    virtual_step2_overlay = None
    difficulty_inflection_checkpoint = str(workdir / "step2_loss_100tokens" / "checkpoint-inflection")
    if requires_saved_step2_model and step2_surface != "state_dict" and not step2_force_save:
        descriptor_path = workdir / "step2_loss_100tokens" / "checkpoint-inflection" / "perturbation_overlay.json"
        if not descriptor_path.exists():
            raise RuntimeError(
                f"--cluster-sort {sort_mode} needs the Step 2 perturbed model for "
                "1k-token loss computation. Step 2 used "
                f"step2_perturbation_surface={step2_surface!r} without saving a "
                f"checkpoint, but no virtual overlay descriptor exists at {descriptor_path}. "
                "Rerun Step 2 with the current code to create the descriptor, use "
                "--step2-perturbation-surface state_dict, or add "
                "--step2-force-save-perturbed-checkpoint."
            )
        with open(descriptor_path, "r", encoding="utf-8") as f:
            virtual_step2_overlay = json.load(f)
        virtual_step2_overlay["descriptor_path"] = str(descriptor_path)
        difficulty_inflection_checkpoint = args.original_model_name

    if sort_mode == "learn-diff-base":
        jobs.append(LossJob(
            name="base_unperturbed",
            checkpoint=args.original_model_name,
            max_tokens=1000,
            is_adapter=False,
            loss_file=spec.clustering_loss_file,
        ))
        jobs.append(LossJob(
            name="difficulty_inflection",
            checkpoint=difficulty_inflection_checkpoint,
            max_tokens=1000,
            is_adapter=False,
            loss_file=spec.clustering_loss_file,
            perturbation_overlay=virtual_step2_overlay,
        ))
    elif sort_mode == "diff_perturb_1k":
        jobs.append(LossJob(
            name="difficulty_inflection",
            checkpoint=difficulty_inflection_checkpoint,
            max_tokens=1000,
            is_adapter=False,
            loss_file=spec.clustering_loss_file,
            perturbation_overlay=virtual_step2_overlay,
        ))
    elif sort_mode == "learn-diff-statedict-base":
        statedict_ckpt_dir = workdir / "step4_3_statedict_perturb" / "checkpoint-inflection"
        if not statedict_ckpt_dir.exists():
            raise RuntimeError(
                f"--cluster-sort learn-diff-statedict-base needs the Step 4.3 "
                f"perturbed checkpoint at {statedict_ckpt_dir}, but it is missing. "
                "Step 4.3 should have produced this before Step 5."
            )
        jobs.append(LossJob(
            name="base_unperturbed",
            checkpoint=args.original_model_name,
            max_tokens=STEP4_3_SORT_EVAL_MAX_TOKENS,
            is_adapter=False,
            loss_file=STEP4_3_SORT_LOSS_FILE,
        ))
        jobs.append(LossJob(
            name="statedict_difficulty_inflection",
            checkpoint=str(statedict_ckpt_dir),
            max_tokens=STEP4_3_SORT_EVAL_MAX_TOKENS,
            is_adapter=False,
            loss_file=STEP4_3_SORT_LOSS_FILE,
        ))

    run_group = Path(args.run_name).parts[0] if getattr(args, "run_name", None) else None
    scan_root = (WORKDIR_BASE / run_group) if run_group else WORKDIR_BASE
    merged_inputs = {**config, **vars(args._args)}
    link_reusable_step5_jobs(
        workdir=workdir,
        current_inputs=merged_inputs,
        jobs=jobs,
        scan_root=scan_root,
    )

    dataset_path = workdir / "step4_continued_1k.jsonl"
    subset_indices_path = workdir / "step3_difficult_samples.npy"

    step1_path = workdir / "step1_generation_100tokens.jsonl"
    with open(step1_path, "r") as f:
        full_size = sum(1 for _ in f)

    step5_new_run(
        jobs=jobs,
        output_dir=workdir / "step5_loss_1k_tokens",
        dataset_path=dataset_path,
        subset_indices_path=subset_indices_path,
        full_dataset_size=full_size,
        reasoning_marker=getattr(args, "reasoning_marker", None),
    )


def _run_step6_discovery(config: Dict, workdir: Path, args) -> None:
    """"""
    from pipeline_src.step6_prepare import prepare_selection_spec
    from pipeline_src.step6_select import run as step6_select_run

    spec = prepare_selection_spec(config, workdir, args)

    step6_select_run(
        spec=spec,
        step2_loss_dir=workdir / "step2_loss_100tokens",
        step5_loss_dir=workdir / "step5_loss_1k_tokens",
        prompt_source=config["prompt_source"],
        subset_indices_path=workdir / "step3_difficult_samples.npy",
        workdir=workdir,
        max_repetition=getattr(args, "max_repetition", 1),
    )


def _materialize_direct_index_selection(workdir: Path, config: Dict, args) -> None:
    """"""
    import numpy as np

    def _is_sharegpt_turn_list(turns) -> bool:
        if not isinstance(turns, list) or not turns:
            return False
        for turn in turns:
            if not isinstance(turn, dict):
                return False
            if "role" not in turn or "content" not in turn:
                return False
        return True

    index_path = Path(str(args.direct_index)).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"--direct-index file not found: {index_path}")

    arr = np.load(str(index_path))
    if arr.ndim != 1:
        raise ValueError(f"--direct-index must be a 1-D array; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("--direct-index array is empty")
    if np.any(arr < 0):
        raise ValueError("--direct-index contains negative value(s)")

    prompt_source = Path(config["prompt_source"]).expanduser().resolve()
    if not prompt_source.is_file():
        raise FileNotFoundError(f"prompt_source for task is not a file: {prompt_source}")

    max_idx = int(np.max(arr))
    with open(prompt_source, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    if max_idx >= n_lines:
        raise ValueError(
            f"--direct-index max index is {max_idx} but {prompt_source} has {n_lines} "
            f"line(s) (0-based indices must be < {n_lines})."
        )

    selected_indices = {int(i) for i in arr.tolist()}
    with open(prompt_source, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if row_idx not in selected_indices:
                continue
            if not line.strip():
                raise ValueError(
                    f"--direct-index selected line {row_idx} in {prompt_source} is empty."
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"--direct-index selected line {row_idx} in {prompt_source} is not valid JSON."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"--direct-index selected line {row_idx} in {prompt_source} is not a JSON object."
                )
            if not _is_sharegpt_turn_list(row.get("conversations")):
                raise ValueError(
                    "--direct-index requires selected rows to already contain "
                    "ShareGPT-style `conversations` with `{role, content}` turns. "
                    f"Row {row_idx} in {prompt_source} is not compatible; use "
                    "--direct-file for datasets that need normalization."
                )

    step7_path = workdir / "step7_full_traces.jsonl"
    step6_path = workdir / "step6_labeled_idx.npy"
    for p in (step7_path, step6_path):
        if p.is_symlink() or p.exists():
            p.unlink()

    step7_path.symlink_to(prompt_source)
    step6_path.symlink_to(index_path)
    log(
        f"[direct-index] Linked {step7_path.name} -> {prompt_source} and "
        f"{step6_path.name} -> {index_path} ({arr.size} indices, max={max_idx})"
    )


def _from_value_turn_to_sharegpt(turn: dict) -> dict:
    """"""
    from_val = str(turn.get("from", "") or "").strip().lower()
    text = turn.get("value", turn.get("content", ""))
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    if from_val in ("human", "user"):
        role = "user"
    elif from_val in ("gpt", "assistant", "bing", "chatgpt"):
        role = "assistant"
    elif from_val == "system":
        role = "system"
    elif from_val in ("observation", "function", "tool"):
        role = "assistant"
    else:
        role = "assistant"

    return {"role": role, "content": text}


def _normalize_row_sharegpt_conversations(row: dict) -> dict:
    """"""
    out = dict(row)
    conv = out.get("conversations")
    msgs = out.get("messages")

    def _first_turn(dlist):
        if not isinstance(dlist, list) or not dlist:
            return None
        t0 = dlist[0]
        return t0 if isinstance(t0, dict) else None

    t_conv = _first_turn(conv)
    t_msg = _first_turn(msgs)

    if t_conv is not None and "role" in t_conv:
        fixed = []
        for t in conv:
            if not isinstance(t, dict):
                raise ValueError("--direct-file: conversations turn is not an object")
            body = t.get("content")
            if body is None:
                body = t.get("value", "")
            fixed.append({"role": str(t["role"]), "content": "" if body is None else str(body)})
        out["conversations"] = fixed
        return out

    if t_conv is not None and "from" in t_conv and "value" in t_conv:
        out["conversations"] = [_from_value_turn_to_sharegpt(t) for t in conv if isinstance(t, dict)]
        return out

    if t_msg is not None and "role" in t_msg:
        fixed = []
        for t in msgs:
            if not isinstance(t, dict):
                continue
            body = t.get("content")
            if body is None:
                body = t.get("value", "")
            fixed.append({"role": str(t["role"]), "content": "" if body is None else str(body)})
        out["conversations"] = fixed
        return out

    if t_msg is not None and "from" in t_msg and "value" in t_msg:
        out["conversations"] = [_from_value_turn_to_sharegpt(t) for t in msgs if isinstance(t, dict)]
        return out

    raise ValueError(
        "--direct-file row needs list ``conversations`` or ``messages`` with turns "
        "either {{role, content}} or {{from, value}}."
    )


def _materialize_direct_file_selection(workdir: Path, args) -> None:
    """"""
    import numpy as np

    src = Path(str(args.direct_file)).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"--direct-file not found: {src}")

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"--direct-file root must be a JSON array; got {type(data).__name__}"
        )
    if len(data) == 0:
        raise ValueError("--direct-file array is empty")

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(
                f"--direct-file element at index {i} must be a JSON object, "
                f"not {type(row).__name__}"
            )

    step7_path = workdir / "step7_full_traces.jsonl"
    step6_path = workdir / "step6_labeled_idx.npy"
    for p in (step7_path, step6_path):
        if p.is_symlink() or p.exists():
            p.unlink()

    with open(step7_path, "w", encoding="utf-8") as out:
        for row in data:
            normalized = _normalize_row_sharegpt_conversations(row)
            out.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    np.save(str(step6_path), np.arange(len(data), dtype=np.int64))
    log(
        f"[direct-file] Wrote {step7_path.name} ({len(data)} lines) and "
        f"{step6_path.name} (indices 0..{len(data) - 1}) from {src.name}"
    )


def run_pipeline(inputs: Dict, config: Dict, workdir: Path, args, skip_linker: SkipAndLink):
    """"""
    from pipeline_src.step_registry import (
        build_step_specs,
        params_from_inputs,
        specs_in_group,
        GROUP_COMBINED_4_7_4_8_5,
    )

    direct_index_path = inputs.get("direct_index")
    direct_file_path = inputs.get("direct_file")
    if direct_index_path:
        _materialize_direct_index_selection(workdir, config, args)
    elif direct_file_path:
        _materialize_direct_file_selection(workdir, args)

    if not direct_index_path and not direct_file_path:
        _mode = inputs["lora_perturb_mode"]
        _mod = "pipeline_src.step4_8_perturb_lora_inflection" if _mode == "inflection" else "pipeline_src.step4_8_perturb_lora"
        log(f"[dispatch] Step 4.8 -> {_mod} (mode={_mode})")

    specs = build_step_specs(
        step5_runner=_run_step5_discovery,
        step6_runner=_run_step6_discovery,
    )
    if direct_index_path or direct_file_path:
        specs = [s for s in specs if s.name in ("step8", "step9")]
        if direct_index_path:
            log("[direct-index] Running steps 8 and 9 only (skipped steps 1–7).")
        else:
            log("[direct-file] Running steps 8 and 9 only (skipped steps 1–7).")

    group_handlers = {
        GROUP_COMBINED_4_7_4_8_5: _run_combined_4_7_4_8_5_group,
    }
    handled_groups: set = set()

    for spec in specs:
        if not spec.predicate(inputs):
            continue

        if spec.group is not None:
            if spec.group in handled_groups:
                continue
            group_specs = [s for s in specs_in_group(specs, spec.group)
                           if s.predicate(inputs)]
            handler = group_handlers.get(spec.group)
            if handler is None:
                raise RuntimeError(
                    f"No group handler registered for group {spec.group!r} "
                    f"(spec={spec.name!r})."
                )
            handler(inputs, config, workdir, args, skip_linker, group_specs)
            handled_groups.add(spec.group)
            continue

        params = params_from_inputs(spec, inputs)
        if skip_linker.register_step(
            spec.name, params, list(spec.outputs),
            side_branch=spec.side_branch,
            side_branch_cascade_anchor=spec.side_branch_cascade_anchor,
        ):
            continue
        run_timed_step(
            spec.label, spec.run, config, workdir,
            StepArgs(args, spec.compat_keys, spec.name),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Data Curation Pipeline for LLM Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data-curation-pipeline.py --task openthoughts-math
  python data-curation-pipeline.py --task numina
  python data-curation-pipeline.py --task m23k --num_samples 500
  python data-curation-pipeline.py --task openthoughts-math --run_name my_experiment
        """
    )

    for dest, spec in ARG_REGISTRY.items():
        flags = spec.get("flags", [f"--{dest}"])
        kwargs = {"default": spec["default"], "help": spec["help"]}
        if "type" in spec:
            kwargs["type"] = spec["type"]
        if "choices" in spec:
            kwargs["choices"] = spec["choices"]
        if "action" in spec:
            kwargs["action"] = spec["action"]
        if "dest" in spec:
            kwargs["dest"] = spec["dest"]
        parser.add_argument(*flags, **kwargs)

    parser.add_argument(
        "--task", type=str, required=True,
        choices=[
            "openthoughts-math", "numina", "numina-dup16",
            "physics", "physics-dup1", "physics-dup2", "physics-dup4", "physics-dup8", "physics-dup16",
            "m23k", "m192k", "m192k-dup16", "m38k", "m38k-dup4", "m38k-dup8", "m38k-dup16",
        ],
        help="Task/dataset to process"
    )
    parser.add_argument(
        "--teacher", type=str, default="openai/gpt-oss-120b",
        help="Teacher model for generation (default: openai/gpt-oss-120b; "
             "m192k/m38k/m38k-dup4/m38k-dup8/m38k-dup16/m192k-dup16 default to Qwen/QwQ-32B when this flag is left at the global default)"
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Custom run name (default: auto-generated)"
    )
    parser.add_argument(
        "--resample", action="store_true",
        help="Enable resample mode: multi-turn approve/deny loop for steps 6+7"
    )
    parser.add_argument(
        "--reasoning_loss_only", action="store_true",
        help="Calculate computation loss explicitly over reasoning tokens only"
    )
    parser.add_argument(
        "--force-rerun", action="store_true", dest="force_rerun",
        help=(
            "Disable cross-folder auto-discovery. Each step runs fresh unless "
            "its output already exists locally or is explicitly overridden via "
            "--step / --up_to_step."
        ),
    )
    parser.add_argument(
        "--no-standardize-kmeans", action="store_false", dest="standardize_kmeans",
        help="Do not standardize all loss dimensions before k-means clustering in train.py"
    )
    parser.add_argument(
        "--sort-ckpts", type=str, default=None,
        help="Comma-separated paths for custom checkpoints used by the chosen sort method (e.g. for checkpoint_diff)."
    )
    parser.add_argument(
        "--no-lora-perturb-extend-b",
        dest="lora_perturb_extend_b_if_no_inflection",
        action="store_false", default=True,
        help="Step 4.8 (inflection mode): disable the auto-extension of the right bound. "
             "By default, if the smoothing-spline fit has no curvature-threshold crossing "
             "in the current range, the finder grows the right bound by "
             "--lora_perturb_b_extend_factor (up to --lora_perturb_max_b_extensions "
             "times) and evaluates the new outer scale. This is the expansion counterpart "
             "of the post-min-monotone contraction: contract when we overshot into the "
             "random regime, expand when the observed window was too narrow to contain any "
             "inflection. Pass --no-lora-perturb-extend-b to force the finder to stay "
             "within its initial per-direction bound."
    )
    for s in ["step1", "step2", "step3", "step4", "step4_3", "step4_5", "step4_7", "step4_8", "step5", "step6", "step7", "step8", "step9"]:
        parser.add_argument(f"--{s}", type=str, default=None, help=f"Force reuse of {s} from the provided path and skip the check during linking.")
    step_names_with_floats = [
        ("step1", 1.0), ("step2", 2.0), ("step3", 3.0), ("step4", 4.0),
        ("step4_3", 4.3),
        ("step4_5", 4.5), ("step4_7", 4.7), ("step4_8", 4.8),
        ("step5", 5.0), ("step6", 6.0), ("step7", 7.0), ("step8", 8.0), ("step9", 9.0)
    ]
    for s, _ in step_names_with_floats:
        parser.add_argument(
            f"--up_to_{s}", type=str, default=None,
            help=(
                f"Force reuse all steps up to {s} from the provided path. "
                f"Mutually exclusive with manual overrides for same/lower steps."
            ),
        )
    parser.add_argument(
        "--direct-index",
        type=str,
        default=None,
        dest="direct_index",
        metavar="PATH",
        help=(
            "Path to a .npy file of int64/int32 global line indices into the task "
            "prompt_source JSONL. Supported for --teacher original on tasks whose "
            "prompt_source is already ShareGPT-style (`conversations` with "
            "`{role, content}` turns), including m23k and openthoughts-math. "
            "Symlinks prompt_source to step7_full_traces.jsonl and this file to "
            "step6_labeled_idx.npy, then runs only steps 8 and 9. Mutually exclusive "
            "with --up_to_* and with --step1 … --step7 overrides."
        ),
    )
    parser.add_argument(
        "--direct-file",
        type=str,
        default=None,
        dest="direct_file",
        metavar="PATH",
        help=(
            "Path to a JSON file whose root is an array of training objects (e.g. step3 labeled.json). "
            "Writes step7_full_traces.jsonl as JSONL and step6_labeled_idx.npy as np.arange(n), then "
            "runs only steps 8 and 9. Any --task is allowed; requires --teacher original and no "
            "--resample. Mutually exclusive with --direct-index, --up_to_*, and --step1 … --step7 overrides."
        ),
    )

    args = parser.parse_args()

    task_cfg = TASK_CONFIGS[args.task]
    if "default_teacher" in task_cfg and "--teacher" not in sys.argv:
        args.teacher = task_cfg["default_teacher"]
    if "default_step7_max_tokens" in task_cfg and "--step7_max_tokens" not in sys.argv:
        args.step7_max_tokens = int(task_cfg["default_step7_max_tokens"])
    
    if args.resample and args.teacher.lower() == "original":
        raise ValueError("Cannot use --resample with --teacher original.")

    if getattr(args, "direct_index", None) and getattr(args, "direct_file", None):
        raise ValueError("Cannot use --direct-index and --direct-file together.")

    if getattr(args, "direct_index", None):
        dpath = Path(args.direct_index).expanduser()
        if args.task not in {"m23k", "openthoughts-math"}:
            raise ValueError(
                "--direct-index is only supported with --task m23k or --task openthoughts-math."
            )
        if args.teacher.lower() != "original":
            raise ValueError("--direct-index requires --teacher original.")
        if args.resample:
            raise ValueError("--direct-index cannot be used with --resample.")
        if not dpath.is_file():
            raise ValueError(f"--direct-index path is not a file: {dpath}")
        args.direct_index = str(dpath.resolve())

    if getattr(args, "direct_file", None):
        fpath = Path(args.direct_file).expanduser()
        if args.teacher.lower() != "original":
            raise ValueError("--direct-file requires --teacher original.")
        if args.resample:
            raise ValueError("--direct-file cannot be used with --resample.")
        if not fpath.is_file():
            raise ValueError(f"--direct-file path is not a file: {fpath}")
        args.direct_file = str(fpath.resolve())

    if getattr(args, "step9_eval_temperature", 0.0) != 0.0 and task_cfg.get("eval_type") != "m23k":
        raise ValueError("--step9_eval_temperature is only supported for medical tasks (eval_type=m23k).")

    if args.num_token_for_difficulty <= 0:
        raise ValueError("--num_token_for_difficulty must be a positive integer.")

    if args.partial_x is not None and args.partial_x <= 0:
        raise ValueError("--partial_x must be > 0 when provided.")
    if args.partial_y < 0:
        raise ValueError("--partial_y must be >= 0.")
    if args.step7_max_tokens <= 0:
        raise ValueError("--step7_max_tokens must be a positive integer.")
    if args.step8_cutoff_len is not None and args.step8_cutoff_len <= 0:
        raise ValueError("--step8_cutoff_len must be a positive integer when provided.")
    if args.lora_train_steps <= 0:
        raise ValueError("--lora_train_steps must be > 0.")
    if args.lora_train_rank is not None and args.lora_train_rank <= 0:
        raise ValueError("--lora_train_rank must be > 0 when provided.")
    if args.lora_bs is not None and args.lora_bs <= 0:
        raise ValueError("--lora_bs must be > 0 when provided.")
    if args.max_repetition < 1:
        raise ValueError("--max-repetition must be >= 1.")

    up_to_args = [(s, n, getattr(args, f"up_to_{s}")) for s, n in step_names_with_floats if getattr(args, f"up_to_{s}") is not None]
    if len(up_to_args) > 1:
        raise ValueError(
            "Only one --up_to_step option may be provided at a time. "
            f"Got: {[f'up_to_{s}' for s, _, _ in up_to_args]}"
        )

    if up_to_args:
        up_to_s, up_to_n, up_to_path = up_to_args[0]
        up_to_path = str(Path(up_to_path).expanduser())

        if not Path(up_to_path).exists():
            raise ValueError(f"--up_to_{up_to_s} path does not exist: {up_to_path}")

        conflict_steps = [
            s for s, n in step_names_with_floats
            if n <= up_to_n and getattr(args, s, None) is not None
        ]
        if conflict_steps:
            raise ValueError(
                f"--up_to_{up_to_s} is mutually exclusive with manual overrides for the same/lower steps: "
                f"{', '.join('--' + s for s in conflict_steps)}"
            )

        for s, n in step_names_with_floats:
            if n <= up_to_n:
                setattr(args, s, up_to_path)

        log(f"Using --up_to_{up_to_s}: forcing manual reuse for all steps <= {up_to_s} from {up_to_path}")

    if getattr(args, "direct_index", None) or getattr(args, "direct_file", None):
        if up_to_args:
            raise ValueError(
                "--direct-index / --direct-file cannot be combined with any --up_to_* option."
            )
        early_reuse = [
            s for s, n in step_names_with_floats
            if n <= 7.0 and getattr(args, s, None) is not None
        ]
        if early_reuse:
            raise ValueError(
                "--direct-index / --direct-file cannot be combined with reuse overrides for steps 1–7: "
                + ", ".join(f"--{x}" for x in early_reuse)
            )

    if args.lora_perturb_ix == "inflection":
        args.lora_perturb_ix = "infl,half-infl"
    args.lora_perturb_ix = ",".join(_parse_ix_values(args.lora_perturb_ix))
    
    if TASK_CONFIGS[args.task].get("eval_type") == "openthoughts-math":
        if "qwq" not in args.teacher.lower() and "gpt-oss" not in args.teacher.lower() and args.teacher.lower() != "original":
            raise ValueError(f"For tasks with openthoughts-math-style completeness checks, only qwq and gpt-oss teachers are supported. Got: {args.teacher}")

    if args.run_name is None:
        teacher_name = args.teacher.split('/')[-1]
        base_name = f"{args.task}-{teacher_name}"
        run_name_parts = []

        for dest, spec in ARG_REGISTRY.items():
            fmt = spec.get("run_name_fmt")
            if fmt is None:
                continue
            value = getattr(args, dest)
            test_fn = spec.get("run_name_test", lambda v: v != spec["default"])
            if test_fn(value):
                if callable(fmt):
                    formatted = fmt(value).replace(",", "-")
                else:
                    formatted = fmt.format(v=value).replace(",", "-")
                run_name_parts.append(formatted)

        lmode = getattr(args, "lora_perturb_mode", ARG_REGISTRY["lora_perturb_mode"]["default"])
        if lmode == "inflection":
            curvature_ratio = getattr(args, "lora_perturb_inflection_curvature_ratio",
                                      ARG_REGISTRY["lora_perturb_inflection_curvature_ratio"]["default"])
            if curvature_ratio != ARG_REGISTRY["lora_perturb_inflection_curvature_ratio"]["default"]:
                sign = "-" if curvature_ratio < 0 else "+"
                run_name_parts.append(f"lmode_inflection{sign}{abs(curvature_ratio):g}c")
            elif lmode != ARG_REGISTRY["lora_perturb_mode"]["default"]:
                run_name_parts.append("lmode_inflection")
        else:
            if lmode != ARG_REGISTRY["lora_perturb_mode"]["default"]:
                run_name_parts.append(f"lmode_{lmode}")

        if not getattr(args, "standardize_kmeans", True):
            run_name_parts.append("no-std")

        if getattr(args, "direct_file", None):
            run_name_parts.append(f"dfil_{Path(args.direct_file).stem}")
        elif getattr(args, "direct_index", None):
            run_name_parts.append(f"didx_{Path(args.direct_index).stem}")

        if not run_name_parts:
            args.run_name = f"{base_name}/default"
        else:
            args.run_name = f"{base_name}/" + "_".join(run_name_parts)

        if args.resample and not args.run_name.endswith("-resample"):
            args.run_name = args.run_name + "-resample"

        if getattr(args, "reasoning_loss_only", False) and not args.run_name.endswith("-r_loss"):
            args.run_name = args.run_name + "-r_loss"

    if getattr(args, "reasoning_loss_only", False):
        from pipeline_src.utils import _derive_reasoning_marker
        log(f"Reasoning loss only enabled. Mapped reasoning_marker to: {repr(_derive_reasoning_marker(args))}")

    inputs: Dict = dict(TASK_CONFIGS[args.task])

    for _arg_name, _arg_value in vars(args).items():
        if _arg_value is None and _arg_name in inputs:
            continue
        inputs[_arg_name] = _arg_value

    inputs.setdefault("generation_prompt_template", None)

    if inputs.get("partial_x") is None:
        inputs["partial_x"] = int(inputs["lora_bs"]) * int(inputs["lora_train_steps"])
    else:
        inputs["partial_x"] = int(inputs["partial_x"])

    if args.task == "m23k" and args.teacher.lower() != "original":
        inputs["prompt_source"] = str(BASE_DIR / "dataset" / "m23k-prd" / "train.jsonl")

    prompt_source_path = Path(str(inputs.get("prompt_source", ""))).expanduser()
    if prompt_source_path.is_file():
        with open(prompt_source_path, "r", encoding="utf-8") as _psf:
            inputs["prompt_source_nrows"] = sum(1 for _ in _psf)
    else:
        inputs["prompt_source_nrows"] = None

    override_steps = []
    for s in ["step1", "step2", "step3", "step4", "step4_3", "step4_5", "step4_7", "step4_8", "step5", "step6", "step7", "step8", "step9"]:
        if getattr(args, s, None):
            override_steps.append(s)
    if override_steps:
        inputs["manual_override_steps"] = override_steps
    inputs["perturbation_surface_version"] = 4

    task_keys = set(TASK_CONFIGS[args.task].keys()) | {
        "generation_prompt_template",
        "perturbation_surface_version",
    }
    config = {k: inputs[k] for k in task_keys if k in inputs}

    cli_keys = set(inputs.keys()) - set(config.keys())
    args = argparse.Namespace(**{k: inputs[k] for k in cli_keys})

    workdir = WORKDIR_BASE / args.run_name
    workdir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("DATA CURATION PIPELINE")
    log("=" * 60)
    log(f"Task:        {args.task}")
    log(f"Teacher:     {args.teacher}")
    log(f"Run Name:    {args.run_name}")
    log(f"Seed:        {args.seed}")
    log(f"Num Samples: {args.num_samples}")
    log(f"Work Dir:    {workdir}")
    log("=" * 60)

    config_file = workdir / "pipeline_config.json"
    config_bak_file = workdir / "pipeline_config.json.bak"
    backed_up_existing_config = False
    if config_file.exists():
        shutil.copy2(config_file, config_bak_file)
        log(f"Existing config detected. Backed up to: {config_bak_file}")
        backed_up_existing_config = True
    elif config_bak_file.exists():
        config_bak_file.unlink()

    _persisted_inputs = dict(inputs)

    with open(config_file, "w") as f:
        json.dump({
            "inputs": _persisted_inputs,
            "defaults": DEFAULTS,
        }, f, indent=2)

    cmd_file = workdir / "cmd.txt"
    with open(cmd_file, "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")

    run_group = Path(args.run_name).parts[0] if args.run_name else None
    scan_root = (WORKDIR_BASE / run_group) if run_group else WORKDIR_BASE
    if run_group:
        log(f"[AUTO-DISCOVER] Restricting prior-run scan to group: {scan_root}")

    skip_linker = SkipAndLink(
        workdir, inputs,
        scan_root=scan_root,
        force_rerun=getattr(args, "force_rerun", False),
    )

    try:
        run_pipeline(inputs, config, workdir, args, skip_linker)

        log("=" * 60)
        log("PIPELINE COMPLETE!")
        log("=" * 60)
        log(f"Results saved to: {workdir}")

    except subprocess.CalledProcessError as e:
        if backed_up_existing_config and config_bak_file.exists():
            try:
                shutil.copy2(config_bak_file, config_file)
                log(f"Restored config from backup after failure: {config_file}")
            except Exception:
                pass
        log(f"ERROR: Step failed with exit code {e.returncode}")
        log(f"Command: {e.cmd}")
        sys.exit(1)
    except Exception:
        if backed_up_existing_config and config_bak_file.exists():
            try:
                shutil.copy2(config_bak_file, config_file)
                log(f"Restored config from backup after failure: {config_file}")
            except Exception:
                pass
        raise
    except KeyboardInterrupt:
        log("Pipeline interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
