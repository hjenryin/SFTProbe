""""""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple


RESUME_COMMON: FrozenSet[str] = frozenset({
    "task", "teacher", "seed",
    "num_token_for_difficulty",
    "original_model_name",
    "reasoning_loss_only",
    "generation_prompt_template",
    "prompt_source_nrows",
})


@dataclass(frozen=True)
class StepSpec:
    """"""

    name: str
    label: str
    compat_keys: FrozenSet[str]
    outputs: Tuple[str, ...]
    run: Optional[Callable[..., Any]]
    predicate: Callable[[Dict[str, Any]], bool] = field(
        default=lambda inputs: True
    )
    group: Optional[str] = None
    side_branch: bool = False
    side_branch_cascade_anchor: Optional[str] = None


def _resample_mode(inputs: Dict[str, Any]) -> bool:
    return bool(inputs.get("resample", False))


def _not_resample_mode(inputs: Dict[str, Any]) -> bool:
    return not bool(inputs.get("resample", False))


def _step4_3_mode(inputs: Dict[str, Any]) -> bool:
    """"""
    return inputs.get("cluster_sort") == "learn-diff-statedict-base"


GROUP_COMBINED_4_7_4_8_5 = "combined_4_7_4_8_5"


_STEP1_KEYS: FrozenSet[str] = frozenset({
    "task", "teacher", "seed", "num_token_for_difficulty",
    "generation_prompt_template",
    "prompt_source_nrows",
})

_STEP2_KEYS: FrozenSet[str] = frozenset({
    "task", "teacher", "seed",
    "original_model_name",
    "reasoning_loss_only",
    "num_token_for_difficulty",
    "generation_prompt_template",
    "perturbation_surface_version",
    "step2_perturbation_surface",
    "step2_force_save_perturbed_checkpoint",
    "step2_final_eval_from_saved_checkpoint",
    "step2_inflection_ytol", "step2_inflection_max_iter",
    "step2_inflection_curvature_ratio",
    "step2_inflection_lower_bound", "step2_inflection_upper_bound",
})

_STEP3_KEYS: FrozenSet[str] = RESUME_COMMON | frozenset({"difficulty_filter_without_source"})

_STEP4_KEYS: FrozenSet[str] = RESUME_COMMON

_STEP4_3_KEYS: FrozenSet[str] = RESUME_COMMON | frozenset({
    "perturbation_surface_version",
    "step4_3_inflection_ytol",
    "step4_3_inflection_max_iter",
    "step4_3_inflection_curvature_ratio",
    "step4_3_inflection_lower_bound",
    "step4_3_inflection_upper_bound",
    "step4_3_inflection_extend_b_if_no_inflection",
    "step4_3_inflection_b_extend_factor",
    "step4_3_inflection_max_b_extensions",
})

_STEP4_5_KEYS: FrozenSet[str] = RESUME_COMMON | frozenset({
    "partial_x",
    "partial_x_sample",
    "partial_y",
})

_STEP4_7_KEYS: FrozenSet[str] = RESUME_COMMON | frozenset({
    "partial_x",
    "partial_x_sample",
    "partial_y",
    "lora_train_steps", "lora_train_rank", "lora_bs",
})

_STEP4_8_KEYS: FrozenSet[str] = _STEP4_7_KEYS | frozenset({
    "lora_perturb_ix",
    "lora_perturb_mode",
    "lora_perturb_inflection_curvature_ratio",
    "lora_perturb_post_min_decrease_frac",
    "lora_perturb_inflection_ytol",
    "lora_perturb_inflection_max_iter",
    "lora_perturb_eval_batch_size",
    "lora_perturb_gpu_threshold",
    "lora_perturb_seed_count",
    "lora_perturb_b_extend_factor",
    "lora_perturb_max_b_extensions",
    "lora_perturb_extend_b_if_no_inflection",
    "lora_perturb_post_min_decrease_tol",
})

_STEP5_KEYS: FrozenSet[str] = _STEP4_8_KEYS | _STEP2_KEYS | _STEP4_3_KEYS | frozenset({
    "cluster_sort",
    "num_samples",
    "sort_ckpts",
})

_STEP6_KEYS: FrozenSet[str] = RESUME_COMMON | frozenset({
    "num_samples",
    "cluster_sort",
    "sort_ckpts",
    "max_repetition",
    "standardize_kmeans",
    "partial_x_sample",
})

_STEP6_7_KEYS: FrozenSet[str] = _STEP6_KEYS | frozenset({
    "step7_max_tokens",
    "max_repetition",
})
_STEP7_KEYS: FrozenSet[str] = _STEP6_KEYS | frozenset({
    "step7_max_tokens",
    "max_repetition",
})

_STEP8_KEYS: FrozenSet[str] = RESUME_COMMON | frozenset({
    "num_samples",
    "resample",
    "step7_max_tokens",
    "step8_cutoff_len",
    "max_repetition",
    "partial_x_sample",
    "direct_index",
    "direct_file",
})

_STEP9_KEYS: FrozenSet[str] = _STEP8_KEYS | frozenset({
    "step9_eval_temperature",
})


def _step4_8_dispatch(config, workdir, step_args):
    """"""
    mode = step_args.lora_perturb_mode
    if mode == "inflection":
        from pipeline_src.step4_8_perturb_lora_inflection import run as _fn
    elif mode == "legacy":
        from pipeline_src.step4_8_perturb_lora import run as _fn
    else:
        raise ValueError(
            f"Unknown lora_perturb_mode: {mode!r}. Expected 'legacy' or 'inflection'."
        )
    return _fn(config, workdir, step_args)


def build_step_specs(
    *,
    step5_runner: Callable[..., Any],
    step6_runner: Callable[..., Any],
) -> List[StepSpec]:
    """"""
    from pipeline_src.step1_generate_100_tokens import run as step1_run
    from pipeline_src.step2_perturb_difficulty_model import run as step2_run
    from pipeline_src.step3_difficulty_filtering import run as step3_run
    from pipeline_src.step4_continue_to_1k_tokens import run as step4_run
    from pipeline_src.step4_3_perturb_statedict import run as step4_3_run
    from pipeline_src.step4_5_partial_completion import run as step4_5_run
    from pipeline_src.step4_7_train_lora import run as step4_7_run
    from pipeline_src.step6_7_resample import run as step6_7_run
    from pipeline_src.step7_full_generation import run as step7_run
    from pipeline_src.step8_sft_training import run as step8_run
    from pipeline_src.step9_evaluation import run as step9_run

    return [
        StepSpec(
            name="step1",
            label="Step 1 (Generate 100 Tokens)",
            compat_keys=_STEP1_KEYS,
            outputs=("step1_generation_100tokens.jsonl",),
            run=step1_run,
        ),
        StepSpec(
            name="step2",
            label="Step 2 (Perturb Difficulty Model)",
            compat_keys=_STEP2_KEYS,
            outputs=("step2_loss_100tokens",),
            run=step2_run,
        ),
        StepSpec(
            name="step3",
            label="Step 3 (Difficulty Filtering)",
            compat_keys=_STEP3_KEYS,
            outputs=("step3_difficult_samples.npy", "step3_difficulty"),
            run=step3_run,
        ),
        StepSpec(
            name="step4",
            label="Step 4 (Continue to 1k Tokens)",
            compat_keys=_STEP4_KEYS,
            outputs=("step4_continued_1k.jsonl",),
            run=step4_run,
        ),
        StepSpec(
            name="step4_3",
            label="Step 4.3 (State-dict Perturb Inflection)",
            compat_keys=_STEP4_3_KEYS,
            outputs=(
                "step4_3_statedict_perturb/checkpoint-inflection",
                "step4_3_statedict_perturb/inflection_metadata.json",
            ),
            run=step4_3_run,
            predicate=_step4_3_mode,
            side_branch=True,
            side_branch_cascade_anchor="step5",
        ),
        StepSpec(
            name="step4_5",
            label="Step 4.5 (Partial Completion)",
            compat_keys=_STEP4_5_KEYS,
            outputs=(
                "step4_5_partial_completion.jsonl",
                "step4_5_selected_global_indices.npy",
                "step4_5_metadata.json",
            ),
            run=step4_5_run,
        ),
        StepSpec(
            name="step4_7",
            label="Step 4.7 (Train LoRA)",
            compat_keys=_STEP4_7_KEYS,
            outputs=("step4_7_lora/adapter", "step4_7_metadata.json"),
            run=step4_7_run,
            group=GROUP_COMBINED_4_7_4_8_5,
        ),
        StepSpec(
            name="step4_8",
            label="Step 4.8 (Perturb LoRA)",
            compat_keys=_STEP4_8_KEYS,
            outputs=("step4_8_perturbed_lora", "step4_8_metadata.json"),
            run=_step4_8_dispatch,
            group=GROUP_COMBINED_4_7_4_8_5,
        ),
        StepSpec(
            name="step5",
            label="Step 5 (Compute Losses)",
            compat_keys=_STEP5_KEYS,
            outputs=("step5_loss_1k_tokens", "step5_metadata.json"),
            run=step5_runner,
            group=GROUP_COMBINED_4_7_4_8_5,
        ),
        StepSpec(
            name="step6_7",
            label="Step 6+7 (Resample)",
            compat_keys=_STEP6_7_KEYS,
            outputs=("step7_selected_traces.jsonl", "step7_full_traces.jsonl"),
            run=step6_7_run,
            predicate=_resample_mode,
        ),
        StepSpec(
            name="step6",
            label="Step 6 (Selection)",
            compat_keys=_STEP6_KEYS,
            outputs=("step6_labeled_idx.npy",),
            run=step6_runner,
            predicate=_not_resample_mode,
        ),
        StepSpec(
            name="step7",
            label="Step 7 (Full Generation)",
            compat_keys=_STEP7_KEYS,
            outputs=("step7_full_traces.jsonl", "step7_selected_traces.jsonl"),
            run=step7_run,
            predicate=_not_resample_mode,
        ),
        StepSpec(
            name="step8",
            label="Step 8 (SFT Training)",
            compat_keys=_STEP8_KEYS,
            outputs=("step8_sft_model",),
            run=step8_run,
        ),
        StepSpec(
            name="step9",
            label="Step 9 (Evaluation)",
            compat_keys=_STEP9_KEYS,
            outputs=("step9_eval",),
            run=step9_run,
        ),
    ]


def params_from_inputs(spec: StepSpec, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """"""
    missing = [k for k in spec.compat_keys if k not in inputs]
    if missing:
        raise KeyError(
            f"[step_registry] Spec '{spec.name}' references keys missing from "
            f"inputs: {sorted(missing)}."
        )
    return {k: inputs[k] for k in sorted(spec.compat_keys)}


def specs_in_group(specs: List[StepSpec], group: str) -> List[StepSpec]:
    """"""
    return [s for s in specs if s.group == group]
