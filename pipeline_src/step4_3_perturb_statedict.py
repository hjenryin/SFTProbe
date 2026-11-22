""""""

import gc
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

from .inflection import LLMEvaluator
from .inflection_search import (
    build_scan_subset,
    load_eval_history,
    scale_key,
    search_inflection_points,
)
from .utils import LOCAL_CACHE, log, setup_model_symlinks


STEP4_3_SCAN_SUBSET_SIZE = 1000
STEP4_3_SORT_EVAL_MAX_TOKENS = 100
STEP4_3_SCAN_TOKENS = STEP4_3_SORT_EVAL_MAX_TOKENS
STEP4_3_SORT_LOSS_FILE = f"loss-{STEP4_3_SORT_EVAL_MAX_TOKENS}token-r-losses.pt"
STEP4_3_PERTURBATION_SURFACE = "state_dict"


def _metadata_matches_current_settings(
    metadata_path: Path,
    *,
    curvature_ratio: float,
    lower_bound: float,
    upper_bound: float,
    ytol: float,
    max_iter: int,
    extend_b_if_no_inflection: bool,
    b_extend_factor: float,
    max_b_extensions: int,
) -> bool:
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return False

    try:
        scan_range = metadata.get("scan_range", [None, None])
        meta_lower = float(scan_range[0])
        meta_upper = float(scan_range[1])
        meta_curv = float(metadata.get("curvature_ratio"))
        meta_ytol = float(metadata.get("ytol"))
        meta_max_iter = int(metadata.get("max_iter"))
        search = metadata.get("search") if isinstance(metadata.get("search"), dict) else {}
        meta_extend = bool(
            metadata.get(
                "extend_b_if_no_inflection",
                search.get("extend_b_if_no_inflection", False),
            )
        )
        meta_b_fac = float(
            metadata.get("b_extend_factor", search.get("b_extend_factor", 1.5))
        )
        meta_b_ext_n = int(
            metadata.get("max_b_extensions", search.get("max_b_extensions", 3))
        )
    except (TypeError, ValueError, IndexError):
        return False

    if "scan_tokens" not in metadata:
        return False
    try:
        meta_scan_tokens = int(metadata["scan_tokens"])
    except (TypeError, ValueError):
        return False

    return (
        metadata.get("perturbation_surface") == STEP4_3_PERTURBATION_SURFACE
        and meta_curv == float(curvature_ratio)
        and meta_lower == float(lower_bound)
        and meta_upper == float(upper_bound)
        and meta_ytol == float(ytol)
        and meta_max_iter == int(max_iter)
        and meta_extend == bool(extend_b_if_no_inflection)
        and meta_b_fac == float(b_extend_factor)
        and meta_b_ext_n == int(max_b_extensions)
        and meta_scan_tokens == int(STEP4_3_SCAN_TOKENS)
    )


def _remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    """"""
    out_dir = workdir / "step4_3_statedict_perturb"
    ckpt_dir = out_dir / "checkpoint-inflection"
    metadata_path = out_dir / "inflection_metadata.json"
    history_path = workdir / "step4_3_eval_history.json"
    debug_plot_dir = workdir / "step4_3_debug_inflection_plot"

    curvature_ratio = float(getattr(args, "step4_3_inflection_curvature_ratio", 0.1))
    lower_bound = float(getattr(args, "step4_3_inflection_lower_bound", 0.0003))
    upper_bound = float(getattr(args, "step4_3_inflection_upper_bound", 0.01))
    ytol = float(getattr(args, "step4_3_inflection_ytol", 1.0))
    max_iter = int(getattr(args, "step4_3_inflection_max_iter", 8))
    extend_b_if_no_inflection = bool(
        getattr(args, "step4_3_inflection_extend_b_if_no_inflection", True)
    )
    b_extend_factor = float(getattr(args, "step4_3_inflection_b_extend_factor", 1.5))
    max_b_extensions = int(getattr(args, "step4_3_inflection_max_b_extensions", 3))

    if lower_bound < 0:
        raise ValueError(f"step4_3_inflection_lower_bound must be >= 0, got {lower_bound}")
    if upper_bound <= lower_bound:
        raise ValueError(
            "step4_3_inflection_upper_bound must be greater than "
            f"step4_3_inflection_lower_bound, got {upper_bound} <= {lower_bound}"
        )
    if ytol <= 0:
        raise ValueError(f"step4_3_inflection_ytol must be > 0, got {ytol}")
    if max_iter <= 0:
        raise ValueError(f"step4_3_inflection_max_iter must be > 0, got {max_iter}")
    if b_extend_factor <= 1.0:
        raise ValueError(
            f"step4_3_inflection_b_extend_factor must be > 1.0, got {b_extend_factor}"
        )
    if max_b_extensions < 0:
        raise ValueError(
            f"step4_3_inflection_max_b_extensions must be >= 0, got {max_b_extensions}"
        )

    if ckpt_dir.is_dir() and metadata_path.exists():
        if _metadata_matches_current_settings(
            metadata_path,
            curvature_ratio=curvature_ratio,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            ytol=ytol,
            max_iter=max_iter,
            extend_b_if_no_inflection=extend_b_if_no_inflection,
            b_extend_factor=b_extend_factor,
            max_b_extensions=max_b_extensions,
        ):
            log("[SKIP] Step 4.3 (state-dict inflection): checkpoint already prepared")
            return ckpt_dir, True

        log(
            "[RECOMPUTE] Step 4.3 outputs predate current settings "
            f"(curvature_ratio={curvature_ratio:g}, "
            f"scan_range=[{lower_bound:g}, {upper_bound:g}], "
            f"ytol={ytol:g}, max_iter={max_iter}, "
            f"extend_b_if_no_inflection={extend_b_if_no_inflection}, "
            f"b_extend_factor={b_extend_factor:g}, max_b_extensions={max_b_extensions}); "
            "removing stale artifacts"
        )
        for stale_path in [out_dir, history_path, debug_plot_dir,
                           workdir / "step4_3_scan_subset.jsonl"]:
            if stale_path.exists() or stale_path.is_symlink():
                _remove_if_exists(stale_path)

    log("=" * 60)
    log(
        "STEP 4.3: State-dict Perturb Inflection "
        f"(filtered Step 4 data, {STEP4_3_SCAN_TOKENS}-token loss eval)"
    )
    log("=" * 60)

    input_data = workdir / "step4_continued_1k.jsonl"
    if not input_data.exists():
        raise FileNotFoundError(
            f"Step 4.3 input missing: {input_data}. Step 4 must produce "
            "step4_continued_1k.jsonl before Step 4.3 can run."
        )

    base_model = args.original_model_name
    reasoning_marker = getattr(args, "reasoning_marker", None)

    scan_input_data, scan_count, full_count = build_scan_subset(
        input_data=input_data,
        workdir=workdir,
        subset_size=STEP4_3_SCAN_SUBSET_SIZE,
        seed=int(args.seed),
        subset_filename="step4_3_scan_subset.jsonl",
    )
    if scan_input_data != input_data:
        log(
            f"Step 4.3 scan uses sampled subset: {scan_count}/{full_count} examples "
            f"(subset_size={STEP4_3_SCAN_SUBSET_SIZE})"
        )
    else:
        log(f"Step 4.3 scan uses full filtered dataset: {full_count} examples")

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        eval_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        eval_devices = ["cpu"]
    log(f"Step 4.3 evaluator devices: {eval_devices}")
    log(f"Step 4.3 perturbation tensor surface: {STEP4_3_PERTURBATION_SURFACE} (pinned)")
    log(f"Step 4.3 inflection curvature ratio: {curvature_ratio:g}")
    log(f"Step 4.3 inflection scan range: [{lower_bound:g}, {upper_bound:g}]")
    log(
        f"Step 4.3 inflection extend_b_if_no_inflection={extend_b_if_no_inflection}, "
        f"b_extend_factor={b_extend_factor:g}, max_b_extensions={max_b_extensions}"
    )

    evaluator = LLMEvaluator(
        base_model=base_model,
        perturbation_direction=None,
        data=None,
        device=eval_devices,
        seed=int(args.seed),
        max_eval_batch_size=16,
        perturbation_tensor_surface=STEP4_3_PERTURBATION_SURFACE,
    )
    memory_plan = getattr(evaluator, "memory_plan", {})
    if isinstance(memory_plan, dict):
        log(
            "Step 4.3 memory plan: "
            f"mode={memory_plan.get('placement_mode')}, "
            f"model_bf16_gib={memory_plan.get('model_size_bf16_gib')}, "
            f"selected_devices={memory_plan.get('selected_devices')}"
        )

    log("Step 4.3 tokenizing scan dataset for evaluator")
    scan_prep_start = time.time()
    scan_prepared_data = evaluator.prepare_data(
        dataset_path=scan_input_data,
        max_tokens=STEP4_3_SCAN_TOKENS,
        reasoning_marker=reasoning_marker,
    )
    log(
        f"Step 4.3 tokenized scan dataset: {len(scan_prepared_data)} examples "
        f"in {time.time() - scan_prep_start:.2f}s"
    )

    eval_cache: Dict[float, Dict] = load_eval_history(history_path)

    log("Step 4.3 starting inflection search")
    search_summary = search_inflection_points(
        evaluator=evaluator,
        prepared_data=scan_prepared_data,
        ytol=ytol,
        max_iter=max_iter,
        curvature_ratio=curvature_ratio,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        eval_cache=eval_cache,
        history_path=history_path,
        debug_plot_dir=debug_plot_dir,
        log_prefix="Step 4.3",
        title_prefix="Step 4.3 Inflection Debug",
        summary_label="Step 4.3",
        extend_b_if_no_inflection=extend_b_if_no_inflection,
        b_extend_factor=b_extend_factor,
        max_b_extensions=max_b_extensions,
    )
    search_summary["scan_dataset"] = {
        "path": str(scan_input_data),
        "used_subset": bool(scan_input_data != input_data),
        "sampled_count": int(scan_count),
        "full_count": int(full_count),
        "subset_size": int(STEP4_3_SCAN_SUBSET_SIZE),
    }

    chosen_scale = float(search_summary["x_opt"])
    chosen_key = scale_key(chosen_scale)
    chosen = eval_cache.get(chosen_key)
    if chosen is None:
        chosen = min(eval_cache.values(), key=lambda r: abs(float(r["scale"]) - chosen_scale))

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_ckpt_dir = LOCAL_CACHE / "perturbed_ckpts" / str(args.task) / "step4_3_checkpoint_inflection"
    if cache_ckpt_dir.exists():
        shutil.rmtree(cache_ckpt_dir)
    cache_ckpt_dir.parent.mkdir(parents=True, exist_ok=True)

    log(f"Step 4.3 materializing selected checkpoint to local cache: {cache_ckpt_dir}")
    materialize_start = time.time()
    evaluator.materialize_checkpoint(scale=float(chosen["scale"]), output_dir=cache_ckpt_dir)
    log(f"Step 4.3 finished checkpoint materialization in {time.time() - materialize_start:.2f}s")

    log(f"Step 4.3 syncing checkpoint symlinks into pipeline result: {ckpt_dir}")
    symlink_start = time.time()
    setup_model_symlinks(cache_ckpt_dir, ckpt_dir)
    log(f"Step 4.3 finished checkpoint symlink sync in {time.time() - symlink_start:.2f}s")

    try:
        evaluator.models_by_device.clear()
        evaluator.base_model = None
        evaluator.tokenizer = None
        evaluator.base_state_dict_by_device.clear()
        evaluator.perturbation_direction_by_device.clear()
        evaluator.base_state_dict_cpu.clear()
        evaluator.perturbation_direction.clear()
        del evaluator
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    log("Step 4.3 cleanup complete: released evaluator references to free GPUs")

    ckpt_rows = sorted(eval_cache.values(), key=lambda x: float(x["scale"]))
    metadata = {
        "seed": int(args.seed),
        "base_model": base_model,
        "perturbation_surface": STEP4_3_PERTURBATION_SURFACE,
        "scan_range": [float(lower_bound), float(upper_bound)],
        "ytol": float(ytol),
        "max_iter": int(max_iter),
        "curvature_ratio": float(curvature_ratio),
        "extend_b_if_no_inflection": bool(extend_b_if_no_inflection),
        "b_extend_factor": float(b_extend_factor),
        "max_b_extensions": int(max_b_extensions),
        "scan_tokens": int(STEP4_3_SCAN_TOKENS),
        "memory_plan": memory_plan,
        "search": search_summary,
        "history_file": str(history_path),
        "evaluations": [
            {
                "name": r["name"],
                "scale": float(r["scale"]),
                "mean_loss": float(r["mean_loss"]),
            }
            for r in ckpt_rows
        ],
        "selected": {
            "checkpoint": chosen["name"],
            "scale": float(chosen["scale"]),
            "mean_loss_scan": float(chosen["mean_loss"]),
            "local_cache_checkpoint_dir": str(cache_ckpt_dir),
            "pipeline_checkpoint_symlink_dir": str(ckpt_dir),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log(f"Saved Step 4.3 metadata to {metadata_path}")

    return ckpt_dir, False
