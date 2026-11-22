""""""

import json
import math
import shutil
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .inflection import (
    LLMEvaluator,
    compute_tensor_surface_fingerprint,
)
from .inflection_search import (
    build_scan_subset,
    load_eval_history,
    scale_key,
    search_inflection_points,
)
from .utils import (
    BASE_DIR,
    LOCAL_CACHE,
    get_num_token_for_difficulty,
    log,
    setup_model_symlinks,
)


INFLECTION_LOW = 0.0
INFLECTION_HIGH = 0.01
STEP2_SCAN_SUBSET_SIZE = 1000
DEFAULT_PERTURBATION_SURFACE = "named_parameters_and_buffers"
PERTURBATION_SURFACE_VERSION = 4
STEP2_FINAL_EVAL_SEMANTICS_VERSION = 1

STEP2_VIRTUAL_OVERLAY_FILENAME = "perturbation_overlay.json"
STEP2_INTERNAL_OVERLAY_FILENAME = "_perturbation_overlay.json"


def _remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _get_perturbation_surface(args) -> str:
    surface = str(getattr(args, "step2_perturbation_surface", DEFAULT_PERTURBATION_SURFACE))
    valid = {"state_dict", "named_parameters_and_buffers"}
    if surface not in valid:
        raise ValueError(
            f"step2_perturbation_surface must be one of {sorted(valid)}, got {surface!r}"
        )
    return surface


def _metadata_matches_current_step2_settings(
    metadata_path: Path,
    perturbation_surface: str,
    curvature_ratio: float,
    lower_bound: float,
    upper_bound: float,
    final_eval_from_saved_checkpoint: bool,
) -> bool:
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return False

    try:
        surface_version = int(metadata.get("perturbation_surface_version", 0))
    except (TypeError, ValueError):
        return False

    try:
        metadata_curvature_ratio = float(metadata.get("curvature_ratio", 0.1))
    except (TypeError, ValueError):
        return False

    try:
        semantics_ver = int(metadata.get("final_eval_semantics_version", 0))
    except (TypeError, ValueError):
        semantics_ver = 0
    if semantics_ver != STEP2_FINAL_EVAL_SEMANTICS_VERSION:
        return False

    meta_from_saved = bool(metadata.get("final_eval_loads_saved_checkpoint", False))
    if meta_from_saved != bool(final_eval_from_saved_checkpoint):
        return False
    scan_range = metadata.get("scan_range", [INFLECTION_LOW, INFLECTION_HIGH])
    try:
        metadata_lower_bound = float(scan_range[0])
        metadata_upper_bound = float(scan_range[1])
    except (TypeError, ValueError, IndexError):
        return False

    if (
        perturbation_surface == "named_parameters_and_buffers"
        and not bool(metadata.get("perturbed_checkpoint_saved", False))
        and metadata.get("final_loss_method") != "optimized_in_memory_overlay"
    ):
        return False
    if (
        perturbation_surface == "named_parameters_and_buffers"
        and not bool(metadata.get("perturbed_checkpoint_saved", False))
    ):
        descriptor = metadata.get("selected", {}).get("virtual_checkpoint_descriptor")
        if not descriptor or not Path(str(descriptor)).exists():
            return False

    return (
        metadata.get("perturbation_surface") == perturbation_surface
        and surface_version == PERTURBATION_SURFACE_VERSION
        and metadata_curvature_ratio == float(curvature_ratio)
        and metadata_lower_bound == float(lower_bound)
        and metadata_upper_bound == float(upper_bound)
    )


def _write_perturbation_overlay_descriptor(
    evaluator: LLMEvaluator,
    *,
    descriptor_path: Path,
    base_model: str,
    perturbation_surface: str,
    chosen_scale: float,
    seed: int,
) -> None:
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_payload = None
    try:
        evaluator._apply_scale_to_model(
            evaluator.base_model,
            device=evaluator.primary_device,
            scale=float(chosen_scale),
        )
        fingerprint = compute_tensor_surface_fingerprint(
            evaluator.base_model,
            tensor_surface=perturbation_surface,
        )
        descriptor_payload = {
            "schema_version": 1,
            "kind": "step2_perturbation_overlay",
            "base_model": str(base_model),
            "perturbation_surface": str(perturbation_surface),
            "perturbation_surface_version": int(PERTURBATION_SURFACE_VERSION),
            "seed": int(seed),
            "scale": float(chosen_scale),
            "fingerprint": fingerprint,
        }
    finally:
        evaluator._restore_model_to_base(evaluator.base_model, device=evaluator.primary_device)

    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor_payload, f, indent=2)


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0 or sy == 0:
        return float("nan")
    return float(sum(a * b for a, b in zip(dx, dy)) / (sx * sy))


def _merge_physics_difficulty(value) -> Optional[int]:
    if value is None:
        return None
    try:
        difficulty = int(value)
    except (TypeError, ValueError):
        return None

    if difficulty in (1, 2):
        return 2
    if difficulty in (8, 9, 10):
        return 8
    if 3 <= difficulty <= 7:
        return difficulty
    return None


def _load_physics_eval_train_difficulties() -> List[Optional[int]]:
    path = BASE_DIR / "dataset" / "PhysicsEval" / "train.jsonl"
    labels: List[Optional[int]] = []
    if not path.exists():
        return labels

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                labels.append(None)
                continue
            labels.append(_merge_physics_difficulty(item.get("problem_difficulty")))
    return labels


def _compute_m23k_correlation(losses: torch.Tensor) -> Dict:
    score_path = BASE_DIR / "dataset" / "m23k-pd" / "gemini-response.jsonl"
    if not score_path.exists():
        return {"type": "m23k", "error": f"score file missing: {score_path}"}

    scores: List[int] = []
    with open(score_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            scores.append(data.get("score"))

    arr = losses.detach().cpu().float().numpy().tolist()
    grouped = {d: [] for d in range(1, 6)}
    for i in range(min(len(arr), len(scores))):
        s = scores[i]
        if s is None or s in (-1, 0):
            continue
        s = int(s)
        if s in grouped:
            grouped[s].append(float(arr[i]))

    means = {str(d): (float(np.mean(grouped[d])) if grouped[d] else float("nan")) for d in range(1, 6)}
    valid = [(float(d), means[str(d)]) for d in range(1, 6) if not math.isnan(means[str(d)])]
    corr = _pearson([x for x, _ in valid], [y for _, y in valid]) if len(valid) >= 2 else float("nan")

    return {
        "type": "m23k",
        "score_file": str(score_path),
        "per_difficulty_mean_loss": means,
        "correlation": corr,
    }


def _compute_physics_correlation(prompt_source: Path, losses: torch.Tensor) -> Dict:
    if not prompt_source.exists():
        return {"type": "physics", "error": f"prompt source missing: {prompt_source}"}

    labels: List[Optional[int]] = []
    raw_label_counts: Dict[int, int] = {}
    merged_label_counts: Dict[int, int] = {}
    source_counts: Dict[str, int] = {}
    fallback_labels = _load_physics_eval_train_difficulties()
    fallback_used = 0

    with open(prompt_source, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            item = json.loads(line)

            source_key = str(item.get("domain") or item.get("dataset") or "unknown")
            source_counts[source_key] = source_counts.get(source_key, 0) + 1

            raw_difficulty = item.get("problem_difficulty")
            if raw_difficulty is not None:
                try:
                    raw_int = int(raw_difficulty)
                    raw_label_counts[raw_int] = raw_label_counts.get(raw_int, 0) + 1
                except (TypeError, ValueError):
                    pass

            merged = _merge_physics_difficulty(item.get("problem_difficulty_merged"))
            if merged is None:
                merged = _merge_physics_difficulty(raw_difficulty)
            if (
                merged is None
                and source_key == "PhysicsEval"
                and row_idx < len(fallback_labels)
            ):
                merged = fallback_labels[row_idx]
                if merged is not None:
                    fallback_used += 1

            if merged is not None:
                merged_label_counts[merged] = merged_label_counts.get(merged, 0) + 1
            labels.append(merged)

    arr = losses.detach().cpu().float().numpy().tolist()
    grouped: Dict[int, List[float]] = {}
    used_count = 0
    missing_label_count = 0
    loss_label_length_mismatch = abs(len(arr) - len(labels))

    for i in range(min(len(arr), len(labels))):
        lab = labels[i]
        if lab is None:
            missing_label_count += 1
            continue
        grouped.setdefault(int(lab), []).append(float(arr[i]))
        used_count += 1

    if len(labels) > len(arr):
        missing_label_count += len(labels) - len(arr)

    means = {str(k): float(np.mean(v)) for k, v in sorted(grouped.items(), key=lambda kv: kv[0])}
    counts = {str(k): len(v) for k, v in sorted(grouped.items(), key=lambda kv: kv[0])}
    xs = [float(k) for k in means.keys()]
    ys = [float(v) for v in means.values()]
    corr = _pearson(xs, ys) if len(xs) >= 2 else None
    if isinstance(corr, float) and math.isnan(corr):
        corr = None

    return {
        "type": "physics",
        "prompt_source": str(prompt_source),
        "difficulty_source": (
            "problem_difficulty_merged with fallback to merged problem_difficulty, "
            "then PhysicsEval train.jsonl by row index for unlabeled PhysicsEval rows"
        ),
        "merge_rule": {"1": 2, "2": 2, "3-7": "unchanged", "8": 8, "9": 8, "10": 8},
        "num_losses": len(arr),
        "num_labels": len(labels),
        "num_used": used_count,
        "num_skipped_missing_label": missing_label_count,
        "num_labels_from_physicseval_fallback": fallback_used,
        "loss_label_length_mismatch": loss_label_length_mismatch,
        "source_counts": dict(sorted(source_counts.items())),
        "raw_difficulty_counts": {str(k): v for k, v in sorted(raw_label_counts.items())},
        "merged_difficulty_counts": {str(k): v for k, v in sorted(merged_label_counts.items())},
        "per_difficulty_counts": counts,
        "per_difficulty_mean_loss": means,
        "correlation": corr,
    }


def _maybe_write_correlation(config: Dict, args, selected_loss_file: Path, out_dir: Path) -> None:
    try:
        losses = torch.load(selected_loss_file, map_location="cpu")
        if not isinstance(losses, torch.Tensor):
            losses = torch.tensor(losses)

        task_lower = str(args.task).lower()
        source_lower = str(config.get("prompt_source", "")).lower()

        payload = None
        if task_lower == "m23k":
            payload = _compute_m23k_correlation(losses)
        elif "physics" in task_lower or "physicseval" in source_lower:
            payload = _compute_physics_correlation(Path(config["prompt_source"]), losses)

        if payload is not None:
            corr_path = out_dir / "correlation_analysis.json"
            with open(corr_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            log(f"Saved correlation analysis to {corr_path}")
    except Exception as exc:
        log(f"WARNING: Failed to compute correlation analysis: {exc}")


def run(config: Dict, workdir: Path, args):
    """"""
    difficulty_tokens = get_num_token_for_difficulty(args)
    step2_loss_dir = workdir / "step2_loss_100tokens"
    loss_target_dir = step2_loss_dir / "checkpoint-inflection"

    selected_loss_file = loss_target_dir / f"loss-{difficulty_tokens}token-r-losses.pt"
    metadata_path = step2_loss_dir / "inflection_metadata.json"
    history_path = workdir / "step2_eval_history.json"
    perturbation_surface = _get_perturbation_surface(args)
    force_save_perturbed_checkpoint = bool(
        getattr(args, "step2_force_save_perturbed_checkpoint", False)
    )
    final_eval_from_saved_checkpoint = bool(
        getattr(args, "step2_final_eval_from_saved_checkpoint", False)
    )
    save_perturbed_checkpoint = (
        perturbation_surface == "state_dict" or force_save_perturbed_checkpoint
    )

    curvature_ratio = float(getattr(args, "step2_inflection_curvature_ratio", 0.1))
    lower_bound = float(getattr(args, "step2_inflection_lower_bound", INFLECTION_LOW))
    upper_bound = float(getattr(args, "step2_inflection_upper_bound", INFLECTION_HIGH))
    if lower_bound < 0:
        raise ValueError(f"step2_inflection_lower_bound must be >= 0, got {lower_bound}")
    if upper_bound <= lower_bound:
        raise ValueError(
            "step2_inflection_upper_bound must be greater than "
            f"step2_inflection_lower_bound, got {upper_bound} <= {lower_bound}"
        )

    if selected_loss_file.exists() and metadata_path.exists():
        if _metadata_matches_current_step2_settings(
            metadata_path,
            perturbation_surface,
            curvature_ratio,
            lower_bound,
            upper_bound,
            final_eval_from_saved_checkpoint,
        ):
            log("[SKIP] Step 2 (difficulty perturbation): inflection checkpoint already prepared")
            _maybe_write_correlation(config, args, selected_loss_file, step2_loss_dir)
            return selected_loss_file, True

        log(
            "[RECOMPUTE] Step 2 outputs predate current Step 2 settings "
            f"(surface={perturbation_surface} v{PERTURBATION_SURFACE_VERSION}, "
            f"curvature_ratio={curvature_ratio:g}, "
            f"scan_range=[{lower_bound:g}, {upper_bound:g}]); "
            "removing stale inflection artifacts"
        )
        for stale_path in [
            step2_loss_dir,
            history_path,
            workdir / "step2_scan_subset.jsonl",
            workdir / "step2_debug_inflection_plot",
        ]:
            if stale_path.exists() or stale_path.is_symlink():
                _remove_if_exists(stale_path)

    log("=" * 60)
    log("STEP 2: Perturb Difficulty Model (Find Inflection)")
    log("=" * 60)

    base_model = args.original_model_name
    input_data = workdir / "step1_generation_100tokens.jsonl"
    reasoning_marker = getattr(args, "reasoning_marker", None)

    scan_input_data, scan_count, full_count = build_scan_subset(
        input_data=input_data,
        workdir=workdir,
        subset_size=STEP2_SCAN_SUBSET_SIZE,
        seed=int(args.seed),
        subset_filename="step2_scan_subset.jsonl",
    )
    if scan_input_data != input_data:
        log(
            f"Step 2 scan uses sampled subset: {scan_count}/{full_count} examples "
            f"(subset_size={STEP2_SCAN_SUBSET_SIZE})"
        )
    else:
        log(f"Step 2 scan uses full dataset: {full_count} examples")

    ytol = float(getattr(args, "step2_inflection_ytol", 0.2))
    max_iter = int(getattr(args, "step2_inflection_max_iter", 8))
    if ytol <= 0:
        raise ValueError(f"step2_inflection_ytol must be > 0, got {ytol}")
    if max_iter <= 0:
        raise ValueError(f"step2_inflection_max_iter must be > 0, got {max_iter}")

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        eval_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        eval_devices = ["cpu"]
    log(f"Step 2 evaluator devices: {eval_devices}")
    log(f"Step 2 perturbation tensor surface: {perturbation_surface}")
    log(f"Step 2 inflection curvature ratio: {curvature_ratio:g}")
    log(f"Step 2 inflection scan range: [{lower_bound:g}, {upper_bound:g}]")
    if perturbation_surface != "state_dict" and force_save_perturbed_checkpoint:
        log(
            "WARNING: Step 2 force-save is enabled with named_parameters_and_buffers; "
            "non-persistent buffers cannot be serialized by save_pretrained"
        )
    elif not save_perturbed_checkpoint:
        log(
            "Step 2 will not save a perturbed model checkpoint because the "
            "selected tensor surface includes named buffers that may be non-persistent"
        )

    evaluator = LLMEvaluator(
        base_model=base_model,
        perturbation_direction=None,
        data=None,
        device=eval_devices,
        seed=int(args.seed),
        max_eval_batch_size=16,
        perturbation_tensor_surface=perturbation_surface,
    )
    print(f"Initialized LLMEvaluator with model {base_model} on devices {eval_devices}")
    memory_plan = getattr(evaluator, "memory_plan", {})
    if isinstance(memory_plan, dict):
        log(
            "Step 2 memory plan: "
            f"mode={memory_plan.get('placement_mode')}, "
            f"model_bf16_gib={memory_plan.get('model_size_bf16_gib')}, "
            f"required_4x_gib={memory_plan.get('required_single_gpu_gib')}, "
            f"selected_devices={memory_plan.get('selected_devices')}, "
            f"keep_resident_by_device={memory_plan.get('keep_resident_by_device')}"
        )

    log("Step 2 tokenizing scan dataset for evaluator")
    scan_prep_start = time.time()
    scan_prepared_data = evaluator.prepare_data(
        dataset_path=scan_input_data,
        max_tokens=difficulty_tokens,
        reasoning_marker=reasoning_marker,
    )
    log(
        f"Step 2 tokenized scan dataset: {len(scan_prepared_data)} examples "
        f"in {time.time() - scan_prep_start:.2f}s"
    )

    eval_cache: Dict[float, Dict] = load_eval_history(history_path)

    log("Step 2 starting inflection search")
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
        debug_plot_dir=history_path.parent / "step2_debug_inflection_plot",
        log_prefix="Step 2",
        title_prefix="Step 2 Inflection Debug",
        summary_label="Step 2",
    )
    search_summary["scan_dataset"] = {
        "path": str(scan_input_data),
        "used_subset": bool(scan_input_data != input_data),
        "sampled_count": int(scan_count),
        "full_count": int(full_count),
        "subset_size": int(STEP2_SCAN_SUBSET_SIZE),
    }

    chosen_scale = float(search_summary["x_opt"])
    chosen_key = scale_key(chosen_scale)
    chosen = eval_cache.get(chosen_key)
    if chosen is None:
        chosen = min(eval_cache.values(), key=lambda r: abs(float(r["scale"]) - chosen_scale))

    ckpt_rows = sorted(eval_cache.values(), key=lambda x: float(x["scale"]))


    loss_target_dir.mkdir(parents=True, exist_ok=True)

    cache_ckpt_dir: Optional[Path] = None
    selected_loss_file = loss_target_dir / f"loss-{difficulty_tokens}token-r-losses.pt"
    final_loss_method = "optimized_checkpoint"
    virtual_checkpoint_descriptor: Optional[Path] = None
    internal_overlay_descriptor: Optional[Path] = None

    if save_perturbed_checkpoint:
        cache_ckpt_dir = LOCAL_CACHE / "perturbed_ckpts" / str(args.task) / "checkpoint-inflection"
        if cache_ckpt_dir.exists():
            shutil.rmtree(cache_ckpt_dir)
        cache_ckpt_dir.parent.mkdir(parents=True, exist_ok=True)

        log(f"Step 2 materializing selected checkpoint to local cache: {cache_ckpt_dir}")
        materialize_start = time.time()
        evaluator.materialize_checkpoint(scale=float(chosen["scale"]), output_dir=cache_ckpt_dir)
        log(f"Step 2 finished checkpoint materialization in {time.time() - materialize_start:.2f}s")

        log(f"Step 2 syncing checkpoint symlinks into pipeline result: {loss_target_dir}")
        symlink_start = time.time()
        setup_model_symlinks(cache_ckpt_dir, loss_target_dir)
        log(f"Step 2 finished checkpoint symlink sync in {time.time() - symlink_start:.2f}s")

        if not final_eval_from_saved_checkpoint:
            internal_overlay_descriptor = loss_target_dir / STEP2_INTERNAL_OVERLAY_FILENAME
            log(
                "Step 2 writing internal perturbation descriptor for final evaluation: "
                f"{internal_overlay_descriptor}"
            )
            descriptor_write_start = time.time()
            _write_perturbation_overlay_descriptor(
                evaluator,
                descriptor_path=internal_overlay_descriptor,
                base_model=base_model,
                perturbation_surface=perturbation_surface,
                chosen_scale=float(chosen["scale"]),
                seed=int(args.seed),
            )
            log(
                "Step 2 wrote internal perturbation descriptor in "
                f"{time.time() - descriptor_write_start:.2f}s"
            )
            final_loss_method = "optimized_in_memory_overlay"

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
        log("Step 2 cleanup complete: released evaluator references to free GPUs")

        log(
            "Step 2 starting final selected-scale loss evaluation using optimized script: "
            f"scale={float(chosen['scale']):.8f}, examples={full_count}"
        )
        final_eval_start = time.time()

        import subprocess
        import sys
        get_mean_loss_script = BASE_DIR / "m1" / "src" / "process_loss" / "get_mean_loss_optimized.py"

        cmd = [
            sys.executable,
            str(get_mean_loss_script),
        ]
        if final_eval_from_saved_checkpoint:
            cmd.extend(["--model_path", str(loss_target_dir)])
        else:
            cmd.extend(
                [
                    "--model_path",
                    str(base_model),
                    "--perturbation_overlay_descriptor",
                    str(internal_overlay_descriptor),
                ]
            )
        cmd.extend(
            [
                "--dataset_path",
                str(input_data),
                "--max_tokens",
                str(difficulty_tokens),
                "--output_name",
                str(selected_loss_file),
            ]
        )
        if reasoning_marker:
            cmd.extend(["--reasoning_marker", reasoning_marker])

        subprocess.run(cmd, check=True)

        if not selected_loss_file.exists():
            raise FileNotFoundError(f"Selected inflection loss file missing: {selected_loss_file}")

        try:
            final_losses = torch.load(selected_loss_file, map_location="cpu")
            if isinstance(final_losses, torch.Tensor):
                full_mean_loss = float(torch.mean(final_losses.float()))
            else:
                full_mean_loss = float(np.mean(final_losses))
        except Exception as e:
            log(f"WARNING: failed to read mean loss from final output: {e}")
            full_mean_loss = 0.0
    else:
        final_loss_method = "optimized_in_memory_overlay"
        virtual_checkpoint_descriptor = loss_target_dir / STEP2_VIRTUAL_OVERLAY_FILENAME
        stale_cache_ckpt_dir = LOCAL_CACHE / "perturbed_ckpts" / str(args.task) / "checkpoint-inflection"
        if stale_cache_ckpt_dir.exists():
            shutil.rmtree(stale_cache_ckpt_dir)

        log(
            "Step 2 starting final selected-scale loss evaluation using optimized "
            "in-memory perturbation overlay "
            f"(no checkpoint save): scale={float(chosen['scale']):.8f}, examples={full_count}"
        )
        final_eval_start = time.time()

        log(f"Step 2 writing virtual perturbation descriptor: {virtual_checkpoint_descriptor}")
        descriptor_start = time.time()
        _write_perturbation_overlay_descriptor(
            evaluator,
            descriptor_path=virtual_checkpoint_descriptor,
            base_model=base_model,
            perturbation_surface=perturbation_surface,
            chosen_scale=float(chosen["scale"]),
            seed=int(args.seed),
        )
        log(
            "Step 2 wrote virtual perturbation descriptor in "
            f"{time.time() - descriptor_start:.2f}s"
        )

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
        log("Step 2 cleanup complete: released evaluator references to free GPUs")

        import subprocess
        import sys
        get_mean_loss_script = BASE_DIR / "m1" / "src" / "process_loss" / "get_mean_loss_optimized.py"

        cmd = [
            sys.executable,
            str(get_mean_loss_script),
            "--model_path", str(base_model),
            "--dataset_path", str(input_data),
            "--max_tokens", str(difficulty_tokens),
            "--output_name", str(selected_loss_file),
            "--perturbation_overlay_descriptor", str(virtual_checkpoint_descriptor),
        ]
        if reasoning_marker:
            cmd.extend(["--reasoning_marker", reasoning_marker])

        subprocess.run(cmd, check=True)

        if not selected_loss_file.exists():
            raise FileNotFoundError(f"Selected inflection loss file missing: {selected_loss_file}")

        try:
            final_losses = torch.load(selected_loss_file, map_location="cpu")
            if isinstance(final_losses, torch.Tensor):
                full_mean_loss = float(torch.mean(final_losses.float()))
            else:
                full_mean_loss = float(np.mean(final_losses))
        except Exception as e:
            log(f"WARNING: failed to read mean loss from final output: {e}")
            full_mean_loss = 0.0

    log(
        f"Step 2 completed final selected-scale loss evaluation in {time.time() - final_eval_start:.2f}s; "
        f"mean_loss_full={float(full_mean_loss):.8f}"
    )

    loads_saved_for_final = bool(
        save_perturbed_checkpoint and final_eval_from_saved_checkpoint
    )
    metadata = {
        "seed": int(args.seed),
        "base_model": base_model,
        "perturbation_surface": perturbation_surface,
        "perturbation_surface_version": PERTURBATION_SURFACE_VERSION,
        "final_eval_semantics_version": STEP2_FINAL_EVAL_SEMANTICS_VERSION,
        "perturbed_checkpoint_saved": bool(save_perturbed_checkpoint),
        "final_loss_method": final_loss_method,
        "final_eval_loads_saved_checkpoint": loads_saved_for_final,
        "force_save_perturbed_checkpoint": bool(force_save_perturbed_checkpoint),
        "saved_checkpoint_may_omit_nonpersistent_buffers": bool(
            perturbation_surface != "state_dict" and save_perturbed_checkpoint
        ),
        "scan_range": [float(lower_bound), float(upper_bound)],
        "tolerance_y_axis": float(ytol),
        "curvature_ratio": float(curvature_ratio),
        "difficulty_tokens": int(difficulty_tokens),
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
            "mean_loss_full": float(full_mean_loss),
            "portable_loss_dir": str(loss_target_dir),
            "local_cache_checkpoint_dir": str(cache_ckpt_dir) if cache_ckpt_dir is not None else None,
            "pipeline_checkpoint_symlink_dir": str(loss_target_dir) if save_perturbed_checkpoint else None,
            "virtual_checkpoint_descriptor": (
                str(virtual_checkpoint_descriptor)
                if virtual_checkpoint_descriptor is not None
                else None
            ),
            "internal_perturbation_overlay_descriptor": (
                str(internal_overlay_descriptor)
                if internal_overlay_descriptor is not None
                else None
            ),
            "loss_dir": str(loss_target_dir),
            "loss_file": str(selected_loss_file),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log(f"Saved Step 2 metadata to {metadata_path}")

    _maybe_write_correlation(config, args, selected_loss_file, step2_loss_dir)

    return selected_loss_file, False
