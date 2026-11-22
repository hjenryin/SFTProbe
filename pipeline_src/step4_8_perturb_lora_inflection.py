""""""

from __future__ import annotations

import contextlib
import gc
import json
import queue as queue_module
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .inflection import InflectionPointFinder, LLMEvaluator, _parse_cuda_index
from .inflection_plot import (
    write_debug_inflection_plot_snapshot,
    write_debug_inflection_plots,
)
from .lora_decomp import decompose_lora, save_rank1_adapter
from .step4_8_perturb_lora import (
    _discover_over_checkpoints,
    _ix_to_tag,
    _parse_ix_values,
    _write_sampled_subset,
)
from .utils import BASE_DIR, log

_gpu_utils_root = str(BASE_DIR)
if _gpu_utils_root not in sys.path:
    sys.path.insert(0, _gpu_utils_root)
from gpu_utils import get_available_gpus


def _max_scale_bucket(dir_idx: int) -> float:
    """"""
    if dir_idx == 0:
        return 15.0
    if dir_idx == 1:
        return 100.0
    if dir_idx == 2:
        return 300.0
    if dir_idx == 3:
        return 400.0
    return 15.0 * (3.0 ** int(dir_idx))


def _linear_seed_points(
    upper_bound: float,
    n: int = 5,
    low: float = 1.0,
) -> List[float]:
    """"""
    if upper_bound <= 0:
        return []
    lo = float(low)
    hi = float(upper_bound)
    if hi <= lo:
        return [hi]
    n_points = max(2, int(n))
    return [float(x) for x in np.linspace(lo, hi, n_points).tolist()]


def _resolve_ratio(ix_str: str, base_loss: float, y_opt: float) -> float:
    """"""
    if ix_str == "infl":
        return float(max(y_opt / base_loss, 1.01))
    if ix_str == "half-infl":
        return float(max((base_loss + y_opt) / (2.0 * base_loss), 1.01))
    return float(ix_str)


def _history_u_region(
    history: Dict[float, float],
    post_min_decrease_tol: float,
    post_min_decrease_frac: float,
) -> List[Tuple[float, float]]:
    """"""
    if not history:
        return []
    items = sorted(history.items(), key=lambda kv: float(kv[0]))
    if len(items) < 2:
        return [(float(k), float(v)) for k, v in items]
    losses = [float(v) for _, v in items]
    m = min(range(len(losses)), key=lambda i: losses[i])
    cutoff = len(items)
    running_max_post_min = losses[m]
    for i in range(m + 1, len(items)):
        rel_threshold = float(post_min_decrease_frac) * running_max_post_min
        abs_threshold = running_max_post_min - float(post_min_decrease_tol)
        if losses[i] < rel_threshold and losses[i] < abs_threshold:
            cutoff = i
            break
        if losses[i] > running_max_post_min:
            running_max_post_min = losses[i]
    return [(float(k), float(v)) for k, v in items[:cutoff]]


def _find_scale_for_target(
    *,
    evaluator: LLMEvaluator,
    history: Dict[float, float],
    target_loss: float,
    base_loss: float,
    x_opt: float,
    y_opt: float,
    upper_bound: float,
    ytol: float,
    post_min_decrease_tol: float,
    post_min_decrease_frac: float,
    max_new_evals: int = 10,
    log_prefix: str = "",
) -> Tuple[float, float]:
    """"""
    abs_ytol = max(1e-9, float(ytol) * float(base_loss))

    if abs(y_opt - target_loss) <= abs_ytol and x_opt > 0:
        return float(x_opt), float(y_opt)

    for it in range(int(max_new_evals) + 1):
        valid = _history_u_region(
            history, post_min_decrease_tol, post_min_decrease_frac
        )
        bracket_pts = [(s, L) for s, L in valid if 0.0 < s <= upper_bound]
        if not bracket_pts:
            raise RuntimeError(
                f"{log_prefix}no usable (positive-scale) history for "
                f"target_loss={target_loss:.4f} (U-region size={len(valid)})"
            )

        best_pt = min(bracket_pts, key=lambda sl: abs(sl[1] - target_loss))
        if abs(best_pt[1] - target_loss) <= abs_ytol:
            return float(best_pt[0]), float(best_pt[1])

        below = [(s, L) for s, L in bracket_pts if L <= target_loss]
        above = [(s, L) for s, L in bracket_pts if L > target_loss]

        if below and above:
            s_lo, L_lo = max(below, key=lambda sl: sl[1])
            s_hi, L_hi = min(above, key=lambda sl: sl[1])
            if abs(s_hi - s_lo) < 1e-6:
                return float(s_hi), float(L_hi)
            denom = (L_hi - L_lo)
            if abs(denom) < 1e-12:
                s_new = 0.5 * (s_lo + s_hi)
            else:
                s_new = s_lo + (target_loss - L_lo) * (s_hi - s_lo) / denom
            lo, hi = (s_lo, s_hi) if s_lo < s_hi else (s_hi, s_lo)
            margin = 1e-6 * max(1.0, hi - lo)
            s_new = float(max(lo + margin, min(hi - margin, s_new)))
        elif below and not above:
            s_max_obs = max(s for s, _ in bracket_pts)
            L_max_obs = max(L for _, L in bracket_pts)
            if s_max_obs >= upper_bound - 1e-9:
                raise RuntimeError(
                    f"{log_prefix}target loss {target_loss:.4f} unreachable within "
                    f"U-region up to upper_bound={upper_bound:.4f}: "
                    f"max-observed loss={L_max_obs:.4f} at scale={s_max_obs:.4f}. "
                    f"The post-min-monotone guard likely truncated the search before "
                    f"the target was reached."
                )
            s_new = float(min(upper_bound, s_max_obs * 2.0))
            if s_new <= s_max_obs + 1e-9:
                s_new = float(upper_bound)
        else:
            s_min_above = min(s for s, _ in above)
            s_new = float(max(1e-6, s_min_above / 2.0))

        existing = list(history.keys())
        if existing:
            nearest = min(existing, key=lambda k: abs(k - s_new))
            if abs(nearest - s_new) < 1e-6:
                bump = max(1e-5, 0.05 * max(upper_bound - 0.0, 1.0))
                s_new = float(nearest + bump)
        s_new = float(max(1e-6, min(upper_bound, s_new)))

        log(
            f"{log_prefix}ix-target iter {it}: target={target_loss:.4f} "
            f"below={(float(below[-1][0]), float(below[-1][1])) if below else None} "
            f"above={(float(above[0][0]), float(above[0][1])) if above else None} "
            f"-> evaluate s={s_new:.4f}"
        )
        L_new = float(evaluator.evaluate(float(s_new)))
        history[float(s_new)] = float(L_new)

    valid = _history_u_region(
        history, post_min_decrease_tol, post_min_decrease_frac
    )
    bracket_pts = [(s, L) for s, L in valid if 0.0 < s <= upper_bound]
    if not bracket_pts:
        raise RuntimeError(
            f"{log_prefix}no bracket points after {max_new_evals} evals"
        )
    best_pt = min(bracket_pts, key=lambda sl: abs(sl[1] - target_loss))
    log(
        f"{log_prefix}exhausted budget ({max_new_evals} new evals); "
        f"returning closest observation s={best_pt[0]:.4f}, L={best_pt[1]:.4f} "
        f"(target={target_loss:.4f})"
    )
    return float(best_pt[0]), float(best_pt[1])


def _persist_direction_history(
    history_path: Path,
    *,
    dir_idx: int,
    base_loss: float,
    upper_bound_initial: float,
    upper_bound_final: float,
    b_extensions_applied: int,
    b_extension_history: List[Tuple[float, float]],
    x_opt: float,
    y_opt: float,
    eval_count: int,
    history: Dict[float, float],
    trace: List[Dict[str, float]],
) -> None:
    payload = {
        "dir_idx": int(dir_idx),
        "base_loss": float(base_loss),
        "upper_bound_initial": float(upper_bound_initial),
        "upper_bound_final": float(upper_bound_final),
        "b_extensions_applied": int(b_extensions_applied),
        "b_extension_history": [
            {"old_b": float(o), "new_b": float(n)}
            for (o, n) in b_extension_history
        ],
        "x_opt": float(x_opt),
        "y_opt": float(y_opt),
        "eval_count": int(eval_count),
        "history": {f"{float(k):.8f}": float(v) for k, v in sorted(history.items())},
        "trace": [
            {
                "iteration": int(t.get("iteration", 0)),
                "x": float(t.get("x", 0.0)),
                "y": float(t.get("y", 0.0)),
                "search_b": float(t.get("search_b", 0.0)),
                "b": float(t.get("b", t.get("search_b", 0.0))),
                "threshold": float(t.get("threshold", 0.0)),
                "extended_this_iter": bool(float(t.get("extended_this_iter", 0.0)) > 0.5),
            }
            for t in trace
        ],
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = history_path.with_suffix(history_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(history_path)


def _load_prior_history(history_path: Path, base_loss: float) -> Dict[float, float]:
    """"""
    prior: Dict[float, float] = {0.0: float(base_loss)}
    if not history_path.exists():
        return prior
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(
            f"[step4_8-inflection] could not parse prior history at "
            f"{history_path}: {exc}; starting fresh"
        )
        return prior

    prior_history = raw.get("history") or {}
    for k, v in prior_history.items():
        try:
            prior[float(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return prior


def _process_direction(
    *,
    dir_idx: int,
    evaluator: LLMEvaluator,
    shared: Dict[str, Any],
) -> Dict[str, Any]:
    """"""
    dir_label = f"dir{int(dir_idx)}"
    t0 = time.time()

    deltas: Dict[int, Dict[str, torch.Tensor]] = shared["deltas"]
    rank1_factors: Dict[int, Dict[str, Any]] = shared["rank1_factors"]
    adapter_template: Dict[str, Any] = shared["adapter_template"]
    tokenizer = shared["tokenizer"]
    out_dir: Path = shared["out_dir"]
    history_root_dir: Path = shared["history_root_dir"]
    debug_plot_root_dir: Path = shared["debug_plot_root_dir"]
    ix_values: List[str] = shared["ix_values"]
    base_loss: float = float(shared["base_loss"])
    ytol: float = float(shared["ytol"])
    finder_max_iter: int = int(shared["finder_max_iter"])
    post_min_decrease_tol: float = float(shared["post_min_decrease_tol"])
    post_min_decrease_frac: float = float(shared["post_min_decrease_frac"])
    curvature_ratio: float = float(shared["curvature_ratio"])
    seed_count: int = int(shared["seed_count"])
    extend_b_if_no_inflection: bool = bool(shared["extend_b_if_no_inflection"])
    b_extend_factor: float = float(shared["b_extend_factor"])
    max_b_extensions: int = int(shared["max_b_extensions"])

    upper = _max_scale_bucket(dir_idx)
    initial_scales = _linear_seed_points(upper, n=seed_count)
    history_path = history_root_dir / f"{dir_label}.json"
    prior_history = _load_prior_history(history_path, base_loss)

    evaluator.set_perturbation_direction(deltas[int(dir_idx)])

    plot_dir = debug_plot_root_dir / dir_label
    plot_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = f"Step 4.8 {dir_label} Inflection Debug"

    finder = InflectionPointFinder(
        evaluator=evaluator,
        a=1.0,
        b=float(upper),
        history=prior_history,
        debug_snapshot_callback=lambda snap: write_debug_inflection_plot_snapshot(
            snap, plot_dir, title_prefix=title_prefix
        ),
        curvature_ratio=curvature_ratio,
        enforce_post_min_monotone=True,
        post_min_decrease_tol=post_min_decrease_tol,
        post_min_decrease_frac=post_min_decrease_frac,
        extend_b_if_no_inflection=extend_b_if_no_inflection,
        b_extend_factor=b_extend_factor,
        max_b_extensions=max_b_extensions,
    )

    log(
        f"[{dir_label}] start: upper={upper:.4f}, "
        f"initial_scales={[round(s, 4) for s in initial_scales]}, "
        f"prior_history={len(prior_history)}, "
        f"post_min_decrease_frac={post_min_decrease_frac:.4f}, "
        f"post_min_decrease_tol={post_min_decrease_tol:.2e}, "
        f"extend_b_if_no_inflection={extend_b_if_no_inflection}, "
        f"b_extend_factor={b_extend_factor}, max_b_extensions={max_b_extensions}"
    )
    x_opt, y_opt = finder.run(
        ytol=float(ytol),
        max_iter=int(finder_max_iter),
        initial_points=initial_scales,
    )
    log(
        f"[{dir_label}] inflection: x_opt={x_opt:.4f}, y_opt={y_opt:.4f}, "
        f"evals={finder.eval_count}, total_history={len(finder.history)}, "
        f"b_final={finder.b:.4f} (initial={finder.b_initial:.4f}), "
        f"b_extensions={finder.b_extensions_applied}"
    )

    write_debug_inflection_plots(
        finder,
        plot_dir,
        title_prefix=title_prefix,
        summary_label=f"Step 4.8 {dir_label}",
    )

    _persist_direction_history(
        history_path,
        dir_idx=int(dir_idx),
        base_loss=float(base_loss),
        upper_bound_initial=float(upper),
        upper_bound_final=float(finder.b),
        b_extensions_applied=int(finder.b_extensions_applied),
        b_extension_history=list(finder.b_extension_history),
        x_opt=float(x_opt),
        y_opt=float(y_opt),
        eval_count=int(finder.eval_count),
        history=finder.history,
        trace=finder.trace,
    )

    ix_results: List[Dict[str, Any]] = []
    for ix_str in ix_values:
        ratio = _resolve_ratio(ix_str, float(base_loss), float(y_opt))
        target_loss = float(base_loss) * float(ratio)
        ix_tag = _ix_to_tag(ratio)
        out_path = out_dir / f"{dir_label}_over_{ix_tag}"
        log(
            f"[{dir_label} {ix_str}] ratio={ratio:.4f} -> target_loss={target_loss:.4f}"
            f" -> {out_path.name}"
        )
        final_scale, final_loss = _find_scale_for_target(
            evaluator=evaluator,
            history=finder.history,
            target_loss=target_loss,
            base_loss=float(base_loss),
            x_opt=float(x_opt),
            y_opt=float(y_opt),
            upper_bound=float(finder.b),
            ytol=float(ytol),
            post_min_decrease_tol=float(post_min_decrease_tol),
            post_min_decrease_frac=float(post_min_decrease_frac),
            log_prefix=f"[{dir_label} {ix_str}] ",
        )
        save_rank1_adapter(
            output_dir=out_path,
            adapter_template=adapter_template,
            rank1_factors_for_direction=rank1_factors[int(dir_idx)],
            scale=float(final_scale),
            tokenizer=tokenizer,
        )
        ix_results.append(
            {
                "ix_str": str(ix_str),
                "ratio": float(ratio),
                "target_loss": float(target_loss),
                "final_scale": float(final_scale),
                "final_loss": float(final_loss),
                "output_dir": str(out_path),
            }
        )

    _persist_direction_history(
        history_path,
        dir_idx=int(dir_idx),
        base_loss=float(base_loss),
        upper_bound_initial=float(upper),
        upper_bound_final=float(finder.b),
        b_extensions_applied=int(finder.b_extensions_applied),
        b_extension_history=list(finder.b_extension_history),
        x_opt=float(x_opt),
        y_opt=float(y_opt),
        eval_count=int(finder.eval_count),
        history=finder.history,
        trace=finder.trace,
    )

    elapsed = time.time() - t0
    log(f"[{dir_label}] done in {elapsed:.1f}s; {len(ix_results)} ix targets saved")
    return {
        "dir_idx": int(dir_idx),
        "direction": dir_label,
        "x_opt": float(x_opt),
        "y_opt": float(y_opt),
        "upper_bound_initial": float(upper),
        "upper_bound_final": float(finder.b),
        "b_extensions_applied": int(finder.b_extensions_applied),
        "b_extension_history": [
            {"old_b": float(o), "new_b": float(n)}
            for (o, n) in finder.b_extension_history
        ],
        "eval_count": int(finder.eval_count),
        "elapsed_seconds": float(elapsed),
        "ix_targets": ix_results,
    }


def _worker_loop(
    *,
    worker_idx: int,
    gpu_str: str,
    task_queue: "queue_module.Queue[int]",
    shared: Dict[str, Any],
    results: List[Dict[str, Any]],
    results_lock: threading.Lock,
    error_event: threading.Event,
) -> None:
    """"""
    gpu_idx = _parse_cuda_index(gpu_str)
    device_ctx: contextlib.AbstractContextManager[Any]
    if gpu_idx is not None and torch.cuda.is_available():
        device_ctx = torch.cuda.device(gpu_idx)
    else:
        device_ctx = contextlib.nullcontext()

    evaluator: LLMEvaluator | None = None
    try:
        with device_ctx:
            model_init_lock = shared["model_init_lock"]
            log(
                f"[worker {worker_idx} gpu={gpu_str}] waiting for model-init "
                f"lock to construct LLMEvaluator"
            )
            with model_init_lock:
                log(
                    f"[worker {worker_idx} gpu={gpu_str}] initializing "
                    f"LLMEvaluator (current_cuda_device="
                    f"{torch.cuda.current_device() if torch.cuda.is_available() else 'n/a'})"
                )
                evaluator = LLMEvaluator(
                    base_model=shared["base_model_path"],
                    perturbation_direction={},
                    data=shared["prepared_data"],
                    device=[gpu_str],
                    seed=int(shared["seed"]),
                    max_eval_batch_size=int(shared["max_eval_batch_size"]),
                )
            evaluator.data = shared["prepared_data"]
            log(
                f"[worker {worker_idx} gpu={gpu_str}] LLMEvaluator ready; "
                f"released model-init lock"
            )

            while not error_event.is_set():
                try:
                    dir_idx = task_queue.get_nowait()
                except queue_module.Empty:
                    break

                log(f"[worker {worker_idx} gpu={gpu_str}] dequeued dir{dir_idx}")
                result = _process_direction(
                    dir_idx=int(dir_idx),
                    evaluator=evaluator,
                    shared=shared,
                )
                with results_lock:
                    results.append(result)

            log(f"[worker {worker_idx} gpu={gpu_str}] queue drained")
    except BaseException:
        error_event.set()
        raise
    finally:
        if evaluator is not None:
            try:
                evaluator.models_by_device.clear()
                evaluator.base_state_dict_by_device.clear()
                evaluator.perturbation_direction_by_device.clear()
                evaluator.base_state_dict_cpu.clear()
                evaluator.perturbation_direction.clear()
                evaluator.base_model = None
            except Exception:
                pass
            del evaluator
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def run(config: Dict[str, Any], workdir: Path, args) -> Tuple[Path, bool]:
    log("=" * 60)
    log("STEP 4.8 (inflection): Perturb LoRA (Over Only)")
    log("=" * 60)

    lora_adapter_path = workdir / "step4_7_lora" / "adapter"
    if not lora_adapter_path.exists():
        legacy_link = workdir / "step4_7_lora_adapter"
        if legacy_link.exists():
            lora_adapter_path = (
                legacy_link.resolve() if legacy_link.is_symlink() else legacy_link
            )
    if not lora_adapter_path.exists():
        raise FileNotFoundError(
            "Missing Step 4.7 LoRA adapter path. "
            f"Checked: {workdir / 'step4_7_lora' / 'adapter'}"
        )

    source_dataset_file = workdir / "step4_continued_1k.jsonl"
    if not source_dataset_file.exists():
        raise FileNotFoundError(f"Missing Step 4.8 source dataset: {source_dataset_file}")

    sampled_subset_file = workdir / "step4_8_sampled_subset.jsonl"
    sampled_count = _write_sampled_subset(
        source_dataset_file,
        sampled_subset_file,
        seed=int(args.seed),
        max_samples=int(args.partial_x),
    )
    log(
        f"Step 4.8 using {sampled_count} sampled examples "
        f"(partial_x={int(args.partial_x)}) from {source_dataset_file}"
    )

    ix_mode_raw = getattr(args, "lora_perturb_ix_values", None)
    if ix_mode_raw is None:
        raise ValueError(
            "Step 4.8 (inflection mode) requires explicit ix values via "
            "`args.lora_perturb_ix_values`"
        )
    ix_values = _parse_ix_values(ix_mode_raw)

    out_dir = workdir / "step4_8_perturbed_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = workdir / "step4_8_metadata.json"

    existing = _discover_over_checkpoints(out_dir, ix_values)
    if existing and metadata_file.exists():
        log("[SKIP] Step 4.8 (inflection): over checkpoints already exist")
        return out_dir, True

    ytol = float(getattr(args, "lora_perturb_inflection_ytol", 0.2))
    finder_max_iter = int(getattr(args, "lora_perturb_inflection_max_iter", 10))
    curvature_ratio = float(getattr(args, "lora_perturb_inflection_curvature_ratio", 0.1))
    post_min_decrease_tol = float(
        getattr(args, "lora_perturb_post_min_decrease_tol", 1e-4)
    )
    post_min_decrease_frac = float(
        getattr(args, "lora_perturb_post_min_decrease_frac", 0.8)
    )
    max_eval_batch_size = int(getattr(args, "lora_perturb_eval_batch_size", 16))
    gpu_threshold = float(getattr(args, "lora_perturb_gpu_threshold", 10.0))
    seed_count = int(getattr(args, "lora_perturb_seed_count", 5))
    extend_b_if_no_inflection = bool(
        getattr(args, "lora_perturb_extend_b_if_no_inflection", True)
    )
    b_extend_factor = float(getattr(args, "lora_perturb_b_extend_factor", 1.5))
    max_b_extensions = int(getattr(args, "lora_perturb_max_b_extensions", 3))
    if ytol <= 0:
        raise ValueError(f"lora_perturb_inflection_ytol must be > 0, got {ytol}")
    if finder_max_iter <= 0:
        raise ValueError(
            f"lora_perturb_inflection_max_iter must be > 0, got {finder_max_iter}"
        )
    if not (0.0 < post_min_decrease_frac < 1.0):
        raise ValueError(
            "lora_perturb_post_min_decrease_frac must be in (0, 1), got "
            f"{post_min_decrease_frac}"
        )
    if b_extend_factor <= 1.0:
        raise ValueError(
            f"lora_perturb_b_extend_factor must be > 1.0, got {b_extend_factor}"
        )
    if max_b_extensions < 0:
        raise ValueError(
            f"lora_perturb_max_b_extensions must be >= 0, got {max_b_extensions}"
        )

    max_length = 1000
    base_model_path = str(args.original_model_name)

    gpus = get_available_gpus(threshold=gpu_threshold)
    if not gpus:
        raise RuntimeError(
            "Step 4.8 (inflection) requires at least one GPU below "
            f"{gpu_threshold:.1f}% memory utilization"
        )
    gpu_strs = [f"cuda:{int(g)}" for g in gpus]
    log(f"Step 4.8 (inflection) available GPUs: {gpu_strs}")

    setup_gpu = gpu_strs[0]
    log(f"Step 4.8 loading base model on {setup_gpu} for decomposition and base-loss")
    setup_evaluator = LLMEvaluator(
        base_model=base_model_path,
        perturbation_direction={},
        data=None,
        device=[setup_gpu],
        seed=int(args.seed),
        max_eval_batch_size=max_eval_batch_size,
    )

    reasoning_marker = getattr(args, "reasoning_marker", None)
    log(
        "Step 4.8 tokenizing subset "
        f"(max_length={max_length}, reasoning_marker={reasoning_marker!r})"
    )
    prep_start = time.time()
    prepared_data = setup_evaluator.prepare_data(
        dataset_path=sampled_subset_file,
        max_tokens=int(max_length),
        reasoning_marker=reasoning_marker,
    )
    log(
        f"Step 4.8 tokenized {len(prepared_data)} examples in "
        f"{time.time() - prep_start:.2f}s"
    )
    setup_evaluator.data = prepared_data

    log(f"Step 4.8 decomposing LoRA adapter at {lora_adapter_path}")
    decomp_start = time.time()
    deltas, lora_rank, adapter_template, rank1_factors = decompose_lora(
        str(lora_adapter_path),
        setup_evaluator.base_model,
        device=setup_gpu,
    )
    direction_ids = list(range(int(lora_rank)))
    log(
        f"Step 4.8 decomposed into {lora_rank} orthogonal directions "
        f"({direction_ids}) in {time.time() - decomp_start:.2f}s"
    )

    log("Step 4.8 computing base_loss (scale=0)")
    base_start = time.time()
    base_loss = float(setup_evaluator.evaluate(0.0))
    log(f"Step 4.8 base_loss={base_loss:.6f} (in {time.time() - base_start:.2f}s)")

    tokenizer = setup_evaluator.tokenizer

    log("Step 4.8 releasing setup evaluator to free GPU memory")
    try:
        setup_evaluator.models_by_device.clear()
        setup_evaluator.base_state_dict_by_device.clear()
        setup_evaluator.perturbation_direction_by_device.clear()
        setup_evaluator.base_state_dict_cpu.clear()
        setup_evaluator.perturbation_direction.clear()
        setup_evaluator.base_model = None
    except Exception:
        pass
    del setup_evaluator
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    task_queue: "queue_module.Queue[int]" = queue_module.Queue()
    for i in direction_ids:
        task_queue.put(int(i))

    history_root_dir = workdir / "step4_8_inflection_history"
    history_root_dir.mkdir(parents=True, exist_ok=True)
    debug_plot_root_dir = workdir / "step4_8_debug_inflection_plot"
    debug_plot_root_dir.mkdir(parents=True, exist_ok=True)

    model_init_lock = threading.Lock()

    shared: Dict[str, Any] = {
        "base_model_path": base_model_path,
        "prepared_data": prepared_data,
        "deltas": deltas,
        "rank1_factors": rank1_factors,
        "adapter_template": adapter_template,
        "tokenizer": tokenizer,
        "out_dir": out_dir,
        "history_root_dir": history_root_dir,
        "debug_plot_root_dir": debug_plot_root_dir,
        "ix_values": ix_values,
        "base_loss": float(base_loss),
        "ytol": float(ytol),
        "finder_max_iter": int(finder_max_iter),
        "curvature_ratio": float(curvature_ratio),
        "post_min_decrease_tol": float(post_min_decrease_tol),
        "post_min_decrease_frac": float(post_min_decrease_frac),
        "seed": int(args.seed),
        "max_eval_batch_size": int(max_eval_batch_size),
        "seed_count": int(seed_count),
        "extend_b_if_no_inflection": bool(extend_b_if_no_inflection),
        "b_extend_factor": float(b_extend_factor),
        "max_b_extensions": int(max_b_extensions),
        "model_init_lock": model_init_lock,
    }

    num_workers = min(len(gpu_strs), len(direction_ids))
    log(
        f"Step 4.8 dispatching {num_workers} worker thread(s) across "
        f"{gpu_strs[:num_workers]} for {len(direction_ids)} direction(s)"
    )

    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()
    error_event = threading.Event()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                _worker_loop,
                worker_idx=i,
                gpu_str=gpu_strs[i],
                task_queue=task_queue,
                shared=shared,
                results=results,
                results_lock=results_lock,
                error_event=error_event,
            )
            for i in range(num_workers)
        ]
        for fut in futures:
            fut.result()

    results.sort(key=lambda r: int(r["dir_idx"]))

    over_ckpts = _discover_over_checkpoints(out_dir, ix_values)
    if not over_ckpts:
        raise RuntimeError(f"No valid *_over_* checkpoints found in {out_dir}")

    metadata = {
        "mode": "inflection",
        "seed": int(args.seed),
        "sampled_subset_file": str(sampled_subset_file),
        "sampled_count": int(sampled_count),
        "ix_values": list(ix_values),
        "ix_mode": (
            "mixed"
            if any(v in ("infl", "half-infl") for v in ix_values)
            else "numeric"
        ),
        "lora_adapter_path": str(lora_adapter_path),
        "lora_rank": int(lora_rank),
        "dataset_for_eval": str(source_dataset_file),
        "output_dir": str(out_dir),
        "checkpoint_format": "adapter",
        "over_checkpoints": sorted(str(p) for p in over_ckpts),
        "base_loss": float(base_loss),
        "gpus_used": list(gpu_strs[:num_workers]),
        "per_direction_results": results,
        "finder_config": {
            "ytol": float(ytol),
            "max_iter": int(finder_max_iter),
            "post_min_decrease_tol": float(post_min_decrease_tol),
            "post_min_decrease_frac": float(post_min_decrease_frac),
            "max_eval_batch_size": int(max_eval_batch_size),
            "seed_count": int(seed_count),
            "gpu_threshold": float(gpu_threshold),
            "max_length": int(max_length),
            "extend_b_if_no_inflection": bool(extend_b_if_no_inflection),
            "b_extend_factor": float(b_extend_factor),
            "max_b_extensions": int(max_b_extensions),
        },
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log(f"Step 4.8 metadata written: {metadata_file}")

    return out_dir, False
