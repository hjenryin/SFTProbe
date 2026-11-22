""""""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .inflection import InflectionPointFinder, LLMEvaluator
from .inflection_plot import (
    write_debug_inflection_plot_snapshot,
    write_debug_inflection_plots,
)
from .utils import log


def scale_key(scale: float) -> float:
    return round(float(scale), 10)


def scale_to_ckpt_name(scale: float) -> str:
    token = f"{float(scale):.8f}".replace("-", "m").replace(".", "p")
    return f"checkpoint-{token}"


def build_scan_subset(
    *,
    input_data: Path,
    workdir: Path,
    subset_size: int,
    seed: int,
    subset_filename: str,
) -> Tuple[Path, int, int]:
    """"""
    with open(input_data, "r", encoding="utf-8") as f:
        rows = [line for line in f if line.strip()]

    total = len(rows)
    if total == 0 or subset_size <= 0 or subset_size >= total:
        return input_data, total, total

    count = min(subset_size, total)
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(total), k=count))

    subset_path = workdir / subset_filename
    with open(subset_path, "w", encoding="utf-8") as f:
        for idx in picked:
            f.write(rows[idx])

    return subset_path, count, total


def load_eval_history(history_path: Path) -> Dict[float, Dict]:
    if not history_path.exists():
        return {}

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    rows = []
    if isinstance(payload, dict):
        rows = payload.get("evaluations", [])
    elif isinstance(payload, list):
        rows = payload

    out: Dict[float, Dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            scale = float(row.get("scale"))
            mean_loss = float(row.get("mean_loss"))
        except Exception:
            continue

        item = {
            "name": str(row.get("name") or scale_to_ckpt_name(scale)),
            "scale": float(scale),
            "mean_loss": float(mean_loss),
        }
        out[scale_key(scale)] = item

    return out


def persist_eval_history(history_path: Path, eval_cache: Dict[float, Dict]) -> None:
    rows = sorted(eval_cache.values(), key=lambda x: float(x["scale"]))
    payload = {
        "evaluations": [
            {
                "name": str(row.get("name") or scale_to_ckpt_name(float(row["scale"]))),
                "scale": float(row["scale"]),
                "mean_loss": float(row["mean_loss"]),
            }
            for row in rows
        ]
    }

    history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = history_path.with_suffix(history_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(history_path)


def evaluate_scale_point(
    *,
    scale: float,
    evaluator: LLMEvaluator,
    prepared_data: List[Dict[str, List[int]]],
    eval_cache: Dict[float, Dict],
    history_path: Path,
    log_prefix: str,
) -> Dict:
    key = scale_key(scale)
    if key in eval_cache:
        return eval_cache[key]

    for row in eval_cache.values():
        if abs(float(row["scale"]) - float(scale)) <= 1e-8:
            return row

    eval_start = time.time()
    log(f"{log_prefix} evaluating scale={float(scale):.8f}")
    _, mean_loss = evaluator.evaluate_scale(scale=float(scale), data=prepared_data)

    row = {
        "name": scale_to_ckpt_name(scale),
        "scale": float(scale),
        "mean_loss": float(mean_loss),
    }
    eval_cache[key] = row
    persist_eval_history(history_path, eval_cache)
    log(
        f"{log_prefix} finished scale={float(scale):.8f} "
        f"mean_loss={float(mean_loss):.8f} in {time.time() - eval_start:.2f}s"
    )
    return row


def search_inflection_points(
    *,
    evaluator: LLMEvaluator,
    prepared_data: List[Dict[str, List[int]]],
    ytol: float,
    max_iter: int,
    curvature_ratio: float,
    lower_bound: float,
    upper_bound: float,
    eval_cache: Dict[float, Dict],
    history_path: Path,
    debug_plot_dir: Path,
    log_prefix: str,
    title_prefix: str,
    summary_label: str,
    extend_b_if_no_inflection: bool = False,
    b_extend_factor: float = 1.5,
    max_b_extensions: int = 3,
) -> Dict:
    """"""
    seed_scales = np.linspace(float(lower_bound), float(upper_bound), 4).tolist()

    class _CacheBackedEvaluator:
        def evaluate(self, x):
            row = evaluate_scale_point(
                scale=float(x),
                evaluator=evaluator,
                prepared_data=prepared_data,
                eval_cache=eval_cache,
                history_path=history_path,
                log_prefix=log_prefix,
            )
            return float(row["mean_loss"])

    finder_history = {float(row["scale"]): float(row["mean_loss"]) for row in eval_cache.values()}
    finder = InflectionPointFinder(
        evaluator=_CacheBackedEvaluator(),
        a=float(lower_bound),
        b=float(upper_bound),
        history=finder_history,
        curvature_ratio=float(curvature_ratio),
        extend_b_if_no_inflection=bool(extend_b_if_no_inflection),
        b_extend_factor=float(b_extend_factor),
        max_b_extensions=int(max_b_extensions),
        debug_snapshot_callback=lambda snap: write_debug_inflection_plot_snapshot(
            snap, debug_plot_dir, title_prefix=title_prefix
        ),
    )

    log(
        f"{log_prefix} inflection finder start: seed_points={len(seed_scales)}, "
        f"cached_points={len(finder_history)}"
    )
    search_start = time.time()

    x_opt, y_opt = finder.run(
        ytol=float(ytol),
        max_iter=int(max_iter),
        initial_points=seed_scales,
    )
    log(
        f"{log_prefix} inflection finder done in {time.time() - search_start:.2f}s; "
        f"eval_count={finder.eval_count}, total_points={len(finder.history)}"
    )

    for scale, mean_loss in finder.history.items():
        key = scale_key(scale)
        if key not in eval_cache:
            eval_cache[key] = {
                "name": scale_to_ckpt_name(scale),
                "scale": float(scale),
                "mean_loss": float(mean_loss),
            }

    persist_eval_history(history_path, eval_cache)
    write_debug_inflection_plots(
        finder,
        debug_plot_dir,
        title_prefix=title_prefix,
        summary_label=summary_label,
    )

    trace = [
        {
            "iteration": int(t["iteration"]),
            "x": float(t["x"]),
            "y": float(t["y"]),
            "num_points": int(t["num_points"]),
        }
        for t in finder.trace
    ]

    converged = False
    if len(trace) >= 2:
        converged = abs(float(trace[-1]["y"]) - float(trace[-2]["y"])) < float(ytol)

    return {
        "initial_scales": seed_scales,
        "scan_range": [float(lower_bound), float(upper_bound)],
        "ytol": float(ytol),
        "max_iter": int(max_iter),
        "curvature_ratio": float(curvature_ratio),
        "extend_b_if_no_inflection": bool(extend_b_if_no_inflection),
        "b_extend_factor": float(b_extend_factor),
        "max_b_extensions": int(max_b_extensions),
        "b_final": float(getattr(finder, "b", upper_bound)),
        "b_extensions_applied": int(getattr(finder, "b_extensions_applied", 0)),
        "converged": bool(converged),
        "trace": trace,
        "x_opt": float(x_opt),
        "y_opt": float(y_opt),
    }
