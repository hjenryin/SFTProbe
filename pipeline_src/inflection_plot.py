""""""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pipeline_src.inflection import InflectionPointFinder
from pipeline_src.utils import log


def write_debug_inflection_plot_snapshot(
    snapshot: Dict,
    out_dir: Path,
    *,
    title_prefix: str = "Inflection Debug",
) -> bool:
    """"""
    if not snapshot:
        return False

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception as exc:
        log(f"WARNING: Failed to import matplotlib for debug plots: {exc}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        iteration = int(float(snapshot.get("iteration", 0)))
        x_all = snapshot.get("x_all", None)
        y_all = snapshot.get("y_all", None)
        if x_all is None or y_all is None:
            x_all = snapshot.get("x_data", [])
            y_all = snapshot.get("y_data", [])
        x_data = [float(x) for x in x_all]
        y_data = [float(y) for y in y_all]
        x_fit = [float(x) for x in snapshot.get("x_fit", [])]
        y_fit = [float(y) for y in snapshot.get("y_fit", [])]

        fig = Figure(figsize=(8, 5))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        if x_fit and y_fit:
            ax.plot(x_fit, y_fit, label="Cubic spline fit", linewidth=2)
        if x_data and y_data:
            ax.scatter(x_data, y_data, label="Evaluations", s=40)
        ax.set_title(f"{title_prefix} - Iteration {iteration}")
        ax.set_xlabel("Scale")
        ax.set_ylabel("Mean loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"iter{iteration:03d}.png", dpi=150)
        return True
    except Exception as exc:
        log(f"WARNING: Failed to write debug plot for snapshot: {exc}")
        return False


def write_debug_inflection_plots(
    finder: InflectionPointFinder,
    out_dir: Path,
    *,
    title_prefix: str = "Inflection Debug",
    summary_label: Optional[str] = None,
) -> int:
    """"""
    snapshots = getattr(finder, "debug_snapshots", None)
    if not snapshots:
        return 0

    saved = 0
    for snap in snapshots:
        if write_debug_inflection_plot_snapshot(
            snap, out_dir, title_prefix=title_prefix
        ):
            saved += 1
    label = summary_label if summary_label else title_prefix
    log(f"Saved {saved} {label} debug inflection plots to {out_dir}")
    return saved
