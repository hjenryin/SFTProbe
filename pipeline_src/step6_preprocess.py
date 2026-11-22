""""""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import log


class SortScore(ABC):
    """"""

    def __init__(
        self,
        step5_dir: Path,
        step2_dir: Path,
        loss_pattern: str,
        subset_indices: np.ndarray,
        full_size: int,
        difficulty_loss_pattern: Optional[str] = None,
        clustering_losses: Optional[List[str]] = None,
    ):
        self.step5_dir = step5_dir
        self.step2_dir = step2_dir
        self.loss_pattern = loss_pattern
        self.subset_indices = subset_indices
        self.full_size = full_size
        self.difficulty_loss_pattern = difficulty_loss_pattern
        self.clustering_losses = set(clustering_losses) if clustering_losses is not None else None

    @abstractmethod
    def describe(self) -> str:
        """"""

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """"""

    def run(self, output_path: Path) -> Path:
        """"""
        log("=" * 60)
        log(f"SORT SCORE: {self.__class__.__name__}")
        log("=" * 60)
        log(self.describe())

        scores = self.compute()
        assert scores.ndim == 1, f"Expected 1-D tensor, got shape {scores.shape}"
        assert scores.shape[0] == len(self.subset_indices), (
            f"Score length {scores.shape[0]} != subset size {len(self.subset_indices)}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(scores, output_path)
        log(f"Saved sort scores ({scores.shape}) to {output_path}")
        return output_path


def _load_loss(dir_path: Path, pattern: str, subset_indices: np.ndarray) -> torch.Tensor:
    """"""
    matches = list(dir_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No loss file matching '{pattern}' found in {dir_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected exactly one loss file matching '{pattern}' in {dir_path}, but found {len(matches)}: {[m.name for m in matches]}")
    
    path = matches[0]
    raw = torch.tensor(torch.load(path), dtype=torch.float32)
    return raw[subset_indices]


class HighLossLastCkpt(SortScore):
    """"""

    def _get_difficulty_ckpts(self) -> List[str]:
        """"""
        return sorted(
            d.name for d in self.step2_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def describe(self) -> str:
        ckpts = self._get_difficulty_ckpts()
        last = ckpts[-1] if ckpts else "(none)"
        return (
            f"Strategy: use the difficulty checkpoint's loss\n"
            f"  Checkpoints found: {ckpts}\n"
            f"  Selected: {last}\n"
            f"  Loss pattern: {self.difficulty_loss_pattern}\n"
            f"  Sort order: high to low (descending)"
        )

    def compute(self) -> torch.Tensor:
        ckpts = self._get_difficulty_ckpts()
        if not ckpts:
            raise RuntimeError(f"No difficulty checkpoints in {self.step2_dir}")
        last = ckpts[-1]
        return _load_loss(self.step2_dir / last, self.difficulty_loss_pattern, self.subset_indices)


class AvgMaxScalePerDirection(SortScore):
    """"""

    _DIR_PATTERN = re.compile(r'^(dir\d+)_over_(\d+(?:\.\d+)?)x$')

    def _group_by_direction(self) -> Dict[str, List[Tuple[float, str]]]:
        """"""
        groups: Dict[str, List[Tuple[float, str]]] = {}
        for d in sorted(self.step5_dir.iterdir()):
            if not d.is_dir():
                continue
            if self.clustering_losses is not None and d.name not in self.clustering_losses:
                continue
            m = self._DIR_PATTERN.match(d.name)
            if not m:
                continue
            prefix, scale = m.group(1), float(m.group(2))
            groups.setdefault(prefix, []).append((scale, d.name))
        return groups

    def describe(self) -> str:
        groups = self._group_by_direction()
        if not groups:
            return "WARNING: No dir*_over_*x checkpoints found in step5 dir"

        lines = [
            "Strategy: average of highest-scale checkpoint per direction",
            f"  Loss pattern: {self.loss_pattern}",
        ]
        selected_names = []
        for prefix in sorted(groups):
            scales = sorted(groups[prefix], key=lambda x: x[0])
            _, max_name = scales[-1]
            all_names = [name for _, name in scales]
            lines.append(f"  {prefix}: {all_names} → selected {max_name}")
            selected_names.append(max_name)
        lines.append(f"  Score = mean({', '.join(selected_names)})")
        lines.append(f"  Sort order: high to low (descending)")
        return "\n".join(lines)

    def compute(self) -> torch.Tensor:
        groups = self._group_by_direction()
        if not groups:
            raise RuntimeError(
                f"No dir*_over_*x checkpoints found in {self.step5_dir}. "
                "AvgMaxScalePerDirection requires checkpoints like dir0_over_2x."
            )

        tensors = []
        for prefix in sorted(groups):
            scales = sorted(groups[prefix], key=lambda x: x[0])
            _, max_name = scales[-1]
            t = _load_loss(
                self.step5_dir / max_name,
                self.loss_pattern,
                self.subset_indices,
            )
            tensors.append(t)
            log(f"  Loaded {max_name}: shape={t.shape}, mean={t.float().mean():.4f}")

        stacked = torch.stack(tensors)
        avg = stacked.float().mean(dim=0)
        log(f"  Averaged across {len(tensors)} directions: mean={avg.mean():.4f}")
        return avg


class CheckpointDiffSort(SortScore):
    """"""

    def _find_ckpt_dirs(self) -> Tuple[str, str]:
        c0 = next(self.step5_dir.glob("sort_ckpt_0_*"), None)
        c1 = next(self.step5_dir.glob("sort_ckpt_1_*"), None)
        if not c0 or not c1:
            raise RuntimeError(
                "Missing custom checkpoints. "
                "Step 5 must evaluate sort_ckpt_0_* and sort_ckpt_1_* (via --sort-ckpts)."
            )
        return c0.name, c1.name

    def describe(self) -> str:
        try:
            name0, name1 = self._find_ckpt_dirs()
            disp0 = name0.replace("sort_ckpt_0_", "")
            disp1 = name1.replace("sort_ckpt_1_", "")
        except RuntimeError:
            disp0, disp1 = "<ckpt1>", "<ckpt2>"

        return (
            f"Strategy: check point loss difference\n"
            f"  Score formula: loss({disp0}) - loss({disp1})\n"
            f"  Loss pattern: {self.loss_pattern}\n"
            f"  Sort order: high to low (descending)"
        )

    def compute(self) -> torch.Tensor:
        name0, name1 = self._find_ckpt_dirs()
        loss1 = _load_loss(self.step5_dir / name0, self.loss_pattern, self.subset_indices)
        loss2 = _load_loss(self.step5_dir / name1, self.loss_pattern, self.subset_indices)
        
        diff = loss1 - loss2
        log(f"  Loaded {name0}: mean={loss1.float().mean():.4f}")
        log(f"  Loaded {name1}: mean={loss2.float().mean():.4f}")
        log(f"  Diff score mean: {diff.float().mean():.4f}")
        return diff


class AvgMaxScaleMinusBase(AvgMaxScalePerDirection):
    """"""

    BASE_DIR_NAME = "base_unperturbed"

    def describe(self) -> str:
        return (
            super().describe()
            + f"\n  MINUS base loss from step5_loss_1k_tokens/{self.BASE_DIR_NAME}/"
            + "\n  Interpretation: perturbation lift vs. unperturbed base model"
        )

    def compute(self) -> torch.Tensor:
        avg = super().compute()
        base_dir = self.step5_dir / self.BASE_DIR_NAME
        if not base_dir.is_dir():
            raise RuntimeError(
                f"Missing {base_dir}. Step 5 must compute the base unperturbed "
                "loss when --cluster-sort avg_max_scale_minus_base is selected."
            )
        base = _load_loss(base_dir, self.loss_pattern, self.subset_indices)
        log(f"  Loaded {self.BASE_DIR_NAME}: mean={base.float().mean():.4f}")
        diff = avg - base
        log(f"  Sort score (avg - base): mean={diff.float().mean():.4f}")
        return diff


class LearnDiffBase(SortScore):
    """"""

    DIFFICULTY_DIR_NAME = "difficulty_inflection"
    BASE_DIR_NAME = "base_unperturbed"
    _CLUSTER_SORT_LABEL = "learn-diff-base"
    _STRATEGY_HEADLINE = "Step2 difficulty checkpoint minus base pretrained loss"

    def describe(self) -> str:
        return (
            f"Strategy: {self._STRATEGY_HEADLINE}\n"
            f"  Uses step5_loss_1k_tokens/{self.DIFFICULTY_DIR_NAME}/\n"
            f"  MINUS step5_loss_1k_tokens/{self.BASE_DIR_NAME}/\n"
            f"  Loss pattern: {self.loss_pattern}\n"
            f"  Score formula: loss({self.DIFFICULTY_DIR_NAME}) - loss({self.BASE_DIR_NAME})\n"
            "  Sort order: high to low (descending)"
        )

    def compute(self) -> torch.Tensor:
        difficulty_dir = self.step5_dir / self.DIFFICULTY_DIR_NAME
        if not difficulty_dir.is_dir():
            raise RuntimeError(
                f"Missing {difficulty_dir}. Step 5 must compute the "
                f"{self.DIFFICULTY_DIR_NAME} loss when "
                f"--cluster-sort {self._CLUSTER_SORT_LABEL} is selected."
            )
        base_dir = self.step5_dir / self.BASE_DIR_NAME
        if not base_dir.is_dir():
            raise RuntimeError(
                f"Missing {base_dir}. Step 5 must compute the {self.BASE_DIR_NAME} "
                f"loss when --cluster-sort {self._CLUSTER_SORT_LABEL} is selected."
            )

        difficulty = _load_loss(difficulty_dir, self.loss_pattern, self.subset_indices)
        base = _load_loss(base_dir, self.loss_pattern, self.subset_indices)
        log(f"  Loaded {self.DIFFICULTY_DIR_NAME}: mean={difficulty.float().mean():.4f}")
        log(f"  Loaded {self.BASE_DIR_NAME}: mean={base.float().mean():.4f}")
        diff = difficulty - base
        log(f"  Sort score (difficulty - base): mean={diff.float().mean():.4f}")
        return diff


class LearnDiffStatedictBase(LearnDiffBase):
    """"""

    DIFFICULTY_DIR_NAME = "statedict_difficulty_inflection"
    _CLUSTER_SORT_LABEL = "learn-diff-statedict-base"
    _STRATEGY_HEADLINE = "Step 4.3 state-dict perturbed checkpoint minus base pretrained loss"


class DiffPerturb1k(SortScore):
    """"""

    DIFFICULTY_DIR_NAME = "difficulty_inflection"

    def describe(self) -> str:
        return (
            "Strategy: Step2 difficulty checkpoint 1000-token loss (no base subtraction)\n"
            f"  Uses step5_loss_1k_tokens/{self.DIFFICULTY_DIR_NAME}/\n"
            f"  Loss pattern: {self.loss_pattern}\n"
            "  Score formula: loss(difficulty_inflection)\n"
            "  Sort order: high to low (descending)"
        )

    def compute(self) -> torch.Tensor:
        difficulty_dir = self.step5_dir / self.DIFFICULTY_DIR_NAME
        if not difficulty_dir.is_dir():
            raise RuntimeError(
                f"Missing {difficulty_dir}. Step 5 must compute the difficulty_inflection "
                "loss when --cluster-sort diff_perturb_1k is selected."
            )
        difficulty = _load_loss(difficulty_dir, self.loss_pattern, self.subset_indices)
        log(f"  Loaded {self.DIFFICULTY_DIR_NAME}: mean={difficulty.float().mean():.4f}")
        return difficulty


SORT_SCORES: Dict[str, type] = {
    "high_loss": HighLossLastCkpt,
    "avg_max_scale": AvgMaxScalePerDirection,
    "avg_max_scale_minus_base": AvgMaxScaleMinusBase,
    "learn-diff-base": LearnDiffBase,
    "learn-diff-statedict-base": LearnDiffStatedictBase,
    "diff_perturb_1k": DiffPerturb1k,
    "checkpoint_diff": CheckpointDiffSort,
}
