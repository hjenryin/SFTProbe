""""""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from budget_utils import budget_allocation

from .step3_difficulty_filtering import _discover_difficulty_ckpt_name
from .utils import (
    BASE_DIR,
    DEFAULTS,
    get_num_gpus,
    get_num_token_for_difficulty,
    get_physical_num_gpus,
    log,
    run_cmd,
)


def _write_selected_lines(source_path: Path, out_path: Path, selected_indices: List[int]) -> int:
    selected_set = set(selected_indices)
    written = 0
    with open(source_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i in selected_set:
                fout.write(line)
                written += 1
    return written


def _load_prompt_meta(prompt_source: Path, indices: List[int]) -> Tuple[Dict[int, str], Dict[int, str]]:
    """"""
    wanted = set(indices)
    sources: Dict[int, str] = {}
    domains: Dict[int, str] = {}
    with open(prompt_source, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in wanted:
                continue
            if not line.strip():
                continue
            row = json.loads(line)
            src = row.get("source")
            if not src:
                raise RuntimeError(
                    f"Missing 'source' at line {i} of {prompt_source}; "
                    "difficulty_source sampling requires per-example source labels."
                )
            sources[i] = str(src)
            domains[i] = str(row.get("domain") or "unknown")
            if len(sources) == len(wanted):
                break
    missing = wanted - set(sources)
    if missing:
        raise RuntimeError(
            f"Could not load source labels for {len(missing)} indices from {prompt_source} "
            f"(e.g. {sorted(missing)[:5]})"
        )
    return sources, domains


def _discover_difficulty_loss_file(ckpt_dir: Path, difficulty_tokens: int) -> Path:
    preferred = ckpt_dir / f"loss-{difficulty_tokens}token-r-losses.pt"
    if preferred.exists():
        return preferred
    matches = sorted(ckpt_dir.glob("loss-*token-r-losses.pt"))
    if not matches:
        raise FileNotFoundError(
            f"No difficulty loss file under {ckpt_dir} "
            f"(expected loss-{difficulty_tokens}token-r-losses.pt)"
        )
    if len(matches) > 1:
        log(
            f"WARNING: multiple difficulty loss files in {ckpt_dir.name}; "
            f"using {matches[0].name}"
        )
    return matches[0]


def _softmax_weights(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
    names = list(scores.keys())
    if not names:
        return {}
    vals = np.array([scores[n] for n in names], dtype=np.float64)
    t = float(temperature) if temperature else 1.0
    shifted = vals - vals.max()
    exps = np.exp(shifted / t)
    probs = exps / exps.sum()
    return {n: float(p) for n, p in zip(names, probs)}


def _allocate_difficulty_source_quotas(
    difficult_indices: List[int],
    sources: Dict[int, str],
    domains: Dict[int, str],
    losses: torch.Tensor,
    total_budget: int,
    softmax_t: float,
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
    """"""
    if losses.ndim != 1:
        raise ValueError(f"Expected 1-D difficulty losses, got shape {tuple(losses.shape)}")

    domain_sum: Dict[str, float] = defaultdict(float)
    domain_count: Dict[str, int] = defaultdict(int)
    source_sum: Dict[Tuple[str, str], float] = defaultdict(float)
    source_count: Dict[Tuple[str, str], int] = defaultdict(int)
    domain_to_sources: Dict[str, set] = defaultdict(set)

    for idx in difficult_indices:
        src = sources[idx]
        dom = domains[idx]
        loss_v = float(losses[idx].item())
        domain_sum[dom] += loss_v
        domain_count[dom] += 1
        source_sum[(dom, src)] += loss_v
        source_count[(dom, src)] += 1
        domain_to_sources[dom].add(src)

    domain_mean = {d: domain_sum[d] / domain_count[d] for d in domain_sum}
    domain_sizes = {d: domain_count[d] for d in domain_sum}
    domain_weights = _softmax_weights(domain_mean, softmax_t)
    domain_alloc = budget_allocation(domain_weights, domain_sizes, total_budget)

    final_alloc: Dict[str, int] = defaultdict(int)
    source_weights_out: Dict[str, float] = {}
    source_mean_out: Dict[str, float] = {}

    for dom, dom_budget in domain_alloc.items():
        src_mean = {}
        src_sizes = {}
        for src in domain_to_sources[dom]:
            key = (dom, src)
            src_mean[src] = source_sum[key] / source_count[key]
            src_sizes[src] = source_count[key]
            source_mean_out[src] = src_mean[src]
        src_weights = _softmax_weights(src_mean, softmax_t)
        source_weights_out.update(src_weights)
        src_alloc = budget_allocation(src_weights, src_sizes, int(dom_budget))
        for src, n in src_alloc.items():
            final_alloc[src] += int(n)

    return dict(final_alloc), source_weights_out, source_mean_out


def _sample_difficulty_source(
    difficult_global_indices: List[int],
    prompt_source: Path,
    workdir: Path,
    target_x: int,
    seed: int,
    difficulty_tokens: int,
    softmax_t: float = 1.0,
) -> Tuple[List[int], Dict]:
    """"""
    step2_dir = workdir / "step2_loss_100tokens"
    ckpt_name = _discover_difficulty_ckpt_name(step2_dir)
    loss_file = _discover_difficulty_loss_file(step2_dir / ckpt_name, difficulty_tokens)
    losses = torch.load(loss_file, map_location="cpu", weights_only=False)
    if not torch.is_tensor(losses):
        losses = torch.as_tensor(losses, dtype=torch.float32)
    losses = losses.detach().float().reshape(-1)

    sources, domains = _load_prompt_meta(prompt_source, difficult_global_indices)
    actual_x = min(int(target_x), len(difficult_global_indices))

    quotas, source_weights, source_means = _allocate_difficulty_source_quotas(
        difficult_global_indices,
        sources,
        domains,
        losses,
        actual_x,
        softmax_t,
    )

    by_source: Dict[str, List[int]] = defaultdict(list)
    for idx in difficult_global_indices:
        by_source[sources[idx]].append(idx)

    rng = random.Random(int(seed))
    selected: List[int] = []
    realized: Dict[str, int] = {}
    for src in sorted(quotas):
        need = int(quotas[src])
        pool = by_source.get(src, [])
        if need > len(pool):
            raise RuntimeError(
                f"Source '{src}' quota {need} exceeds available difficult samples {len(pool)}"
            )
        picked = rng.sample(pool, k=need) if need else []
        selected.extend(picked)
        realized[src] = len(picked)

    if len(selected) < actual_x:
        selected_set = set(selected)
        remainder = [i for i in difficult_global_indices if i not in selected_set]
        rng.shuffle(remainder)
        need = actual_x - len(selected)
        extra = remainder[:need]
        selected.extend(extra)
        for idx in extra:
            src = sources[idx]
            realized[src] = realized.get(src, 0) + 1

    selected.sort()
    meta = {
        "sample_mode": "difficulty_source",
        "difficulty_ckpt": ckpt_name,
        "difficulty_loss_file": str(loss_file),
        "softmax_t": float(softmax_t),
        "softmax_score": "loss",
        "source_mean_difficulty": source_means,
        "source_softmax_weights": source_weights,
        "source_quotas": quotas,
        "source_realized": realized,
    }
    log(
        "difficulty_source quotas: "
        + ", ".join(f"{s}={quotas[s]}" for s in sorted(quotas))
    )
    return selected, meta


def _sample_random(difficult_global_indices: List[int], target_x: int, seed: int) -> Tuple[List[int], Dict]:
    rng = random.Random(int(seed))
    actual_x = min(int(target_x), len(difficult_global_indices))
    selected = rng.sample(difficult_global_indices, k=actual_x)
    selected.sort()
    return selected, {"sample_mode": "random"}


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    output_file = workdir / "step4_5_partial_completion.jsonl"
    selected_idx_file = workdir / "step4_5_selected_global_indices.npy"
    metadata_file = workdir / "step4_5_metadata.json"

    if output_file.exists() and selected_idx_file.exists() and metadata_file.exists():
        log("[SKIP] Step 4.5 (partial completion): outputs already exist")
        return output_file, True

    log("=" * 60)
    log("STEP 4.5: Partial Completion")
    log("=" * 60)

    step4_file = workdir / "step4_continued_1k.jsonl"
    step3_idx_file = workdir / "step3_difficult_samples.npy"

    if not step4_file.exists():
        raise FileNotFoundError(f"Missing Step 4 output: {step4_file}")
    if not step3_idx_file.exists():
        raise FileNotFoundError(f"Missing Step 3 indices: {step3_idx_file}")

    difficult_global_indices = np.load(step3_idx_file).tolist()
    difficult_global_indices = [int(i) for i in difficult_global_indices]
    if not difficult_global_indices:
        raise RuntimeError("step3_difficult_samples.npy is empty; Step 4.5 cannot sample from an empty set")

    sample_mode = getattr(args, "partial_x_sample", "random")
    target_x = int(args.partial_x)
    if sample_mode == "random":
        selected_global_indices, sample_meta = _sample_random(
            difficult_global_indices, target_x, int(args.seed)
        )
    elif sample_mode == "difficulty_source":
        prompt_source = Path(config["prompt_source"])
        if not prompt_source.exists():
            raise FileNotFoundError(f"Missing prompt_source for difficulty_source sampling: {prompt_source}")
        selected_global_indices, sample_meta = _sample_difficulty_source(
            difficult_global_indices,
            prompt_source,
            workdir,
            target_x,
            int(args.seed),
            difficulty_tokens=get_num_token_for_difficulty(args),
            softmax_t=float(DEFAULTS["softmax_t"]),
        )
    else:
        raise ValueError(
            f"Unknown partial_x_sample={sample_mode!r}; expected 'random' or 'difficulty_source'"
        )

    actual_x = len(selected_global_indices)
    np.save(selected_idx_file, np.array(selected_global_indices, dtype=np.int64))
    log(
        f"Selected {actual_x} indices from {len(difficult_global_indices)} "
        f"Step 4-continued samples (mode={sample_mode})"
    )

    y_tokens = int(args.partial_y)

    if args.teacher.lower() == "original":
        written = _write_selected_lines(step4_file, output_file, selected_global_indices)
        if y_tokens > 0:
            log("teacher=original -> skipped additional generation (partial_y ignored) and exported selected samples directly")
        else:
            log("teacher=original -> exported selected samples directly")
    elif y_tokens == 0:
        written = _write_selected_lines(step4_file, output_file, selected_global_indices)
        log("y=0 -> skipped additional generation and exported selected samples directly")
    else:
        tmp_dir = workdir / "step4_5_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        num_gpus = get_num_gpus()
        physical_gpus = get_physical_num_gpus()
        base_dp = 4
        ratio = (num_gpus / physical_gpus) if physical_gpus > 0 else 1
        data_parallel = max(1, min(num_gpus, int(base_dp * ratio)))

        tmp_idx_file = tmp_dir / "step4_5_selected_global_indices.npy"
        np.save(tmp_idx_file, np.array(selected_global_indices, dtype=np.int64))

        cmd = [
            sys.executable,
            str(BASE_DIR / "m1" / "src" / "generation" / "generate.py"),
            "--input_file", str(step4_file),
            "--output_dir", str(tmp_dir),
            "--model_name", args.teacher,
            "--max_tokens", str(y_tokens),
            "--num_gpus", str(num_gpus),
            "--data_parallel", str(data_parallel),
            "--seed", str(args.seed),
            "--mode", "continuation",
            "--continuation_indices", str(tmp_idx_file),
            "--chat_template_args", json.dumps(DEFAULTS["chat_template_args"]),
        ]
        run_cmd(cmd)

        generated_full = tmp_dir / "continuation_results.jsonl"
        if not generated_full.exists():
            raise FileNotFoundError(f"Expected generation output missing: {generated_full}")

        written = _write_selected_lines(generated_full, output_file, selected_global_indices)

    metadata = {
        "seed": int(args.seed),
        "x_requested": int(args.partial_x),
        "x_selected": int(actual_x),
        "y_tokens": int(y_tokens),
        "num_step4_continued": int(len(difficult_global_indices)),
        "selected_global_indices_file": str(selected_idx_file),
        **sample_meta,
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if written != actual_x:
        raise RuntimeError(f"Step 4.5 wrote {written} records, expected {actual_x}")

    return output_file, False
