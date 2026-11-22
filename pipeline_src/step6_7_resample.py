""""""

import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import requests as req

from .step7_reuse import (
    APPROVED,
    DENIED,
    INCOMPLETE,
    PENDING,
    collect_generation_candidates_from_store,
    extract_final_answer_content as _extract_final_answer_content,
    get_final_answer_delimiter as _get_final_answer_delimiter,
    load_jsonl as _load_jsonl,
    normalize_candidate,
    resolve_best_candidates,
    step4_prefixes_for_indices,
    write_jsonl,
)
from .generation_store import GenerationStore
from .utils import (
    BASE_DIR,

    check_output_exists,
    find_free_port,
    get_num_gpus,
    get_physical_num_gpus,
    get_step7_max_tokens,
    log,
    run_cmd,
    DEFAULTS,
    STEP7_SELECTED_TRACES,
)

RESAMPLE_MAX_RETRIES = 3
JUDGE_MALFORMED_THRESHOLD = 0.10


def get_final_answer_delimiter(teacher: str) -> str:
    """"""
    return _get_final_answer_delimiter(teacher)


def extract_final_answer_content(assistant_content: str, teacher: str) -> str:
    """"""
    return _extract_final_answer_content(assistant_content, teacher)


def _compute_budget_allocations(cluster_orderings, source_weights, denied, total_budget,
                                 max_repetition=1, approved_reps=None):
    """"""
    from budget_utils import budget_allocation
    if approved_reps is None:
        approved_reps = {}

    def _idx_capacity(idx):
        if idx in denied:
            return 0
        return max(0, max_repetition - approved_reps.get(idx, 0))

    source_sizes = {}
    for src, clusters in cluster_orderings.items():
        size = 0
        for ordering in clusters.values():
            size += sum(_idx_capacity(idx) for idx in ordering)
        source_sizes[src] = size

    source_alloc = budget_allocation(source_weights, source_sizes, total_budget)

    cluster_alloc = {}
    for src, clusters in cluster_orderings.items():
        src_budget = source_alloc.get(src, 0)
        n_clusters = len(clusters)
        if n_clusters == 0 or src_budget == 0:
            cluster_alloc[src] = {}
            continue

        cluster_weights = {cid: 1.0 / n_clusters for cid in clusters}
        cluster_sizes = {
            cid: sum(_idx_capacity(idx) for idx in ordering)
            for cid, ordering in clusters.items()
        }
        cluster_alloc[src] = budget_allocation(cluster_weights, cluster_sizes, src_budget)

    return source_alloc, cluster_alloc


def _compute_multi_round_allocation_and_selection(
    cluster_orderings, source_weights, approved, denied, total_budget,
    max_repetition=1, approved_reps=None, rep_denied=None
):
    """"""
    if approved_reps is None:
        approved_reps = {}
    if rep_denied is None:
        rep_denied = {}

    from budget_utils import budget_allocation

    index_usage = {}
    final_allocated_list = []

    cluster_total_alloc = {}
    cluster_approved_frontier = {}
    cluster_inspect = {}
    for src, clusters in cluster_orderings.items():
        cluster_total_alloc[src] = {cid: 0 for cid in clusters}
        cluster_approved_frontier[src] = {cid: 0 for cid in clusters}
        cluster_inspect[src] = {cid: [] for cid in clusters}

    for rep in range(1, max_repetition + 1):
        remaining_budget = total_budget - len(final_allocated_list)
        if remaining_budget <= 0:
            break

        source_sizes = {}
        cluster_sizes = {}
        for src, clusters in cluster_orderings.items():
            src_size = 0
            cluster_sizes[src] = {}
            for cid, ordering in clusters.items():
                c_size = 0
                for idx in ordering:
                    idx = int(idx)
                    if idx in denied:
                        continue
                    if index_usage.get(idx, 0) < rep:
                        c_size += 1
                cluster_sizes[src][cid] = c_size
                src_size += c_size
            source_sizes[src] = src_size

        source_alloc = budget_allocation(source_weights, source_sizes, remaining_budget)

        for src, clusters in cluster_orderings.items():
            src_budget = source_alloc.get(src, 0)
            if src_budget <= 0:
                continue

            n_clusters = len(clusters)
            if n_clusters == 0:
                continue

            c_weights = {cid: 1.0 / n_clusters for cid in clusters}
            c_alloc = budget_allocation(c_weights, cluster_sizes[src], src_budget)

            for cid, ordering in clusters.items():
                alloc = c_alloc.get(cid, 0)
                picked = 0
                for idx in ordering:
                    if picked >= alloc:
                        break
                    idx = int(idx)
                    if idx in denied:
                        continue

                    if index_usage.get(idx, 0) < rep:
                        index_usage[idx] = index_usage.get(idx, 0) + 1
                        picked += 1
                        final_allocated_list.append(idx)

                        cluster_total_alloc[src][cid] += 1

                        reps_done = approved_reps.get(idx, 0)
                        if rep <= reps_done:
                            cluster_approved_frontier[src][cid] += 1
                        else:
                            next_rep = rep
                            idx_rep_denied = rep_denied.get(idx, set())
                            if next_rep not in idx_rep_denied:
                                cluster_inspect[src][cid].append(idx)

    cluster_remaining_cap = {}
    for src, clusters in cluster_orderings.items():
        cluster_remaining_cap[src] = {}
        for cid, ordering in clusters.items():
            total_cap = 0
            for idx in ordering:
                idx = int(idx)
                if idx in denied:
                    continue
                total_cap += max(0, max_repetition - approved_reps.get(idx, 0))
            cluster_remaining_cap[src][cid] = max(0, total_cap - cluster_total_alloc[src][cid])

    return (
        final_allocated_list,
        cluster_total_alloc,
        cluster_approved_frontier,
        cluster_inspect,
        cluster_remaining_cap
    )


def _select_approved_by_allocation(cluster_orderings, source_weights, approved, denied,
                                    total_budget, max_repetition=1, approved_reps=None):
    """"""
    if approved_reps is None:
        approved_reps = {}

    final_allocated_list, _, _, _, _ = \
        _compute_multi_round_allocation_and_selection(
            cluster_orderings, source_weights, approved, denied, total_budget,
            max_repetition=max_repetition, approved_reps=approved_reps
        )

    final_indices = []
    allocated_counts = {}
    for idx in final_allocated_list:
        allocated_counts[idx] = allocated_counts.get(idx, 0) + 1
        
        if allocated_counts[idx] <= approved_reps.get(idx, 0):
            final_indices.append(idx)

    return final_indices


def rebalance_budget(cluster_orderings, source_weights, approved, denied, total_budget,
                     max_repetition=1, approved_reps=None,
                     rep_denied=None):
    """"""
    if approved_reps is None:
        approved_reps = {}
    if rep_denied is None:
        rep_denied = {}

    final_allocated_list, cluster_total_alloc, cluster_approved_frontier, cluster_inspect, cluster_remaining_cap = \
        _compute_multi_round_allocation_and_selection(
            cluster_orderings, source_weights, approved, denied, total_budget,
            max_repetition=max_repetition, approved_reps=approved_reps, rep_denied=rep_denied
        )

    batch_indices = []
    cluster_stats = {}

    for src, clusters in cluster_orderings.items():
        src_stats = {}
        for cid, ordering in clusters.items():
            total_alloc = cluster_total_alloc[src][cid]
            approved_in_frontier = cluster_approved_frontier[src][cid]
            inspect_indices = list(cluster_inspect[src][cid])
            
            batch_indices.extend(inspect_indices)

            src_stats[cid] = {
                "total_alloc": total_alloc,
                "approved": approved_in_frontier,
                "to_inspect": len(inspect_indices),
                "picked": len(inspect_indices),
                "remaining_after": cluster_remaining_cap[src][cid],
            }
        cluster_stats[src] = src_stats

    return batch_indices, cluster_stats


def _summarize_frontier_stats(cluster_stats):
    """"""
    allocated_slots = 0
    approved_frontier = 0
    inspect_frontier = 0

    for stats in cluster_stats.values():
        for row in stats.values():
            allocated_slots += int(row.get("total_alloc", 0))
            approved_frontier += int(row.get("approved", 0))
            inspect_frontier += int(row.get("to_inspect", 0))

    skipped_frontier = max(0, allocated_slots - approved_frontier - inspect_frontier)
    return {
        "allocated_slots": allocated_slots,
        "approved_frontier": approved_frontier,
        "inspect_frontier": inspect_frontier,
        "skipped_frontier": skipped_frontier,
    }


def _count_phrase(count: int, singular: str, plural: str = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def _format_counts(counts: Dict[str, int], order: List[str]) -> str:
    parts = [f"{name}={int(counts.get(name, 0))}" for name in order if counts.get(name, 0)]
    return ", ".join(parts) if parts else "none"


def _format_delta_plan(indices: List[int], delta_generation_cache: Dict[int, Tuple[Dict, int]]) -> str:
    by_delta: Dict[int, int] = {}
    for idx in indices:
        _row, delta = delta_generation_cache[idx]
        by_delta[int(delta)] = by_delta.get(int(delta), 0) + 1
    if not by_delta:
        return "none"
    return ", ".join(f"+{delta}: {count}" for delta, count in sorted(by_delta.items()))


def _load_failure_counts(path: Path) -> Dict[int, int]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(idx): int(count) for idx, count in raw.items()}


def _save_failure_counts(path: Path, counts: Dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(int(idx)): int(count) for idx, count in sorted(counts.items()) if int(count) > 0}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)


def _failure_attempt_key(idx: int, normalized) -> Tuple:
    attempt_id = normalized.candidate.attempt_id
    if attempt_id is not None:
        return ("attempt", int(attempt_id))
    row = normalized.candidate.row
    return (
        "fallback",
        int(idx),
        str(normalized.relation),
        int(normalized.candidate.source_max_tokens),
        str(row.get("trace_hash") or normalized.candidate.source_path),
    )


def _is_local_failure_evidence(normalized) -> bool:
    verdict = normalized.candidate.verdict
    if normalized.relation == "longer":
        return True
    if verdict in {DENIED, INCOMPLETE}:
        return True
    return (not normalized.complete) and normalized.remaining_tokens <= 0


def _is_approved_fit(normalized) -> bool:
    return (
        normalized.candidate.verdict == APPROVED
        and normalized.complete
        and normalized.relation != "longer"
    )


def _is_pending_fit(normalized) -> bool:
    return (
        normalized.candidate.verdict == PENDING
        and normalized.complete
        and normalized.relation != "longer"
        and normalized.candidate.attempt_id is not None
    )


def _is_continuable(normalized) -> bool:
    return (
        normalized.candidate.verdict != DENIED
        and not normalized.complete
        and normalized.relation == "shorter"
        and normalized.remaining_tokens > 0
    )


def _pending_attempt_sort_key(item: Tuple[Dict, int]) -> Tuple[int, str]:
    _row, attempt_id = item
    return (int(attempt_id), str(attempt_id))


def _continuable_sort_key(normalized) -> Tuple[int, int, str]:
    return (
        -int(normalized.candidate.source_max_tokens),
        int(normalized.remaining_tokens),
        str(normalized.candidate.source_path),
    )


def _add_local_failure_events(
    failure_attempt_counts: Dict[int, int],
    failure_attempt_keys_seen: Dict[int, Set[Tuple]],
    approved: Set[int],
    denied: Set[int],
    failed_events: List[Tuple[int, Tuple]],
) -> Tuple[List[int], List[int]]:
    retrying = []
    exhausted = []
    for idx_raw, failure_key in failed_events:
        idx = int(idx_raw)
        if idx in approved:
            continue
        seen = failure_attempt_keys_seen.setdefault(idx, set())
        if failure_key in seen:
            continue
        seen.add(failure_key)
        failure_attempt_counts[idx] = int(failure_attempt_counts.get(idx, 0)) + 1
        if failure_attempt_counts[idx] >= RESAMPLE_MAX_RETRIES:
            if idx not in denied:
                exhausted.append(idx)
            denied.add(idx)
        else:
            retrying.append(idx)
    return retrying, exhausted


def oversample_batch(cluster_orderings, batch_indices, approved, denied,
                     acceptance_rate, original_batch_size, dp_size, idx_to_cluster):
    """"""
    if original_batch_size == 0 or acceptance_rate <= 0:
        return batch_indices

    use_ceil = original_batch_size < 50 or original_batch_size < dp_size
    pad_to_dp = original_batch_size < dp_size

    cluster_needed = {}
    for idx in batch_indices:
        key = idx_to_cluster.get(idx)
        if key:
            cluster_needed[key] = cluster_needed.get(key, 0) + 1

    oversampled_indices = []
    extras_allocated = {}

    for (src, cid), needed in cluster_needed.items():
        if use_ceil:
            target = math.ceil(needed / acceptance_rate)
        else:
            target = math.floor(needed / acceptance_rate)

        extras_needed = max(0, target - needed)
        extras_allocated[(src, cid)] = extras_needed

    if pad_to_dp:
        current_total = original_batch_size + sum(extras_allocated.values())
        remainder = current_total % dp_size
        if remainder != 0:
            padding_needed = dp_size - remainder
            sorted_clusters = sorted(cluster_needed.keys(), key=lambda k: cluster_needed[k], reverse=True)
            if sorted_clusters:
                idx = 0
                for _ in range(padding_needed):
                    key = sorted_clusters[idx % len(sorted_clusters)]
                    extras_allocated[key] = extras_allocated.get(key, 0) + 1
                    idx += 1

    for (src, cid), extras_to_pick in extras_allocated.items():
        if extras_to_pick <= 0:
            continue

        ordering = cluster_orderings.get(src, {}).get(cid, [])
        picked = 0

        skip_set = approved | denied | set(batch_indices) | set(oversampled_indices)

        for idx in ordering:
            if picked >= extras_to_pick:
                break
            if idx not in skip_set:
                oversampled_indices.append(idx)
                picked += 1

    return batch_indices + oversampled_indices


def _read_jsonl_from(path: Path, start_line: int) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line or not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_m23k_ground_truth(config: Dict) -> Dict[int, str]:
    m23k_gt = {}
    gt_path = config["prompt_source"]
    log(f"Loading m23k ground truth from {gt_path}")
    with open(gt_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                obj = json.loads(line)
                m23k_gt[i] = obj.get("answer_letter", "").strip().upper()
    return m23k_gt


def _load_openthoughts_math_ground_truth(config: Dict) -> Tuple[Dict[int, str], Dict[int, str]]:
    ot_math_gt = {}
    ot_math_prompt = {}
    gt_path = config["prompt_source"]
    log(f"Loading openthoughts-math ground truth from {gt_path}")
    with open(gt_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                obj = json.loads(line)
                ot_math_gt[i] = obj.get("solution", "")
                ot_math_prompt[i] = obj.get("prompt", "")
    return ot_math_gt, ot_math_prompt


def _update_attempt_status(
    store: GenerationStore,
    attempt_id: int,
    status: str,
    args=None,
    workdir: Path = None,
) -> None:
    if attempt_id is None:
        raise RuntimeError(f"Cannot update generation verdict to {status}: missing attempt_id")
    store.update_status(
        attempt_id,
        status,
        source_run=getattr(args, "run_name", "") if args is not None else "",
        source_path=str(workdir) if workdir is not None else "",
    )


def _judge_template_kwargs(model_name: str) -> Dict:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    model_l = str(model_name).lower()
    if "gemma" in model_l or "qwen" in model_l or "glm" in model_l:
        kwargs["enable_thinking"] = False
    return kwargs


def _strict_judge_verdict(response_text: str, judge_strict_verdict) -> Tuple[str, Dict]:
    parsed = judge_strict_verdict(response_text)
    final_verdict = parsed.get("final_verdict")
    if str(final_verdict).lower() == "correct":
        return APPROVED, parsed
    if str(final_verdict).lower() == "incorrect":
        return DENIED, parsed
    return PENDING, parsed


def _judge_allowed_token_ids(tokenizer) -> List[int]:
    correct_ids = tokenizer.encode("Correct", add_special_tokens=False)
    incorrect_ids = tokenizer.encode("Incorrect", add_special_tokens=False)
    if len(correct_ids) != 1 or len(incorrect_ids) != 1:
        raise RuntimeError(
            "Judge verdict token enforcement requires single-token "
            f"Correct/Incorrect labels, got {correct_ids}/{incorrect_ids}"
        )
    return [int(correct_ids[0]), int(incorrect_ids[0])]


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    """"""
    final_idx_file = workdir / "step6_labeled_idx.npy"
    final_traces_file = workdir / STEP7_SELECTED_TRACES

    if check_output_exists(final_traces_file, "Step 6+7 Resample") and check_output_exists(final_idx_file, "Step 6+7 Resample"):
        return final_traces_file, True

    log("=" * 60)
    log("STEP 6+7: Resample Mode (Multi-turn Approve/Deny)")
    log("=" * 60)

    step6_dir = workdir / "step6_diversity"
    data_dir = step6_dir / "data"
    orderings_file = data_dir / "cluster_orderings.json"
    budgets_file = data_dir / "source_budgets.json"
    weights_file = data_dir / "source_weights.json"
    approved_file = step6_dir / "step6_approved.npy"
    denied_file = step6_dir / "step6_denied.npy"
    failure_counts_file = step6_dir / "resample_failure_counts.json"
    resample_log = step6_dir / "resample_log.txt"

    if not orderings_file.exists():
        log("Step 6a: Computing cluster orderings via step6_select (resample mode)")
        from .step6_prepare import prepare_selection_spec
        from .step6_select import run as step6_select_run

        spec = prepare_selection_spec(config, workdir, args)

        step6_select_run(
            spec=spec,
            step2_loss_dir=workdir / "step2_loss_100tokens",
            step5_loss_dir=workdir / "step5_loss_1k_tokens",
            prompt_source=config["prompt_source"],
            subset_indices_path=workdir / "step3_difficult_samples.npy",
            workdir=workdir,
            resample=True,
        )
    else:
        log(f"Step 6a: Cluster orderings already exist at {orderings_file}")

    with open(orderings_file) as f:
        cluster_orderings = json.load(f)
    with open(budgets_file) as f:
        source_budgets = json.load(f)

    if weights_file.exists():
        with open(weights_file) as f:
            source_weights = json.load(f)
        log(f"Loaded source weights: {source_weights}")
    else:
        total_original = sum(source_budgets.values())
        if total_original > 0:
            source_weights = {src: budget / total_original for src, budget in source_budgets.items()}
        else:
            source_weights = {src: 1.0 / len(source_budgets) for src in source_budgets}
        log(f"No source_weights.json found, derived from budgets: {source_weights}")

    total_budget = args.num_samples
    step7_max_tokens = get_step7_max_tokens(args)
    max_repetition = getattr(args, "max_repetition", 1)
    log(f"Total budget: {total_budget}")
    log(f"Step 7 max_new_tokens: {step7_max_tokens}")
    log(f"Max repetition: {max_repetition}")
    log(f"Sources: {list(cluster_orderings.keys())}")
    log(f"Source budgets: {source_budgets}")

    idx_to_cluster = {}
    for src, clusters in cluster_orderings.items():
        for cid, ordering in clusters.items():
            for idx in ordering:
                idx_to_cluster[idx] = (src, cid)

    resume_started = time.perf_counter()
    log("[GEN-STORE] Resume: loading Step 4 rows and building candidate prefix map")
    step4_rows = _load_jsonl(workdir / "step4_continued_1k.jsonl")
    candidate_index_set = set(idx_to_cluster.keys())
    prefixes_by_idx = step4_prefixes_for_indices(step4_rows, candidate_index_set)
    log(
        f"[GEN-STORE] Resume: loaded {len(step4_rows)} Step 4 rows, "
        f"{len(candidate_index_set)} candidate indices, {len(prefixes_by_idx)} usable prefixes "
        f"in {time.perf_counter() - resume_started:.1f}s"
    )
    log("[GEN-STORE] Resume: opening shared SQLite generation store")
    store = GenerationStore.for_run(config, args)

    failure_attempt_counts = _load_failure_counts(failure_counts_file)
    denied = {
        int(idx)
        for idx, count in failure_attempt_counts.items()
        if int(count) >= RESAMPLE_MAX_RETRIES
    }
    log(
        f"[GEN-STORE] Resume: loaded run-local failure counts for "
        f"{len(failure_attempt_counts)} indices ({len(denied)} locally exhausted)"
    )
    approved: Set[int] = set()
    approved_reps: Dict[int, int] = {}
    rep_denied: Dict[int, Set[int]] = {}
    pending_generation_cache: Dict[int, List[Tuple[Dict, int]]] = {}
    delta_generation_cache: Dict[int, Tuple[Dict, int]] = {}
    failure_attempt_keys_seen: Dict[int, Set[Tuple]] = {}

    if args.force_rerun:
        log("[GEN-STORE] --force-rerun set; skipping shared generation reuse")
    else:
        from transformers import AutoTokenizer
        tokenizer_started = time.perf_counter()
        log(f"[GEN-STORE] Resume: loading tokenizer for reuse normalization ({args.teacher})")
        tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        log(f"[GEN-STORE] Resume: tokenizer loaded in {time.perf_counter() - tokenizer_started:.1f}s")

        collect_started = time.perf_counter()
        log("[GEN-STORE] Resume: fetching reusable generations from SQLite")
        candidates = collect_generation_candidates_from_store(
            store, step4_rows, candidate_index_set
        )
        log(
            f"[GEN-STORE] Resume: fetched reusable candidates for {len(candidates)} indices "
            f"in {time.perf_counter() - collect_started:.1f}s"
        )
        stats = {
            "approved": 0,
            "pending": 0,
            "delta": 0,
            "local_failures": 0,
            "locally_exhausted": 0,
            "ignored_exhausted": 0,
        }
        classify_started = time.perf_counter()
        log("[GEN-STORE] Resume: normalizing and classifying all compatible attempts")
        for row_num, (idx, idx_candidates) in enumerate(sorted(candidates.items()), start=1):
            idx = int(idx)
            if idx < 0 or idx >= len(step4_rows):
                continue

            normalized = [
                normalize_candidate(c, step4_rows[idx], tokenizer, args.teacher, step7_max_tokens)
                for c in idx_candidates
            ]
            if not normalized:
                continue

            approved_fits = [item for item in normalized if _is_approved_fit(item)]
            if approved_fits:
                n_approved = len(approved_fits)
                if max_repetition > 1:
                    approved_reps[idx] = min(n_approved, max_repetition)
                    if approved_reps[idx] >= max_repetition:
                        approved.add(idx)
                else:
                    approved.add(idx)
                    approved_reps[idx] = 1
                denied.discard(idx)
                stats["approved"] += 1
                continue

            failure_keys = {
                _failure_attempt_key(idx, item)
                for item in normalized
                if _is_local_failure_evidence(item)
            }
            loaded_failure_count = len(failure_keys)
            if loaded_failure_count:
                stats["local_failures"] += loaded_failure_count
                failure_attempt_keys_seen[idx] = set(failure_keys)
                failure_attempt_counts[idx] = max(
                    int(failure_attempt_counts.get(idx, 0)),
                    int(loaded_failure_count),
                )

            if idx in denied:
                stats["ignored_exhausted"] += 1
                continue

            if failure_attempt_counts.get(idx, 0) >= RESAMPLE_MAX_RETRIES:
                denied.add(idx)
                stats["locally_exhausted"] += 1
                continue

            pending_fits = [
                (item.row, int(item.candidate.attempt_id))
                for item in normalized
                if _is_pending_fit(item)
            ]
            if pending_fits:
                pending_generation_cache[idx] = sorted(pending_fits, key=_pending_attempt_sort_key)
                stats["pending"] += len(pending_fits)
                continue

            continuable = [item for item in normalized if _is_continuable(item)]
            if continuable:
                best = sorted(continuable, key=_continuable_sort_key)[0]
                delta_generation_cache[idx] = (best.row, int(best.remaining_tokens))
                stats["delta"] += 1

            if row_num % 2000 == 0:
                log(
                    f"[GEN-STORE] Resume: classified {row_num}/{len(candidates)} candidate indices "
                    f"(approved={stats['approved']}, pending_attempts={stats['pending']}, "
                    f"delta={stats['delta']}, local_failures={stats['local_failures']}, "
                    f"locally_exhausted={stats['locally_exhausted']})"
                )
        log(
            f"[GEN-STORE] Resume: classified {len(candidates)} candidate indices "
            f"in {time.perf_counter() - classify_started:.1f}s"
        )

        log(
            "[GEN-STORE] Resumed shared state: "
            f"{len(approved)} saturated, {len(denied)} locally exhausted, "
            f"{sum(len(v) for v in pending_generation_cache.values())} pending complete attempts, "
            f"{len(delta_generation_cache)} delta continuations "
            f"({stats['local_failures']} loaded local failure evidence, "
            f"{stats['locally_exhausted']} newly exhausted from loaded evidence, "
            f"{stats['ignored_exhausted']} already exhausted)"
        )
        log(f"[GEN-STORE] Resume completed in {time.perf_counter() - resume_started:.1f}s")

    _save_failure_counts(failure_counts_file, failure_attempt_counts)
    np.save(approved_file, np.array(sorted(approved)))
    np.save(denied_file, np.array(sorted(denied)))

    port = find_free_port()
    num_gpus = get_num_gpus()
    physical_gpus = get_physical_num_gpus()

    base_dp = 4
    if physical_gpus > 0:
        ratio = num_gpus / physical_gpus
        calculated_dp = int(base_dp * ratio)
    else:
        calculated_dp = 1

    dp_size = max(1, calculated_dp)

    if dp_size > num_gpus:
        dp_size = num_gpus

    tp_size = max(1, num_gpus // dp_size)

    log(f"GPU Config: Visible={num_gpus}, Physical={physical_gpus}, DP={dp_size}, TP={tp_size}")

    log(f"Starting generation server on port {port} (TP={tp_size}, DP={dp_size})...")
    server_proc = subprocess.Popen(
        [
            sys.executable,
            str(BASE_DIR / "m1" / "src" / "generation" / "generate-server.py"),
            "serve",
            "--model_name", args.teacher,
            "--tensor_parallel_size", str(tp_size),
            "--data_parallel_size", str(dp_size),
            "--port", str(port),
            "--workdir", str(workdir),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        for attempt in range(150):
            try:
                resp = req.get(f"http://localhost:{port}/health", timeout=5)
                if resp.status_code == 200:
                    log("Generation server is ready")
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Generation server did not start within 5 minutes")

        tokenizer = None
        round_num = 0
        total_generated = 0
        total_accepted = 0

        while True:
            round_num += 1
            log(f"\n{'='*40} ROUND {round_num} {'='*40}")
            retryable_failed = sum(
                1
                for idx, count in failure_attempt_counts.items()
                if idx not in approved and idx not in denied and 0 < count <= RESAMPLE_MAX_RETRIES
            )
            log(
                f"Verdict state: {len(approved)} saturated, {len(denied)} locally exhausted, "
                f"{retryable_failed} retryable failed; "
                f"target selected approvals: {total_budget}"
            )

            batch_indices, cluster_stats = rebalance_budget(
                cluster_orderings, source_weights, approved, denied, total_budget,
                max_repetition=max_repetition, approved_reps=approved_reps,
                rep_denied=rep_denied,
            )
            frontier_summary = _summarize_frontier_stats(cluster_stats)
            log(
                "Current allocation frontier: "
                f"{frontier_summary['approved_frontier']} already-approved slots, "
                f"{frontier_summary['inspect_frontier']} candidates to inspect this round, "
                f"{frontier_summary['skipped_frontier']} blocked/skipped slots"
            )

            original_batch_size = len(batch_indices)
            acceptance_rate = total_accepted / total_generated if total_generated > 0 else 0.0

            if original_batch_size > 0 and original_batch_size < 100 and total_generated > 0:
                batch_indices = oversample_batch(
                    cluster_orderings, batch_indices, approved, denied,
                    acceptance_rate, original_batch_size, dp_size, idx_to_cluster
                )
                log(
                    f"Oversampled request batch: {original_batch_size} frontier candidates "
                    f"-> {len(batch_indices)} total requests "
                    f"(historical frontier approval rate: {acceptance_rate:.3f})"
                )

            if not batch_indices:
                final_ready = _select_approved_by_allocation(
                    cluster_orderings, source_weights, approved, denied, total_budget,
                    max_repetition=max_repetition, approved_reps=approved_reps,
                )
                if len(final_ready) >= total_budget:
                    log("Budget met under current allocation. Finishing resample loop.")
                    break
                raise RuntimeError(
                    f"All allocation frontiers exhausted! Current allocation has "
                    f"{len(final_ready)}/{total_budget} approved selections, "
                    f"Saturated={len(approved)}, Locally exhausted={len(denied)}. Cannot continue."
                )

            pending_cached_indices = [i for i in batch_indices if i in pending_generation_cache]
            pending_cached_attempt_count = sum(
                len(pending_generation_cache[i]) for i in pending_cached_indices
            )
            delta_cached_indices = [
                i for i in batch_indices
                if i in delta_generation_cache and i not in pending_generation_cache
            ]
            uncached_indices = [
                i for i in batch_indices
                if i not in pending_generation_cache and i not in delta_generation_cache
            ]
            judge_count = len(batch_indices)
            oversampled_count = judge_count - original_batch_size
            generation_count = len(delta_cached_indices) + len(uncached_indices)

            selected_label = (
                f"{original_batch_size} frontier + {oversampled_count} oversampled"
                if oversampled_count > 0
                else f"{original_batch_size} frontier"
            )
            log(f"Round {round_num} selection: {judge_count} candidates to inspect ({selected_label})")

            log(
                f"  Already have complete cached traces for: "
                f"{_count_phrase(pending_cached_attempt_count, 'pending attempt')}"
            )
            log(
                f"  Need generation before inspection can finish: "
                f"{_count_phrase(generation_count, 'candidate')} "
                f"({_count_phrase(len(uncached_indices), 'fresh generation')}, "
                f"{_count_phrase(len(delta_cached_indices), 'cached continuation')})"
            )
            if delta_cached_indices:
                log(f"  Continuation delta plan by max_new_tokens: {_format_delta_plan(delta_cached_indices, delta_generation_cache)}")
            if uncached_indices:
                log(f"  Fresh generation budget: max_new_tokens={step7_max_tokens}")

            unique_in_round = len(set(batch_indices))
            max_reps_queued_in_round = max((batch_indices.count(idx) for idx in set(batch_indices)), default=0)
            unique_approved_cumulative = len(approved_reps)
            max_reps_approved_cumulative = max(approved_reps.values(), default=0)

            stats_msg = (
                f"Resample Statistics (Plan):\n"
                f"  - Unique prompts in this round's batch: {unique_in_round}\n"
                f"  - Max repetitions queued for a single prompt: {max_reps_queued_in_round}\n"
                f"  - Cumulative unique prompts with approved attempts: {unique_approved_cumulative}\n"
                f"  - Cumulative max approved repetitions for a single prompt: {max_reps_approved_cumulative}"
            )
            log(stats_msg)

            with open(resample_log, "a") as f:
                f.write(f"\n{'='*40} ROUND {round_num} PLAN (Before Inference) {'='*40}\n")
                f.write(f"{stats_msg}\n\n")
                if oversampled_count > 0:
                    f.write(
                        f"Planned inspection: {judge_count} samples "
                        f"({original_batch_size} frontier + {oversampled_count} oversampled)\n"
                    )
                else:
                    f.write(f"Planned inspection: {judge_count} frontier samples\n")
                f.write(
                    "Already have complete cached traces for: "
                    f"{_count_phrase(pending_cached_attempt_count, 'pending attempt')}\n"
                )
                f.write(
                    "Planned generation before inspection can finish: "
                    f"{_count_phrase(generation_count, 'sample')} "
                    f"({_count_phrase(len(uncached_indices), 'fresh generation')}, "
                    f"{_count_phrase(len(delta_cached_indices), 'cached continuation')})\n"
                )
                if delta_cached_indices:
                    f.write(
                        "Continuation delta plan by max_new_tokens: "
                        f"{_format_delta_plan(delta_cached_indices, delta_generation_cache)}\n"
                    )
                if uncached_indices:
                    f.write(f"Fresh generation budget: max_new_tokens={step7_max_tokens}\n")
                for src, stats in cluster_stats.items():
                    src_alloc = sum(s['total_alloc'] for s in stats.values())
                    src_picked = sum(s['to_inspect'] for s in stats.values())
                    f.write(f"  Source: {src} (alloc={src_alloc}, to_inspect={src_picked})\n")
                    for cid, s in stats.items():
                        if s['to_inspect'] > 0:
                            f.write(f"    Cluster {cid}: alloc={s['total_alloc']} approved={s['approved']} "
                                    f"to_inspect={s['to_inspect']} remaining_after_selection={s['remaining_after']}\n")

            round_generations = []
            if pending_cached_indices:
                for idx in pending_cached_indices:
                    if idx in pending_generation_cache:
                        attempts = pending_generation_cache[idx]
                        if attempts:
                            attempt = attempts.pop(0)
                            round_generations.append(attempt)
                            if not attempts:
                                pending_generation_cache.pop(idx)

            resolved_generation_verdicts: List[Tuple[int, int, str, Tuple]] = []
            with tempfile.TemporaryDirectory(prefix=f"step6_7_round{round_num}_", dir=str(workdir)) as tmpdir:
                tmpdir = Path(tmpdir)

                def insert_generated_rows(generated_rows: List[Dict], label: str) -> None:
                    inserted_count = 0
                    reused_counts = {APPROVED: 0, DENIED: 0, INCOMPLETE: 0, PENDING: 0}
                    for gen in generated_rows:
                        idx = int(gen["original_index"])
                        inserted = store.insert_attempt_from_trace(
                            trace_row=gen,
                            step4_row=step4_rows[idx],
                            global_index=idx,
                            requested_max_new_tokens=step7_max_tokens,
                            status=PENDING,
                            source_run=args.run_name,
                            source_path=str(workdir),
                        )
                        if not inserted.inserted and inserted.status in {APPROVED, DENIED, INCOMPLETE}:
                            failure_key = (
                                "round",
                                int(round_num),
                                str(label),
                                int(idx),
                                int(inserted.attempt_id),
                                str(inserted.status),
                            )
                            resolved_generation_verdicts.append(
                                (idx, inserted.attempt_id, inserted.status, failure_key)
                            )
                            reused_counts[inserted.status] += 1
                        else:
                            round_generations.append((gen, inserted.attempt_id))
                            if inserted.inserted:
                                inserted_count += 1
                            else:
                                reused_counts[PENDING] += 1
                    log(
                        f"[GEN-STORE] {label}: {len(generated_rows)} rows processed, "
                        f"{inserted_count} new attempts, "
                        f"{_format_counts(reused_counts, [APPROVED, DENIED, INCOMPLETE, PENDING])} reused canonical rows"
                    )

                if delta_cached_indices:
                    log(
                        f"Round {round_num} generation phase: continuing "
                        f"{len(delta_cached_indices)} cached partial traces "
                        f"({_format_delta_plan(delta_cached_indices, delta_generation_cache)})"
                    )
                    claimed_delta, blocked_delta = store.claim_indices(
                        prefixes_by_idx,
                        delta_cached_indices,
                        owner=args.run_name,
                    )
                    if blocked_delta:
                        log(
                            f"[GEN-STORE] Warning: ignoring active claims for {len(blocked_delta)} "
                            f"delta continuations. First blocked indices: {blocked_delta[:10]}"
                        )
                    try:
                        by_delta: Dict[int, List[Tuple[int, Dict]]] = {}
                        for idx in delta_cached_indices:
                            row, delta = delta_generation_cache[idx]
                            by_delta.setdefault(int(delta), []).append((idx, row))

                        for delta, items in sorted(by_delta.items()):
                            dense_input = tmpdir / f"delta_input_{delta}.jsonl"
                            dense_index = tmpdir / f"delta_indices_{delta}.npy"
                            dense_output = tmpdir / f"delta_output_{delta}.jsonl"
                            rows = []
                            for idx, row in items:
                                row = dict(row)
                                row["original_index"] = int(idx)
                                rows.append(row)
                            write_jsonl(dense_input, rows)
                            np.save(dense_index, np.arange(len(rows), dtype=int))

                            delta_cmd = [
                                sys.executable,
                                str(BASE_DIR / "m1" / "src" / "generation" / "generate-server.py"),
                                "index",
                                "--index", str(dense_index),
                                "--input_file", str(dense_input),
                                "--output_file", str(dense_output),
                                "--model_name", args.teacher,
                                "--workdir", str(workdir),
                                "--chat_template_args", json.dumps(DEFAULTS["chat_template_args"]),
                                "--max_tokens", str(delta),
                            ]
                            run_cmd(delta_cmd)
                            delta_rows = _read_jsonl_from(dense_output, 0)
                            if len(delta_rows) != len(rows):
                                raise RuntimeError(
                                    f"Expected {len(rows)} delta generations for max_tokens={delta}, "
                                    f"got {len(delta_rows)}"
                                )
                            insert_generated_rows(
                                delta_rows,
                                f"delta max_tokens={delta}",
                            )
                    finally:
                        store.release_claims(prefixes_by_idx, claimed_delta, owner=args.run_name)

                if uncached_indices:
                    log(
                        f"Round {round_num} generation phase: fresh-generating "
                        f"{len(uncached_indices)} traces with max_new_tokens={step7_max_tokens}"
                    )
                    claimed_fresh, blocked_fresh = store.claim_indices(
                        prefixes_by_idx,
                        uncached_indices,
                        owner=args.run_name,
                    )
                    if blocked_fresh:
                        log(
                            f"[GEN-STORE] Warning: ignoring active claims for {len(blocked_fresh)} "
                            f"fresh generations. First blocked indices: {blocked_fresh[:10]}"
                        )
                    try:
                        uncached_batch = tmpdir / "uncached_indices.npy"
                        uncached_output = tmpdir / "uncached_output.jsonl"
                        np.save(uncached_batch, np.array(uncached_indices, dtype=int))
                        gen_cmd = [
                            sys.executable,
                            str(BASE_DIR / "m1" / "src" / "generation" / "generate-server.py"),
                            "index",
                            "--index", str(uncached_batch),
                            "--input_file", str(workdir / "step4_continued_1k.jsonl"),
                            "--output_file", str(uncached_output),
                            "--model_name", args.teacher,
                            "--workdir", str(workdir),
                            "--chat_template_args", json.dumps(DEFAULTS["chat_template_args"]),
                            "--max_tokens", str(step7_max_tokens),
                        ]
                        run_cmd(gen_cmd)
                        fresh_rows = _read_jsonl_from(uncached_output, 0)
                        if len(fresh_rows) != len(uncached_indices):
                            raise RuntimeError(
                                f"Expected {len(uncached_indices)} fresh generations, "
                                f"got {len(fresh_rows)}"
                            )
                        insert_generated_rows(fresh_rows, "fresh")
                    finally:
                        store.release_claims(prefixes_by_idx, claimed_fresh, owner=args.run_name)

            new_generations = round_generations

            newly_approved = []
            newly_denied = []
            newly_incomplete = []
            newly_pending = []
            judge_candidates = []
            local_failure_events: List[Tuple[int, Tuple]] = []
            for idx, attempt_id, status, failure_key in resolved_generation_verdicts:
                if status == APPROVED:
                    newly_approved.append(idx)
                    approved_reps[idx] = approved_reps.get(idx, 0) + 1
                    if approved_reps[idx] >= max_repetition:
                        approved.add(idx)
                elif status == DENIED:
                    newly_denied.append(idx)
                    local_failure_events.append((idx, failure_key))
                elif status == INCOMPLETE:
                    newly_incomplete.append(idx)
                    local_failure_events.append((idx, failure_key))

            for gen, attempt_id in new_generations:
                idx = gen["original_index"]
                if config["dataset_type"] == "m23k":
                    if "m23k_gt" not in locals():
                        m23k_gt = _load_m23k_ground_truth(config)

                    conversations = gen.get("conversations", [])
                    final_content = None

                    if conversations:
                        for msg in reversed(conversations):
                            if msg.get("role") == "assistant":
                                content = msg.get("content", "")
                                try:
                                    final_content = extract_final_answer_content(content, args.teacher)
                                    break
                                except ValueError:
                                    pass

                    is_approved = False
                    if final_content:
                        matches = re.findall(r'\\boxed\{([A-Za-z])\}', final_content)
                        if matches:
                            pred_letter = matches[-1].upper()
                            gt_letter = m23k_gt.get(idx, "")
                            if pred_letter == gt_letter:
                                is_approved = True

                    if final_content is None:
                        newly_incomplete.append(idx)
                        _update_attempt_status(store, attempt_id, INCOMPLETE, args=args, workdir=workdir)
                        local_failure_events.append((idx, ("attempt", int(attempt_id))))
                    elif is_approved:
                        newly_approved.append(idx)
                        approved_reps[idx] = approved_reps.get(idx, 0) + 1
                        if approved_reps[idx] >= max_repetition:
                            approved.add(idx)
                        _update_attempt_status(store, attempt_id, APPROVED, args=args, workdir=workdir)
                    else:
                        newly_denied.append(idx)
                        _update_attempt_status(store, attempt_id, DENIED, args=args, workdir=workdir)
                        local_failure_events.append((idx, ("attempt", int(attempt_id))))

                elif config.get("eval_type", config.get("dataset_type")) == "openthoughts-math":
                    if "ot_math_gt" not in locals():
                        ot_math_gt, ot_math_prompt = _load_openthoughts_math_ground_truth(config)

                    conversations = gen.get("conversations", [])
                    if conversations:
                        last_content = str(conversations[-1].get("content", ""))
                        try:
                            final_answer = extract_final_answer_content(last_content, args.teacher)
                        except ValueError:
                            newly_incomplete.append(idx)
                            _update_attempt_status(store, attempt_id, INCOMPLETE, args=args, workdir=workdir)
                            local_failure_events.append((idx, ("attempt", int(attempt_id))))
                        else:
                            judge_candidates.append((gen, final_answer, attempt_id))
                    else:
                        newly_incomplete.append(idx)
                        _update_attempt_status(store, attempt_id, INCOMPLETE, args=args, workdir=workdir)
                        local_failure_events.append((idx, ("attempt", int(attempt_id))))

                else:
                    conversations = gen.get("conversations", [])
                    if conversations:
                        last_content = str(conversations[-1].get("content", ""))
                        try:
                            extract_final_answer_content(last_content, args.teacher)
                        except ValueError:
                            newly_incomplete.append(idx)
                            _update_attempt_status(store, attempt_id, INCOMPLETE, args=args, workdir=workdir)
                            local_failure_events.append((idx, ("attempt", int(attempt_id))))
                        else:
                            newly_approved.append(idx)
                            approved_reps[idx] = approved_reps.get(idx, 0) + 1
                            if approved_reps[idx] >= max_repetition:
                                approved.add(idx)
                            _update_attempt_status(store, attempt_id, APPROVED, args=args, workdir=workdir)
                    else:
                        newly_incomplete.append(idx)
                        _update_attempt_status(store, attempt_id, INCOMPLETE, args=args, workdir=workdir)
                        local_failure_events.append((idx, ("attempt", int(attempt_id))))

            resolved_counts: Dict[str, int] = {}
            for _idx, _attempt_id, status, _failure_key in resolved_generation_verdicts:
                resolved_counts[status] = resolved_counts.get(status, 0) + 1
            if resolved_generation_verdicts or new_generations:
                log(
                    f"Round {round_num} trace inspection: "
                    f"{len(new_generations)} selected trace attempts checked, "
                    f"{len(judge_candidates)} complete traces still need LLM judge, "
                    f"{len(newly_incomplete)} marked incomplete, "
                    f"{_format_counts(resolved_counts, [APPROVED, DENIED, INCOMPLETE])} reused existing final verdicts"
                )

            if config.get("eval_type", config.get("dataset_type")) == "openthoughts-math" and judge_candidates:
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

                if str(BASE_DIR / "m1" / "src" / "judge") not in sys.path:
                    sys.path.insert(0, str(BASE_DIR / "m1" / "src" / "judge"))
                from judge import (
                    JUDGE_SYSTEM_PROMPT_NO_THINK,
                    JUDGE_USER_TEMPLATE,
                    strict_judge_verdict as judge_strict_verdict,
                )

                judge_prompts = []
                for gen, final_answer, _attempt_id in judge_candidates:
                    idx = gen["original_index"]

                    user_content = JUDGE_USER_TEMPLATE.format(
                        question=ot_math_prompt.get(idx, ""),
                        generated_answer=final_answer,
                        gt_solution=ot_math_gt.get(idx, ""),
                    )
                    messages = [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT_NO_THINK},
                        {"role": "user", "content": user_content},
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        **_judge_template_kwargs(args.teacher),
                    )
                    judge_prompts.append(text)

                allowed_token_ids = _judge_allowed_token_ids(tokenizer)
                log(
                    f"Round {round_num} judge request: {len(judge_prompts)} prompts to port {port}, "
                    f"max_tokens=1, temperature=0.0, allowed verdict tokens={allowed_token_ids} "
                    "(Correct/Incorrect), malformed_fail_threshold="
                    f"{JUDGE_MALFORMED_THRESHOLD:.0%}"
                )

                resp = req.post(
                    f"http://localhost:{port}/v1/completions",
                    json={
                        "model": "teacher",
                        "prompt": judge_prompts,
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "skip_special_tokens": False,
                        "allowed_token_ids": allowed_token_ids,
                    },
                    timeout=None,
                )
                resp.raise_for_status()
                resp_data = resp.json()

                choices = resp_data["choices"]
                choices_sorted = sorted(choices, key=lambda c: c["index"])
                if len(choices_sorted) != len(judge_candidates):
                    raise RuntimeError(
                        f"Judge returned {len(choices_sorted)} choices for "
                        f"{len(judge_candidates)} prompts"
                    )

                parsed_verdicts = []
                malformed = []
                for (gen, _, attempt_id), choice in zip(judge_candidates, choices_sorted):
                    idx = gen["original_index"]
                    response_text = choice["text"]
                    verdict, parsed = _strict_judge_verdict(response_text, judge_strict_verdict)
                    parsed_verdicts.append((idx, attempt_id, verdict))
                    if verdict == PENDING:
                        malformed.append((idx, response_text, parsed.get("verdict", "")))

                malformed_rate = len(malformed) / max(1, len(judge_candidates))
                if malformed_rate > JUDGE_MALFORMED_THRESHOLD:
                    samples = [
                        {"idx": idx, "raw": raw[:300], "parsed_verdict": str(parsed)[:100]}
                        for idx, raw, parsed in malformed[:5]
                    ]
                    raise RuntimeError(
                        "Malformed judge responses exceeded threshold: "
                        f"{len(malformed)}/{len(judge_candidates)} "
                        f"({malformed_rate:.1%}) > {JUDGE_MALFORMED_THRESHOLD:.0%}. "
                        f"Sample malformed responses: {samples}"
                    )

                judge_correct = 0
                judge_incorrect = 0
                for idx, attempt_id, verdict in parsed_verdicts:
                    if verdict == APPROVED:
                        judge_correct += 1
                        newly_approved.append(idx)
                        approved_reps[idx] = approved_reps.get(idx, 0) + 1
                        if approved_reps[idx] >= max_repetition:
                            approved.add(idx)
                        _update_attempt_status(store, attempt_id, APPROVED, args=args, workdir=workdir)
                    elif verdict == DENIED:
                        judge_incorrect += 1
                        newly_denied.append(idx)
                        _update_attempt_status(store, attempt_id, DENIED, args=args, workdir=workdir)
                        local_failure_events.append((idx, ("attempt", int(attempt_id))))
                    else:
                        newly_pending.append(idx)

                log(
                    f"[GEN-STORE] Wrote {len(judge_candidates)} judge verdicts "
                    "immediately after round-level judge call"
                )
                log(
                    f"Round {round_num} judge results: {judge_correct} approved, {judge_incorrect} denied, "
                    f"{len(malformed)} malformed/pending (out of {len(judge_candidates)} complete)"
                )
            elif config.get("eval_type", config.get("dataset_type")) == "openthoughts-math":
                log(
                    f"Round {round_num} judge request skipped: "
                    f"{len(judge_candidates)} complete traces queued for LLM judge"
                )

            orig_set = set(batch_indices[:original_batch_size])
            orig_approved = sum(1 for idx in newly_approved if idx in orig_set)

            total_generated += original_batch_size
            total_accepted += orig_approved

            log(
                f"Round {round_num}: {len(newly_approved)} approved, "
                f"{len(newly_denied)} denied, {len(newly_incomplete)} incomplete, "
                f"{len(newly_pending)} pending"
            )
            if original_batch_size != len(batch_indices):
                log(f"  (From original {original_batch_size} samples: {orig_approved} approved)")

            if newly_approved:
                for idx in newly_approved:
                    denied.discard(idx)

            retrying_failed, exhausted_failed = _add_local_failure_events(
                failure_attempt_counts,
                failure_attempt_keys_seen,
                approved,
                denied,
                local_failure_events,
            )

            if retrying_failed:
                log(
                    f"  Retrying {len(retrying_failed)} failed examples fresh from Step 4 "
                    f"(local failure threshold={RESAMPLE_MAX_RETRIES})"
                )
            if exhausted_failed:
                log(
                    f"  Locally exhausted {len(exhausted_failed)} examples after "
                    f"{RESAMPLE_MAX_RETRIES} failures"
                )

            _save_failure_counts(failure_counts_file, failure_attempt_counts)
            np.save(approved_file, np.array(sorted(approved)))
            np.save(denied_file, np.array(sorted(denied)))

            newly_approved_by_cluster = {}
            newly_denied_by_cluster = {}
            for idx in newly_approved:
                key = idx_to_cluster.get(idx)
                if key:
                    newly_approved_by_cluster[key] = newly_approved_by_cluster.get(key, 0) + 1
            for idx in newly_denied:
                key = idx_to_cluster.get(idx)
                if key:
                    newly_denied_by_cluster[key] = newly_denied_by_cluster.get(key, 0) + 1

            unique_approved_cumulative_post = len(approved_reps)
            max_reps_approved_cumulative_post = max(approved_reps.values(), default=0)

            stats_msg_post = (
                f"Resample Statistics (Actual Results):\n"
                f"  - Cumulative unique prompts with approved attempts: {unique_approved_cumulative_post}\n"
                f"  - Cumulative max approved repetitions for a single prompt: {max_reps_approved_cumulative_post}"
            )
            log(stats_msg_post)

            with open(resample_log, "a") as f:
                f.write(f"\n{'='*40} ROUND {round_num} RESULTS (After Inference) {'='*40}\n")
                f.write(f"{stats_msg_post}\n\n")
                f.write(f"Batch size: {original_batch_size} (oversampled to {len(batch_indices)})\n")
                f.write(f"Newly approved: {len(newly_approved)}\n")
                f.write(f"Newly denied: {len(newly_denied)}\n")
                f.write(f"Newly incomplete: {len(newly_incomplete)}\n")
                f.write(f"Still pending: {len(newly_pending)}\n")
                f.write(f"New local failures: {len(retrying_failed) + len(exhausted_failed)}\n")
                f.write(f"Retrying failed: {len(retrying_failed)}\n")
                f.write(f"Newly locally exhausted: {len(exhausted_failed)}\n")
                f.write(
                    "Local failure counts at threshold: "
                    f"{sum(1 for idx, count in failure_attempt_counts.items() if idx not in approved and int(count) >= RESAMPLE_MAX_RETRIES)}\n"
                )
                current_approved_slots = len(_select_approved_by_allocation(
                    cluster_orderings, source_weights, approved, denied, total_budget,
                    max_repetition=max_repetition, approved_reps=approved_reps,
                ))
                f.write(f"Total approved slots: {current_approved_slots}/{total_budget}\n")
                f.write(f"Total saturated prompts: {len(approved)}\n")
                f.write(f"Total locally exhausted: {len(denied)}\n")

                all_keys = set(newly_approved_by_cluster.keys()) | set(newly_denied_by_cluster.keys())
                for (src, cid) in sorted(all_keys):
                    n_app = newly_approved_by_cluster.get((src, cid), 0)
                    n_den = newly_denied_by_cluster.get((src, cid), 0)
                    f.write(f"    {src} / Cluster {cid}: +{n_app} approved, -{n_den} denied\n")

    finally:
        log("Stopping generation server...")
        stop_cmd = [
            sys.executable,
            str(BASE_DIR / "m1" / "src" / "generation" / "generate-server.py"),
            "stop",
            "--workdir", str(workdir),
        ]
        try:
            run_cmd(stop_cmd)
        except Exception as e:
            log(f"Warning: Failed to stop server cleanly: {e}")
        try:
            server_proc.terminate()
            server_proc.wait(timeout=10)
        except Exception:
            server_proc.kill()

    total_approved_reps = sum(approved_reps.values())
    log(
        f"\nResample complete: {total_approved_reps} total approved repetitions across "
        f"{len(approved_reps)} unique indices (target budget {total_budget}), "
        f"{len(denied)} locally exhausted"
    )

    final_indices_list = _select_approved_by_allocation(
        cluster_orderings, source_weights, approved, denied, total_budget,
        max_repetition=max_repetition, approved_reps=approved_reps,
    )

    if len(final_indices_list) < total_budget:
        pending_indices, _ = rebalance_budget(
            cluster_orderings, source_weights, approved, denied, total_budget,
            max_repetition=max_repetition, approved_reps=approved_reps,
            rep_denied=rep_denied,
        )
        raise RuntimeError(
            f"Final allocation yielded {len(final_indices_list)} indices, "
            f"expected {total_budget}. Pending allocation candidates: "
            f"{len(pending_indices)}. Refusing to fall back to unrelated approvals."
        )

    final_indices = np.array(sorted(final_indices_list))

    np.save(final_idx_file, final_indices)
    log(f"Saved {len(final_indices)} indices (with repetitions) to {final_idx_file}")

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    final_unique = set(int(idx) for idx in final_indices.tolist())
    final_candidates = collect_generation_candidates_from_store(
        store, step4_rows, final_unique
    )

    from collections import Counter
    final_counts = Counter(int(idx) for idx in final_indices_list)

    missing = []
    with open(final_traces_file, "w", encoding="utf-8") as f:
        for idx in sorted(final_unique):
            needed = final_counts.get(idx, 1)
            idx_cands = final_candidates.get(idx, [])

            approved_cands = []
            for c in idx_cands:
                norm = normalize_candidate(c, step4_rows[idx], tokenizer, args.teacher, step7_max_tokens)
                if _is_approved_fit(norm):
                    approved_cands.append(norm)

            if len(approved_cands) < needed:
                final_resolved = resolve_best_candidates(
                    {idx: idx_cands}, step4_rows, tokenizer, args.teacher, step7_max_tokens
                )
                best = final_resolved.get(idx)
                if (
                    best is None
                    or best.candidate.verdict != APPROVED
                    or not best.complete
                    or best.relation == "longer"
                ):
                    missing.append(idx)
                    continue
                f.write(json.dumps(best.row, ensure_ascii=False) + "\n")
                if needed > 1:
                    log(
                        f"WARNING: Index {idx} needed {needed} approved traces "
                        f"but only 1 available from store"
                    )
            else:
                for cand in approved_cands[:needed]:
                    f.write(json.dumps(cand.row, ensure_ascii=False) + "\n")

    if missing:
        raise RuntimeError(
            f"Could not materialize {len(missing)} approved traces from generation store. "
            f"First missing indices: {missing[:10]}"
        )

    total_written = sum(final_counts.values())
    log(f"Saved {total_written} approved traces ({len(final_unique)} unique) to {final_traces_file}")
    store.close()

    return final_traces_file, False
