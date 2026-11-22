""""""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .utils import WORKDIR_BASE, get_step7_max_tokens, inputs_from_run_config, log


APPROVED = "approved"
DENIED = "denied"
INCOMPLETE = "incomplete"
PENDING = "pending"


@dataclass(frozen=True)
class GenerationSource:
    path: Path
    inputs: Dict[str, Any]
    step7_max_tokens: int
    has_verdicts: bool


@dataclass(frozen=True)
class GenerationCandidate:
    original_index: int
    source_path: Path
    source_max_tokens: int
    row: Dict[str, Any]
    requested_max_tokens: Optional[int] = None
    verdict: str = PENDING
    attempt_id: Optional[int] = None


@dataclass(frozen=True)
class NormalizedCandidate:
    candidate: GenerationCandidate
    row: Dict[str, Any]
    complete: bool
    relation: str
    remaining_tokens: int


def get_final_answer_delimiter(teacher: str) -> str:
    """"""
    teacher_l = teacher.lower()
    if "qwq" in teacher_l or "qwen3" in teacher_l:
        return "</think>"
    if teacher_l == "original":
        return "<|im_start|>answer\n"
    if "gpt-oss" in teacher_l or "gpt_oss" in teacher_l:
        return "<|end|><|start|>assistant<|channel|>final<|message|>"
    if "gemma" in teacher_l:
        return "<channel|>"
    raise ValueError(
        f"Unsupported reasoning/final-answer delimiter for teacher '{teacher}'. "
        "Add this model family before using Step 7 completion checks."
    )


def _gemma_trace_has_final_answer(text: str) -> bool:
    """"""
    return r"\boxed{" in text


def has_final_answer_delimiter(row: Dict[str, Any], teacher: str) -> bool:
    try:
        delimiter = get_final_answer_delimiter(teacher)
    except ValueError:
        return False
    content = assistant_content(row)
    if delimiter in content:
        return True
    if "gemma" in teacher.lower():
        return _gemma_trace_has_final_answer(content)
    return False


def extract_final_answer_content(text: str, teacher: str) -> str:
    """"""
    delimiter = get_final_answer_delimiter(teacher)
    if delimiter in text:
        return text.split(delimiter)[-1].strip()
    if "gemma" in teacher.lower() and _gemma_trace_has_final_answer(text):
        return text.strip()
    raise ValueError(
        f"Expected final-answer delimiter {delimiter!r} was not found for teacher '{teacher}'."
    )


def assistant_message(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conversations = row.get("conversations") or []
    for msg in reversed(conversations):
        role = str(msg.get("role", msg.get("from", ""))).lower()
        if role == "assistant":
            return msg
    return None


def assistant_content(row: Dict[str, Any]) -> str:
    msg = assistant_message(row)
    if msg is None:
        return ""
    return str(msg.get("content", msg.get("value", "")))


def set_assistant_content(row: Dict[str, Any], content: str) -> None:
    msg = assistant_message(row)
    if msg is None:
        raise ValueError("Cannot set assistant content: row has no assistant message")
    msg["content"] = content
    if "value" in msg:
        msg["value"] = content


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _decode_prefix(tokenizer, text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(tokens[:max_tokens])


def step4_assistant_token_count(step4_row: Dict[str, Any], tokenizer) -> int:
    return _token_count(tokenizer, assistant_content(step4_row))


def truncate_row_to_step7_budget(
    row: Dict[str, Any],
    step4_row: Dict[str, Any],
    tokenizer,
    current_step7_max_tokens: int,
    prefix_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """"""
    normalized = copy.deepcopy(row)
    if prefix_tokens is None:
        prefix_tokens = step4_assistant_token_count(step4_row, tokenizer)
    target_total = prefix_tokens + int(current_step7_max_tokens)
    clipped = _decode_prefix(tokenizer, assistant_content(normalized), target_total)
    set_assistant_content(normalized, clipped)

    step4_max_tokens = step4_row.get("max_tokens", prefix_tokens)
    if not isinstance(step4_max_tokens, int):
        step4_max_tokens = prefix_tokens
    normalized["max_tokens"] = int(step4_max_tokens) + int(current_step7_max_tokens)
    return normalized


def normalize_candidate(
    candidate: GenerationCandidate,
    step4_row: Dict[str, Any],
    tokenizer,
    teacher: str,
    current_step7_max_tokens: int,
) -> NormalizedCandidate:
    current_cap = int(current_step7_max_tokens)
    source_cap = int(candidate.source_max_tokens)
    requested_cap = (
        int(candidate.requested_max_tokens)
        if candidate.requested_max_tokens is not None
        else source_cap
    )
    prefix_tokens: Optional[int] = None

    if source_cap > current_cap:
        prefix_tokens = step4_assistant_token_count(step4_row, tokenizer)
        row = truncate_row_to_step7_budget(
            candidate.row, step4_row, tokenizer, current_cap, prefix_tokens
        )
        relation = "exact" if requested_cap == current_cap else "longer"
        remaining = 0
    else:
        row = copy.deepcopy(candidate.row)
        relation = "exact" if source_cap == current_cap else "shorter"
        remaining = max(0, current_cap - source_cap)

    if "original_index" not in row or row["original_index"] is None:
        row["original_index"] = int(candidate.original_index)

    step4_max_tokens = step4_row.get("max_tokens")
    if not isinstance(step4_max_tokens, int):
        if prefix_tokens is None:
            prefix_tokens = step4_assistant_token_count(step4_row, tokenizer)
        step4_max_tokens = prefix_tokens
    row["max_tokens"] = int(step4_max_tokens) + min(source_cap, current_cap)

    complete = has_final_answer_delimiter(row, teacher)
    return NormalizedCandidate(
        candidate=candidate,
        row=row,
        complete=complete,
        relation=relation,
        remaining_tokens=remaining,
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_trace_map(path: Path, candidate_indices: Optional[Set[int]] = None) -> Dict[int, Dict[str, Any]]:
    """"""
    result: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return result

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = row.get("original_index")
            if idx is None:
                idx = line_num
            idx = int(idx)
            if candidate_indices is not None and idx not in candidate_indices:
                continue
            if "original_index" not in row or row["original_index"] is None:
                row["original_index"] = idx
            result[idx] = row
    return result


def validate_prefix_compatibility(
    current_step4: Path,
    other_step4: Path,
    sample_size: int = 50,
) -> bool:
    """"""
    try:
        if current_step4.resolve() == other_step4.resolve():
            log("[MERGE]   Prefix check: same file (symlink), skipping content check")
            return True
    except OSError:
        pass

    line_count = 0
    with open(current_step4, "r", encoding="utf-8") as f:
        for _ in f:
            line_count += 1

    if line_count == 0:
        return False

    sample_indices = set(random.sample(range(line_count), min(sample_size, line_count)))

    def _sample(path: Path) -> Dict[int, str]:
        data: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i not in sample_indices or not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = assistant_content(row)
                if content:
                    data[i] = content[:1020]
        return data

    current_data = _sample(current_step4)
    other_data = _sample(other_step4)
    shared = set(current_data.keys()) & set(other_data.keys())
    if not shared:
        log("[MERGE]   Prefix check: no shared indices to compare")
        return False

    mismatches = 0
    for idx in shared:
        if current_data[idx] != other_data[idx]:
            mismatches += 1
            if mismatches == 1:
                log(f"[MERGE]   Prefix MISMATCH at index {idx}")

    if mismatches > 0:
        log(f"[MERGE]   Prefix check FAILED: {mismatches}/{len(shared)} mismatches")
        return False

    log(f"[MERGE]   Prefix check PASSED: {len(shared)} indices verified")
    return True


def _is_step7_trusted(candidate_inputs: Dict[str, Any]) -> bool:
    override_steps = candidate_inputs.get("manual_override_steps", []) or []
    if not override_steps:
        return True
    order = {
        "step1": 1.0, "step2": 2.0, "step3": 3.0, "step4": 4.0,
        "step4_5": 4.5, "step4_7": 4.7, "step4_8": 4.8,
        "step5": 5.0, "step6": 6.0, "step6_7": 6.5, "step7": 7.0,
        "step8": 8.0, "step9": 9.0,
    }
    min_override = min(order.get(step, float("inf")) for step in override_steps)
    return 7.0 < min_override


def scan_generation_reuse_sources(workdir: Path, config: Dict[str, Any], args) -> List[GenerationSource]:
    """"""
    sources: List[GenerationSource] = []
    if args.force_rerun:
        log("[MERGE] --force-rerun set; skipping historic Step 7 generation reuse")
        return sources

    current_resolved = workdir.resolve()

    try:
        rel_parts = workdir.resolve().relative_to(WORKDIR_BASE.resolve()).parts
        scan_root = WORKDIR_BASE / rel_parts[0] if rel_parts else WORKDIR_BASE
    except ValueError:
        scan_root = WORKDIR_BASE

    current_step4 = workdir / "step4_continued_1k.jsonl"

    for config_file in scan_root.rglob("pipeline_config.json"):
        candidate_dir = config_file.parent
        if candidate_dir.resolve() == current_resolved:
            continue

        has_traces = (
            (candidate_dir / "step7_full_traces_tmp.jsonl").exists() or
            (candidate_dir / "step7_full_traces.jsonl").exists() or
            (candidate_dir / "step7_selected_traces.jsonl").exists()
        )
        if not has_traces:
            continue

        try:
            with open(config_file, encoding="utf-8") as f:
                other_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        other_inputs = inputs_from_run_config(other_config)
        if not _is_step7_trusted(other_inputs):
            continue

        checks = [
            other_inputs.get("task") == args.task,
            other_inputs.get("teacher") == args.teacher,
            other_inputs.get("seed") == args.seed,
            other_inputs.get("prompt_source") == config.get("prompt_source"),
        ]
        if not all(checks):
            continue

        other_step4 = candidate_dir / "step4_continued_1k.jsonl"
        if not other_step4.exists():
            log(f"[MERGE] Skipping {candidate_dir.name}: no step4_continued_1k.jsonl")
            continue
        if not validate_prefix_compatibility(current_step4, other_step4):
            log(f"[MERGE] Skipping {candidate_dir.name}: prefix mismatch")
            continue

        has_verdicts = (
            (candidate_dir / "step6_diversity" / "step6_approved.npy").exists() and
            (candidate_dir / "step6_diversity" / "step6_denied.npy").exists()
        )
        source = GenerationSource(
            path=candidate_dir,
            inputs=other_inputs,
            step7_max_tokens=get_step7_max_tokens(other_inputs),
            has_verdicts=has_verdicts,
        )
        sources.append(source)
        log(
            f"[MERGE] Found compatible folder: {candidate_dir.name} "
            f"(resample={other_inputs.get('resample')}, n={other_inputs.get('num_samples')}, "
            f"step7_max_tokens={source.step7_max_tokens})"
        )

    return sources


def collect_generation_candidates(
    sources: Iterable[GenerationSource],
    candidate_indices: Optional[Set[int]] = None,
) -> Dict[int, List[GenerationCandidate]]:
    """"""
    import numpy as np

    by_idx: Dict[int, List[GenerationCandidate]] = {}

    for source in sources:
        if source.has_verdicts:
            approved_file = source.path / "step6_diversity" / "step6_approved.npy"
            denied_file = source.path / "step6_diversity" / "step6_denied.npy"
            approved = set(np.load(approved_file).tolist())
            denied = set(np.load(denied_file).tolist())

            trace_rows: Dict[int, Dict[str, Any]] = {}
            trace_rows.update(
                load_trace_map(source.path / "step7_full_traces_tmp.jsonl", candidate_indices)
            )
            trace_rows.update(
                load_trace_map(source.path / "step7_full_traces.jsonl", candidate_indices)
            )
            trace_rows.update(
                load_trace_map(source.path / "step7_selected_traces.jsonl", candidate_indices)
            )

            for idx, row in trace_rows.items():
                if candidate_indices is not None and idx not in candidate_indices:
                    continue
                if idx in approved:
                    verdict = APPROVED
                elif idx in denied:
                    verdict = DENIED
                else:
                    verdict = PENDING
                by_idx.setdefault(idx, []).append(
                    GenerationCandidate(
                        original_index=idx,
                        source_path=source.path,
                        source_max_tokens=source.step7_max_tokens,
                        row=row,
                        verdict=verdict,
                    )
                )
        else:
            selected_file = source.path / "step7_selected_traces.jsonl"
            indexed_file = source.path / "step7_full_traces.jsonl"
            traces_file = selected_file if selected_file.exists() else indexed_file
            labeled_idx_file = source.path / "step6_labeled_idx.npy"
            if not traces_file.exists() or not labeled_idx_file.exists():
                continue
            fully_generated = set(np.load(labeled_idx_file).tolist())
            if candidate_indices is not None:
                fully_generated &= set(candidate_indices)
            trace_rows = load_trace_map(traces_file, fully_generated)
            for idx, row in trace_rows.items():
                if idx not in fully_generated:
                    continue
                by_idx.setdefault(idx, []).append(
                    GenerationCandidate(
                        original_index=idx,
                        source_path=source.path,
                        source_max_tokens=source.step7_max_tokens,
                        row=row,
                        verdict=PENDING,
                    )
                )

    return by_idx


def step4_prefixes_for_indices(
    step4_rows: List[Dict[str, Any]],
    candidate_indices: Optional[Set[int]] = None,
) -> Dict[int, str]:
    """"""
    if candidate_indices is None:
        indices = range(len(step4_rows))
    else:
        indices = sorted(int(idx) for idx in candidate_indices)

    prefixes: Dict[int, str] = {}
    for idx in indices:
        if idx < 0 or idx >= len(step4_rows):
            continue
        prefix = assistant_content(step4_rows[idx])
        if prefix:
            prefixes[int(idx)] = prefix
    return prefixes


def collect_generation_candidates_from_store(
    store,
    step4_rows: List[Dict[str, Any]],
    candidate_indices: Optional[Set[int]] = None,
) -> Dict[int, List[GenerationCandidate]]:
    """"""
    started = time.perf_counter()
    prefixes = step4_prefixes_for_indices(step4_rows, candidate_indices)
    log(f"[GEN-STORE] Reuse scan: built {len(prefixes)} Step 4 prefix hashes")
    by_idx: Dict[int, List[GenerationCandidate]] = {}

    attempts = store.fetch_attempts_for_prefixes(prefixes)
    log(
        f"[GEN-STORE] Reuse scan: fetched {len(attempts)} matching attempt rows "
        f"from SQLite in {time.perf_counter() - started:.1f}s"
    )
    for row_num, attempt in enumerate(attempts, start=1):
        idx = int(attempt["global_index"])
        try:
            row = json.loads(attempt["trace_json"])
        except json.JSONDecodeError:
            continue
        if "original_index" not in row or row["original_index"] is None:
            row["original_index"] = idx
        source_path = Path(attempt.get("source_path") or str(store.path))
        by_idx.setdefault(idx, []).append(
            GenerationCandidate(
                original_index=idx,
                source_path=source_path,
                source_max_tokens=int(attempt.get("generated_tokens") or attempt["requested_max_new_tokens"]),
                requested_max_tokens=int(attempt["requested_max_new_tokens"]),
                row=row,
                verdict=str(attempt["status"]),
                attempt_id=int(attempt["attempt_id"]),
            )
        )
        if row_num % 5000 == 0:
            log(
                f"[GEN-STORE] Reuse scan: parsed {row_num}/{len(attempts)} rows "
                f"covering {len(by_idx)} indices"
            )

    log(
        f"[GEN-STORE] Reuse scan: collected candidates for {len(by_idx)} indices "
        f"in {time.perf_counter() - started:.1f}s"
    )
    return by_idx


def _normalized_rank_key(normalized: NormalizedCandidate, current_cap: int) -> Tuple[Any, ...]:
    cand = normalized.candidate
    source_cap = int(cand.source_max_tokens)
    extendable = normalized.relation == "shorter" and normalized.remaining_tokens > 0

    if normalized.complete and cand.verdict == APPROVED:
        category = 0
    elif normalized.complete and cand.verdict == PENDING:
        category = 1
    elif not normalized.complete and extendable and cand.verdict != DENIED:
        category = 2
    elif normalized.complete and cand.verdict == DENIED:
        category = 3
    elif not normalized.complete and cand.verdict in {PENDING, INCOMPLETE}:
        category = 4
    else:
        category = 5

    if normalized.complete:
        relation_order = {"exact": 0, "longer": 1, "shorter": 2}.get(normalized.relation, 3)
        length_preference = abs(source_cap - current_cap)
    elif extendable:
        relation_order = 0
        length_preference = -source_cap
    else:
        relation_order = {"exact": 0, "longer": 1, "shorter": 2}.get(normalized.relation, 3)
        length_preference = abs(source_cap - current_cap)

    return (
        category,
        relation_order,
        length_preference,
        abs(source_cap - current_cap),
        str(cand.source_path),
    )


def resolve_best_candidates(
    candidates_by_idx: Dict[int, List[GenerationCandidate]],
    step4_rows: List[Dict[str, Any]],
    tokenizer,
    teacher: str,
    current_step7_max_tokens: int,
) -> Dict[int, NormalizedCandidate]:
    """"""
    started = time.perf_counter()
    resolved: Dict[int, NormalizedCandidate] = {}
    current_cap = int(current_step7_max_tokens)
    total = len(candidates_by_idx)
    for row_num, (idx, candidates) in enumerate(candidates_by_idx.items(), start=1):
        if idx < 0 or idx >= len(step4_rows):
            continue
        normalized = [
            normalize_candidate(c, step4_rows[idx], tokenizer, teacher, current_cap)
            for c in candidates
        ]
        if not normalized:
            continue
        resolved[idx] = min(normalized, key=lambda item: _normalized_rank_key(item, current_cap))
        if row_num % 1000 == 0:
            log(
                f"[GEN-STORE] Reuse resolve: normalized {row_num}/{total} indices "
                f"({len(resolved)} reusable so far, {time.perf_counter() - started:.1f}s)"
            )
    log(
        f"[GEN-STORE] Reuse resolve: selected best candidates for {len(resolved)}/{total} "
        f"indices in {time.perf_counter() - started:.1f}s"
    )
    return resolved
