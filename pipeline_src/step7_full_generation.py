""""""

import json
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .step7_reuse import (
    DENIED,
    INCOMPLETE,
    collect_generation_candidates_from_store,
    has_final_answer_delimiter,
    load_jsonl,
    resolve_best_candidates,
    step4_prefixes_for_indices,
    write_jsonl,
)
from .generation_store import GenerationStore, PENDING
from .utils import (
    BASE_DIR,
    check_output_exists,
    get_num_gpus,
    get_physical_num_gpus,
    get_step7_max_tokens,
    log,
    run_cmd,
    DEFAULTS,
    STEP7_INDEXED_TRACES,
    STEP7_SELECTED_TRACES,
)

from collections import Counter


def _run_continuation_group(
    *,
    workdir: Path,
    working_rows: List[Dict],
    indices: List[int],
    max_tokens: int,
    args,
    num_gpus: int,
    data_parallel: int,
) -> List[Dict]:
    """"""
    with tempfile.TemporaryDirectory(prefix=f"step7_gen_{max_tokens}_", dir=str(workdir)) as tmp:
        tmp_dir = Path(tmp)
        input_file = tmp_dir / "input.jsonl"
        output_dir = tmp_dir / "out"
        index_file = tmp_dir / "indices.npy"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(input_file, working_rows)
        np.save(index_file, np.array(sorted(indices), dtype=int))

        cmd = [
            sys.executable,
            str(BASE_DIR / "m1" / "src" / "generation" / "generate.py"),
            "--input_file", str(input_file),
            "--output_dir", str(output_dir),
            "--model_name", args.teacher,
            "--max_tokens", str(max_tokens),
            "--num_gpus", str(num_gpus),
            "--data_parallel", str(data_parallel),
            "--seed", str(args.seed),
            "--mode", "continuation",
            "--continuation_indices", str(index_file),
            "--chat_template_args", json.dumps(DEFAULTS["chat_template_args"]),
        ]
        run_cmd(cmd)
        return load_jsonl(output_dir / "continuation_results.jsonl")


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    """"""
    max_repetition = getattr(args, "max_repetition", 1)
    output_filename = STEP7_SELECTED_TRACES if max_repetition > 1 else STEP7_INDEXED_TRACES
    output_file = workdir / output_filename

    if check_output_exists(output_file, "Step 7 (Full generation)"):
        return output_file, True

    log("=" * 60)
    log("STEP 7: Full Generation for Selected Samples")
    if max_repetition > 1:
        log(f"  max_repetition={max_repetition}")
    log("=" * 60)

    if args.teacher.lower() == "original":
        log("[BYPASS] Linking original dataset instead of full generation")
        input_file = workdir / "step1_generation_100tokens.jsonl"
        if output_file.is_file() or output_file.is_symlink():
            output_file.unlink()
        output_file.symlink_to(input_file.name)
        return output_file, False

    num_gpus = get_num_gpus()
    physical_gpus = get_physical_num_gpus()
    step7_max_tokens = get_step7_max_tokens(args)

    base_dp = 4
    if physical_gpus > 0:
        ratio = num_gpus / physical_gpus
        calculated_dp = int(base_dp * ratio)
    else:
        calculated_dp = 1

    data_parallel = max(1, calculated_dp)

    if data_parallel > num_gpus:
        data_parallel = num_gpus

    log(f"GPU Config: Visible={num_gpus}, Physical={physical_gpus}, DP={data_parallel} (TP={num_gpus//data_parallel})")
    print(num_gpus)

    step4_file = workdir / "step4_continued_1k.jsonl"
    step4_rows = load_jsonl(step4_file)
    working_rows = list(step4_rows)
    raw_indices = np.load(workdir / "step6_labeled_idx.npy").tolist()
    index_counts = Counter(int(i) for i in raw_indices)
    selected_indices = sorted(index_counts.keys())
    selected_set = set(selected_indices)

    has_repetitions = any(c > 1 for c in index_counts.values())
    if has_repetitions:
        total_traces = sum(index_counts.values())
        log(f"Step 7: {len(selected_indices)} unique indices, {total_traces} total traces (with repetitions)")
    else:
        log(f"Step 7: {len(selected_indices)} unique indices")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    store = GenerationStore.for_run(config, args)
    prefixes_by_idx = step4_prefixes_for_indices(step4_rows, selected_set)

    if args.force_rerun:
        log("[GEN-STORE] --force-rerun set; skipping shared generation reuse")
        candidates = {}
    else:
        candidates = collect_generation_candidates_from_store(store, step4_rows, selected_set)

    resolved = resolve_best_candidates(
        candidates, step4_rows, tokenizer, args.teacher, step7_max_tokens
    )

    generation_groups: Dict[int, List[int]] = {}
    reused = 0
    continued = 0
    fresh = 0

    for idx in selected_indices:
        best = resolved.get(idx)
        if best is None or best.candidate.verdict == DENIED:
            generation_groups.setdefault(step7_max_tokens, []).append(idx)
            fresh += 1
            continue

        working_rows[idx] = best.row
        source_cap = best.candidate.source_max_tokens
        if source_cap < step7_max_tokens and not has_final_answer_delimiter(best.row, args.teacher):
            delta = step7_max_tokens - source_cap
            generation_groups.setdefault(delta, []).append(idx)
            continued += 1
        elif best.candidate.verdict == INCOMPLETE and not has_final_answer_delimiter(best.row, args.teacher):
            generation_groups.setdefault(step7_max_tokens, []).append(idx)
            fresh += 1
        else:
            reused += 1

    log(
        f"Step 7 reuse plan (rep 1): {reused} reused, {continued} continued from shorter runs, "
        f"{fresh} fresh generations (max_new_tokens={step7_max_tokens})"
    )

    for max_tokens, indices in sorted(generation_groups.items()):
        if not indices:
            continue
        log(f"Generating {len(indices)} Step 7 traces with max_new_tokens={max_tokens}")
        claimed, blocked = store.claim_indices(
            prefixes_by_idx,
            indices,
            owner=args.run_name,
        )
        if blocked:
            raise RuntimeError(
                f"Generation store has active claims for {len(blocked)} selected indices; "
                f"try again after the claim lease expires. First blocked indices: {blocked[:10]}"
            )
        try:
            generated_rows = _run_continuation_group(
                workdir=workdir,
                working_rows=working_rows,
                indices=claimed,
                max_tokens=max_tokens,
                args=args,
                num_gpus=num_gpus,
                data_parallel=data_parallel,
            )
            for idx in claimed:
                working_rows[idx] = generated_rows[idx]
                store.insert_attempt_from_trace(
                    trace_row=generated_rows[idx],
                    step4_row=step4_rows[idx],
                    global_index=idx,
                    requested_max_new_tokens=step7_max_tokens,
                    status=PENDING,
                    source_run=args.run_name,
                    source_path=str(output_file),
                )
        finally:
            store.release_claims(prefixes_by_idx, claimed, owner=args.run_name)

    if not has_repetitions:
        write_jsonl(output_file, working_rows)
    else:
        rep_indices = [idx for idx, count in index_counts.items() if count > 1]
        extra_needed = {idx: index_counts[idx] - 1 for idx in rep_indices}
        total_extra = sum(extra_needed.values())

        log(f"Step 7 multi-rep: generating {total_extra} additional traces for {len(rep_indices)} repeated indices")

        extra_rows: Dict[int, List[Dict]] = {idx: [] for idx in rep_indices}

        for rep_round in range(2, max_repetition + 1):
            round_indices = [idx for idx in rep_indices if len(extra_rows[idx]) < extra_needed[idx]]
            if not round_indices:
                break

            log(f"  Repetition round {rep_round}: generating for {len(round_indices)} indices")

            round_working = list(step4_rows)
            for idx in round_indices:
                round_working[idx] = step4_rows[idx]

            claimed, blocked = store.claim_indices(
                prefixes_by_idx,
                round_indices,
                owner=args.run_name,
            )
            if blocked:
                raise RuntimeError(
                    f"Generation store has active claims for {len(blocked)} selected indices in rep round {rep_round}; "
                    f"try again after the claim lease expires. First blocked indices: {blocked[:10]}"
                )

            try:
                round_generated = _run_continuation_group(
                    workdir=workdir,
                    working_rows=round_working,
                    indices=claimed,
                    max_tokens=step7_max_tokens,
                    args=args,
                    num_gpus=num_gpus,
                    data_parallel=data_parallel,
                )

                for idx in claimed:
                    extra_rows[idx].append(round_generated[idx])
                    store.insert_attempt_from_trace(
                        trace_row=round_generated[idx],
                        step4_row=step4_rows[idx],
                        global_index=idx,
                        requested_max_new_tokens=step7_max_tokens,
                        status=PENDING,
                        source_run=args.run_name,
                        source_path=str(output_file),
                    )
            finally:
                store.release_claims(prefixes_by_idx, claimed, owner=args.run_name)

        log(f"Writing {sum(index_counts.values())} total traces to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for idx in sorted(working_rows.keys()) if isinstance(working_rows, dict) else range(len(working_rows)):
                if idx not in selected_set:
                    continue
                f.write(json.dumps(working_rows[idx], ensure_ascii=False) + "\n")
                for extra_row in extra_rows.get(idx, []):
                    f.write(json.dumps(extra_row, ensure_ascii=False) + "\n")

    store.close()

    return output_file, False
