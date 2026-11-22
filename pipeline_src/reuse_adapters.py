""""""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _replace_path(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        _remove_path(dst)
    src.replace(dst)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assistant_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for msg in reversed(messages):
        role = str(msg.get("role", msg.get("from", ""))).lower()
        if role == "assistant":
            return msg
    return None


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except FileNotFoundError:
        return a.absolute() == b.absolute()


def _tokenizer_name_for_partial_y(current_inputs: Dict[str, Any]) -> str:
    teacher = str(current_inputs.get("teacher", ""))
    if teacher.lower() == "original":
        return str(current_inputs["original_model_name"])
    return teacher


def materialize_step4_5_partial_y(
    *,
    source_workdir: Path,
    target_workdir: Path,
    current_inputs: Dict[str, Any],
    source_inputs: Dict[str, Any],
) -> None:
    """"""
    source_jsonl = source_workdir / "step4_5_partial_completion.jsonl"
    source_idx = source_workdir / "step4_5_selected_global_indices.npy"
    target_jsonl = target_workdir / "step4_5_partial_completion.jsonl"
    target_idx = target_workdir / "step4_5_selected_global_indices.npy"
    target_meta = target_workdir / "step4_5_metadata.json"
    source_meta = _load_json(source_workdir / "step4_5_metadata.json") or {}

    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_name_for_partial_y(current_inputs),
        trust_remote_code=True,
    )
    target_token_budget = 100 + 1000 + int(current_inputs["partial_y"])

    tmp_jsonl = target_workdir / "step4_5_partial_completion.jsonl.tmp"
    tmp_jsonl.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(source_jsonl, "r", encoding="utf-8") as fin, open(tmp_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            payload = json.loads(line)
            messages = payload.get("conversations") or []
            assistant = _assistant_message(messages)
            if assistant is None:
                raise ValueError("step4_5 partial-completion row has no assistant message")

            content = str(assistant.get("content", assistant.get("value", "")))
            tokens = tokenizer.encode(content, add_special_tokens=False)
            clipped = tokenizer.decode(tokens[:target_token_budget])
            assistant["content"] = clipped
            if "value" in assistant:
                assistant["value"] = clipped
            original_max_tokens = payload.get("max_tokens", len(tokens))
            if not isinstance(original_max_tokens, int):
                original_max_tokens = len(tokens)
            payload["max_tokens"] = min(int(original_max_tokens), target_token_budget)

            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            row_count += 1

    _replace_path(tmp_jsonl, target_jsonl)
    if not _same_path(source_idx, target_idx):
        shutil.copy2(source_idx, target_idx)

    metadata = dict(source_meta)
    metadata.update(
        {
            "y_tokens": int(current_inputs["partial_y"]),
            "requested_partial_y": int(current_inputs["partial_y"]),
            "source_partial_y": int(source_inputs["partial_y"]),
            "materialized_from": str(source_workdir),
            "materialized_row_count": int(row_count),
            "clip_token_budget": int(target_token_budget),
            "selected_global_indices_file": str(target_idx),
        }
    )
    _write_json(target_meta, metadata)


def _copy_filtered_dir_tree(
    *,
    source_dir: Path,
    target_dir: Path,
    keep_names: Iterable[str],
) -> None:
    keep = set(keep_names)
    tmp_dir = target_dir.with_name(target_dir.name + ".tmp-materialize")
    if tmp_dir.exists() or tmp_dir.is_symlink():
        _remove_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if source_dir.exists():
        for child in sorted(source_dir.iterdir()):
            if child.name not in keep:
                continue
            dst = tmp_dir / child.name
            if child.is_dir():
                src_tree = child.resolve() if child.is_symlink() else child
                shutil.copytree(src_tree, dst, symlinks=True)
            else:
                src_file = child.resolve() if child.is_symlink() else child
                shutil.copy2(src_file, dst)

    _replace_path(tmp_dir, target_dir)


def materialize_step4_8_rank(
    *,
    source_workdir: Path,
    target_workdir: Path,
    current_inputs: Dict[str, Any],
    source_inputs: Dict[str, Any],
) -> None:
    """"""
    requested_rank = int(current_inputs["lora_train_rank"])
    source_rank = int(source_inputs["lora_train_rank"])

    source_out_dir = source_workdir / "step4_8_perturbed_lora"
    target_out_dir = target_workdir / "step4_8_perturbed_lora"
    source_meta_path = source_workdir / "step4_8_metadata.json"
    target_meta_path = target_workdir / "step4_8_metadata.json"
    source_meta = _load_json(source_meta_path) or {}

    keep_dir_names: List[str] = []
    for child in sorted(source_out_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("dir") or "_over_" not in child.name:
            continue
        dir_tok = child.name.split("_over_", 1)[0]
        try:
            dir_idx = int(dir_tok[len("dir"):])
        except ValueError:
            continue
        if dir_idx < requested_rank:
            keep_dir_names.append(child.name)

    if not keep_dir_names:
        raise RuntimeError(
            f"Cannot materialize narrowed step4_8 output from {source_workdir}: "
            f"no dir<i>_over_* checkpoints found for requested rank {requested_rank}"
        )

    _copy_filtered_dir_tree(
        source_dir=source_out_dir,
        target_dir=target_out_dir,
        keep_names=keep_dir_names,
    )

    source_subset = source_workdir / "step4_8_sampled_subset.jsonl"
    target_subset = target_workdir / "step4_8_sampled_subset.jsonl"
    if source_subset.exists():
        if not _same_path(source_subset, target_subset):
            shutil.copy2(source_subset, target_subset)
    elif target_subset.exists() or target_subset.is_symlink():
        _remove_path(target_subset)

    source_history = source_workdir / "step4_8_inflection_history"
    target_history = target_workdir / "step4_8_inflection_history"
    if source_history.exists():
        keep_history = [f"dir{i}.json" for i in range(requested_rank)]
        _copy_filtered_dir_tree(
            source_dir=source_history,
            target_dir=target_history,
            keep_names=keep_history,
        )
    elif target_history.exists() or target_history.is_symlink():
        _remove_path(target_history)

    source_plots = source_workdir / "step4_8_debug_inflection_plot"
    target_plots = target_workdir / "step4_8_debug_inflection_plot"
    if source_plots.exists():
        keep_plot_dirs = [f"dir{i}" for i in range(requested_rank)]
        _copy_filtered_dir_tree(
            source_dir=source_plots,
            target_dir=target_plots,
            keep_names=keep_plot_dirs,
        )
    elif target_plots.exists() or target_plots.is_symlink():
        _remove_path(target_plots)

    metadata = dict(source_meta)
    target_ckpts = [str(target_out_dir / name) for name in keep_dir_names]
    metadata["over_checkpoints"] = sorted(target_ckpts)
    metadata["requested_lora_rank"] = int(requested_rank)
    metadata["source_lora_rank"] = int(source_rank)
    metadata["materialized_from"] = str(source_workdir)
    metadata["lora_rank"] = int(requested_rank)

    results = metadata.get("per_direction_results")
    if isinstance(results, list):
        filtered_results = []
        for row in results:
            try:
                dir_idx = int(row.get("dir_idx"))
            except (TypeError, ValueError):
                continue
            if dir_idx >= requested_rank:
                continue
            row_copy = dict(row)
            ix_targets = row_copy.get("ix_targets")
            if isinstance(ix_targets, list):
                new_ix_targets = []
                for item in ix_targets:
                    item_copy = dict(item)
                    output_dir = item_copy.get("output_dir")
                    if output_dir:
                        item_copy["output_dir"] = str(target_out_dir / Path(str(output_dir)).name)
                    new_ix_targets.append(item_copy)
                row_copy["ix_targets"] = new_ix_targets
            filtered_results.append(row_copy)
        metadata["per_direction_results"] = filtered_results

    _write_json(target_meta_path, metadata)
