#!/usr/bin/env python3
""""""

from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing
import os
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "m1" / "output" / "teacher-selection"
LOSS_SCRIPT = REPO_ROOT / "m1" / "src" / "process_loss" / "get_mean_loss_optimized.py"
DEFAULT_STUDENT = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATA_PATH = REPO_ROOT / "dataset" / "numina-math-4-sources" / "train.jsonl"
DEEPSEEK_R1_MODEL = "deepseek-ai/deepseek-r1-0528-maas"


def deepseek_r1_base_url() -> str:
    url = os.environ.get("DEEPSEEK_R1_BASE_URL")
    if url:
        return url
    project = os.environ.get("GCP_PROJECT")
    if not project:
        raise RuntimeError("Set DEEPSEEK_R1_BASE_URL or GCP_PROJECT")
    return (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        f"{project}/locations/us-central1/endpoints/openapi/chat/completions"
    )


DEEPSEEK_R1_TEACHER_ALIASES = {
    "deepseek-r1": DEEPSEEK_R1_MODEL,
    "deepseek-r1-novita": DEEPSEEK_R1_MODEL,
    DEEPSEEK_R1_MODEL: DEEPSEEK_R1_MODEL,
}
DEEPSEEK_R1_CONTROL_TOKENS = {
    "parser": "think_tags",
    "think_start": "<think>",
    "think_end": "</think>",
    "completion_starts_inside_think": True,
    "special_tokens": ["<think>", "</think>"],
}
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_utils import get_available_gpus, resolve_physical_gpu
from teachers import (
    TEACHERS,
    apply_chat_template,
    count_tokens,
    extract_question,
    filter_supported_kwargs,
    predownload_models,
    sanitize_teacher_name,
    supports_arg,
    utc_now,
)


TEACHER_GENERATION_INSTRUCTION = "In your own words, explain what the problem is asking."
STUDENT_EVAL_INSTRUCTION = "Return your answer within \\boxed{}."


def flatten_teachers(values: Iterable[str]) -> List[str]:
    teachers: List[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                teachers.append(part)
    return teachers


def resolve_teachers(values: Sequence[str]) -> List[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return list(TEACHERS.keys()) + [DEEPSEEK_R1_MODEL]
    teachers = flatten_teachers(values)
    resolved: List[str] = []
    unknown: List[str] = []
    for teacher in teachers:
        alias = DEEPSEEK_R1_TEACHER_ALIASES.get(teacher.lower())
        if alias is not None:
            resolved.append(alias)
            continue
        if teacher in TEACHERS:
            resolved.append(teacher)
            continue
        unknown.append(teacher)
    if unknown:
        known = "\n  ".join(list(TEACHERS.keys()) + sorted(DEEPSEEK_R1_TEACHER_ALIASES.keys()))
        raise ValueError(f"Unknown teacher(s): {unknown}\nKnown teachers:\n  {known}")
    return list(dict.fromkeys(resolved))


def is_deepseek_r1_teacher(model_name: str) -> bool:
    return model_name.lower() in DEEPSEEK_R1_TEACHER_ALIASES or model_name == DEEPSEEK_R1_MODEL


def teacher_config(model_name: str) -> Dict[str, Any]:
    if is_deepseek_r1_teacher(model_name):
        return {
            "inference_backend": "vertex_openapi",
            "endpoint": deepseek_r1_base_url(),
            "control_tokens": DEEPSEEK_R1_CONTROL_TOKENS,
        }
    return TEACHERS[model_name]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(item).__name__}")
            rows.append(item)
    return rows


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}, got {type(item).__name__}")
                rows.append(item)
        return rows

    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "samples", "train", "rows"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Expected {path} to contain a JSON list, JSONL rows, or a dict with data/samples/train/rows.")


def infer_answer_type(data_path: Path, answer_type: str) -> str:
    if answer_type != "auto":
        return answer_type
    lower_path = str(data_path).lower()
    if "m23k" in lower_path or "medmcqa" in lower_path:
        return "choice"
    return "others"


def infer_prompt_policy(data_path: Path, prompt_policy: str, answer_type: str) -> str:
    if prompt_policy != "auto":
        return prompt_policy
    lower_path = str(data_path).lower()
    if "m23k" in lower_path or answer_type == "choice":
        return "end"
    if "numina" in lower_path or "openthoughts-math" in lower_path or "physics" in lower_path:
        return "begin"
    return "none"


def build_teacher_and_student_prompts(original_prompt: str) -> Tuple[str, str]:
    """"""
    return (
        f"{TEACHER_GENERATION_INSTRUCTION} {original_prompt}",
        f"{STUDENT_EVAL_INSTRUCTION} {original_prompt}",
    )


def sample_data(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_json_or_jsonl(args.data)
    if args.num_examples > len(rows):
        raise ValueError(f"--num-examples={args.num_examples} but {args.data} only has {len(rows)} rows")

    rng = random.Random(args.seed)
    selected_indices = rng.sample(range(len(rows)), args.num_examples)
    answer_type = infer_answer_type(args.data, args.answer_type)
    prompt_policy = infer_prompt_policy(args.data, args.prompt_policy, answer_type)

    samples = []
    for sample_id, original_index in enumerate(selected_indices):
        row = rows[original_index]
        original_prompt = extract_question(row)
        teacher_prompt, student_eval_prompt = build_teacher_and_student_prompts(original_prompt)
        samples.append(
            {
                "sample_id": sample_id,
                "dataset_path": str(args.data),
                "dataset_name": args.data.parent.name if args.data.parent.name else args.data.stem,
                "original_index": original_index,
                "original_prompt": original_prompt,
                "teacher_prompt": teacher_prompt,
                "student_eval_prompt": student_eval_prompt,
                "prompt_policy": prompt_policy,
                "answer_type": answer_type,
                "box_prompt_added": True,
                "source": row.get("source"),
                "category": row.get("category"),
                "domain": row.get("domain"),
                "id": row.get("id"),
            }
        )

    metadata = {
        "data_path": str(args.data),
        "num_rows": len(rows),
        "num_examples": args.num_examples,
        "seed": args.seed,
        "selected_indices": selected_indices,
        "prompt_policy": prompt_policy,
        "answer_type": answer_type,
        "teacher_generation_instruction": TEACHER_GENERATION_INSTRUCTION,
        "student_eval_instruction": STUDENT_EVAL_INSTRUCTION,
    }
    return samples, metadata


def build_user_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def clean_exact_control_text(text: str, control: Dict[str, Any]) -> str:
    cleaned = text
    for pattern in control.get("full_control_patterns", []):
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)
    for token in sorted(control.get("special_tokens", []), key=len, reverse=True):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def parse_harmony(text: str, control: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    warnings: List[str] = []
    segment_re = re.compile(
        r"(?:<\|start\|>assistant)?<\|channel\|>([A-Za-z_]+)<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|start\|>|$)",
        flags=re.DOTALL,
    )
    matches = list(segment_re.finditer(text))
    if not matches:
        return "", clean_exact_control_text(text, control), warnings

    reasoning_channels = {channel.lower() for channel in control.get("reasoning_channels", [])}
    final_channels = {channel.lower() for channel in control.get("final_channels", [])}
    reasoning_parts = []
    final_parts = []

    for match in matches:
        channel = match.group(1).lower()
        content = clean_exact_control_text(match.group(2), control)
        if not content:
            continue
        if channel in reasoning_channels:
            reasoning_parts.append(content)
        elif channel in final_channels:
            final_parts.append(content)

    if not final_parts:
        remaining = clean_exact_control_text(segment_re.sub("", text), control)
        if remaining:
            final_parts.append(remaining)
        else:
            warnings.append("no_harmony_final_channel")

    return "\n\n".join(reasoning_parts).strip(), "\n\n".join(final_parts).strip(), warnings


def parse_gemma_channel(text: str, control: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    warnings: List[str] = []
    pattern = re.compile(
        r"<\|channel>(thought|analysis|reasoning)\s*(.*?)(<channel\|>|$)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return "", clean_exact_control_text(text, control), warnings

    reasoning_parts = []
    for match in matches:
        reasoning_parts.append(clean_exact_control_text(match.group(2), control))
        if match.group(3) != "<channel|>":
            warnings.append("missing_gemma_channel_end")

    remaining = clean_exact_control_text(pattern.sub("", text), control)
    return "\n\n".join(part for part in reasoning_parts if part).strip(), remaining, warnings


def parse_tagged_think(text: str, control: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    warnings: List[str] = []
    start = control.get("think_start", "<think>")
    end = control.get("think_end", "</think>")
    pattern = re.compile(
        rf"{re.escape(start)}\s*(.*?)\s*{re.escape(end)}",
        flags=re.DOTALL | re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        if control.get("completion_starts_inside_think") and end in text:
            reasoning, remaining = text.split(end, 1)
            return clean_exact_control_text(reasoning, control), clean_exact_control_text(remaining, control), warnings
        if control.get("allow_unclosed_think") and start in text:
            start_idx = text.find(start)
            warnings.append("missing_think_end")
            final_before = clean_exact_control_text(text[:start_idx], control)
            reasoning = clean_exact_control_text(text[start_idx + len(start) :], control)
            return reasoning, final_before, warnings
        return "", clean_exact_control_text(text, control), warnings

    reasoning_parts = [clean_exact_control_text(match.group(1), control) for match in matches]
    remaining = clean_exact_control_text(pattern.sub("", text), control)
    return "\n\n".join(part for part in reasoning_parts if part).strip(), remaining, warnings


def parse_huatuo_markdown(text: str, control: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    warnings: List[str] = []
    pattern = re.compile(
        r"##\s*Thinking\s*(.*?)\s*##\s*Final Response\s*(.*)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return (
            clean_exact_control_text(match.group(1), control),
            clean_exact_control_text(match.group(2), control),
            warnings,
        )

    think_only = re.search(r"##\s*Thinking\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if think_only:
        warnings.append("missing_huatuo_final_response")
        return clean_exact_control_text(think_only.group(1), control), "", warnings

    return "", clean_exact_control_text(text, control), warnings


def parse_generation_text(model_name: str, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    control = config.get("control_tokens", {})
    parser = control.get("parser", "plain")
    if parser == "harmony":
        reasoning_text, final_answer_text, warnings = parse_harmony(text, control)
    elif parser == "gemma_channel":
        reasoning_text, final_answer_text, warnings = parse_gemma_channel(text, control)
    elif parser == "huatuo_markdown":
        reasoning_text, final_answer_text, warnings = parse_huatuo_markdown(text, control)
    elif parser in {"think_tags", "mistral_think"}:
        reasoning_text, final_answer_text, warnings = parse_tagged_think(text, control)
    elif parser == "plain":
        reasoning_text, final_answer_text, warnings = "", clean_exact_control_text(text, control), []
    else:
        reasoning_text, final_answer_text, warnings = "", clean_exact_control_text(text, control), [f"unknown_parser:{parser}"]

    normalized_parts = [part for part in (reasoning_text, final_answer_text) if part]
    normalized_response = "\n\n".join(normalized_parts).strip()
    if not normalized_response and text.strip():
        normalized_response = clean_exact_control_text(text, control)
        warnings.append("empty_parse_fallback")

    leftovers = [
        token
        for token in control.get("special_tokens", [])
        if token and token in normalized_response
    ]
    if leftovers:
        warnings.append(f"leftover_control_tokens:{','.join(leftovers)}")

    return {
        "model": model_name,
        "parser": parser,
        "reasoning_text": reasoning_text,
        "final_answer_text": final_answer_text,
        "normalized_response": normalized_response,
        "warnings": warnings,
        "leftover_control_tokens": leftovers,
    }


def teacher_generation_max_tokens(args_or_payload: Any) -> int:
    return int(args_or_payload["num_tokens"]) + 50


def make_sampling_params(config: Dict[str, Any], args_payload: Dict[str, Any]) -> Any:
    from vllm import SamplingParams

    kwargs = dict(config.get("sampling", {}))
    kwargs["max_tokens"] = teacher_generation_max_tokens(args_payload)
    kwargs["skip_special_tokens"] = False
    kwargs.setdefault("seed", args_payload["seed"])
    return SamplingParams(**filter_supported_kwargs(SamplingParams, kwargs))


def make_llm(model_path: str, config: Dict[str, Any], args_payload: Dict[str, Any], tensor_parallel_size: int, model_name: Optional[str] = None) -> Any:
    from vllm import LLM
    from teachers import resolve_teacher_llm_kwargs

    config_llm_kwargs = resolve_teacher_llm_kwargs(model_name or model_path, config)
    trust_remote_code = bool(args_payload["trust_remote_code"] or config_llm_kwargs.get("trust_remote_code"))
    kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": args_payload["max_model_len"],
        "gpu_memory_utilization": args_payload["gpu_memory_utilization"],
        "trust_remote_code": trust_remote_code,
    }
    if args_payload.get("dtype"):
        kwargs["dtype"] = args_payload["dtype"]
    if args_payload.get("download_dir"):
        kwargs["download_dir"] = args_payload["download_dir"]
    if args_payload.get("max_num_seqs"):
        kwargs["max_num_seqs"] = args_payload["max_num_seqs"]
    if supports_arg(LLM, "generation_config"):
        kwargs["generation_config"] = "vllm"
    kwargs.update(config_llm_kwargs)
    return LLM(**filter_supported_kwargs(LLM, kwargs))


def worker_payload_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "num_tokens": args.num_tokens,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "download_dir": str(args.download_dir) if args.download_dir else None,
        "trust_remote_code": args.trust_remote_code,
        "max_num_seqs": args.max_num_seqs,
        "include_rendered_prompts": args.include_rendered_prompts,
    }


def teacher_generation_worker(
    model_name: str,
    model_path: str,
    config: Dict[str, Any],
    samples: List[Dict[str, Any]],
    physical_gpu_ids: List[int],
    output_file: str,
    args_payload: Dict[str, Any],
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in physical_gpu_ids)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        tensor_parallel_size = len(physical_gpu_ids)
        print(f"[teacher:{model_name}] Loading on physical GPUs {physical_gpu_ids} (DP=1, TP={tensor_parallel_size})")
        payload = dict(args_payload)
        if payload.get("max_num_seqs") is None and config.get("max_num_seqs") is not None:
            payload["max_num_seqs"] = int(config["max_num_seqs"])
        llm = make_llm(
            model_path,
            config,
            payload,
            tensor_parallel_size=tensor_parallel_size,
            model_name=model_name,
        )
        tokenizer = llm.get_tokenizer()
        sampling_params = make_sampling_params(config, payload)

        rendered_prompts = [
            apply_chat_template(
                tokenizer,
                build_user_messages(sample["teacher_prompt"], config.get("system_prompt")),
                config.get("chat_template_args", {}),
            )
            for sample in samples
        ]

        started_at = time.perf_counter()
        outputs = llm.generate(rendered_prompts, sampling_params, use_tqdm=True)
        generate_seconds = time.perf_counter() - started_at

        generations = []
        generated_tokens = 0
        for sample, rendered_prompt, output in zip(samples, rendered_prompts, outputs):
            completion = output.outputs[0]
            raw_text = completion.text
            token_ids = getattr(completion, "token_ids", None)
            parsed = parse_generation_text(model_name, raw_text, config)
            total_tokens = len(token_ids) if token_ids is not None else count_tokens(tokenizer, raw_text)
            generated_tokens += total_tokens
            reasoning_tokens = count_tokens(tokenizer, parsed["reasoning_text"])
            final_tokens = count_tokens(tokenizer, parsed["final_answer_text"])
            normalized_tokens = count_tokens(tokenizer, parsed["normalized_response"])

            generations.append(
                {
                    **sample,
                    "teacher": model_name,
                    "teacher_safe_name": sanitize_teacher_name(model_name),
                    "prompt_rendered": rendered_prompt if args_payload["include_rendered_prompts"] else None,
                    "raw_generation_text": raw_text,
                    "reasoning_text": parsed["reasoning_text"],
                    "final_answer_text": parsed["final_answer_text"],
                    "normalized_response": parsed["normalized_response"],
                    "parser": parsed["parser"],
                    "parser_warnings": parsed["warnings"],
                    "leftover_control_tokens": parsed["leftover_control_tokens"],
                    "token_counts": {
                        "reasoning": reasoning_tokens,
                        "final_answer": final_tokens,
                        "normalized_response": normalized_tokens,
                        "generated_total": total_tokens,
                    },
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                }
            )

        result = {
            "model": model_name,
            "created_at": utc_now(),
            "dp_size": 1,
            "tensor_parallel_size": tensor_parallel_size,
            "physical_gpu_ids": physical_gpu_ids,
            "model_path": model_path,
            "generation_parameters": {
                "sampling": {
                    **config.get("sampling", {}),
                    "max_tokens": teacher_generation_max_tokens(args_payload),
                    "skip_special_tokens": False,
                },
                "chat_template_args": config.get("chat_template_args", {}),
                "llm_kwargs": config.get("llm_kwargs", {}),
                "max_model_len": args_payload["max_model_len"],
                "gpu_memory_utilization": args_payload["gpu_memory_utilization"],
                "control_tokens": config.get("control_tokens", {}),
            },
            "throughput": {
                "num_prompts": len(samples),
                "generated_tokens": generated_tokens,
                "generate_seconds": generate_seconds,
                "tokens_per_second": generated_tokens / generate_seconds if generate_seconds > 0 else None,
            },
            "generations": generations,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        write_json(result, tmp_path)
        os.replace(tmp_path, output_path)

        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        result_queue.put((model_name, {"ok": True, "output_file": output_file, "num_generations": len(generations)}))
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result_queue.put((model_name, {"ok": False, "error": repr(exc)}))


def teacher_num_gpus(model_name: str) -> int:
    if is_deepseek_r1_teacher(model_name):
        return 0
    config = TEACHERS[model_name]
    return int(config.get("num_gpus") or config.get("parallel", {}).get("tensor_parallel_size") or 1)


def load_deepseek_generation_map(path: Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return records
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("generations", []) if isinstance(payload, dict) else []
    else:
        rows = load_jsonl(path)
    for row in rows:
        sample_id = row.get("sample_id")
        if isinstance(sample_id, int):
            records[sample_id] = row
    return records


def aggregate_deepseek_teacher_outputs(result_path: Path, thread_paths: Sequence[Path]) -> Dict[int, Dict[str, Any]]:
    records = load_deepseek_generation_map(result_path)
    for thread_path in thread_paths:
        records.update(load_deepseek_generation_map(thread_path))
    return records


def generate_for_deepseek_r1(args: argparse.Namespace, model_name: str, samples: List[Dict[str, Any]], output_file: Path) -> Path:
    del model_name
    workdir = args.output_dir / "deepseek_r1" / sanitize_teacher_name(DEEPSEEK_R1_MODEL)
    threads_dir = workdir / "thread_outputs"
    threads_dir.mkdir(parents=True, exist_ok=True)

    thread_count = max(1, min(args.deepseek_threads, len(samples)))
    thread_output_paths = [threads_dir / f"thread_{idx:02d}.jsonl" for idx in range(thread_count)]
    thread_failure_paths = [threads_dir / f"thread_{idx:02d}.failed.jsonl" for idx in range(thread_count)]

    if args.rerun_existing:
        for path in [output_file, *thread_output_paths, *thread_failure_paths]:
            if path.exists():
                path.unlink()

    expected_by_sample_id = {sample["sample_id"]: sample for sample in samples}
    records = aggregate_deepseek_teacher_outputs(output_file, thread_output_paths)
    pending_sample_ids = [sample_id for sample_id in sorted(expected_by_sample_id) if sample_id not in records]
    resumed_count = len(expected_by_sample_id) - len(pending_sample_ids)

    if resumed_count > 0:
        print(
            f"[teacher:{DEEPSEEK_R1_MODEL}] Resuming partial output: "
            f"reused={resumed_count}/{len(expected_by_sample_id)} pending={len(pending_sample_ids)}"
        )

    if not pending_sample_ids:
        ordered_rows = [records[sample_id] for sample_id in sorted(records) if sample_id in expected_by_sample_id]
        write_json({"model": DEEPSEEK_R1_MODEL, "created_at": utc_now(), "generations": ordered_rows}, output_file)
        print(f"[teacher:{DEEPSEEK_R1_MODEL}] Reusing ready output: {output_file}")
        return output_file

    if args.dry_run:
        print(
            f"[teacher:{DEEPSEEK_R1_MODEL}] Would run {len(pending_sample_ids)} pending prompts "
            f"with {thread_count} threads via Vertex OpenAPI endpoint"
        )
        return output_file

    lock = threading.Lock()
    failures = 0
    work_items = [expected_by_sample_id[sample_id] for sample_id in pending_sample_ids]
    deepseek_config = teacher_config(DEEPSEEK_R1_MODEL)

    def fetch_gcloud_access_token() -> str:
        cmd = [os.environ.get("GCLOUD_BIN", "gcloud"), "auth", "print-access-token"]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"Failed to fetch access token via gcloud: {stderr or exc}") from exc
        token = (proc.stdout or "").strip()
        if not token:
            raise RuntimeError("gcloud returned an empty access token.")
        return token

    def call_deepseek_vertex(prompt: str) -> str:
        max_attempts = 5
        base_backoff_seconds = 8.0
        payload = {
            "model": DEEPSEEK_R1_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": teacher_generation_max_tokens(vars(args)),
        }
        body = json.dumps(payload).encode("utf-8")
        timeout_seconds = max(1, int(args.deepseek_request_timeout))
        raw = ""
        for attempt in range(1, max_attempts + 1):
            access_token = fetch_gcloud_access_token()
            request = urllib.request.Request(
                deepseek_r1_base_url(),
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < max_attempts:
                    sleep_seconds = base_backoff_seconds * (2 ** (attempt - 1))
                    print(
                        f"[teacher:{DEEPSEEK_R1_MODEL}] 429 rate limited "
                        f"(attempt {attempt}/{max_attempts}); retrying in {sleep_seconds:.1f}s"
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise
        data = json.loads(raw)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Invalid completion payload: missing choices in {data!r}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            content = "".join(text_parts)
        if not isinstance(content, str):
            content = str(content)
        return content

    def build_generation(sample: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        parsed = parse_generation_text(DEEPSEEK_R1_MODEL, raw_text, deepseek_config)
        prompt_rendered = None
        if args.include_rendered_prompts:
            prompt_rendered = json.dumps([{"role": "user", "content": sample["teacher_prompt"]}], ensure_ascii=False)
        return {
            **sample,
            "teacher": DEEPSEEK_R1_MODEL,
            "teacher_safe_name": sanitize_teacher_name(DEEPSEEK_R1_MODEL),
            "prompt_rendered": prompt_rendered,
            "raw_generation_text": raw_text,
            "reasoning_text": parsed["reasoning_text"],
            "final_answer_text": parsed["final_answer_text"],
            "normalized_response": parsed["normalized_response"],
            "parser": parsed["parser"],
            "parser_warnings": parsed["warnings"],
            "leftover_control_tokens": parsed["leftover_control_tokens"],
            "token_counts": {
                "reasoning": None,
                "final_answer": None,
                "normalized_response": None,
                "generated_total": None,
            },
            "finish_reason": None,
            "stop_reason": None,
        }

    def worker(thread_idx: int, items: Sequence[Dict[str, Any]]) -> None:
        nonlocal failures
        out_path = thread_output_paths[thread_idx]
        fail_path = thread_failure_paths[thread_idx]
        successes = 0
        local_failures = 0
        total = len(items)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as out_handle, fail_path.open("a", encoding="utf-8") as fail_handle:
            for sample in items:
                try:
                    content = call_deepseek_vertex(sample["teacher_prompt"])
                    generation = build_generation(sample, content)
                    out_handle.write(json.dumps(generation, ensure_ascii=False, default=json_default) + "\n")
                    out_handle.flush()
                    successes += 1
                    if successes % args.deepseek_progress_every == 0 or successes == total:
                        print(
                            f"[teacher:{DEEPSEEK_R1_MODEL}:thread-{thread_idx:02d}] "
                            f"progress success={successes}/{total} failures={local_failures}"
                        )
                except Exception as exc:
                    fail_payload = {
                        "sample_id": sample["sample_id"],
                        "original_index": sample["original_index"],
                        "error": str(exc),
                    }
                    fail_handle.write(json.dumps(fail_payload, ensure_ascii=False) + "\n")
                    fail_handle.flush()
                    local_failures += 1
                    with lock:
                        failures += 1
        print(
            f"[teacher:{DEEPSEEK_R1_MODEL}:thread-{thread_idx:02d}] "
            f"done success={successes} failures={local_failures} total={total}"
        )

    started_at = time.perf_counter()
    chunks = [work_items[i::thread_count] for i in range(thread_count)]
    threads: List[threading.Thread] = []
    for thread_idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        thread = threading.Thread(target=worker, args=(thread_idx, chunk), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    generate_seconds = time.perf_counter() - started_at

    final_records = aggregate_deepseek_teacher_outputs(output_file, thread_output_paths)
    ordered_rows = [final_records[sample_id] for sample_id in sorted(final_records) if sample_id in expected_by_sample_id]

    if len(ordered_rows) != len(samples):
        missing = len(samples) - len(ordered_rows)
        raise RuntimeError(
            f"DeepSeek generation incomplete: {missing} rows missing. "
            f"Rerun to continue partial job. Failed calls this run: {failures}."
        )

    write_json(
        {
            "model": DEEPSEEK_R1_MODEL,
            "created_at": utc_now(),
            "dp_size": None,
            "tensor_parallel_size": 0,
            "physical_gpu_ids": [],
            "model_path": DEEPSEEK_R1_MODEL,
            "generation_parameters": {
                "inference_backend": "vertex_openapi",
                "endpoint": deepseek_r1_base_url(),
                "threads": thread_count,
                "request_timeout_seconds": args.deepseek_request_timeout,
                "max_tokens": teacher_generation_max_tokens(vars(args)),
                "control_tokens": DEEPSEEK_R1_CONTROL_TOKENS,
            },
            "throughput": {
                "num_prompts": len(samples),
                "generated_tokens": None,
                "generate_seconds": generate_seconds,
                "tokens_per_second": None,
            },
            "generations": ordered_rows,
        },
        output_file,
    )
    print(f"[teacher:{DEEPSEEK_R1_MODEL}] Finished; output={output_file}")
    return output_file


def load_ready_teacher_output(path: Path, expected_count: int) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    generations = payload.get("generations")
    if not isinstance(generations, list) or len(generations) != expected_count:
        return None
    if any(
        not isinstance(row, dict)
        or not row.get("normalized_response")
        or not row.get("student_eval_prompt")
        for row in generations
    ):
        return None
    return payload


def terminate_running(running: Dict[str, Dict[str, Any]]) -> None:
    for info in running.values():
        process = info["process"]
        if process.is_alive():
            process.terminate()
    for info in running.values():
        info["process"].join(timeout=5)


def run_teacher_jobs(args: argparse.Namespace, samples: List[Dict[str, Any]]) -> Dict[str, Path]:
    teacher_outputs: Dict[str, Path] = {}
    pending_local = []
    pending_deepseek = []
    for model_name in args.teachers:
        output_file = args.output_dir / f"{sanitize_teacher_name(model_name)}.json"
        ready = None if args.rerun_existing else load_ready_teacher_output(output_file, len(samples))
        if ready is not None:
            print(f"[teacher:{model_name}] Reusing ready output: {output_file}")
            teacher_outputs[model_name] = output_file
        elif is_deepseek_r1_teacher(model_name):
            pending_deepseek.append(model_name)
        else:
            pending_local.append(model_name)

    for model_name in pending_deepseek:
        output_file = args.output_dir / f"{sanitize_teacher_name(model_name)}.json"
        teacher_outputs[model_name] = generate_for_deepseek_r1(args, model_name, samples, output_file)

    if not pending_local:
        return teacher_outputs

    model_paths = predownload_models(pending_local, args)
    available_logical = get_available_gpus(threshold=args.gpu_threshold)
    if not available_logical:
        raise RuntimeError("No available GPUs found for teacher generation.")

    max_needed = max(teacher_num_gpus(model_name) for model_name in pending_local)
    if max_needed > len(available_logical):
        raise RuntimeError(
            f"A teacher requires {max_needed} GPUs, but only {len(available_logical)} GPUs are available: {available_logical}"
        )

    free_logical = list(available_logical)
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    running: Dict[str, Dict[str, Any]] = {}
    args_payload = worker_payload_from_args(args)

    try:
        while pending_local or running:
            launched_any = True
            while launched_any:
                launched_any = False
                for model_name in list(pending_local):
                    need = teacher_num_gpus(model_name)
                    if len(free_logical) < need:
                        continue
                    logical_group = free_logical[:need]
                    del free_logical[:need]
                    physical_group = [resolve_physical_gpu(gpu_id) for gpu_id in logical_group]
                    output_file = args.output_dir / f"{sanitize_teacher_name(model_name)}.json"
                    process = multiprocessing.Process(
                        target=teacher_generation_worker,
                        args=(
                            model_name,
                            model_paths.get(model_name, model_name),
                            TEACHERS[model_name],
                            samples,
                            physical_group,
                            str(output_file),
                            args_payload,
                            result_queue,
                        ),
                    )
                    process.start()
                    running[model_name] = {
                        "process": process,
                        "logical_gpus": logical_group,
                        "physical_gpus": physical_group,
                        "output_file": output_file,
                    }
                    pending_local.remove(model_name)
                    launched_any = True
                    print(
                        f"[scheduler] Launched {model_name} on logical GPUs {logical_group} "
                        f"(physical {physical_group}); {len(pending_local)} pending."
                    )
                    break

            if not running:
                raise RuntimeError("No running teacher jobs and no pending job can fit on available GPUs.")

            try:
                model_name, payload = result_queue.get(timeout=5)
            except queue.Empty:
                finished_without_result = [
                    (name, info["process"].exitcode)
                    for name, info in running.items()
                    if not info["process"].is_alive() and info["process"].exitcode is not None
                ]
                if finished_without_result:
                    name, exitcode = finished_without_result[0]
                    raise RuntimeError(f"Teacher worker for {name} exited with code {exitcode} without a result.")
                continue

            info = running.pop(model_name)
            info["process"].join()
            free_logical.extend(info["logical_gpus"])
            free_logical.sort(key=lambda gpu_id: available_logical.index(gpu_id))

            if info["process"].exitcode != 0:
                raise RuntimeError(f"Teacher worker for {model_name} exited with code {info['process'].exitcode}")
            if not payload.get("ok"):
                raise RuntimeError(f"Teacher worker for {model_name} failed: {payload.get('error')}")
            teacher_outputs[model_name] = Path(payload["output_file"])
            print(f"[scheduler] Finished {model_name}; freed logical GPUs {info['logical_gpus']}.")
    except Exception:
        terminate_running(running)
        raise

    return teacher_outputs


def merge_teacher_outputs(args: argparse.Namespace, samples: List[Dict[str, Any]], teacher_outputs: Dict[str, Path]) -> List[Dict[str, Any]]:
    rows = []
    row_index = 0
    for model_name in args.teachers:
        output_path = teacher_outputs[model_name]
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        generations = payload.get("generations", [])
        if len(generations) != len(samples):
            raise RuntimeError(f"{output_path} has {len(generations)} generations, expected {len(samples)}")
        for generation in generations:
            response = generation.get("normalized_response", "")
            if not response:
                raise RuntimeError(
                    f"Empty normalized response for teacher={model_name}, sample_id={generation.get('sample_id')}"
                )
            student_prompt = generation.get("student_eval_prompt")
            if not student_prompt:
                raise RuntimeError(
                    f"Missing student_eval_prompt for teacher={model_name}, sample_id={generation.get('sample_id')} "
                    "(regenerate teachers or re-sample with an updated script)."
                )
            rows.append(
                {
                    "row_index": row_index,
                    "teacher": model_name,
                    "teacher_safe_name": sanitize_teacher_name(model_name),
                    "sample_id": generation["sample_id"],
                    "original_index": generation["original_index"],
                    "dataset_path": generation["dataset_path"],
                    "dataset_name": generation["dataset_name"],
                    "prompt": student_prompt,
                    "teacher_prompt": generation["teacher_prompt"],
                    "original_prompt": generation["original_prompt"],
                    "completion": response,
                    "conversations": [
                        {"role": "user", "content": student_prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "source": generation.get("source"),
                    "category": generation.get("category"),
                    "domain": generation.get("domain"),
                    "parser": generation.get("parser"),
                    "parser_warnings": generation.get("parser_warnings", []),
                    "token_counts": generation.get("token_counts", {}),
                    "raw_generation_text": generation.get("raw_generation_text"),
                    "reasoning_text": generation.get("reasoning_text"),
                    "final_answer_text": generation.get("final_answer_text"),
                }
            )
            row_index += 1
    return rows


def run_loss(args: argparse.Namespace, merged_path: Path) -> Path:
    output_path = args.output_dir / "student_mean_losses.pt"
    if args.rerun_existing and output_path.exists():
        output_path.unlink()

    cmd = [
        sys.executable,
        str(LOSS_SCRIPT),
        "--model_path",
        args.student,
        "--dataset_path",
        str(merged_path),
        "--max_tokens",
        str(args.num_tokens),
        "--output_name",
        str(output_path),
        "--gpu_threshold",
        str(args.gpu_threshold),
    ]
    print(f"[loss] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not output_path.exists():
        raise RuntimeError(f"Loss script completed but did not create {output_path}")
    return output_path


def load_losses(loss_path: Path) -> List[float]:
    import torch

    tensor = torch.load(loss_path, map_location="cpu")
    return [float(value) for value in tensor.tolist()]


def summarize_losses(rows: List[Dict[str, Any]], losses: List[float]) -> Dict[str, Any]:
    if len(rows) != len(losses):
        raise RuntimeError(f"Loss count mismatch: {len(losses)} losses for {len(rows)} merged rows")

    grouped: Dict[str, List[float]] = defaultdict(list)
    for row, loss in zip(rows, losses):
        row["mean_loss"] = loss
        grouped[row["teacher"]].append(loss)

    teacher_summary = {}
    for teacher, values in grouped.items():
        teacher_summary[teacher] = {
            "count": len(values),
            "mean_loss": sum(values) / len(values) if values else None,
            "std_loss": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_loss": min(values) if values else None,
            "max_loss": max(values) if values else None,
        }
    return teacher_summary


def write_loss_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "teacher", "sample_id", "original_index", "mean_loss", "parser_warnings"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_index": row["row_index"],
                    "teacher": row["teacher"],
                    "sample_id": row["sample_id"],
                    "original_index": row["original_index"],
                    "mean_loss": row["mean_loss"],
                    "parser_warnings": ";".join(row.get("parser_warnings", [])),
                }
            )


def plot_teacher_losses(teacher_summary: Dict[str, Any], output_path: Path) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"Could not import matplotlib: {exc}"

    labels = list(teacher_summary.keys())
    means = [teacher_summary[label]["mean_loss"] for label in labels]
    stds = [teacher_summary[label]["std_loss"] for label in labels]
    short_labels = [sanitize_teacher_name(label).replace("__", "\n") for label in labels]

    fig_width = max(8.0, 1.4 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=5, color="#4C78A8")
    ax.set_ylabel("Mean student loss")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return None


def inspect_templates(args: argparse.Namespace, samples: List[Dict[str, Any]]) -> Path:
    records = {}
    prompt = samples[0]["teacher_prompt"] if samples else TEACHER_GENERATION_INSTRUCTION
    for model_name in args.teachers:
        config = teacher_config(model_name)
        if is_deepseek_r1_teacher(model_name):
            records[model_name] = {
                "request_messages": [{"role": "user", "content": prompt}],
                "endpoint": deepseek_r1_base_url(),
                "control_tokens": config.get("control_tokens", {}),
            }
            continue
        try:
            from teachers import load_teacher_tokenizer

            tokenizer = load_teacher_tokenizer(
                model_name,
                config,
                trust_remote_code=bool(
                    args.trust_remote_code or config.get("llm_kwargs", {}).get("trust_remote_code")
                ),
            )
            rendered = apply_chat_template(
                tokenizer,
                build_user_messages(prompt, config.get("system_prompt")),
                config.get("chat_template_args", {}),
            )
            records[model_name] = {
                "rendered_prompt": rendered,
                "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
                "control_tokens": config.get("control_tokens", {}),
            }
        except Exception as exc:
            records[model_name] = {
                "error": repr(exc),
                "control_tokens": config.get("control_tokens", {}),
            }

    output_path = args.output_dir / "template_inspection.json"
    write_json({"created_at": utc_now(), "teachers": records}, output_path)
    return output_path


def selected_teacher_config_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        model_name: {
            **teacher_config(model_name),
            "effective_num_gpus_for_teacher_selection": teacher_num_gpus(model_name),
        }
        for model_name in args.teachers
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=list(TEACHERS.keys()),
        help="Teacher model ids, comma separated or space separated. Use 'all' for every configured teacher plus DeepSeek R1.",
    )
    parser.add_argument(
        "--student",
        default=DEFAULT_STUDENT,
        help="Student model path/name to evaluate and perturb "
        f"(default: {DEFAULT_STUDENT!r}, same as output/.../math).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="JSON/JSONL dataset file to sample from "
        f"(default: {DEFAULT_DATA_PATH}, same as output/.../math).",
    )
    parser.add_argument("--num-examples", "--num_examples", dest="num_examples", type=int, default=100)
    parser.add_argument("--num_tokens", "--num-tokens", dest="num_tokens", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_threshold", "--gpu-threshold", dest="gpu_threshold", type=float, default=10.0)
    parser.add_argument("--prompt-policy", choices=["auto", "begin", "end", "none"], default="auto")
    parser.add_argument("--answer-type", choices=["auto", "choice", "others"], default="others")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-download-progress", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--include-rendered-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-templates", action="store_true")
    parser.add_argument("--deepseek-threads", type=int, default=8)
    parser.add_argument("--deepseek-request-timeout", type=int, default=1200)
    parser.add_argument("--deepseek-progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    if args.num_tokens <= 0:
        raise ValueError("--num_tokens must be positive")
    if args.deepseek_threads <= 0:
        raise ValueError("--deepseek-threads must be positive")
    if args.deepseek_request_timeout <= 0:
        raise ValueError("--deepseek-request-timeout must be positive")
    if args.deepseek_progress_every <= 0:
        raise ValueError("--deepseek-progress-every must be positive")
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"run-{timestamp}"
    args.teachers = resolve_teachers(args.teachers)
    return args


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, sample_metadata = sample_data(args)
    sample_path = args.output_dir / "sampled_questions.json"
    write_json({"created_at": utc_now(), "metadata": sample_metadata, "samples": samples}, sample_path)

    config_path = args.output_dir / "teacher_config_snapshot.json"
    write_json({"created_at": utc_now(), "teachers": selected_teacher_config_snapshot(args)}, config_path)

    inspection_path = None
    if args.inspect_templates:
        inspection_path = inspect_templates(args, samples)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "sample_file": str(sample_path),
                    "teacher_config_snapshot": str(config_path),
                    "template_inspection": str(inspection_path) if inspection_path else None,
                    "teachers": [
                        {"model": model_name, "num_gpus": teacher_num_gpus(model_name)}
                        for model_name in args.teachers
                    ],
                },
                indent=2,
            )
        )
        return

    teacher_outputs = run_teacher_jobs(args, samples)
    merged_rows = merge_teacher_outputs(args, samples, teacher_outputs)
    merged_path = args.output_dir / "merged_teacher_generations.jsonl"
    write_jsonl(merged_rows, merged_path)

    loss_path = run_loss(args, merged_path)
    losses = load_losses(loss_path)
    teacher_summary = summarize_losses(merged_rows, losses)

    merged_with_losses_path = args.output_dir / "merged_teacher_generations_with_losses.jsonl"
    loss_csv_path = args.output_dir / "generation_losses.csv"
    summary_path = args.output_dir / "teacher_loss_summary.json"
    plot_path = args.output_dir / "teacher_loss_barplot.png"

    write_jsonl(merged_rows, merged_with_losses_path)
    write_loss_csv(merged_rows, loss_csv_path)
    plot_warning = plot_teacher_losses(teacher_summary, plot_path)

    write_json(
        {
            "created_at": utc_now(),
            "args": vars(args),
            "sample_file": str(sample_path),
            "teacher_config_snapshot": str(config_path),
            "template_inspection": str(inspection_path) if inspection_path else None,
            "merged_generations": str(merged_path),
            "merged_generations_with_losses": str(merged_with_losses_path),
            "loss_tensor": str(loss_path),
            "loss_csv": str(loss_csv_path),
            "barplot": str(plot_path) if plot_warning is None else None,
            "plot_warning": plot_warning,
            "teachers": teacher_summary,
        },
        summary_path,
    )
    print(f"Wrote teacher selection summary to {summary_path}")
    if plot_warning:
        print(f"Plot warning: {plot_warning}")
    else:
        print(f"Wrote teacher loss barplot to {plot_path}")


if __name__ == "__main__":
    main()
