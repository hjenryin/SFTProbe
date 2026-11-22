#!/usr/bin/env python3
"""Generate a small teacher comparison set with vLLM.

Samples five questions from each configured dataset, runs each teacher model,
and writes per-teacher JSON plus a summary of reasoning/final-answer lengths.
"""

import argparse
import gc
import inspect
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
torch.set_float32_matmul_precision('high')
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "m1" / "output" / "teacher-generations-extensive-think"
GLOBAL_SYSTEM_PROMPT = "Think step-by-step extensively and double-check everything."

DEFAULT_DATASETS = [
    {
        "name": "m23k-prd-sharegpt",
        "path": REPO_ROOT / "dataset" / "m23k-prd-sharegpt" / "train.jsonl",
    },
    {
        "name": "numina-math-4-sources",
        "path": REPO_ROOT / "dataset" / "numina-math-4-sources" / "test.jsonl",
    },
    {
        "name": "PhysicsEval",
        "path": REPO_ROOT / "dataset" / "PhysicsEval" / "test.jsonl",
    },
]

CONTROL_TOKEN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "harmony": {
        "parser": "harmony",
        "reasoning_channels": ["analysis", "thought", "reasoning"],
        "final_channels": ["final", "answer", "commentary"],
        "special_tokens": [
            "<|startoftext|>",
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|end|>",
            "<|return|>",
            "<|endoftext|>",
        ],
        "full_control_patterns": [
            r"<\|start\|>(?:system|user|assistant|developer)",
            r"<\|channel\|>[A-Za-z_]+<\|message\|>",
        ],
        "source": "Confirmed with AutoTokenizer.apply_chat_template and cached Harmony chat_template.jinja.",
    },
    "gemma_channel": {
        "parser": "gemma_channel",
        "reasoning_channels": ["thought", "analysis", "reasoning"],
        "reasoning_start_tokens": ["<|channel>thought", "<|channel>analysis", "<|channel>reasoning"],
        "reasoning_end_token": "<channel|>",
        "special_tokens": ["<bos>", "<eos>", "<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>"],
        "full_control_patterns": [
            r"<\|turn>(?:system|user|model)\n?",
            r"<\|channel>(?:thought|analysis|reasoning)\s*",
        ],
        "source": "Confirmed with AutoTokenizer.apply_chat_template, tokenizer special map, and cached model card.",
    },
    "think_chatml": {
        "parser": "think_tags",
        "think_start": "<think>",
        "think_end": "</think>",
        "completion_starts_inside_think": True,
        "special_tokens": [
            "<think>",
            "</think>",
            "<|im_start|>",
            "<|im_sep|>",
            "<|im_end|>",
            "<|endoftext|>",
        ],
        "full_control_patterns": [
            r"<\|im_start\|>(?:system|user|assistant)\n?",
            r"<\|im_sep\|>",
        ],
        "source": "Confirmed with AutoTokenizer.apply_chat_template and tokenizer_config.json.",
    },
}

TEACHERS: Dict[str, Dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "num_gpus": 2,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "chat_template_args": {"reasoning_effort": "high"},
        "control_tokens": CONTROL_TOKEN_CONFIGS["harmony"],
        "notes": "OpenAI recommends temp=1.0/top_p=1.0; high is the largest configurable reasoning level.",
        "parameter_source": "https://huggingface.co/openai/gpt-oss-120b",
    },
    "google/gemma-4-31B-it": {
        "num_gpus": 1,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "chat_template_args": {"enable_thinking": True},
        "control_tokens": CONTROL_TOKEN_CONFIGS["gemma_channel"],
        "notes": "Gemma 4 thinking is enabled through the chat template without passing an explicit system prompt.",
        "parameter_source": "https://huggingface.co/google/gemma-4-31B-it",
    },
    "google/gemma-4-26B-A4B-it": {
        "num_gpus": 1,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.05},
        "chat_template_args": {"enable_thinking": True},
        "control_tokens": CONTROL_TOKEN_CONFIGS["gemma_channel"],
        "notes": "Gemma 4 thinking is enabled through the chat template without passing an explicit system prompt.",
        "parameter_source": "https://huggingface.co/google/gemma-4-26B-A4B-it",
    },
    "Qwen/Qwen3.6-27B": {
        "num_gpus": 1,
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
        "chat_template_args": {"enable_thinking": True},
        "control_tokens": CONTROL_TOKEN_CONFIGS["think_chatml"],
        "notes": "Qwen3.6 runs in thinking mode by default.",
        "parameter_source": "https://huggingface.co/Qwen/Qwen3.6-27B",
    },
    "microsoft/Phi-4-reasoning-plus": {
        "num_gpus": 1,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
        "chat_template_args": {},
        "control_tokens": CONTROL_TOKEN_CONFIGS["think_chatml"],
        "notes": "Microsoft recommends temperature=0.8, top_k=50, top_p=0.95, do_sample=True, and longer max_new_tokens for complex queries.",
        "parameter_source": "https://huggingface.co/microsoft/Phi-4-reasoning-plus",
    },
    "Qwen/QwQ-32B": {
        "num_gpus": 1,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
        "chat_template_args": {},
        "control_tokens": CONTROL_TOKEN_CONFIGS["think_chatml"],
        "notes": "QwQ 32B reasoning model.",
        "parameter_source": "https://huggingface.co/Qwen/QwQ-32B",
    },
}



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def extract_question(item: Dict[str, Any]) -> str:
    for key in ("prompt", "problem", "question", "simplified_problem_statement"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    conversations = item.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("from")
            content = message.get("content") or message.get("value")
            if role in {"user", "human"} and isinstance(content, str) and content.strip():
                return content.strip()

    raise ValueError(f"Could not find a question field in item keys: {sorted(item.keys())}")


def sample_questions(sample_per_dataset: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    for dataset in DEFAULT_DATASETS:
        path = Path(dataset["path"])
        rows = load_jsonl(path)
        if len(rows) < sample_per_dataset:
            raise ValueError(f"{path} has only {len(rows)} rows; need {sample_per_dataset}")

        selected_indices = rng.sample(range(len(rows)), sample_per_dataset)
        for index in selected_indices:
            item = rows[index]
            samples.append(
                {
                    "dataset": dataset["name"],
                    "dataset_path": str(path),
                    "original_index": index,
                    "question": extract_question(item),
                    "source": item.get("source"),
                    "category": item.get("category"),
                    "domain": item.get("domain"),
                }
            )

    return samples


def sanitize_teacher_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_name).strip("_")


def supports_arg(callable_obj: Any, arg_name: str) -> bool:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return True
    return arg_name in signature.parameters


def filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_messages(question: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages = []
    if system_prompt is None:
        system_prompt = GLOBAL_SYSTEM_PROMPT
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{question}"
            ),
        }
    )
    return messages


DEEPSEEK_V4_BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
DEEPSEEK_V4_EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>"
DEEPSEEK_V4_USER = "<\uff5cUser\uff5c>"
DEEPSEEK_V4_ASSISTANT = "<\uff5cAssistant\uff5c>"


def render_deepseek_v4_messages(messages: List[Dict[str, str]], chat_template_args: Dict[str, Any]) -> str:
    thinking_mode = chat_template_args.get("thinking_mode", "thinking")
    reasoning_effort = chat_template_args.get("reasoning_effort")
    if thinking_mode not in {"thinking", "chat"}:
        raise ValueError(f"DeepSeek-V4 thinking_mode must be 'thinking' or 'chat', got {thinking_mode!r}")
    if reasoning_effort not in {None, "high", "max"}:
        raise ValueError(f"DeepSeek-V4 reasoning_effort must be None, 'high', or 'max', got {reasoning_effort!r}")

    rendered = []
    if reasoning_effort == "max":
        rendered.append(
            "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
            "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
            "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
        )

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            rendered.append(content)
        elif role == "user":
            rendered.append(f"{DEEPSEEK_V4_USER}{content}")
        elif role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            if thinking_mode == "thinking":
                rendered.append(f"{DEEPSEEK_V4_ASSISTANT}{reasoning}</think>{content}{DEEPSEEK_V4_EOS}")
            else:
                rendered.append(f"{DEEPSEEK_V4_ASSISTANT}</think>{content}{DEEPSEEK_V4_EOS}")
        else:
            raise ValueError(f"DeepSeek-V4 renderer does not support role {role!r}")

    rendered.append(DEEPSEEK_V4_ASSISTANT)
    rendered.append("<think>" if thinking_mode == "thinking" else "</think>")
    return DEEPSEEK_V4_BOS + "".join(rendered)


TOKENIZER_OVERRIDE_ROOT = Path(__file__).resolve().parent / "tokenizer_overrides"


def ensure_pretrained_fast_tokenizer_dir(model_ref: str) -> Path:
    """Materialize a tokenizer dir that loads via PreTrainedTokenizerFast.

    Some DeepSeek-R1 checkpoints ship byte-level BPE (`tokenizer.json` with Ġ/Ċ) but
    set ``tokenizer_class: LlamaTokenizerFast``. AutoTokenizer / vLLM then leave piece
    markers in decoded text. Copying the files and forcing PreTrainedTokenizerFast
    fixes detok without using vLLM's deepseek_v32 mode (which rewrites V3.2 chat
    templates and is wrong for R1).
    """
    source_dir = Path(model_ref)
    if source_dir.is_dir() and (source_dir / "tokenizer.json").exists():
        safe = source_dir.name
        file_source = source_dir
    else:
        safe = model_ref.replace("/", "__")
        file_source = None

    out_dir = TOKENIZER_OVERRIDE_ROOT / safe
    ready_marker = out_dir / ".ready"
    if ready_marker.exists() and (out_dir / "tokenizer.json").exists() and (out_dir / "tokenizer_config.json").exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    def resolve_file(fname: str) -> Optional[Path]:
        if file_source is not None:
            candidate = file_source / fname
            return candidate if candidate.exists() else None
        from huggingface_hub import hf_hub_download

        try:
            return Path(hf_hub_download(repo_id=model_ref, filename=fname))
        except Exception:
            return None

    for fname in ("tokenizer.json", "special_tokens_map.json"):
        src = resolve_file(fname)
        if src is None:
            continue
        dest = out_dir / fname
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)

    cfg_src = resolve_file("tokenizer_config.json")
    if cfg_src is None:
        raise FileNotFoundError(f"tokenizer_config.json not found for {model_ref}")
    cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg_dest = out_dir / "tokenizer_config.json"
    if cfg_dest.exists() or cfg_dest.is_symlink():
        cfg_dest.unlink()
    cfg_dest.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ready_marker.write_text(f"source={model_ref}\n", encoding="utf-8")
    return out_dir


def resolve_teacher_llm_kwargs(model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return llm_kwargs for vLLM, injecting a fixed tokenizer path when needed."""
    config = config or {}
    kwargs = dict(config.get("llm_kwargs") or {})
    if config.get("fix_mislabeled_llama_tokenizer"):
        kwargs["tokenizer"] = str(ensure_pretrained_fast_tokenizer_dir(model_name))
    return kwargs


def load_teacher_tokenizer(model_name: str, config: Optional[Dict[str, Any]] = None, trust_remote_code: bool = True) -> Any:
    """Load an HF tokenizer that matches the teacher's vLLM tokenizer settings."""
    config = config or {}
    llm_kwargs = resolve_teacher_llm_kwargs(model_name, config)
    trust = bool(trust_remote_code or llm_kwargs.get("trust_remote_code"))
    tokenizer_path = llm_kwargs.get("tokenizer") or model_name
    if config.get("fix_mislabeled_llama_tokenizer") or llm_kwargs.get("tokenizer"):
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast.from_pretrained(tokenizer_path, trust_remote_code=trust)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust)


def apply_chat_template(tokenizer: Any, messages: List[Dict[str, str]], chat_template_args: Dict[str, Any]) -> str:
    template_args = dict(chat_template_args)
    template_format = template_args.pop("template_format", None)
    if template_format == "deepseek_v4":
        return render_deepseek_v4_messages(messages, template_args)
    if template_format is not None:
        raise ValueError(f"Unknown template_format={template_format!r}")

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_args,
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].title()}: {message['content']}")
    rendered.append("Assistant:")
    return "\n\n".join(rendered)


def strip_common_special_tokens(text: str) -> str:
    replacements = [
        "<|end|>",
        "<|eot_id|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<channel|>",
        "<think>",
        "</think>",
        "[gMASK]",
        "<sop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "[SYSTEM_PROMPT]",
        "[/SYSTEM_PROMPT]",
        "[INST]",
        "[/INST]",
        "[THINK]",
        "[/THINK]",
        "[AVAILABLE_TOOLS]",
        "[/AVAILABLE_TOOLS]",
        "[TOOL_CALLS]",
        "[ARGS]",
        "[TOOL_RESULTS]",
        "[/TOOL_RESULTS]",
        DEEPSEEK_V4_BOS,
        DEEPSEEK_V4_EOS,
        DEEPSEEK_V4_USER,
        DEEPSEEK_V4_ASSISTANT,
        "<\uff5clatest_reminder\uff5c>",
        "\uff5cDSML\uff5c",
        "</s>",
    ]
    for token in replacements:
        text = text.replace(token, "")
    text = re.sub(r"<\|start\|>assistant", "", text)
    text = re.sub(r"<\|channel\|>[A-Za-z_]+<\|message\|>", "", text)
    return text.strip()


def parse_reasoning_and_final(text: str) -> Tuple[str, str]:
    def split_at_marker(source: str, marker_pattern: str) -> Optional[Tuple[str, str]]:
        match = re.search(marker_pattern, source, flags=re.IGNORECASE)
        if not match:
            return None
        return source[: match.start()].strip(), source[match.end() :].strip()

    def strip_mistral_wrappers(source: str) -> str:
        source = re.sub(r"^\s*```(?:markdown)?\s*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"\s*```\s*$", "", source)
        source = re.sub(
            r"\[(?:MODEL_)?(?:THOUGHT|THINKING|THINK)\]",
            "",
            source,
            flags=re.IGNORECASE,
        )
        source = re.sub(
            r"\[/(?:MODEL_)?(?:THOUGHT|THINKING|THINK|THOOK)\]",
            "",
            source,
            flags=re.IGNORECASE,
        )
        return source.strip()

    huatuo = re.search(
        r"##\s*Thinking\s*(.*?)\s*##\s*Final Response\s*(.*)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if huatuo:
        return (
            strip_common_special_tokens(huatuo.group(1)),
            strip_common_special_tokens(huatuo.group(2)),
        )

    gemma_channel = re.search(
        r"<\|channel>(thought|analysis|reasoning)\s*(.*?)(?:<channel\|>|$)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if gemma_channel:
        reasoning_text = gemma_channel.group(2).strip()
        final_answer_text = text[gemma_channel.end() :].strip()
        return reasoning_text, strip_common_special_tokens(final_answer_text)

    harmony_messages = re.findall(
        r"<\|channel\|>([A-Za-z_]+)<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|$)",
        text,
        flags=re.DOTALL,
    )
    if harmony_messages:
        reasoning_parts = [
            content.strip()
            for channel, content in harmony_messages
            if channel.lower() in {"analysis", "thought", "reasoning"}
        ]
        final_parts = [
            content.strip()
            for channel, content in harmony_messages
            if channel.lower() in {"final", "answer", "commentary"}
        ]
        if final_parts:
            return "\n\n".join(reasoning_parts).strip(), strip_common_special_tokens(final_parts[-1])

    closing_think = split_at_marker(text, r"</think>")
    if closing_think:
        reasoning_text, final_answer_text = closing_think
        return strip_common_special_tokens(reasoning_text), strip_common_special_tokens(final_answer_text)

    final_marker = split_at_marker(text, r"\[(?:FINAL_ANSWER|FINAL ANSWER|ANSWER|FINAL)\]")
    if final_marker:
        reasoning_text, final_answer_text = final_marker
        return strip_common_special_tokens(strip_mistral_wrappers(reasoning_text)), strip_common_special_tokens(
            strip_mistral_wrappers(final_answer_text)
        )

    reasoning_parts = []
    remaining = text
    patterns = [
        r"<think>\s*(.*?)\s*</think>",
        r"\[THINK\]\s*(.*?)\s*\[/THINK\]",
        r"\[THOUGHT\]\s*(.*?)\s*\[/(?:THOUGHT|THINKING|THOOK)\]",
        r"\[THINKING\]\s*(.*?)\s*\[/(?:THINKING|THOUGHT|THINK|THOOK)\]",
        r"\[MODEL_THINKING\]\s*(.*?)\s*\[/(?:MODEL_THINKING|THINKING|THOUGHT|THINK|THOOK)\]",
        r"<\|channel>thought\s*(.*?)\s*<channel\|>",
        r"<\|channel>analysis\s*(.*?)\s*<channel\|>",
        r"<\|channel>reasoning\s*(.*?)\s*<channel\|>",
        r"<\|channel\|>thought\s*(.*?)\s*<channel\|>",
        r"<\|channel\|>analysis\s*(.*?)\s*<channel\|>",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, remaining, flags=re.DOTALL | re.IGNORECASE))
        if not matches:
            continue
        reasoning_parts.extend(match.group(1).strip() for match in matches)
        remaining = re.sub(pattern, "", remaining, flags=re.DOTALL | re.IGNORECASE)

    return "\n\n".join(part for part in reasoning_parts if part).strip(), strip_common_special_tokens(remaining)


def count_tokens(tokenizer: Any, text: str, fallback_token_ids: Optional[Iterable[int]] = None) -> int:
    if fallback_token_ids is not None and not text:
        return len(list(fallback_token_ids))
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def make_sampling_params(config: Dict[str, Any], max_tokens: int) -> Any:
    from vllm import SamplingParams

    kwargs = dict(config.get("sampling", {}))
    kwargs["max_tokens"] = max_tokens
    kwargs["skip_special_tokens"] = False
    return SamplingParams(**filter_supported_kwargs(SamplingParams, kwargs))


def is_local_model_path(model_name: str) -> bool:
    return model_name.startswith("/") or model_name.startswith("./") or model_name.startswith("../") or Path(model_name).exists()


def predownload_model(model_name: str, args: argparse.Namespace) -> str:
    if args.no_download_progress or is_local_model_path(model_name):
        return model_name

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import enable_progress_bars
    except ImportError:
        print("huggingface_hub is not installed in this Python environment; vLLM will handle model download.")
        return model_name

    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    enable_progress_bars()

    kwargs: Dict[str, Any] = {"repo_id": model_name}
    if args.download_dir:
        kwargs["cache_dir"] = args.download_dir

    print(f"Downloading/checking model snapshot with progress bars: {model_name}")
    snapshot_path = snapshot_download(**kwargs)
    print(f"Using local snapshot: {snapshot_path}")
    return snapshot_path


def predownload_models(model_names: List[str], args: argparse.Namespace) -> Dict[str, str]:
    model_paths = {}
    if args.no_download_progress:
        return {model_name: model_name for model_name in model_names}

    if not model_names:
        return model_paths

    print("\n=== Pre-caching teacher model snapshots ===")
    print("Hugging Face will reuse files that are already in the cache.")
    for model_name in model_names:
        model_paths[model_name] = predownload_model(model_name, args)
    return model_paths


def make_llm(model_name: str, config: Dict[str, Any], args: argparse.Namespace, tensor_parallel_size: int) -> Any:
    from vllm import LLM

    kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.dtype:
        kwargs["dtype"] = args.dtype
    if args.download_dir:
        kwargs["download_dir"] = args.download_dir
    if supports_arg(LLM, "generation_config"):
        kwargs["generation_config"] = "vllm"
    kwargs.update(resolve_teacher_llm_kwargs(model_name, config))
    return LLM(**filter_supported_kwargs(LLM, kwargs))


def get_visible_gpu_ids() -> List[int]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return [int(value) for value in cuda_visible.split(",") if value.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return list(range(len(lines)))
    except Exception:
        return [0]


def build_gpu_groups(args: argparse.Namespace, config: Dict[str, Any]) -> Tuple[List[List[int]], int, int]:
    gpu_ids = get_visible_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("No visible GPUs found.")

    parallel = config.get("parallel", {})
    requested_dp = parallel.get("data_parallel_size", args.data_parallel_size)
    requested_tp = parallel.get("tensor_parallel_size", config.get("num_gpus", args.tensor_parallel_size))

    dp_size = min(max(1, requested_dp), len(gpu_ids))
    if requested_tp is None:
        tp_size = max(1, len(gpu_ids) // dp_size)
    else:
        tp_size = max(1, requested_tp)

    if dp_size * tp_size > len(gpu_ids):
        adjusted_dp = max(1, len(gpu_ids) // tp_size)
        print(
            f"Requested DP={dp_size}, TP={tp_size} needs {dp_size * tp_size} GPUs, "
            f"but only {len(gpu_ids)} are visible. Using DP={adjusted_dp}, TP={tp_size}."
        )
        dp_size = adjusted_dp

    usable = gpu_ids[: dp_size * tp_size]
    groups = [usable[i * tp_size : (i + 1) * tp_size] for i in range(dp_size)]
    return groups, dp_size, tp_size


def worker_generate(
    worker_rank: int,
    gpu_ids: List[int],
    model_name: str,
    model_path: str,
    config: Dict[str, Any],
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    tensor_parallel_size: int,
    queue: multiprocessing.Queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        print(f"[Worker {worker_rank}] Loading {model_name} on GPUs {gpu_ids} (TP={tensor_parallel_size})")
        llm = make_llm(model_path, config, args, tensor_parallel_size)
        tokenizer = llm.get_tokenizer()
        sampling_params = make_sampling_params(config, args.max_tokens)

        prompts = [
            apply_chat_template(
                tokenizer,
                build_messages(sample["question"], config.get("system_prompt")),
                config.get("chat_template_args", {}),
            )
            for sample in samples
        ]
        print(prompts[0])
        generate_started_at = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        generate_seconds = time.perf_counter() - generate_started_at

        generations = []
        generated_tokens = 0
        for sample, prompt, output in zip(samples, prompts, outputs):
            completion = output.outputs[0]
            generated_text = completion.text
            token_ids = getattr(completion, "token_ids", None)
            reasoning_text, final_answer_text = parse_reasoning_and_final(generated_text)
            reasoning_tokens = count_tokens(tokenizer, reasoning_text)
            final_answer_tokens = count_tokens(tokenizer, final_answer_text)
            total_tokens = len(token_ids) if token_ids is not None else count_tokens(tokenizer, generated_text)
            generated_tokens += total_tokens

            generations.append(
                {
                    **sample,
                    "prompt_rendered": prompt if args.include_rendered_prompts else None,
                    "generated_text": generated_text,
                    "reasoning_text": reasoning_text,
                    "final_answer_text": final_answer_text,
                    "token_counts": {
                        "reasoning": reasoning_tokens,
                        "final_answer": final_answer_tokens,
                        "total": total_tokens,
                    },
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                }
            )

        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        throughput = {
            "worker_rank": worker_rank,
            "gpu_ids": gpu_ids,
            "num_prompts": len(samples),
            "generated_tokens": generated_tokens,
            "generate_seconds": generate_seconds,
            "tokens_per_second": generated_tokens / generate_seconds if generate_seconds > 0 else None,
        }
        queue.put((worker_rank, {"generations": generations, "throughput": throughput}))
    except Exception as exc:
        import traceback

        traceback.print_exc()
        queue.put((worker_rank, exc))


def run_teacher(model_name: str, model_path: str, samples: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    from vllm import SamplingParams  # noqa: F401 - imported here to fail early before loading model

    config = TEACHERS[model_name]
    gpu_groups, effective_dp_size, effective_tp_size = build_gpu_groups(args, config)
    print(f"GPU Config: DP={effective_dp_size}, TP={effective_tp_size}, groups={gpu_groups}")

    chunk_size = math.ceil(len(samples) / effective_dp_size)
    chunks = [
        samples[i * chunk_size : min((i + 1) * chunk_size, len(samples))]
        for i in range(effective_dp_size)
    ]

    queue: multiprocessing.Queue = multiprocessing.Queue()
    processes = []
    for worker_rank, chunk in enumerate(chunks):
        if not chunk:
            continue
        process = multiprocessing.Process(
            target=worker_generate,
            args=(
                worker_rank,
                gpu_groups[worker_rank],
                model_name,
                model_path,
                config,
                chunk,
                args,
                effective_tp_size,
                queue,
            ),
        )
        process.start()
        processes.append(process)

    results_by_worker: Dict[int, List[Dict[str, Any]]] = {}
    throughput_by_worker: Dict[int, Dict[str, Any]] = {}
    for _ in processes:
        worker_rank, payload = queue.get()
        if isinstance(payload, Exception):
            for process in processes:
                process.terminate()
            raise payload
        results_by_worker[worker_rank] = payload["generations"]
        throughput_by_worker[worker_rank] = payload["throughput"]

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Worker process {process.pid} exited with code {process.exitcode}")

    generations = []
    for worker_rank in range(len(chunks)):
        generations.extend(results_by_worker.get(worker_rank, []))

    total_generated_tokens = sum(item.get("generated_tokens", 0) for item in throughput_by_worker.values())
    max_generate_seconds = max(
        (item.get("generate_seconds", 0.0) for item in throughput_by_worker.values()),
        default=0.0,
    )
    summed_worker_seconds = sum(item.get("generate_seconds", 0.0) for item in throughput_by_worker.values())
    throughput = {
        "total_generated_tokens": total_generated_tokens,
        "wall_generate_seconds": max_generate_seconds,
        "aggregate_tokens_per_second": (
            total_generated_tokens / max_generate_seconds if max_generate_seconds > 0 else None
        ),
        "summed_worker_generate_seconds": summed_worker_seconds,
        "mean_worker_tokens_per_second": (
            sum(
                item["tokens_per_second"]
                for item in throughput_by_worker.values()
                if item.get("tokens_per_second") is not None
            )
            / len(throughput_by_worker)
            if throughput_by_worker
            else None
        ),
        "workers": [throughput_by_worker[idx] for idx in sorted(throughput_by_worker)],
    }

    return {
        "model": model_name,
        "created_at": utc_now(),
        "generation_parameters": {
            "sampling": config.get("sampling", {}),
            "max_tokens": args.max_tokens,
            "chat_template_args": config.get("chat_template_args", {}),
            "llm_kwargs": {
                "requested_tensor_parallel_size": config.get("parallel", {}).get("tensor_parallel_size", args.tensor_parallel_size),
                "requested_data_parallel_size": config.get("parallel", {}).get("data_parallel_size", args.data_parallel_size),
                "effective_tensor_parallel_size": effective_tp_size,
                "effective_data_parallel_size": effective_dp_size,
                "gpu_groups": gpu_groups,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "download_dir": args.download_dir,
                "model_path": model_path,
                **config.get("llm_kwargs", {}),
            },
            "notes": config.get("notes"),
            "parameter_source": config.get("parameter_source"),
        },
        "throughput": throughput,
        "generations": generations,
    }


def average(values: List[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_teacher(result: Dict[str, Any], output_file: Path) -> Dict[str, Any]:
    generations = result["generations"]
    reasoning = [item["token_counts"]["reasoning"] for item in generations]
    final = [item["token_counts"]["final_answer"] for item in generations]
    total = [item["token_counts"]["total"] for item in generations]
    return {
        "output_file": str(output_file),
        "num_generations": len(generations),
        "avg_reasoning_tokens": average(reasoning),
        "avg_final_answer_tokens": average(final),
        "avg_total_tokens": average(total),
        "throughput": result.get("throughput"),
        "generation_parameters": result["generation_parameters"],
    }


def load_ready_generation(output_file: Path) -> Optional[Dict[str, Any]]:
    if not output_file.exists():
        return None

    try:
        with output_file.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Existing output is not readable, regenerating: {output_file} ({exc})")
        return None

    generations = result.get("generations")
    if not isinstance(generations, list) or not generations:
        print(f"Existing output has no generations, regenerating: {output_file}")
        return None

    missing_token_counts = [
        idx
        for idx, item in enumerate(generations)
        if not isinstance(item, dict) or not isinstance(item.get("token_counts"), dict)
    ]
    if missing_token_counts:
        print(f"Existing output is missing token counts, regenerating: {output_file}")
        return None

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teachers", nargs="*", default=list(TEACHERS.keys()), choices=list(TEACHERS.keys()))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size per data-parallel worker. Defaults to visible_gpus // data_parallel_size.",
    )
    parser.add_argument("--data-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument(
        "--no-download-progress",
        action="store_true",
        help="Skip explicit Hugging Face snapshot_download progress bars and let vLLM download normally.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Regenerate teachers even when their output JSON already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-rendered-prompts",
        action="store_true",
        help="Store the full chat-template-rendered prompt in each output row.",
    )
    return parser.parse_args()


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    args = parse_args()
    samples = sample_questions(args.sample_per_dataset, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = args.output_dir / "sampled_questions.json"
    write_json(
        {
            "created_at": utc_now(),
            "seed": args.seed,
            "sample_per_dataset": args.sample_per_dataset,
            "samples": samples,
        },
        sample_path,
    )

    if args.dry_run:
        print(json.dumps({"sample_file": str(sample_path), "teachers": args.teachers}, indent=2))
        return

    summary = {
        "created_at": utc_now(),
        "seed": args.seed,
        "sample_per_dataset": args.sample_per_dataset,
        "sample_file": str(sample_path),
        "teachers": {},
    }

    models_to_run = []
    for model_name in args.teachers:
        safe_name = sanitize_teacher_name(model_name)
        output_file = args.output_dir / f"{safe_name}.json"
        if not args.rerun_existing and load_ready_generation(output_file) is not None:
            continue
        models_to_run.append(model_name)

    model_paths = predownload_models(models_to_run, args)

    for model_name in args.teachers:
        safe_name = sanitize_teacher_name(model_name)
        output_file = args.output_dir / f"{safe_name}.json"
        result = None if args.rerun_existing else load_ready_generation(output_file)
        if result is not None:
            print(f"Skipping ready teacher output and adding it to summary: {output_file}")
        else:
            print(f"\n=== Running teacher: {model_name} ===")
            result = run_teacher(model_name, model_paths.get(model_name, model_name), samples, args)
            write_json(result, output_file)
        summary["teachers"][model_name] = summarize_teacher(result, output_file)

    write_json(summary, args.output_dir / "summary.json")
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
