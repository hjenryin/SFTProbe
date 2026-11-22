""""""

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

from .utils import (
    BASE_DIR,
    log,
    run_cmd,
)


def _remove_legacy_link(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _count_jsonl_rows(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    log("=" * 60)
    log("STEP 4.7: Train LoRA")
    log("=" * 60)

    input_file = workdir / "step4_5_partial_completion.jsonl"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing Step 4.5 output: {input_file}")

    step_dir = workdir / "step4_7_lora"
    dataset_dir = step_dir / "dataset"
    adapter_dir = step_dir / "adapter"
    metadata_file = workdir / "step4_7_metadata.json"
    legacy_link = workdir / "step4_7_lora_adapter"

    completion_marker = adapter_dir / "adapter_config.json"
    if completion_marker.exists() and metadata_file.exists():
        if legacy_link.exists() or legacy_link.is_symlink():
            _remove_legacy_link(legacy_link)
        log("[SKIP] Step 4.7 (LoRA training): adapter already exists")
        return adapter_dir, True

    dataset_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_file = dataset_dir / "train.jsonl"
    shutil.copy(input_file, train_file)

    dataset_name = "step4_5_partial"
    dataset_info = {
        dataset_name: {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    with open(dataset_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    context_length = 100 + 1000 + int(args.partial_y)

    if args.teacher.lower() == "original":
        log(f"Teacher is original. Manually truncating response length to {context_length} tokens.")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.original_model_name, trust_remote_code=True)
        
        truncated_data = []
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                for msg in data.get("conversations", []):
                    if msg.get("role") == "assistant" or msg.get("from") == "assistant":
                        content = msg.get("content", "")
                        tokens = tokenizer.encode(content, add_special_tokens=False)
                        if len(tokens) > context_length:
                            msg["content"] = tokenizer.decode(tokens[:context_length])
                            if "value" in msg:
                                msg["value"] = msg["content"]
                truncated_data.append(data)
                
        with open(train_file, "w", encoding="utf-8") as f:
            for d in truncated_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    cmd = [
        "llamafactory-cli",
        "train",
        str(BASE_DIR / "m1" / "src" / "train" / "lora_perturb.yaml"),
        f"dataset_dir={dataset_dir}",
        f"dataset={dataset_name}",
        f"output_dir={adapter_dir}",
        f"run_name={args.run_name}_step4_7_lora",
        f"seed={int(args.seed)}",
        f"model_name_or_path={args.original_model_name}",
    ]

    if "llama" in args.original_model_name.lower():
        cmd.append("template=llama3")

    cmd.append("gradient_accumulation_steps=1")

    from gpu_utils import get_num_gpus, get_allowed_gpus

    num_gpus = get_num_gpus()
    world_size = max(1, int(num_gpus))
    run_env = os.environ.copy()
    
    if world_size > int(args.lora_bs):
        log(f"visible num_gpus={world_size} is greater than lora_bs={args.lora_bs}. Restricting GPUs via CUDA_VISIBLE_DEVICES.")
        requested = int(args.lora_bs)
        allowed = get_allowed_gpus()
        if allowed is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in allowed[:requested])
        else:
            run_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(requested))
        world_size = int(args.lora_bs)

    per_device_train_batch_size = max(1, int(math.ceil(int(args.lora_bs) / world_size)))
    cmd.append(f"per_device_train_batch_size={per_device_train_batch_size}")

    if args.lora_train_steps is not None:
        cmd.append(f"max_steps={int(args.lora_train_steps)}")
    if args.lora_train_rank is not None:
        cmd.append(f"lora_rank={int(args.lora_train_rank)}")

    run_cmd(cmd, env=run_env)

    if legacy_link.exists() or legacy_link.is_symlink():
        _remove_legacy_link(legacy_link)

    metadata = {
        "input_file": str(input_file),
        "dataset_dir": str(dataset_dir),
        "adapter_dir": str(adapter_dir),
        "num_train_rows": _count_jsonl_rows(train_file),
        "model_name_or_path": args.original_model_name,
        "seed": int(args.seed),
        "max_steps": int(args.lora_train_steps) if args.lora_train_steps is not None else 200,
        "lora_rank": int(args.lora_train_rank) if args.lora_train_rank is not None else 4,
        "lora_bs": int(args.lora_bs),
        "num_gpus": int(num_gpus),
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": int(per_device_train_batch_size),
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return adapter_dir, False
