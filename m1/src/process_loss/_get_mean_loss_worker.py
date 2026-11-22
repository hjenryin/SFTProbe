import os
import argparse
import json
import torch
import gc
import sys
import math
from typing import Dict, Iterable, List, Tuple
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline_src.inflection import compute_tensor_surface_fingerprint


PERTURBATION_SURFACES = {"state_dict", "named_parameters_and_buffers"}


def _load_tokenizer(model_path):
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def _normalize_perturbation_surface(surface):
    if surface is None:
        return None
    surface = str(surface)
    if surface not in PERTURBATION_SURFACES:
        raise ValueError(
            f"Unknown perturbation surface {surface!r}; expected one of "
            f"{sorted(PERTURBATION_SURFACES)}"
        )
    return surface


def _tensor_dedup_key(tensor: torch.Tensor) -> Tuple:
    detached = tensor.detach()
    try:
        storage = detached.untyped_storage()
    except Exception:
        storage = None

    try:
        storage_ptr = int(storage.data_ptr()) if storage is not None else int(detached.data_ptr())
    except Exception:
        storage_ptr = id(storage) if storage is not None else id(detached)

    return (
        str(detached.device),
        storage_ptr,
        int(detached.storage_offset()),
        tuple(int(x) for x in detached.size()),
        tuple(int(x) for x in detached.stride()),
    )


def _dedupe_named_tensors(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    deduped: Dict[str, torch.Tensor] = {}
    seen = set()
    for name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        key = _tensor_dedup_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        deduped[str(name)] = tensor
    return deduped


def _get_live_model_tensors(model, perturbation_surface: str) -> Dict[str, torch.Tensor]:
    surface = _normalize_perturbation_surface(perturbation_surface)
    if surface == "state_dict":
        return model.state_dict()

    named_tensors: List[Tuple[str, torch.Tensor]] = []
    named_tensors.extend(list(model.named_parameters()))
    named_tensors.extend(list(model.named_buffers()))
    return _dedupe_named_tensors(named_tensors)


def _apply_in_memory_perturbation(
    model,
    *,
    perturbation_surface: str,
    perturbation_seed: int,
    perturbation_scale: float,
    device: str,
) -> None:
    """"""
    surface = _normalize_perturbation_surface(perturbation_surface)
    if surface is None:
        return

    generator = torch.Generator(device=device)
    generator.manual_seed(int(perturbation_seed))
    scale = float(perturbation_scale)
    tensors = _get_live_model_tensors(model, surface)

    print(
        "[PerturbationOverlay] Applying in-memory perturbation: "
        f"surface={surface}, seed={int(perturbation_seed)}, "
        f"scale={scale:.8f}, tensors={len(tensors)}",
        flush=True,
    )

    with torch.no_grad():
        for key, target in tqdm(
            tensors.items(),
            desc="[PerturbationOverlay] Apply",
            leave=False,
        ):
            if not torch.is_floating_point(target):
                continue

            eps = torch.randn(
                target.shape,
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            l2_norm = float(torch.norm(eps, p=2).item())
            d = max(1, eps.numel())
            key_scale = l2_norm / math.sqrt(float(d))
            direction = (key_scale * eps).to(device=target.device, dtype=target.dtype)
            target.add_(scale * direction)

            del eps, direction

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_perturbation_overlay_descriptor(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        descriptor = json.load(f)
    if not isinstance(descriptor, dict):
        raise ValueError(f"Perturbation overlay descriptor must be a JSON object: {path}")
    return descriptor


def _resolve_perturbation_overlay_args(
    *,
    model_path,
    perturbation_surface=None,
    perturbation_seed=None,
    perturbation_scale=None,
    perturbation_overlay_descriptor=None,
):
    descriptor = _load_perturbation_overlay_descriptor(perturbation_overlay_descriptor)
    if descriptor is None:
        return perturbation_surface, perturbation_seed, perturbation_scale, None

    desc_model_path = descriptor.get("base_model")
    if desc_model_path is not None and str(desc_model_path) != str(model_path):
        print(
            "[PerturbationOverlay] Descriptor base_model differs from --model_path: "
            f"descriptor={desc_model_path}, model_path={model_path}",
            flush=True,
        )

    desc_surface = descriptor.get("perturbation_surface")
    desc_seed = descriptor.get("seed")
    desc_scale = descriptor.get("scale")
    if desc_surface is None or desc_seed is None or desc_scale is None:
        raise ValueError(
            "Perturbation overlay descriptor must include perturbation_surface, seed, and scale"
        )

    if perturbation_surface is not None and str(perturbation_surface) != str(desc_surface):
        raise ValueError(
            f"CLI perturbation_surface={perturbation_surface!r} does not match "
            f"descriptor surface={desc_surface!r}"
        )
    if perturbation_seed is not None and int(perturbation_seed) != int(desc_seed):
        raise ValueError(
            f"CLI perturbation_seed={perturbation_seed!r} does not match "
            f"descriptor seed={desc_seed!r}"
        )
    if perturbation_scale is not None and float(perturbation_scale) != float(desc_scale):
        raise ValueError(
            f"CLI perturbation_scale={perturbation_scale!r} does not match "
            f"descriptor scale={desc_scale!r}"
        )

    return str(desc_surface), int(desc_seed), float(desc_scale), descriptor


def _assert_perturbation_overlay_fingerprint(model, perturbation_surface, descriptor):
    if descriptor is None:
        return
    expected = descriptor.get("fingerprint")
    if not isinstance(expected, dict):
        raise ValueError("Perturbation overlay descriptor is missing fingerprint metadata")

    actual = compute_tensor_surface_fingerprint(
        model,
        tensor_surface=perturbation_surface,
        samples_per_tensor=int(expected.get("samples_per_tensor", 16)),
    )
    if actual.get("digest") != expected.get("digest"):
        raise AssertionError(
            "Perturbation overlay fingerprint mismatch: "
            f"expected={expected.get('digest')}, actual={actual.get('digest')}, "
            f"surface={perturbation_surface}"
        )
    print(
        "[PerturbationOverlay] Fingerprint verified: "
        f"digest={actual.get('digest')}, tensors={actual.get('float_tensor_count')}",
        flush=True,
    )


def _load_model(
    model_path,
    device,
    *,
    perturbation_surface=None,
    perturbation_seed=None,
    perturbation_scale=None,
    perturbation_overlay_descriptor=None,
):
    (
        perturbation_surface,
        perturbation_seed,
        perturbation_scale,
        overlay_descriptor,
    ) = _resolve_perturbation_overlay_args(
        model_path=model_path,
        perturbation_surface=perturbation_surface,
        perturbation_seed=perturbation_seed,
        perturbation_scale=perturbation_scale,
        perturbation_overlay_descriptor=perturbation_overlay_descriptor,
    )
    perturbation_surface = _normalize_perturbation_surface(perturbation_surface)
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        if perturbation_surface is not None:
            raise ValueError("In-memory perturbation overlay is not supported for adapter checkpoints")

        from peft import PeftModel

        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}")

        print(f"[PEFT] Loading adapter checkpoint from {model_path} with base model {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()
        return model

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    if perturbation_surface is not None:
        if perturbation_seed is None or perturbation_scale is None:
            raise ValueError(
                "perturbation_seed and perturbation_scale are required when "
                "perturbation_surface is set"
            )
        _apply_in_memory_perturbation(
            model,
            perturbation_surface=perturbation_surface,
            perturbation_seed=int(perturbation_seed),
            perturbation_scale=float(perturbation_scale),
            device=device,
        )
        _assert_perturbation_overlay_fingerprint(model, perturbation_surface, overlay_descriptor)
    return model


def maybe_apply_liger_kernel(model_path):
    """"""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            model_path = adapter_config.get("base_model_name_or_path", model_path)
        except Exception as e:
            print(f"[Liger] Skipped: unable to read adapter config from {model_path}: {e}")
            return False

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[Liger] Skipped: unable to read model config from {model_path}: {e}")
        return False

    model_type = str(getattr(cfg, "model_type", "")).lower()
    if model_type.startswith("qwen2"):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2()
            print("[Liger] Applied Qwen2 kernel optimizations")
            return True
        except Exception as e:
            print(f"[Liger] Skipped: failed to apply Qwen2 kernel: {e}")
            return False
    elif model_type == "llama":
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama()
            print("[Liger] Applied LLaMA kernel optimizations")
            return True
        except Exception as e:
            print(f"[Liger] Skipped: failed to apply LLaMA kernel: {e}")
            return False
    else:
        print(f"[Liger] Skipped: model_type={model_type!r} is not Qwen2.x or LLaMA")
        return False

def process_dataset_split(
    model_path,
    dataset,
    tokenizer,
    output_file,
    device_id,
    process_id,
    total_processes,
    *,
    perturbation_surface=None,
    perturbation_seed=None,
    perturbation_scale=None,
    perturbation_overlay_descriptor=None,
):
    device = f"cuda:{device_id}"
    
    try:
        print(f"[Worker {process_id}] Loading model: {model_path}")
        model = _load_model(
            model_path,
            device,
            perturbation_surface=perturbation_surface,
            perturbation_seed=perturbation_seed,
            perturbation_scale=perturbation_scale,
            perturbation_overlay_descriptor=perturbation_overlay_descriptor,
        )

        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True),
            pin_memory=True
        )

        results = []

        for batch in tqdm(dataloader, desc=f"Worker {process_id}", position=process_id-1):
            index = batch.pop("index").item()
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss.item()
                if torch.isnan(outputs.loss):
                    loss = 0.0
                results.append({
                    'index': index,
                    'mean_loss': loss
                })

        torch.save(results, output_file)
        print(f"[Worker {process_id}] Saved {len(results)} samples to {output_file}")

    except Exception as e:
        print(f"[Worker {process_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'model' in locals(): del model
        gc.collect()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Full path to output .pt file. Defaults to <model_path>/<output_filename>.")
    parser.add_argument("--output_filename", type=str, default="mean_losses.pt",
                        help="Filename within model_path. Used only when --output_file is not set.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--indices_file", type=str, default=None,
                        help="Numpy .npy file of dataset indices to process. "
                             "If omitted, the full dataset is used.")
    parser.add_argument("--process_id", type=int, default=1)
    parser.add_argument("--total_processes", type=int, default=1)
    parser.add_argument("--reasoning_marker", type=str, default=None)
    parser.add_argument("--strip_prefix", type=str, default=None,
                        help="Common string prefix to strip from every response before loss evaluation.")
    parser.add_argument("--perturbation_surface", type=str, choices=sorted(PERTURBATION_SURFACES), default=None,
                        help="Optional in-memory Step 2 perturbation surface to apply after model load.")
    parser.add_argument("--perturbation_seed", type=int, default=None,
                        help="Seed for the optional in-memory Step 2 perturbation overlay.")
    parser.add_argument("--perturbation_scale", type=float, default=None,
                        help="Scale for the optional in-memory Step 2 perturbation overlay.")
    parser.add_argument("--perturbation_overlay_descriptor", type=str, default=None,
                        help="Optional descriptor JSON for an in-memory Step 2 perturbation overlay.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.perturbation_overlay_descriptor is None and args.perturbation_surface is not None and (
        args.perturbation_seed is None or args.perturbation_scale is None
    ):
        raise ValueError(
            "--perturbation_seed and --perturbation_scale are required with "
            "--perturbation_surface"
        )

    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = os.path.join(args.model_path, args.output_filename)

    if os.path.exists(output_file):
        print(f"✅ Already exists, skipping: {output_file}")
        sys.exit(0)

    maybe_apply_liger_kernel(args.model_path)

    tokenizer = _load_tokenizer(args.model_path)

    if "json" in args.dataset_path:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")
    
    import numpy as np

    if args.indices_file is not None:
        worker_indices = np.load(args.indices_file).tolist()
    else:
        worker_indices = list(range(len(dataset)))

    dataset = dataset.select(worker_indices)

    def process_data(examples, indices):
        real_indices = [worker_indices[i] for i in indices]
        
        if "conversations" in examples:
            convos = examples['conversations']
            prompt_msgs = []
            responses = []
            for chat in convos:
                user_content = ""
                asst_content = ""
                for msg in chat:
                    if msg['role'] == 'user':
                        user_content = msg.get('value') or msg.get('content')
                    elif msg['role'] == 'assistant':
                        asst_content = msg.get('value') or msg.get('content')
                prompt_msgs.append([{"role": "user", "content": user_content}])
                responses.append(asst_content)
        else:
            prompt_msgs = [[{"role": "user", "content": p}] for p in examples['prompt']]
            responses = [r[0]['content'] for r in examples['completion']]

        prompts_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_texts = [p + r for p, r in zip(prompts_text, responses)]

        prompt_ids = tokenizer(prompts_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_texts, add_special_tokens=False)["input_ids"]
        
        full_ids_list = []
        labels_list = []
        
        marker_ids = None
        if getattr(args, "reasoning_marker", None) is not None:
            marker_ids = tokenizer(args.reasoning_marker, add_special_tokens=False)["input_ids"]

        prefix_ids = None
        if getattr(args, "strip_prefix", None) is not None:
            prefix_ids = tokenizer(args.strip_prefix, add_special_tokens=False)["input_ids"]
            
        def find_sublist(lst, sublst):
            for i in range(len(lst) - len(sublst) + 1):
                if lst[i:i + len(sublst)] == sublst:
                    return i
            return -1
        
        for full, prompt in zip(full_ids, prompt_ids):
            if marker_ids is not None:
                idx = find_sublist(full, marker_ids)
                if idx != -1:
                    full = full[:idx]

            p_len = len(prompt)

            strip_len = 0
            if prefix_ids is not None:
                resp_start = full[p_len:p_len + len(prefix_ids)]
                if resp_start == prefix_ids:
                    strip_len = len(prefix_ids)

            response_tokens = full[p_len + strip_len:]

            response_len = len(response_tokens)

            if args.max_tokens == -1:
                truncated_response_len = response_len
            else:
                truncated_response_len = min(response_len, args.max_tokens)

            input_seq = full[:p_len] + response_tokens[:truncated_response_len]

            lab = list(input_seq)
            if p_len < len(lab):
                lab[:p_len] = [-100] * p_len

            full_ids_list.append(input_seq)
            labels_list.append(lab)
            
        return {"input_ids": full_ids_list, "labels": labels_list, "index": real_indices}

    processed_dataset = dataset.map(
        process_data, batched=True, batch_size=1000, 
        remove_columns=dataset.column_names, with_indices=True
    )
    process_dataset_split(
        args.model_path,
        processed_dataset,
        tokenizer,
        output_file,
        args.gpu_id,
        args.process_id,
        args.total_processes,
        perturbation_surface=args.perturbation_surface,
        perturbation_seed=args.perturbation_seed,
        perturbation_scale=args.perturbation_scale,
        perturbation_overlay_descriptor=args.perturbation_overlay_descriptor,
    )

if __name__ == "__main__":
    main()
