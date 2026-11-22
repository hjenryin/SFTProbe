import json
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import UnivariateSpline
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq


def maybe_apply_liger_kernel(model_path: Union[str, Path]) -> bool:
    """"""
    model_path = str(model_path)

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[Liger] Skipped: unable to read model config from {model_path}: {e}", flush=True)
        return False

    model_type = str(getattr(cfg, "model_type", "")).lower()
    if model_type.startswith("qwen2"):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            apply_liger_kernel_to_qwen2()
            print("[Liger] Applied Qwen2 kernel optimizations", flush=True)
            return True
        except Exception as e:
            print(f"[Liger] Skipped: failed to apply Qwen2 kernel: {e}", flush=True)
            return False
    elif model_type == "llama":
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            
            apply_liger_kernel_to_llama()
            print("[Liger] Applied LLaMA kernel optimizations", flush=True)
            return True
        except Exception as e:
            print(f"[Liger] Skipped: failed to apply LLaMA kernel: {e}", flush=True)
            return False
    else:
        print(f"[Liger] Skipped: model_type={model_type!r} is not Qwen2.x or LLaMA", flush=True)
        return False


def _normalize_devices(device: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(device, str):
        return [device]
    if isinstance(device, Sequence):
        out = [str(d) for d in device]
        if out:
            return out
    raise ValueError("device must be a non-empty string or sequence of strings")


def _parse_cuda_index(device: str) -> Optional[int]:
    d = str(device).strip().lower()
    if d == "cuda":
        return 0
    if d.startswith("cuda:"):
        try:
            return int(d.split(":", 1)[1])
        except Exception:
            return None
    return None


def _cuda_device_context(device: str):
    idx = _parse_cuda_index(device)
    if idx is None or not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.device(idx)


def _estimate_bf16_model_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    total = 0
    for value in state_dict.values():
        if torch.is_floating_point(value):
            total += int(value.numel()) * 2
    return int(total)


def _get_device_total_vram_bytes(device: str) -> Optional[int]:
    idx = _parse_cuda_index(device)
    if idx is None or not torch.cuda.is_available():
        return None
    if idx < 0 or idx >= torch.cuda.device_count():
        return None
    try:
        return int(torch.cuda.get_device_properties(idx).total_memory)
    except Exception:
        return None


def _bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / float(1024 ** 3)


def _normalize_tensor_surface(tensor_surface: str) -> str:
    surface = str(tensor_surface or "state_dict")
    valid = {"state_dict", "named_parameters_and_buffers"}
    if surface not in valid:
        raise ValueError(
            f"Unknown perturbation tensor surface {tensor_surface!r}; "
            f"expected one of {sorted(valid)}"
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


def _get_live_model_tensors(model, tensor_surface: str = "state_dict") -> Dict[str, torch.Tensor]:
    surface = _normalize_tensor_surface(tensor_surface)
    if surface == "state_dict":
        return model.state_dict()

    named_tensors: List[Tuple[str, torch.Tensor]] = []
    named_tensors.extend(list(model.named_parameters()))
    named_tensors.extend(list(model.named_buffers()))
    return _dedupe_named_tensors(named_tensors)


def _fingerprint_sample_indices(numel: int, take: int, device) -> torch.Tensor:
    if take <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if take >= numel:
        return torch.arange(numel, dtype=torch.long, device=device)
    if take == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    return (
        torch.arange(take, dtype=torch.long, device=device)
        * int(numel - 1)
        // int(take - 1)
    )


def compute_tensor_surface_fingerprint(
    model,
    tensor_surface: str = "state_dict",
    *,
    samples_per_tensor: int = 16,
) -> Dict:
    """"""
    surface = _normalize_tensor_surface(tensor_surface)
    tensors = _get_live_model_tensors(model, tensor_surface=surface)
    hasher = hashlib.sha256()
    rows = []
    float_tensor_count = 0
    sampled_value_count = 0

    for name, tensor in tensors.items():
        if not torch.is_floating_point(tensor):
            continue
        float_tensor_count += 1

        detached = tensor.detach()
        shape = [int(x) for x in detached.shape]
        dtype = str(detached.dtype)
        numel = int(detached.numel())

        hasher.update(str(name).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(dtype.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(",".join(str(x) for x in shape).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(numel).encode("utf-8"))
        hasher.update(b"\0")

        if numel > 0 and samples_per_tensor > 0:
            flat = detached.reshape(-1)
            take = min(int(samples_per_tensor), numel)
            idx = _fingerprint_sample_indices(numel, take, flat.device)
            sample = flat.index_select(0, idx).detach().cpu().float().contiguous()
            sample_bytes = sample.numpy().tobytes()
            hasher.update(sample_bytes)
            sampled_value_count += int(sample.numel())
        else:
            hasher.update(b"<empty>")

        rows.append({
            "name": str(name),
            "dtype": dtype,
            "shape": shape,
            "numel": numel,
        })

    return {
        "algorithm": "sha256_surface_sample_v1",
        "tensor_surface": surface,
        "samples_per_tensor": int(samples_per_tensor),
        "float_tensor_count": int(float_tensor_count),
        "sampled_value_count": int(sampled_value_count),
        "digest": hasher.hexdigest(),
        "tensors": rows,
    }


def _to_chat_prompt_and_response(item: Dict) -> Tuple[str, str]:
    conversations = item.get("conversations")
    if isinstance(conversations, list) and conversations:
        user_content = ""
        assistant_content = ""
        for msg in conversations:
            role = msg.get("role")
            if role == "user":
                user_content = msg.get("value") or msg.get("content") or ""
            elif role == "assistant":
                assistant_content = msg.get("value") or msg.get("content") or ""
        return str(user_content), str(assistant_content)

    prompt = item.get("prompt", "")
    completion = item.get("completion", "")
    response = ""
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            response = first.get("content", "")
        else:
            response = str(first)
    elif isinstance(completion, dict):
        response = completion.get("content", "")
    else:
        response = str(completion)

    return str(prompt), str(response)


def _find_sublist(values: List[int], target: List[int]) -> int:
    if not target or len(target) > len(values):
        return -1
    last = len(values) - len(target) + 1
    for i in range(last):
        if values[i:i + len(target)] == target:
            return i
    return -1


def _load_jsonl_rows(dataset_path: Union[str, Path]) -> List[Dict]:
    rows: List[Dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _prepare_examples(
    rows: Iterable[Dict],
    tokenizer,
    max_tokens: int,
    reasoning_marker: Optional[str],
) -> List[Dict[str, List[int]]]:
    marker_ids = None
    if reasoning_marker is not None:
        marker_ids = tokenizer(reasoning_marker, add_special_tokens=False)["input_ids"]

    examples: List[Dict[str, List[int]]] = []
    for row in rows:
        prompt_text, response_text = _to_chat_prompt_and_response(row)
        prompt_chat = [{"role": "user", "content": prompt_text}]

        prompt_rendered = tokenizer.apply_chat_template(
            prompt_chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_rendered + response_text

        prompt_ids = tokenizer(prompt_rendered, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if marker_ids is not None:
            idx = _find_sublist(full_ids, marker_ids)
            if idx != -1:
                full_ids = full_ids[:idx]

        prompt_len = len(prompt_ids)
        response_tokens = full_ids[prompt_len:]
        if max_tokens != -1:
            response_tokens = response_tokens[:max_tokens]

        input_seq = full_ids[:prompt_len] + response_tokens
        labels = list(input_seq)
        if prompt_len < len(labels):
            labels[:prompt_len] = [-100] * prompt_len
        else:
            labels = [-100] * len(labels)

        examples.append({"input_ids": input_seq, "labels": labels})

    return examples


def _sample_state_key_wise_direction(
    base_state_dict: Dict[str, torch.Tensor],
    seed: int,
    *,
    sample_device: str,
    output_device: str,
    desc: str,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=sample_device)
    generator.manual_seed(int(seed))

    direction: Dict[str, torch.Tensor] = {}
    for key, value in tqdm(
        base_state_dict.items(),
        desc=desc,
        leave=False,
    ):
        if not torch.is_floating_point(value):
            continue

        eps = torch.randn(
            value.shape,
            generator=generator,
            dtype=torch.float32,
            device=sample_device,
        )
        l2_norm = float(torch.norm(eps, p=2).item())
        d = max(1, eps.numel())
        key_scale = l2_norm / math.sqrt(float(d))

        direction_tensor = (key_scale * eps).to(dtype=value.dtype)
        if str(output_device) != str(sample_device):
            direction_tensor = direction_tensor.to(output_device)
        direction[key] = direction_tensor

        del eps

    return direction


def _compute_mean_loss_outputs_loss(
    model,
    examples: List[Dict[str, List[int]]],
    device: str,
    tokenizer,
    batch_size: int = 2,
) -> Tuple[float, int]:
    if not examples:
        return 0.0, 0

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    dataloader = DataLoader(
        examples,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=collator,
    )

    total_loss = 0.0
    total_batches = 0
    model.eval()

    for batch in tqdm(dataloader, desc=f"Eval {device}", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            try:
                outputs = model(**batch, skip_logits=True)
            except TypeError:
                outputs = model(**batch)


        total_loss += float(outputs.loss.detach().float().item())
        total_batches += 1

        del outputs

    return float(total_loss), int(total_batches)


def _compute_per_example_losses(
    model,
    examples: List[Dict[str, List[int]]],
    device: str,
    tokenizer,
) -> torch.Tensor:
    if not examples:
        return torch.empty(0, dtype=torch.float32)

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    dataloader = DataLoader(
        examples,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    losses: List[float] = []
    model.eval()

    for batch in tqdm(dataloader, desc=f"Eval losses {device}", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            try:
                outputs = model(**batch, skip_logits=True)
            except TypeError:
                outputs = model(**batch)

        loss = outputs.loss.detach().float()
        losses.append(0.0 if bool(torch.isnan(loss).item()) else float(loss.item()))

        del outputs

    return torch.tensor(losses, dtype=torch.float32)


def evaluate_model(base_model, perturbation_direction, scale, data, device):
    evaluator = LLMEvaluator(
        base_model=base_model,
        perturbation_direction=perturbation_direction,
        data=data,
        device=device,
    )
    return evaluator.evaluate(float(scale))


class LLMEvaluator:
    def __init__(
        self,
        base_model,
        perturbation_direction: Optional[Dict[str, torch.Tensor]],
        data,
        device,
        *,
        seed: int = 42,
        max_eval_batch_size: int = 1,
        perturbation_tensor_surface: str = "state_dict",
    ):
        requested = _normalize_devices(device)
        deduped: List[str] = []
        for d in requested:
            if d not in deduped:
                deduped.append(d)

        self.requested_devices = deduped
        self.device = list(deduped)
        self.primary_device = self.device[0]
        self.max_eval_batch_size = max(1, int(max_eval_batch_size))
        self.perturbation_tensor_surface = _normalize_tensor_surface(perturbation_tensor_surface)
        print(
            "[LLMEvaluator] init start. "
            f"requested_devices={self.requested_devices}, "
            f"perturbation_tensor_surface={self.perturbation_tensor_surface}",
            flush=True,
        )

        self.base_model_path = base_model if isinstance(base_model, str) else None
        self.models_by_device: Dict[str, object] = {}
        self.keep_resident_by_device: Dict[str, bool] = {}
        self.memory_plan: Dict[str, object] = {}

        self.base_state_dict_by_device: Dict[str, Dict[str, torch.Tensor]] = {}
        self.base_snapshot_fingerprint_by_device: Dict[str, bytes] = {}
        self.perturbation_direction_by_device: Dict[str, Dict[str, torch.Tensor]] = {}

        if isinstance(base_model, str):
            maybe_apply_liger_kernel(base_model)

            print(f"[LLMEvaluator] loading tokenizer from {base_model}", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

            def _load_model_to(target_device: str):
                print(f"[LLMEvaluator] loading model on {target_device}", flush=True)
                return AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map=target_device,
                )

            initial_device = self.primary_device
            initial_model = _load_model_to(initial_device)

            model_size_bf16_bytes = _estimate_bf16_model_bytes(initial_model.state_dict())
            required_single_gpu_bytes = int(4 * model_size_bf16_bytes)
            print(
                "[LLMEvaluator] model memory requirement: "
                f"bf16={_bytes_to_gib(model_size_bf16_bytes):.3f} GiB, "
                f"threshold(4x)={_bytes_to_gib(required_single_gpu_bytes):.3f} GiB",
                flush=True,
            )

            vram_by_device_bytes = {
                dev: _get_device_total_vram_bytes(dev)
                for dev in self.requested_devices
            }
            self.keep_resident_by_device = {
                dev: bool(total_bytes is not None and int(total_bytes) > required_single_gpu_bytes)
                for dev, total_bytes in vram_by_device_bytes.items()
            }
            print(f"[LLMEvaluator] per-device residency={self.keep_resident_by_device}", flush=True)

            self.models_by_device[initial_device] = initial_model
            for dev in self.requested_devices[1:]:
                self.models_by_device[dev] = _load_model_to(dev)

            self.device = list(self.models_by_device.keys())
            self.primary_device = self.device[0]

            self.memory_plan = {
                "model_size_bf16_bytes": int(model_size_bf16_bytes),
                "model_size_bf16_gib": _bytes_to_gib(model_size_bf16_bytes),
                "required_single_gpu_bytes": int(required_single_gpu_bytes),
                "required_single_gpu_gib": _bytes_to_gib(required_single_gpu_bytes),
                "vram_by_device_bytes": {
                    dev: (int(v) if v is not None else None)
                    for dev, v in vram_by_device_bytes.items()
                },
                "vram_by_device_gib": {
                    dev: _bytes_to_gib(v)
                    for dev, v in vram_by_device_bytes.items()
                },
                "placement_mode": "per-device-residency",
                "keep_resident_by_device": dict(self.keep_resident_by_device),
                "selected_devices": list(self.device),
                "perturbation_tensor_surface": self.perturbation_tensor_surface,
            }
        else:
            if len(self.device) != 1:
                raise ValueError("When base_model is a model object, only one device is supported")
            self.models_by_device[self.primary_device] = base_model
            self.tokenizer = None

            model_size_bf16_bytes = _estimate_bf16_model_bytes(base_model.state_dict())
            required_single_gpu_bytes = int(4 * model_size_bf16_bytes)
            vram_total = _get_device_total_vram_bytes(self.primary_device)
            keep_resident = bool(vram_total is not None and int(vram_total) > required_single_gpu_bytes)
            self.keep_resident_by_device = {self.primary_device: keep_resident}

            self.memory_plan = {
                "model_size_bf16_bytes": int(model_size_bf16_bytes),
                "model_size_bf16_gib": _bytes_to_gib(model_size_bf16_bytes),
                "required_single_gpu_bytes": int(required_single_gpu_bytes),
                "required_single_gpu_gib": _bytes_to_gib(required_single_gpu_bytes),
                "vram_by_device_bytes": {
                    self.primary_device: (int(vram_total) if vram_total is not None else None)
                },
                "vram_by_device_gib": {
                    self.primary_device: _bytes_to_gib(vram_total)
                },
                "placement_mode": "provided-model-single-device",
                "keep_resident_by_device": dict(self.keep_resident_by_device),
                "selected_devices": [self.primary_device],
                "perturbation_tensor_surface": self.perturbation_tensor_surface,
            }

        self.base_model = self.models_by_device[self.primary_device]

        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "right"

        self.base_fingerprint_by_device = {
            dev: self._get_fingerprint(model)
            for dev, model in self.models_by_device.items()
        }
        self.model_fingerprint = self.base_fingerprint_by_device.get(self.primary_device, b"")
        need_cpu_base = any(not keep for keep in self.keep_resident_by_device.values())
        self.base_state_dict_cpu: Dict[str, torch.Tensor] = {}
        if need_cpu_base:
            print("[LLMEvaluator] cloning base state to CPU (non-resident devices detected)", flush=True)
            for key, value in tqdm(
                self._get_live_model_tensors(self.base_model).items(),
                desc="[LLMEvaluator] Clone base->CPU",
                leave=False,
            ):
                self.base_state_dict_cpu[key] = value.detach().cpu().clone()
        else:
            print("[LLMEvaluator] skipping CPU base clone (all devices resident)", flush=True)

        print("[LLMEvaluator] preparing perturbation direction", flush=True)
        self.perturbation_direction: Dict[str, torch.Tensor] = {}
        if perturbation_direction is not None:
            self.perturbation_direction = perturbation_direction
        else:
            non_resident_devices = [dev for dev, keep in self.keep_resident_by_device.items() if not keep]
            if non_resident_devices:
                cuda_candidates = [dev for dev in self.device if _parse_cuda_index(dev) is not None]
                sample_device = cuda_candidates[0] if cuda_candidates else "cpu"
                print(
                    "[LLMEvaluator] sampling fallback direction on "
                    f"{sample_device} and offloading each tensor to CPU",
                    flush=True,
                )
                self.perturbation_direction = _sample_state_key_wise_direction(
                    self._get_live_model_tensors(self.base_model),
                    int(seed),
                    sample_device=sample_device,
                    output_device="cpu",
                    desc="[LLMEvaluator] Sample direction (GPU->CPU)",
                )

        print("[LLMEvaluator] materializing resident tensors on eligible devices", flush=True)
        for dev, keep_resident in self.keep_resident_by_device.items():
            if not keep_resident:
                continue
            model = self.models_by_device.get(dev)
            if model is None:
                continue

            model_sd = self._get_live_model_tensors(model)

            resident_base: Dict[str, torch.Tensor] = {}
            if self.base_state_dict_cpu:
                for key, base_cpu in tqdm(
                    self.base_state_dict_cpu.items(),
                    desc=f"[LLMEvaluator] Resident base {dev}",
                    leave=False,
                ):
                    if key not in model_sd:
                        continue
                    target = model_sd[key]
                    resident_base[key] = base_cpu.to(device=target.device, dtype=target.dtype).clone()
            else:
                for key, value in tqdm(
                    model_sd.items(),
                    desc=f"[LLMEvaluator] Resident base {dev}",
                    leave=False,
                ):
                    resident_base[key] = value.detach().clone()
            self.base_state_dict_by_device[dev] = resident_base

            if perturbation_direction is None:
                resident_direction = _sample_state_key_wise_direction(
                    model_sd,
                    int(seed),
                    sample_device=dev,
                    output_device=dev,
                    desc=f"[LLMEvaluator] Resident direction {dev}",
                )
            else:
                resident_direction = {}
                for key, direction in tqdm(
                    self.perturbation_direction.items(),
                    desc=f"[LLMEvaluator] Resident direction {dev}",
                    leave=False,
                ):
                    if key not in model_sd:
                        continue
                    target = model_sd[key]
                    resident_direction[key] = direction.to(device=target.device, dtype=target.dtype).clone()
            self.perturbation_direction_by_device[dev] = resident_direction

        for dev in self.device:
            base_snapshot = self._base_snapshot_for_device(dev)
            if base_snapshot:
                self.base_snapshot_fingerprint_by_device[dev] = self._fingerprint_tensors(base_snapshot)

        print("[LLMEvaluator] init complete", flush=True)

        self.data = data
        self._prepared_cache: Dict[Tuple[str, int, str], List[Dict[str, List[int]]]] = {}

    def _fingerprint_tensors(self, tensors: Dict[str, torch.Tensor]) -> bytes:
        if not tensors:
            return b""

        sampled_chunks: List[torch.Tensor] = []
        for tensor in tensors.values():
            if not torch.is_floating_point(tensor):
                continue
            flat = tensor.detach().cpu().float().reshape(-1)
            if flat.numel() == 0:
                continue
            take = min(16, int(flat.numel()))
            sampled_chunks.append(flat[:take])
            if len(sampled_chunks) >= 8:
                break

        if not sampled_chunks:
            return b""
        return torch.cat(sampled_chunks, dim=0).numpy().tobytes()

    def _get_fingerprint(self, model):
        return self._fingerprint_tensors(self._get_live_model_tensors(model))

    def _get_live_model_tensors(self, model) -> Dict[str, torch.Tensor]:
        return _get_live_model_tensors(
            model,
            tensor_surface=self.perturbation_tensor_surface,
        )

    def _base_snapshot_for_device(self, device: str) -> Dict[str, torch.Tensor]:
        resident_base = self.base_state_dict_by_device.get(device)
        if resident_base:
            return resident_base
        return self.base_state_dict_cpu

    def _assert_base_snapshot_intact(self, device: str) -> None:
        expected = self.base_snapshot_fingerprint_by_device.get(device)
        if expected is None:
            return
        current = self._fingerprint_tensors(self._base_snapshot_for_device(device))
        if current != expected:
            raise AssertionError(
                "Base snapshot fingerprint changed; baseline tensors were mutated "
                f"for device={device}"
            )

    def _assert_model_restored_to_base(self, model, device: str) -> None:
        expected = self.base_snapshot_fingerprint_by_device.get(device)
        if expected is None:
            return
        current = self._get_fingerprint(model)
        if current != expected:
            raise AssertionError(
                "Model restore check failed: live model fingerprint does not match "
                f"base snapshot for device={device}"
            )

    def _restore_model_to_base(self, model, device: str) -> None:
        self._apply_scale_to_model(model, device=device, scale=0.0)
        self._assert_model_restored_to_base(model=model, device=device)

    def _assert_perturbation_applied(self, *, model, device: str, scale: float) -> None:
        if abs(float(scale)) == 0.0:
            return
        base_fp = self.base_fingerprint_by_device.get(device)
        if base_fp is None:
            return
        current_fp = self._get_fingerprint(model)
        if current_fp == base_fp:
            raise AssertionError(
                "Perturbation check failed: model fingerprint is unchanged from base "
                f"on {device} for scale={float(scale):.8f}"
            )

    def _assert_cross_device_fingerprint(self) -> None:
        if len(self.device) <= 1:
            return

        fingerprints: Dict[str, bytes] = {}
        for dev in self.device:
            fingerprints[dev] = self._get_fingerprint(self.models_by_device[dev])

        baseline = next(iter(fingerprints.values()))
        mismatched = [dev for dev, fp in fingerprints.items() if fp != baseline]
        if mismatched:
            raise AssertionError(
                "Cross-device model fingerprint mismatch before evaluation: "
                f"mismatched_devices={mismatched}"
            )

    def _apply_scale_to_model(self, model, device: str, scale: float) -> None:
        scale = float(scale)
        self._assert_base_snapshot_intact(device)
        model_sd = self._get_live_model_tensors(model)
        resident_base = self.base_state_dict_by_device.get(device)
        resident_direction = self.perturbation_direction_by_device.get(device)

        with torch.no_grad():
            for key, target in model_sd.items():
                if resident_base is not None and key in resident_base:
                    target.copy_(resident_base[key])
                elif key in self.base_state_dict_cpu:
                    base_cpu = self.base_state_dict_cpu[key]
                    target.copy_(base_cpu.to(device=target.device, dtype=target.dtype))
                else:
                    continue

                if resident_direction is not None:
                    direction = resident_direction.get(key)
                else:
                    direction = self.perturbation_direction.get(key)
                    if direction is not None:
                        direction = direction.to(device=target.device, dtype=target.dtype)
                if direction is None:
                    continue
                target.add_(scale * direction)

    def _evaluate_chunk_on_device(
        self,
        *,
        scale: float,
        device: str,
        data: List[Dict[str, List[int]]],
        start: int,
        end: int,
    ) -> torch.Tensor:
        if start >= end:
            return torch.empty(0, dtype=torch.float64)

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for loss evaluation batches")

        model = self.models_by_device[device]
        with _cuda_device_context(device):
            try:
                self._apply_scale_to_model(model, device=device, scale=float(scale))
                self._assert_perturbation_applied(model=model, device=device, scale=float(scale))

                loss_sum, batch_count = _compute_mean_loss_outputs_loss(
                    model=model,
                    examples=data[int(start):int(end)],
                    device=device,
                    tokenizer=self.tokenizer,
                    batch_size=self.max_eval_batch_size,
                )
            finally:
                self._restore_model_to_base(model, device=device)
        return torch.tensor([loss_sum, float(batch_count)], dtype=torch.float64)

    def _evaluate_loss_chunk_on_device(
        self,
        *,
        scale: float,
        device: str,
        data: List[Dict[str, List[int]]],
        start: int,
        end: int,
    ) -> torch.Tensor:
        if start >= end:
            return torch.empty(0, dtype=torch.float32)

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for loss evaluation batches")

        model = self.models_by_device[device]
        with _cuda_device_context(device):
            try:
                self._apply_scale_to_model(model, device=device, scale=float(scale))
                self._assert_perturbation_applied(model=model, device=device, scale=float(scale))

                losses = _compute_per_example_losses(
                    model=model,
                    examples=data[int(start):int(end)],
                    device=device,
                    tokenizer=self.tokenizer,
                )
            finally:
                self._restore_model_to_base(model, device=device)
        return losses

    def materialize_checkpoint(self, scale: float, output_dir: Union[str, Path]) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            self._apply_scale_to_model(self.base_model, device=self.primary_device, scale=float(scale))
            self.base_model.save_pretrained(output_path)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_path)
        finally:
            self._restore_model_to_base(self.base_model, device=self.primary_device)
        return output_path

    def set_perturbation_direction(self, direction: Optional[Dict[str, torch.Tensor]]) -> None:
        """"""
        self.perturbation_direction = dict(direction) if direction else {}
        self.perturbation_direction_by_device = {}

        if not self.perturbation_direction:
            return

        for dev, keep_resident in self.keep_resident_by_device.items():
            if not keep_resident:
                continue
            model = self.models_by_device.get(dev)
            if model is None:
                continue
            model_sd = self._get_live_model_tensors(model)
            resident_direction: Dict[str, torch.Tensor] = {}
            for key, value in self.perturbation_direction.items():
                if key not in model_sd:
                    continue
                target = model_sd[key]
                resident_direction[key] = value.to(
                    device=target.device, dtype=target.dtype
                ).clone()
            self.perturbation_direction_by_device[dev] = resident_direction

    def sync_all_devices_to_scale(self, scale: float) -> None:
        """"""
        scale = float(scale)
        for dev, model in self.models_by_device.items():
            self._apply_scale_to_model(model, device=dev, scale=scale)

    def prepare_data(
        self,
        dataset_path: Union[str, Path],
        max_tokens: int,
        reasoning_marker: Optional[str] = None,
    ) -> List[Dict[str, List[int]]]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required to prepare data, but evaluator was initialized without one")

        key = (str(Path(dataset_path)), int(max_tokens), str(reasoning_marker))
        cached = self._prepared_cache.get(key)
        if cached is not None:
            return cached

        rows = _load_jsonl_rows(dataset_path)
        prepared = _prepare_examples(
            rows=rows,
            tokenizer=self.tokenizer,
            max_tokens=int(max_tokens),
            reasoning_marker=reasoning_marker,
        )
        self._prepared_cache[key] = prepared
        return prepared

    def evaluate_scale(
        self,
        scale: float,
        *,
        data: Optional[List[Dict[str, List[int]]]] = None,
    ) -> Tuple[torch.Tensor, float]:
        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No evaluation data was provided")

        self._assert_cross_device_fingerprint()

        scale = float(scale)
        if len(self.device) == 1:
            losses = self._evaluate_chunk_on_device(
                scale=scale,
                device=self.primary_device,
                data=data,
                start=0,
                end=len(data),
            )
        else:
            n = len(data)
            chunk_specs: List[Tuple[str, int, int]] = []
            for i, dev in enumerate(self.device):
                start = (n * i) // len(self.device)
                end = (n * (i + 1)) // len(self.device)
                if start < end:
                    chunk_specs.append((dev, start, end))

            if not chunk_specs:
                losses = torch.empty(0, dtype=torch.float32)
            elif len(chunk_specs) == 1:
                dev, start, end = chunk_specs[0]
                losses = self._evaluate_chunk_on_device(
                    scale=scale,
                    device=dev,
                    data=data,
                    start=start,
                    end=end,
                )
            else:
                results: List[Optional[torch.Tensor]] = [None] * len(chunk_specs)
                with ThreadPoolExecutor(max_workers=len(chunk_specs)) as pool:
                    futures = []
                    for idx, (dev, start, end) in enumerate(chunk_specs):
                        future = pool.submit(
                            self._evaluate_chunk_on_device,
                            scale=scale,
                            device=dev,
                            data=data,
                            start=start,
                            end=end,
                        )
                        futures.append((idx, future))

                    for idx, future in futures:
                        results[idx] = future.result()

                losses = torch.cat([x for x in results if x is not None], dim=0) if results else torch.empty(0, dtype=torch.float32)

        if losses.numel() == 0:
            return torch.empty(0, dtype=torch.float32), float("nan")
        stats = losses.reshape(-1, 2)
        loss_sum = float(stats[:, 0].sum().item())
        batch_count = float(stats[:, 1].sum().item())
        mean_loss = float(loss_sum / batch_count) if batch_count > 0 else float("nan")
        return torch.empty(0, dtype=torch.float32), mean_loss

    def evaluate_scale_losses(
        self,
        scale: float,
        *,
        data: Optional[List[Dict[str, List[int]]]] = None,
    ) -> torch.Tensor:
        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No evaluation data was provided")

        self._assert_cross_device_fingerprint()

        scale = float(scale)
        if len(self.device) == 1:
            return self._evaluate_loss_chunk_on_device(
                scale=scale,
                device=self.primary_device,
                data=data,
                start=0,
                end=len(data),
            )

        n = len(data)
        chunk_specs: List[Tuple[str, int, int]] = []
        for i, dev in enumerate(self.device):
            start = (n * i) // len(self.device)
            end = (n * (i + 1)) // len(self.device)
            if start < end:
                chunk_specs.append((dev, start, end))

        if not chunk_specs:
            return torch.empty(0, dtype=torch.float32)
        if len(chunk_specs) == 1:
            dev, start, end = chunk_specs[0]
            return self._evaluate_loss_chunk_on_device(
                scale=scale,
                device=dev,
                data=data,
                start=start,
                end=end,
            )

        results: List[Optional[torch.Tensor]] = [None] * len(chunk_specs)
        with ThreadPoolExecutor(max_workers=len(chunk_specs)) as pool:
            futures = []
            for idx, (dev, start, end) in enumerate(chunk_specs):
                future = pool.submit(
                    self._evaluate_loss_chunk_on_device,
                    scale=scale,
                    device=dev,
                    data=data,
                    start=start,
                    end=end,
                )
                futures.append((idx, future))

            for idx, future in futures:
                results[idx] = future.result()

        return (
            torch.cat([x for x in results if x is not None], dim=0)
            if results
            else torch.empty(0, dtype=torch.float32)
        )

    def evaluate(self, x):
        _, mean_loss = self.evaluate_scale(float(x), data=self.data)
        return float(mean_loss)


class InflectionPointFinder:
    def __init__(
        self,
        evaluator: LLMEvaluator,
        a,
        b,
        history: Optional[Dict[float, float]] = None,
        debug_snapshot_callback: Optional[Callable[[Dict[str, List[float]]], None]] = None,
        smoothing_factor: Optional[float] = None,
        curvature_ratio: float = 0.1,
        dense_grid_size: int = 400,
        enforce_post_min_monotone: bool = False,
        post_min_decrease_tol: float = 1e-4,
        post_min_decrease_frac: float = 0.9,
        extend_b_if_no_inflection: bool = False,
        b_extend_factor: float = 2.0,
        max_b_extensions: int = 3,
    ):
        """"""
        self.f = evaluator.evaluate
        self.a = float(a)
        self.b = float(b)
        self.b_initial = float(b)
        raw_history = dict(history or {})
        self.history: Dict[float, float] = {
            float(x): float(y)
            for x, y in raw_history.items()
            if self.a <= float(x) <= self.b
        }
        self.eval_count = 0
        self.trace: List[Dict[str, float]] = []
        self.debug_snapshots: List[Dict[str, List[float]]] = []
        self.debug_snapshot_callback = debug_snapshot_callback
        self.smoothing_factor = smoothing_factor
        self.curvature_ratio = float(curvature_ratio)
        self.dense_grid_size = int(max(64, dense_grid_size))
        self.enforce_post_min_monotone = bool(enforce_post_min_monotone)
        self.post_min_decrease_tol = float(post_min_decrease_tol)
        frac_val = float(post_min_decrease_frac)
        if not (0.0 < frac_val < 1.0):
            raise ValueError(
                "post_min_decrease_frac must be in (0, 1); got "
                f"{post_min_decrease_frac!r}"
            )
        self.post_min_decrease_frac = frac_val
        self.extend_b_if_no_inflection = bool(extend_b_if_no_inflection)
        if float(b_extend_factor) <= 1.0:
            raise ValueError(
                f"b_extend_factor must be > 1.0 to grow the bound, got {b_extend_factor}"
            )
        self.b_extend_factor = float(b_extend_factor)
        self.max_b_extensions = int(max(0, int(max_b_extensions)))
        self.b_extensions_applied = 0
        self.b_extension_history: List[Tuple[float, float]] = []

    def _build_smoothing_spline(self, x_data: np.ndarray, y_data: np.ndarray) -> UnivariateSpline:
        k = max(1, min(3, int(len(x_data) - 1)))
        if self.smoothing_factor is None:
            y_var = float(np.var(y_data)) if len(y_data) > 1 else 0.0
            s = max(1e-12, float(len(x_data)) * y_var)
        else:
            s = max(0.0, float(self.smoothing_factor))

        try:
            return UnivariateSpline(x_data, y_data, k=k, s=s)
        except Exception:
            return UnivariateSpline(x_data, y_data, k=k, s=0.0)

    def _first_threshold_crossing(
        self,
        fit_x: np.ndarray,
        second_derivative: np.ndarray,
        threshold: float,
    ) -> Tuple[Optional[float], int]:
        if len(fit_x) < 2:
            return None, -1

        for i in range(len(fit_x) - 1):
            left = float(second_derivative[i])
            right = float(second_derivative[i + 1])
            if left > threshold and right <= threshold:
                if right == left:
                    return float(0.5 * (fit_x[i] + fit_x[i + 1])), i
                ratio = (threshold - left) / (right - left)
                ratio = float(min(1.0, max(0.0, ratio)))
                x_cross = float(fit_x[i] + ratio * (fit_x[i + 1] - fit_x[i]))
                return x_cross, i

        for i, val in enumerate(second_derivative):
            if float(val) <= threshold:
                return float(fit_x[i]), max(0, i - 1)

        return None, -1

    def _shrink_b_for_post_min_decrease(self, current_b: float) -> Optional[float]:
        """"""
        if not self.history:
            return None

        items = [
            (float(x), float(y))
            for x, y in self.history.items()
            if self.a <= float(x) <= float(current_b) + 1e-12
        ]
        if len(items) < 3:
            return None

        items.sort(key=lambda kv: kv[0])
        scales = [x for x, _ in items]
        losses = [y for _, y in items]

        m = min(range(len(losses)), key=lambda i: losses[i])
        running_max_post_min = losses[m]
        for i in range(m + 1, len(losses)):
            rel_threshold = self.post_min_decrease_frac * running_max_post_min
            abs_threshold = running_max_post_min - self.post_min_decrease_tol
            if losses[i] < rel_threshold and losses[i] < abs_threshold:
                new_b = float(scales[i] - 1e-5)
                if new_b > self.a and new_b < float(current_b):
                    return float(new_b)
                return None
            if losses[i] > running_max_post_min:
                running_max_post_min = losses[i]
        return None

    def eval(self, x):
        """"""
        x = float(max(self.a, min(self.b, float(x))))
        if x not in self.history:
            self.history[x] = float(self.f(x))
            self.eval_count += 1
        return self.history[x]

    def run(self, ytol, max_iter=10, initial_points: Optional[Sequence[float]] = None):
        """"""
        ytol = float(ytol)
        max_iter = int(max_iter)

        if initial_points is None:
            initial_points = np.linspace(self.a, self.b, 4).tolist()

        for x in initial_points:
            x = float(max(self.a, min(self.b, x)))
            self.eval(x)

        if not self.history:
            raise RuntimeError("No points were evaluated during inflection search")

        prev_y = None
        x_curr = None
        search_b = float(self.b)

        for it in range(1, max_iter + 1):
            if self.enforce_post_min_monotone:
                shrunk_b = self._shrink_b_for_post_min_decrease(search_b)
                if shrunk_b is not None and shrunk_b < search_b - 1e-9:
                    search_b = float(shrunk_b)

            all_x = np.array(sorted(self.history.keys()), dtype=np.float64)
            all_y = np.array([self.history[x] for x in all_x], dtype=np.float64)
            in_range = (all_x >= self.a) & (all_x <= search_b + 1e-12)
            x_data = all_x[in_range]
            y_data = all_y[in_range]

            if len(x_data) < 3:
                break

            width = float(max(1e-12, search_b - self.a))
            u_data = (x_data - self.a) / width
            spline = self._build_smoothing_spline(u_data, y_data)

            fit_u = np.linspace(0.0, 1.0, self.dense_grid_size, dtype=np.float64)
            fit_x = self.a + fit_u * width
            fit_y = spline(fit_u)
            fit_d2_u = spline.derivative(2)(fit_u)
            fit_d2 = fit_d2_u / (width * width)

            valid_d2 = fit_d2[np.isfinite(fit_d2)]
            k_max = float(np.max(valid_d2)) if valid_d2.size > 0 else 0.0
            if not math.isfinite(k_max) or k_max < 0.0:
                k_max = 0.0
            c_val = float(self.curvature_ratio * k_max)
            threshold = float(-c_val)

            self.debug_snapshots.append(
                {
                    "iteration": float(it),
                    "x_all": all_x.astype(np.float64).tolist(),
                    "y_all": all_y.astype(np.float64).tolist(),
                    "x_data": x_data.astype(np.float64).tolist(),
                    "y_data": y_data.astype(np.float64).tolist(),
                    "x_fit": fit_x.astype(np.float64).tolist(),
                    "y_fit": fit_y.astype(np.float64).tolist(),
                    "d2_fit": fit_d2.astype(np.float64).tolist(),
                    "threshold": [threshold],
                    "search_b": [float(search_b)],
                }
            )
            if self.debug_snapshot_callback is not None:
                try:
                    self.debug_snapshot_callback(self.debug_snapshots[-1])
                except Exception:
                    pass

            x_new, crossing_idx = self._first_threshold_crossing(fit_x, fit_d2, threshold)
            extended_this_iter = False
            if x_new is None:
                can_extend = (
                    self.extend_b_if_no_inflection
                    and self.b_extensions_applied < self.max_b_extensions
                    and search_b >= self.b - 1e-9
                )
                if can_extend:
                    old_b = float(self.b)
                    new_b = float(old_b * self.b_extend_factor)
                    self.b = new_b
                    search_b = new_b
                    self.b_extensions_applied += 1
                    self.b_extension_history.append((old_b, new_b))
                    x_new = new_b
                    extended_this_iter = True
                else:
                    diffs = np.diff(x_data)
                    max_gap_idx = int(np.argmax(diffs))
                    x_new = float(x_data[max_gap_idx] + diffs[max_gap_idx] / 2.0)
            else:
                pass

            x_new = float(max(self.a, min(search_b, x_new)))

            min_dist = float(np.min(np.abs(all_x - x_new)))
            if min_dist < 1e-9:
                if len(x_data) >= 2:
                    diffs = np.diff(x_data)
                    max_gap_idx = int(np.argmax(diffs))
                    alt_x = float(x_data[max_gap_idx] + diffs[max_gap_idx] / 2.0)
                    alt_x = float(max(self.a, min(search_b, alt_x)))
                    if float(np.min(np.abs(all_x - alt_x))) >= 1e-9:
                        x_new = alt_x
                    else:
                        in_range_keys = [k for k in self.history.keys() if self.a <= float(k) <= search_b + 1e-12]
                        if not in_range_keys:
                            in_range_keys = list(self.history.keys())
                        best_x = min(in_range_keys, key=lambda k: abs(k - x_new))
                        return float(best_x), float(self.history[best_x])
                else:
                    in_range_keys = [k for k in self.history.keys() if self.a <= float(k) <= search_b + 1e-12]
                    if not in_range_keys:
                        in_range_keys = list(self.history.keys())
                    best_x = min(in_range_keys, key=lambda k: abs(k - x_new))
                    return float(best_x), float(self.history[best_x])

            y_new = float(self.eval(x_new))
            x_curr = float(x_new)

            self.trace.append(
                {
                    "iteration": float(it),
                    "x": float(x_new),
                    "y": float(y_new),
                    "num_points": float(len(self.history)),
                    "threshold": float(threshold),
                    "search_b": float(search_b),
                    "b": float(self.b),
                    "extended_this_iter": 1.0 if extended_this_iter else 0.0,
                }
            )

            if extended_this_iter:
                prev_y = None
                continue

            if prev_y is not None and abs(y_new - prev_y) < ytol:
                break
            prev_y = y_new

        if x_curr is None:
            candidates = [x for x in sorted(self.history.keys()) if self.a <= float(x) <= search_b + 1e-12]
            if not candidates:
                candidates = sorted(self.history.keys())
            x_curr = float(candidates[len(candidates) // 2])
        return float(x_curr), float(self.history[x_curr])
