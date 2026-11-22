""""""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def get_base_key(lora_key: str) -> str:
    """"""
    k = lora_key.replace("base_model.model.", "")
    k = k.replace(".lora_A.weight", ".weight")
    k = k.replace(".lora_B.weight", ".weight")
    return k


def _clear_checkpoint_artifacts(output_dir: Union[str, Path], save_as_adapter: bool) -> None:
    """"""
    if save_as_adapter:
        stale_patterns = [
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "config.json",
            "generation_config.json",
        ]
    else:
        stale_patterns = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        ]

    for pattern in stale_patterns:
        for path in Path(output_dir).glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()


def decompose_lora(
    lora_dir: Union[str, Path],
    base_model: Any,
    device: str = "cuda",
    *,
    show_progress: bool = True,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], int, Dict[str, Any], Dict[int, Dict[str, Dict[str, Any]]]]:
    """"""
    adapter_config_path = os.path.join(str(lora_dir), "adapter_config.json")
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    lora_rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scaling = alpha / lora_rank

    safetensors_path = os.path.join(str(lora_dir), "adapter_model.safetensors")
    if not os.path.exists(safetensors_path):
        bin_path = os.path.join(str(lora_dir), "adapter_model.bin")
        adapter_sd = torch.load(bin_path, map_location=device)
    else:
        adapter_sd = load_file(safetensors_path)

    base_keys = set(base_model.state_dict().keys())

    lora_A_keys = [k for k in adapter_sd.keys() if "lora_A" in k]

    num_dirs = lora_rank + 1
    deltas: Dict[int, Dict[str, torch.Tensor]] = {i: {} for i in range(num_dirs)}
    rank1_factors: Dict[int, Dict[str, Dict[str, Any]]] = {i: {} for i in range(num_dirs)}

    iterator = tqdm(lora_A_keys) if show_progress else lora_A_keys
    for a_key in iterator:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in adapter_sd:
            continue

        base_k = get_base_key(a_key)
        if base_k not in base_keys:
            continue

        A = adapter_sd[a_key].float().to(device)
        B = adapter_sd[b_key].float().to(device)

        W_combined = scaling * (B @ A).cpu()
        deltas[lora_rank][base_k] = W_combined.bfloat16()

        Q_A, R_A = torch.linalg.qr(A.T)
        Q_B, R_B = torch.linalg.qr(B)

        M = R_B @ R_A.T
        U, S, Vh = torch.linalg.svd(M)

        for i in range(lora_rank):
            U_i = U[:, i]
            V_i = Vh.T[:, i]
            left = Q_B @ U_i
            right = Q_A @ V_i
            W_comp = (scaling * S[i]) * torch.outer(left, right)
            deltas[i][base_k] = W_comp.bfloat16().cpu()
            rank1_factors[i][a_key.replace(".lora_A.weight", "")] = {
                "left": left.detach().cpu().to(torch.float32),
                "right": right.detach().cpu().to(torch.float32),
                "rank_scale": float(S[i].item()) / float(lora_rank),
            }

    return deltas, lora_rank, config, rank1_factors


def save_rank1_adapter(
    output_dir: Union[str, Path],
    adapter_template: Dict[str, Any],
    rank1_factors_for_direction: Dict[str, Dict[str, Any]],
    scale: float,
    tokenizer: Optional[Any],
) -> Path:
    """"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _clear_checkpoint_artifacts(output_path, save_as_adapter=True)

    alpha_orig = float(adapter_template["lora_alpha"])
    adapter_config = copy.deepcopy(adapter_template)
    adapter_config["r"] = 1
    adapter_config["lora_alpha"] = float(alpha_orig * float(scale))
    adapter_config["inference_mode"] = True

    adapter_state: Dict[str, torch.Tensor] = {}
    for prefix, factor_info in rank1_factors_for_direction.items():
        left = factor_info["left"]
        right = factor_info["right"]
        rank_scale = float(factor_info["rank_scale"])

        adapter_state[f"{prefix}.lora_A.weight"] = (
            (rank_scale * right).unsqueeze(0).to(torch.bfloat16)
        )
        adapter_state[f"{prefix}.lora_B.weight"] = left.unsqueeze(1).to(torch.bfloat16)

    save_file(adapter_state, str(output_path / "adapter_model.safetensors"))
    with open(output_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)

    return output_path
