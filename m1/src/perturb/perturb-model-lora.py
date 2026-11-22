import os
import re
import sys
from tqdm import tqdm
import gc
import json
import torch
import argparse
import copy
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from safetensors.torch import load_file, save_file
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from gpu_utils import get_available_gpus


def load_model_and_tokenizer(model_name, device="cuda"):
    print(f"Loading base model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cpu_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    ref_key = next(iter(cpu_state_dict))
    cpu_fingerprint = (ref_key, cpu_state_dict[ref_key].flatten()[0].item())

    model = model.to(device)
    print("✓ Model loaded successfully")
    return model, tokenizer, cpu_state_dict, cpu_fingerprint

def sample_dataset(dataset_path, n_samples=100, seed=42):
    print(f"Loading dataset from {dataset_path}...")
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    np.random.seed(seed)
    total = len(samples)

    if n_samples == -1 or total <= n_samples:
        if total <= n_samples:
            print(f"Dataset has only {total} examples, using all of them")
        else:
            print(f"Dataset has {total} examples, using all (n_samples=-1)")
        indices = list(range(total))
    else:
        indices = np.random.choice(total, n_samples, replace=False).tolist()
    
    sampled = [samples[i] for i in indices]
    print(f"Sampled {len(sampled)} examples")
    return sampled

def compute_loss_on_samples(model, tokenizer, samples, max_length=2048, batch_size=4, device='cuda'):
    model.eval()
    tokenizer.padding_side = 'right'

    dataset_data = []
    for sample in samples:
        if "conversations" in sample:
            convos = sample['conversations']
            user_content = ""
            asst_content = ""
            for msg in convos:
                if msg['role'] == 'user':
                    user_content = msg.get('value') or msg.get('content')
                elif msg['role'] == 'assistant':
                    asst_content = msg.get('value') or msg.get('content')
            
            prompt_msgs = [{"role": "user", "content": user_content}]
            response_text = asst_content
        else:
            if "prompt" in sample and "completion" in sample:
                prompt_msgs = [{"role": "user", "content": sample['prompt']}]
                response_text = sample['completion'][0]['content'] if isinstance(sample['completion'], list) else sample['completion']
            else:
                prompt_msgs = [{"role": "user", "content": sample.get('text', str(sample))}]
                response_text = ""

        p_text = tokenizer.apply_chat_template([prompt_msgs], tokenize=False, add_generation_prompt=True)[0]
        r_text = response_text
        
        p_ids = tokenizer(p_text, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(r_text, add_special_tokens=False)["input_ids"]

        if max_length != -1 and len(r_ids) > max_length:
            r_ids = r_ids[:max_length]
            
        full_ids = p_ids + r_ids
        labels = ([-100] * len(p_ids)) + r_ids

        dataset_data.append({
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        })

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors='pt')
    dataloader = DataLoader(dataset_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=data_collator)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
                
    if num_batches == 0:
        return float('inf')
    return total_loss / num_batches

def get_base_key(lora_key):
    """"""
    k = lora_key.replace("base_model.model.", "")
    k = k.replace(".lora_A.weight", ".weight")
    k = k.replace(".lora_B.weight", ".weight")
    return k

def _clear_checkpoint_artifacts(output_dir, save_as_adapter):
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

def _has_saved_checkpoint(output_dir, save_as_adapter):
    output_path = Path(output_dir)
    if save_as_adapter:
        return (
            (output_path / "adapter_config.json").exists()
            and (
                (output_path / "adapter_model.safetensors").exists()
                or (output_path / "adapter_model.bin").exists()
            )
        )

    return (
        (output_path / "model.safetensors").exists()
        or (output_path / "model.safetensors.index.json").exists()
        or any(path.suffix == ".safetensors" for path in output_path.glob("*.safetensors"))
    )

def _save_rank1_adapter(output_dir, adapter_template, rank1_factors, scale, tokenizer):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _clear_checkpoint_artifacts(output_path, save_as_adapter=True)

    alpha_orig = float(adapter_template["lora_alpha"])
    adapter_config = copy.deepcopy(adapter_template)
    adapter_config["r"] = 1
    adapter_config["lora_alpha"] = float(alpha_orig * float(scale))
    adapter_config["inference_mode"] = True

    adapter_state = {}
    for prefix, factor_info in rank1_factors.items():
        left = factor_info["left"]
        right = factor_info["right"]
        rank_scale = float(factor_info["rank_scale"])

        adapter_state[f"{prefix}.lora_A.weight"] = (rank_scale * right).unsqueeze(0).to(torch.bfloat16)
        adapter_state[f"{prefix}.lora_B.weight"] = left.unsqueeze(1).to(torch.bfloat16)

    save_file(adapter_state, str(output_path / "adapter_model.safetensors"))
    with open(output_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    tokenizer.save_pretrained(output_path)

def decompose_lora(lora_dir, base_model,device="cuda"):
    adapter_config_path = os.path.join(lora_dir, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        config = json.load(f)
    lora_rank = config["r"]
    alpha = config["lora_alpha"]
    scaling = alpha / lora_rank

    safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.exists(safetensors_path):
        safetensors_path = os.path.join(lora_dir, "adapter_model.bin")
        adapter_sd = torch.load(safetensors_path, map_location=device)
    else:
        adapter_sd = load_file(safetensors_path)

    base_keys = [k for k in base_model.state_dict().keys()]

    lora_A_keys = [k for k in adapter_sd.keys() if "lora_A" in k]
    
    num_dirs = lora_rank + 1
    deltas = {i: {} for i in range(num_dirs)}
    rank1_factors = {i: {} for i in range(num_dirs)}
    
    verified_first = False

    for a_key in tqdm(lora_A_keys):
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

            
        dir_matrices = []

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
            
            dir_matrices.append(W_comp.view(-1))
                
    return deltas, lora_rank, config, rank1_factors

def apply_perturbed_weights(model, cpu_state_dict, cpu_fingerprint, delta, scale):
    """"""
    ref_key, ref_value = cpu_fingerprint
    def _assert_cpu_intact():
        assert cpu_state_dict[ref_key].device.type == 'cpu', \
            f"CPU reference tensor '{ref_key}' is no longer on CPU!"
        actual = cpu_state_dict[ref_key].flatten()[0].item()
        assert actual == ref_value, \
            f"CPU reference '{ref_key}' was modified! Expected {ref_value}, got {actual}"
    _assert_cpu_intact()

    sd = model.state_dict()
    with torch.no_grad():
        for k, v in cpu_state_dict.items():
            if k in sd:
                sd[k].copy_(v)
        if scale != 0.0:
            for k in delta:
                if k in sd:
                    sd[k].add_(scale * delta[k].to(sd[k].device))

    _assert_cpu_intact()


def restore_clean_weights(model, cpu_state_dict):
    """"""
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in cpu_state_dict.items():
            if k in sd:
                sd[k].copy_(v)

def find_minimum(model, tokenizer, samples, delta, base_loss, cpu_state_dict, cpu_fingerprint, tol=0.1, evaluated_points=None, start_scale=1.0, save_fn=None, batch_size=4, eval_cache=None, max_length=2048):
    import scipy.optimize

    if evaluated_points is None:
        evaluated_points = []
    if eval_cache is None:
        eval_cache = {}

    def get_loss(s):
        s = float(s)
        for cached_s in list(eval_cache.keys()):
            if abs(cached_s - s) < 1e-6:
                evaluated_points.append((s, eval_cache[cached_s]))
                return eval_cache[cached_s]
                
        if s == 0.0:
            loss = base_loss
        else:
            apply_perturbed_weights(model, cpu_state_dict, cpu_fingerprint, delta, s)
            loss = compute_loss_on_samples(model, tokenizer, samples, max_length=max_length, batch_size=batch_size, device=model.device)
            if save_fn is not None:
                save_fn(model, s, loss)
                
        eval_cache[s] = loss
        evaluated_points.append((s, loss))
        print(f"[{model.device}] Evaluated scale {s:.4f}, loss {loss:.4f}")
        return loss

    print(f"[{model.device}] Finding minimum (starting bounds 0.0, {start_scale * 2.0})...")
    res = scipy.optimize.minimize_scalar(
        get_loss,
        bracket=(0.0, start_scale * 2.0),
        method='brent',
        options={'xtol': tol}
    )
    final_scale = float(res.x)
    final_loss = float(res.fun)
    return final_scale, final_loss

def find_ix_loss(model, tokenizer, samples, delta, target_loss, base_loss, ix, cpu_state_dict, cpu_fingerprint, ytol=0.3, evaluated_points=None, guess1=0.0, guess2=-2.0, save_fn=None, batch_size=4, eval_cache=None, target_type='over', optimizer='parabolic', limit_to_bucket=True, max_length=2048):
    if evaluated_points is None:
        evaluated_points = []
    if eval_cache is None:
        eval_cache = {}

    def get_loss(s):
        s = float(s)
        for cached_s in list(eval_cache.keys()):
            if abs(cached_s - s) < 1e-6:
                cached_loss = eval_cache[cached_s]
                evaluated_points.append((s, cached_loss))
                return cached_loss
                
        if s == 0.0:
            loss = base_loss
        else:
            apply_perturbed_weights(model, cpu_state_dict, cpu_fingerprint, delta, s)
            loss = compute_loss_on_samples(model, tokenizer, samples, max_length=max_length, batch_size=batch_size, device=model.device)
            if save_fn is not None:
                save_fn(model, s, loss)
                
        eval_cache[s] = loss
        evaluated_points.append((s, loss))
        print(f"[{model.device}] Evaluated scale {s:.4f}, loss {loss:.4f}")
        return loss

    print(f"[{model.device}] {ix}x Loss Target ({target_loss:.4f}) using Cache-Driven {optimizer.capitalize()} (search: {target_type})")

    if 0.0 not in eval_cache:
        eval_cache[0.0] = base_loss

    s_limit = float('inf')
    s_best = None
    maxiter = 100
    for iteration in range(maxiter):
        current_cache = dict(eval_cache)
        
        valid_points = []
        for s, l in current_cache.items():
            if target_type == 'over' and s >= 0.0:
                valid_points.append((s, l))
            elif target_type == 'opposite' and s <= 0.0:
                valid_points.append((s, l))
                
        valid_points.sort(key=lambda x: abs(x[0]))
        if s_limit != float('inf'):
            valid_points = [p for p in valid_points if abs(p[0]) <= s_limit]

        if valid_points:
            m = min(range(len(valid_points)), key=lambda i: valid_points[i][1])
            violation_idx = None
            violation_msg = ""
            for i in range(1, m + 1):
                if valid_points[i][1] > valid_points[i-1][1] + 1e-4:
                    violation_idx = i
                    violation_msg = f"Before minimum at scale {valid_points[m][0]:.4f}, loss increased from {valid_points[i-1][1]:.4f} (at scale {valid_points[i-1][0]:.4f}) to {valid_points[i][1]:.4f} (at scale {valid_points[i][0]:.4f})"
                    break
            if violation_idx is None:
                for i in range(m + 1, len(valid_points)):
                    if valid_points[i][1] < valid_points[i-1][1] - 1e-4:
                        violation_idx = i
                        violation_msg = f"After minimum at scale {valid_points[m][0]:.4f}, loss decreased from {valid_points[i-1][1]:.4f} (at scale {valid_points[i-1][0]:.4f}) to {valid_points[i][1]:.4f} (at scale {valid_points[i][0]:.4f})"
                        break
            
            if violation_idx is not None:
                if limit_to_bucket:
                    s_limit = abs(valid_points[violation_idx-1][0])
                    print(f"[{model.device}] U-Shape violation detected: {violation_msg}. Limiting search to first U-region (scale 0 to {s_limit:.4f}).")
                    valid_points = [p for p in valid_points if abs(p[0]) <= s_limit]
                else:
                    raise ValueError(f"Function is not U-shaped! {violation_msg}")
                
        s_under = None
        s_over = None
        
        pts_under = [p for p in valid_points if p[1] <= target_loss]
        if pts_under:
            s_under = max(pts_under, key=lambda x: abs(x[0]))[0]
            
        pts_over = [p for p in valid_points if p[1] > target_loss]
        if pts_over:
            s_over = min(pts_over, key=lambda x: abs(x[0]))[0]
            
        if s_over is not None and s_under is not None:
            l_over = eval_cache[s_over]
            l_under = eval_cache[s_under]
            
            abs_ytol = base_loss * ytol
            
            if abs(l_over - target_loss) <= abs_ytol and abs(s_over) > 1e-4:
                s_best = s_over
                break
            if abs(l_under - target_loss) <= abs_ytol and abs(s_under) > 1e-4:
                s_best = s_under
                break
            if abs(s_over - s_under) < 1e-4:
                s_best = s_over
                break
                
            s_new = None
            if optimizer == 'parabolic' and len(valid_points) >= 3:
                sorted_by_dist = sorted(valid_points, key=lambda x: abs(x[1] - target_loss))
                s1, L1 = sorted_by_dist[0]
                s2, L2 = sorted_by_dist[1]
                s3, L3 = sorted_by_dist[2]
                
                D = (s1 - s2) * (s1 - s3) * (s2 - s3)
                if abs(D) > 1e-12:
                    a = (L1 * (s2 - s3) + L2 * (s3 - s1) + L3 * (s1 - s2)) / D
                    b = (L1 * (s3**2 - s2**2) + L2 * (s1**2 - s3**2) + L3 * (s2**2 - s1**2)) / D
                    c = L1 - a * s1**2 - b * s1
                    
                    if a > 1e-6:
                        discriminant = b**2 - 4 * a * (c - target_loss)
                        if discriminant >= 0:
                            r1 = (-b - math.sqrt(discriminant)) / (2 * a)
                            r2 = (-b + math.sqrt(discriminant)) / (2 * a)
                            if target_type == 'over':
                                proposed_s = max(r1, r2)
                            else:
                                proposed_s = min(r1, r2)
                                
                            min_s, max_s = min(s_under, s_over), max(s_under, s_over)
                            margin = abs(s_over - s_under) * 5.0
                            if (target_type == 'over' and 0 <= proposed_s < max_s + margin) or \
                               (target_type == 'opposite' and min_s - margin < proposed_s <= 0):
                                s_new = proposed_s

            if s_new is None:
                s_new = s_over - (l_over - target_loss) * (s_over - s_under) / (l_over - l_under)
                
                min_s, max_s = min(s_under, s_over), max(s_under, s_over)
                margin = abs(s_over - s_under) * 0.05
                if not (min_s + margin < s_new < max_s - margin):
                    s_new = (s_under + s_over) / 2.0
            
            if target_type == 'over':
                s_new = max(0.0, min(s_new, s_limit))
            else:
                s_new = min(0.0, max(s_new, -s_limit))
                
            _ = get_loss(s_new)
            
        elif s_under is not None and s_over is None:
            if s_under == 0.0:
                s_new = guess2
            else:
                s_new = s_under * 2.0
            
            if target_type == 'over':
                s_new = min(s_new, s_limit)
            else:
                s_new = max(s_new, -s_limit)

            if abs(s_new - s_under) < 1e-6:
                limit_name = "u-shape cap" if s_limit != float('inf') else "bucket bound"
                raise ValueError(f"Target loss {target_loss:.4f} not found within {limit_name} {abs(s_new):.2f}")
                
            _ = get_loss(s_new)
            
        else:
            break
            
    if s_best is None:
        if s_over is not None: s_best = s_over
        else: s_best = s_under if s_under is not None else 0.0
        
    return s_best


from typing import List, Tuple

def compute_second_derivatives(scales: List[float], losses: List[float]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    if len(scales) < 3:
        return out
    for i in range(1, len(scales) - 1):
        s_prev, s_cur, s_next = scales[i - 1], scales[i], scales[i + 1]
        l_prev, l_cur, l_next = losses[i - 1], losses[i], losses[i + 1]
        d1_left = (l_cur - l_prev) / (s_cur - s_prev)
        d1_right = (l_next - l_cur) / (s_next - s_cur)
        denom = (s_next - s_prev) / 2.0
        if denom == 0: continue
        d2 = (d1_right - d1_left) / denom
        out.append((i, float(d2)))
    return out

def choose_inflection_index(scales: List[float], losses: List[float], tol: float = 1.0) -> int:
    d2_vals = compute_second_derivatives(scales, losses)
    if not d2_vals: return max(0, min(len(scales) - 1, len(scales) // 2))
    for idx, d2 in d2_vals:
        if abs(d2) <= tol: return idx
    for (idx_a, d2_a), (idx_b, d2_b) in zip(d2_vals[:-1], d2_vals[1:]):
        if d2_a == 0: return idx_a
        if d2_b == 0: return idx_b
        if d2_a * d2_b < 0: return idx_a if abs(d2_a) <= abs(d2_b) else idx_b
    best_idx, _ = min(d2_vals, key=lambda item: abs(item[1]))
    return best_idx

def save_perturbed_model(model, delta, scale, output_dir, cpu_state_dict, cpu_fingerprint):
    print(f"Saving perturbed model at scale {scale:.4f} to {output_dir}...")
    apply_perturbed_weights(model, cpu_state_dict, cpu_fingerprint, delta, scale)
    model.save_pretrained(output_dir, safe_serialization=True)
    restore_clean_weights(model, cpu_state_dict)

def worker_process(device_id, task_queue, result_queue, global_cache, args, samples):
    """"""
    device = f"cuda:{device_id}"
    print(f"\nWorker [{device}] starting and loading model...")
    model, tokenizer, cpu_state_dict, cpu_fingerprint = load_model_and_tokenizer(args.base_model, device=device)

    if args.lora_model == "random":
        print(f"Worker [{device}] computing random directions...")
        deltas = {}
        lora_rank = 3
        target_keys = [k for k, v in model.state_dict().items() 
                       if k.endswith('.weight') and v.ndim == 2 and not any(x in k for x in ['embed_tokens', 'lm_head', 'norm'])]
        for i in range(lora_rank + 1):
            deltas[i] = {}
            
        torch.manual_seed(42)
        for k in tqdm(target_keys, desc=f"Worker [{device}] Random Dirs"):
            v = model.state_dict()[k]
            noise = torch.randn((lora_rank + 1,) + v.shape, dtype=torch.float32, device=device)
            
            norms = torch.linalg.matrix_norm(noise, ord='fro', dim=(1, 2), keepdim=True)
            noise = (noise / norms).to(v.dtype)
            
            for i in range(lora_rank + 1):
                deltas[i][k] = noise[i].cpu()
        print(f"Worker [{device}] random directions computed (rank={lora_rank}).")
    else:
        print(f"Worker [{device}] computing LoRA directions...")
        deltas, lora_rank, adapter_template, rank1_factors = decompose_lora(args.lora_model, model,device=device)
        print(f"Worker [{device}] directions computed (rank={lora_rank}).")

    print(f"Worker [{device}] ready for tasks.")

    while True:
        task = task_queue.get()
        if task is None:
            break

        dir_idx = task['dir_idx']
        target_type = task['target_type']
        base_loss = task['base_loss']
        delta = deltas[dir_idx]

        dir_name = "original" if (dir_idx == lora_rank and args.lora_model != "random") else f"dir{dir_idx}"
        if args.lora_model == "random":
            scaled_tol = args.ytol
            start_scale = 1.0
        else:
            scaled_tol = args.ytol if dir_name == "original" else args.ytol * (2.0 ** dir_idx)
            start_scale = 1.0 if dir_name == "original" else float(2 ** dir_idx)
        eval_cache = global_cache[dir_idx]
        print(f"\nWorker [{device}] starting direction: {dir_name} ({target_type}, start_scale={start_scale}, ytol={args.ytol}, scaled_xtol={scaled_tol})")

        evaluated_points = []

        def make_save_fn():
            def save_fn(model, s, loss):
                return None
            return save_fn

        def get_loss(s):
            s = float(s)
            for cached_s in list(eval_cache.keys()):
                if abs(cached_s - s) < 1e-6:
                    return eval_cache[cached_s]
            
            if abs(s) < 1e-6:
                loss = base_loss
            else:
                apply_perturbed_weights(model, cpu_state_dict, cpu_fingerprint, delta, s)
                loss = compute_loss_on_samples(
                    model, tokenizer, samples, 
                    max_length=args.max_length, 
                    batch_size=args.batch_size_ix, 
                    device=model.device
                )
            
            eval_cache[s] = loss
            print(f"[{model.device}] Evaluated scale {s:.4f}, loss {loss:.4f}")
            return loss

        if target_type == 'base':
            final_scale = 0.0
            final_loss = base_loss
            print(f"[{dir_name}] Base Loss ({final_loss:.4f}) at scale {final_scale:.4f}")
            out_path = os.path.join(args.output_dir, f"{dir_name}_base")
        elif target_type == 'in':
            final_scale, final_loss = find_minimum(model, tokenizer, samples, delta, base_loss, cpu_state_dict, cpu_fingerprint, tol=scaled_tol, evaluated_points=evaluated_points, start_scale=start_scale, save_fn=make_save_fn(), batch_size=args.batch_size_ix, eval_cache=eval_cache, max_length=args.max_length)
            print(f"[{dir_name}] Minimum Loss {final_loss:.4f} found at scale {final_scale:.4f}")
            out_path = os.path.join(args.output_dir, f"{dir_name}_in")

        elif target_type == 'scan':
            max_scale = task['max_scale']
            max_evals = task.get('max_evals', 15)
            
            if 0 <= dir_idx <= 3 and dir_name != "original":
                s = float(2 ** (dir_idx + 1))
            else:
                s = start_scale
                
            scan_results = [(0.0, base_loss)]
            eval_count = 0
            bracket = None
            bracket_losses = None
            
            while s <= 2 * max_scale and eval_count < max_evals:
                loss = get_loss(s)
                scan_results.append((s, loss))
                eval_count += 1
                
                if len(scan_results) >= 4:
                    (s0, l0), (s1, l1), (s2, l2), (s3, l3) = scan_results[-4:]
                    
                    d1_l1 = (l1 - l0) / (s1 - s0) if (s1 - s0) != 0 else 0
                    d1_r1 = (l2 - l1) / (s2 - s1) if (s2 - s1) != 0 else 0
                    d2_1  = (d1_r1 - d1_l1) / ((s2 - s0)/2) if (s2 - s0) != 0 else 0
                    
                    d1_l2 = (l2 - l1) / (s2 - s1) if (s2 - s1) != 0 else 0
                    d1_r2 = (l3 - l2) / (s3 - s2) if (s3 - s2) != 0 else 0
                    d2_2  = (d1_r2 - d1_l2) / ((s3 - s1)/2) if (s3 - s1) != 0 else 0
                    
                    if d2_1 > 0 and d2_2 <= 0:
                        bracket = (s1, s2)
                        bracket_losses = (l1, l2)
                        break
                s *= 2.0
                
            if bracket is None:
                if len(scan_results) >= 2:
                    bracket = (scan_results[-2][0], scan_results[-1][0])
                    bracket_losses = (scan_results[-2][1], scan_results[-1][1])
                else:
                    bracket = (0.0, scan_results[-1][0])
                    bracket_losses = (base_loss, scan_results[-1][1])
                    
            abs_ytol = base_loss * args.ytol
            while bracket and abs(bracket_losses[1] - bracket_losses[0]) > abs_ytol and eval_count < max_evals:
                mid = (bracket[0] + bracket[1]) / 2.0
                loss_mid = get_loss(mid)
                scan_results.append((mid, loss_mid))
                eval_count += 1
                
                sorted_pts = sorted(scan_results, key=lambda x: x[0])
                idx = next((i for i, (sv, _) in enumerate(sorted_pts) if abs(sv - mid) < 1e-6), None)
                
                if idx is not None and 1 <= idx <= len(sorted_pts) - 2:
                    (sa, la), (sb, lb), (sc, lc) = sorted_pts[idx-1:idx+2]
                    d1_l = (lb - la) / (sb - sa) if (sb - sa) != 0 else 0
                    d1_r = (lc - lb) / (sc - sb) if (sc - sb) != 0 else 0
                    d2_mid = (d1_r - d1_l) / ((sc - sa)/2) if (sc - sa) != 0 else 0
                    
                    if d2_mid > 0:
                        bracket = (mid, bracket[1])
                        bracket_losses = (loss_mid, bracket_losses[1])
                    else:
                        bracket = (bracket[0], mid)
                        bracket_losses = (bracket_losses[0], loss_mid)
                else:
                    break
                    
            final_scale = (bracket[0] + bracket[1]) / 2.0 if bracket else scan_results[-1][0]
            final_loss = get_loss(final_scale)
            
            result_queue.put({
                "dir_idx": dir_idx,
                "direction": dir_name,
                "target_type": target_type,
                "final_scale": final_scale,
                "final_loss": final_loss,
                "scales": [x[0] for x in scan_results],
                "losses": [x[1] for x in scan_results]
            })
            continue
            
        elif target_type == 'opposite':

            task_ix = task['ix']
            target_max_loss = base_loss * task_ix
            guess1 = 0.0
            guess2 = -start_scale * 2.0
            ix_tag = f"{task_ix:g}x"

            opt_to_use = args.optimizer
            if opt_to_use == 'hybrid':
                opt_to_use = 'secant' if task_ix > 3.0 else 'parabolic'

            try:
                final_scale = find_ix_loss(model, tokenizer, samples, delta, target_max_loss, base_loss, task_ix, cpu_state_dict, cpu_fingerprint, ytol=args.ytol, evaluated_points=evaluated_points, guess1=guess1, guess2=guess2, save_fn=make_save_fn(), batch_size=args.batch_size_ix, eval_cache=eval_cache, target_type='opposite', optimizer=opt_to_use, limit_to_bucket=args.limit_to_bucket, max_length=args.max_length)
                final_loss = target_max_loss
                print(f"[{dir_name}] {task_ix}x Loss ({target_max_loss:.4f}) found at scale {final_scale:.4f}")
            except ValueError as e:
                print(f"[{dir_name}] U-Shape Error: {e}. Skipping {ix_tag} target.")
                final_scale = float('nan')
                final_loss = float('nan')
            out_path = os.path.join(args.output_dir, f"{dir_name}_opposite_{ix_tag}")
        elif target_type == 'over':
            task_ix = task['ix']
            target_max_loss = base_loss * task_ix
            if args.lora_model == "random":
                max_scale_bucket = 25.0
            else:
                max_scale_bucket = 25.0 if dir_name == "original" else 25.0 * (3.0 ** dir_idx)
            guess1 = 0.0
            guess2 = max_scale_bucket
            ix_tag = f"{task_ix:g}x"

            opt_to_use = args.optimizer
            if opt_to_use == 'hybrid':
                opt_to_use = 'secant' if task_ix > 3.0 else 'parabolic'

            try:
                final_scale = find_ix_loss(model, tokenizer, samples, delta, target_max_loss, base_loss, task_ix, cpu_state_dict, cpu_fingerprint, ytol=args.ytol, evaluated_points=evaluated_points, guess1=guess1, guess2=guess2, save_fn=make_save_fn(), batch_size=args.batch_size_ix, eval_cache=eval_cache, target_type='over', optimizer=opt_to_use, limit_to_bucket=args.limit_to_bucket, max_length=args.max_length)
                final_loss = target_max_loss
                print(f"[{dir_name}] {task_ix}x Loss ({target_max_loss:.4f}) found at scale {final_scale:.4f}")
            except ValueError as e:
                print(f"[{dir_name}] U-Shape Error: {e}. Skipping {ix_tag} target.")
                final_scale = float('nan')
                final_loss = float('nan')
            out_path = os.path.join(args.output_dir, f"{dir_name}_over_{ix_tag}")

        evaluated_points.append((0.0, base_loss))
        evaluated_points.sort(key=lambda x: x[0])
        
        scales = [x[0] for x in evaluated_points]
        losses = [x[1] for x in evaluated_points]
        
        if not (target_type == 'in' and args.skip_save_in):
            is_completed = False
            if os.path.exists(out_path):
                has_model = _has_saved_checkpoint(out_path, args.save_as_adapter)
                if abs(final_scale) < 1e-6 or not args.save_ckpt:
                    has_model = True
                    
                has_eval = True
                if args.source_eval == "ix":
                    has_eval = os.path.exists(os.path.join(out_path, "eval_losses.txt"))
                if has_model and has_eval:
                    scale_txt_path = os.path.join(out_path, "scale.txt")
                    if os.path.exists(scale_txt_path):
                        with open(scale_txt_path, "r") as f:
                            try:
                                saved_scale = float(f.read().strip())
                                if abs(saved_scale - final_scale) < 1e-4:
                                    is_completed = True
                                else:
                                    print(f"  [{dir_name}] Scale mismatch (saved:{saved_scale:.4f}, new:{final_scale:.4f}). Overwriting.")
                            except:
                                print(f"  [{dir_name}] Could not parse scale.txt. Overwriting.")
                    else:
                        print(f"  [{dir_name}] No scale.txt found. Overwriting for exact verify.")

            if is_completed:
                print(f"[{dir_name}] Target '{target_type}' ({final_scale:.4f}) already saved in {out_path}. Skipping save and eval.")
            else:
                if abs(final_scale) >= 1e-6 and args.save_ckpt:
                    os.makedirs(out_path, exist_ok=True)
                    _clear_checkpoint_artifacts(out_path, args.save_as_adapter)
                    if args.save_as_adapter:
                        _save_rank1_adapter(out_path, adapter_template, rank1_factors[dir_idx], final_scale, tokenizer)
                    else:
                        save_perturbed_model(model, delta, final_scale, out_path, cpu_state_dict, cpu_fingerprint)
                        tokenizer.save_pretrained(out_path)
                    with open(os.path.join(out_path, "scale.txt"), "w") as f:
                        f.write(f"{final_scale:.8f}\n")
                else:
                    reason = "scale is 0.0" if abs(final_scale) < 1e-6 else "--save_ckpt not specified"
                    print(f"  [{dir_name}] Skipping model and tokenizer save ({reason})")
        
        result_queue.put({
            "dir_idx": dir_idx,
            "direction": dir_name,
            "target_type": target_type,
            "final_scale": final_scale,
            "final_loss": final_loss,
            "scales": scales,
            "losses": losses
        })
            
    print(f"Worker [{device}] exiting.")
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    gc.collect()

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    processed_argv = []
    for arg in sys.argv[1:]:
        match = re.match(r'^--(\d*\.?\d+)x$', arg)
        if match:
            processed_argv.extend(['--ix', match.group(1)])
        else:
            processed_argv.append(arg)

    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, choices=['secant', 'parabolic', 'hybrid'], default='hybrid',
                        help='Optimizer for finding target loss checkpoint (default: parabolic)')
    parser.add_argument('--lora_model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=os.environ.get(
        "S2L_DATASET",
        str(Path(os.environ.get("S2L_ROOT", Path(__file__).resolve().parents[3])) / "dataset" / "m23k-prd-sharegpt" / "train.jsonl"),
    ))
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum tokens per sample for loss computation (default: 2048)')
    parser.add_argument('--opposite', action='store_true')
    parser.add_argument('--skip-save-in', action='store_true', dest='skip_save_in',
                        help='Skip saving the _in (minimum-loss) checkpoints')
    parser.add_argument('--in', action='store_true', dest='run_in',
                        help='Only run the -in jobs if this flag is specified')
    parser.add_argument('--ix', type=str, action='append', dest='ix_values',
                        metavar='VAL', help="Loss multiplier or metric; can be repeated (e.g. --2x --ix infl --ix 1.5)")
    parser.add_argument('--scan-points-max', type=int, default=15,
                        help='Max evaluations per direction during inflection scan (default: 15)')
    parser.add_argument('--source-eval', type=str, choices=['none'], default='none',
                        help='Disabled; use --source-eval none.')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip perturbation search and only plot the source-wise trends from existing eval_losses.txt files.')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='Save the perturbed model checkpoints (safetensors). If not specified, only eval is performed.')
    parser.add_argument('--save-as-adapter', '--save-as-adaptor', action='store_true', dest='save_as_adapter',
                        help='Save perturbed checkpoints as LoRA adapter directories instead of merged full models.')
    parser.add_argument('--skip-original-direction', action='store_true',
                        help='Skip the original combined LoRA direction and run only dir0..dir{r-1}.')
    parser.add_argument('--ytol', type=float, default=0.2)
    parser.add_argument('--gpu_threshold', type=float, default=10.0,
                        help='Memory utilization threshold (%%) for GPU availability. '
                             'GPUs with utilization above this value are excluded. Default: 10.0')
    parser.add_argument('--batch_size_ix', type=int, default=5,
                        help='Batch size for loss calculation when finding ix (default: 10)')
    parser.add_argument('--batch_size_eval', type=int, default=10,
                        help='Batch size for loss calculation when saving checkpoints (default: 10)')
    parser.add_argument('--no-limit-to-bucket', action='store_false', dest='limit_to_bucket', default=True,
                        help='Allow search to expand beyond the initial bucket bound (default: False/Limit active)')
    args = parser.parse_args(processed_argv)

    if args.plot_only:
        if not os.path.exists(args.output_dir):
            raise ValueError(f"Output directory does not exist for plot-only: {args.output_dir}")
        print("\n[plot-only] Aggregating existing eval_losses.txt to plot source-wise trends...")
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "process_loss"))
            import plot_txt_losses
            
            checkpoint_glob = os.path.join(args.output_dir, "*")
            selected_ckpts = plot_txt_losses.select_checkpoints(checkpoint_glob, 0.0)
            
            if selected_ckpts:
                csv_path = Path(args.output_dir) / "direction_source_mean_losses.csv"
                plot_path = Path(args.output_dir) / "direction_source_mean_losses.png"
                
                rows = plot_txt_losses.load_means_from_txt(selected_ckpts, "eval_losses.txt", Path(args.output_dir))
                plot_txt_losses.save_csv(rows, csv_path)
                plot_txt_losses.plot_lines(rows, plot_path)
                
                print(f"Saved evaluation trends to {plot_path}")
            else:
                print("No checkpoints matched for plotting.")
        except Exception as e:
            print(f"Skipping two-source evaluation plot due to error: {e}")
        return

    if args.ix_values is None:
        args.ix_values = ["3.0"]

    if args.ix_values is not None:
        for v in args.ix_values:
            if v not in ["infl", "half-infl"] and float(v) < 1.0:
                raise ValueError(f"--ix values must be >= 1.0, got {v}")

    if args.lora_model == "random":
        args.base_model = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Hardcoding base model to: {args.base_model}")
    else:
        adapter_config_path = os.path.join(args.lora_model, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            config = json.load(f)
        args.base_model = config.get("base_model_name_or_path")
        if not args.base_model:
            raise ValueError("base_model could not be found in adapter_config.json")
        print(f"Auto-detected base model: {args.base_model}")

    if args.save_as_adapter and args.lora_model == "random":
        raise ValueError("--save-as-adapter is not supported when lora_model=random")

    os.makedirs(args.output_dir, exist_ok=True)

    available_gpus = get_available_gpus(threshold=args.gpu_threshold)
    if not available_gpus:
        raise RuntimeError("No GPUs available! All GPUs have high memory utilization.")
    
    master_gpu = available_gpus[0]
    print(f"\nExecuting Master Node Baseline Calculations on GPU {master_gpu}...")
    model, tokenizer, _, _ = load_model_and_tokenizer(args.base_model, device=f"cuda:{master_gpu}")

    print(f"\nSampling datasets in main process...")
    samples = sample_dataset(args.dataset, n_samples=args.max_samples)
    print(f"Main process: sampled {len(samples)} training samples")

    prior_logs = None
    search_logs_path = os.path.join(args.output_dir, "search_logs.json")
    if os.path.exists(search_logs_path):
        print(f"\nFound prior search logs at {search_logs_path}. Loading...")
        try:
            with open(search_logs_path, "r") as f:
                prior_logs = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load prior search logs: {e}")

    base_loss = None
    if prior_logs:
        for dir_name, data in prior_logs.items():
            scales = data.get("scales", [])
            losses = data.get("losses", [])
            for s, l in zip(scales, losses):
                if abs(float(s)) < 1e-6:
                    base_loss = float(l)
                    break
            if base_loss is not None:
                break
                
    if base_loss is not None:
        print(f"Loaded Base Loss from prior logs: {base_loss:.4f}")
    else:
        print("Computing baseline loss...")
        base_loss = compute_loss_on_samples(model, tokenizer, samples, max_length=args.max_length, device=f"cuda:{master_gpu}")
        print(f"Base Loss: {base_loss:.4f}")

    if args.lora_model == "random":
        lora_rank = 3
        print(f"LoRA rank for random: {lora_rank}")
    else:
        adapter_config_path = os.path.join(args.lora_model, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            config = json.load(f)
        lora_rank = config["r"]
        print(f"LoRA rank from config: {lora_rank}")

    if args.skip_original_direction and args.lora_model != "random":
        direction_ids = list(range(lora_rank))
        print(f"Skipping original direction; running directions: {[f'dir{i}' for i in direction_ids]}")
    else:
        direction_ids = list(range(lora_rank + 1))

    
    print(f"Clearing Master Node GPU {master_gpu} Memory before spawning workers...")
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    gc.collect()

    results = []
    plot_data = {i: {"scales": [], "losses": []} for i in direction_ids}

    print(f"\nStarting {len(available_gpus)} persistent workers on GPUs: {available_gpus}...")

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    manager = mp.Manager()
    global_cache = manager.dict()
    for i in direction_ids:
        global_cache[i] = manager.dict()
        
    if prior_logs:
        print("Populating cache with prior search evaluations...")
        loaded_points = 0
        for i in direction_ids:
            direction_name = "original" if (i == lora_rank and args.lora_model != "random") else f"dir{i}"
            if direction_name in prior_logs:
                prior_scales = prior_logs[direction_name].get("scales", [])
                prior_losses = prior_logs[direction_name].get("losses", [])
                for s, l in zip(prior_scales, prior_losses):
                    global_cache[i][float(s)] = float(l)
                    loaded_points += 1
        print(f"Successfully loaded {loaded_points} points into cache.")

    workers = []
    for gpu_id in available_gpus:
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, result_queue, global_cache, args, samples))
        p.start()
        workers.append(p)
        
    worker_results = []
    needs_inflection = args.ix_values is not None and ("infl" in args.ix_values or "half-infl" in args.ix_values)
    if needs_inflection:
        phase1_count = len(direction_ids) * 2
        if args.run_in: phase1_count += len(direction_ids)
        
        for dir_idx in direction_ids:
            task_queue.put({"dir_idx": dir_idx, "target_type": "base", "base_loss": base_loss})
            if args.run_in:
                task_queue.put({"dir_idx": dir_idx, "target_type": "in", "base_loss": base_loss})
                
        for dir_idx in direction_ids:
            max_bucket = 25.0 if (dir_idx == lora_rank and args.lora_model != "random") else 25.0 * (3.0 ** dir_idx)
            if args.lora_model == "random": max_bucket = 25.0
                
            task_queue.put({
                "dir_idx": dir_idx, "target_type": "scan",
                "base_loss": base_loss, "max_scale": max_bucket,
                "max_evals": args.scan_points_max,
            })
            
        print(f"Phase 1: Computing base and inflections across {len(direction_ids)} directions...")
        phase1_results = [result_queue.get() for _ in range(phase1_count)]
        worker_results.extend(phase1_results)
        
        derived_ix = {}
        for res in phase1_results:
            if res["target_type"] == "scan":
                dir_idx = res["dir_idx"]
                l = res["final_loss"]
                l0 = base_loss
                
                if l < l0:
                     print(f"Warning: derived inflection loss {l:.4f} is lower than base {l0:.4f}. Clamping to 1.01x.")
                     ix_upper = 1.01
                     ix_lower = 1.01
                else:
                    ix_upper = max(l / l0, 1.01)
                    ix_lower = max((l0 + l) / (2.0 * l0), 1.01)
                
                derived_ix[dir_idx] = (ix_lower, ix_upper)
                print(f"dir{dir_idx} inflection derived: l={l:.4f}, l0={l0:.4f} -> lower={ix_lower:.3f}x, upper={ix_upper:.3f}x")
            
        phase2_count = 0
        for dir_idx in direction_ids:
            targets_for_this_dir = []
            for ix_str in args.ix_values:
                if ix_str == "infl":
                    targets_for_this_dir.append(derived_ix[dir_idx][1])
                elif ix_str == "half-infl":
                    targets_for_this_dir.append(derived_ix[dir_idx][0])
                else:
                    targets_for_this_dir.append(float(ix_str))
                    
            targets_for_this_dir = sorted(list(set(targets_for_this_dir)), reverse=True)
            for ix_val in targets_for_this_dir:
                task_queue.put({
                    "dir_idx": dir_idx, "target_type": "over",
                    "base_loss": base_loss, "ix": ix_val,
                })
                phase2_count += 1
                
        print(f"Phase 2: Computing checkpoints for {phase2_count} target ix values...")
        phase2_results = [result_queue.get() for _ in range(phase2_count)]
        worker_results.extend(phase2_results)
        num_tasks = phase1_count + phase2_count

    else:
        parsed_ix_values = sorted([float(x) for x in set(args.ix_values)], reverse=True)
        num_ix = len(parsed_ix_values)
        num_tasks = len(direction_ids) * (1 + (1 if args.run_in else 0) + num_ix)

        for dir_idx in direction_ids:
            task_queue.put({
                "dir_idx": dir_idx,
                "target_type": 'base',
                "base_loss": base_loss,
            })

        if args.run_in:
            for dir_idx in direction_ids:
                task_queue.put({
                    "dir_idx": dir_idx,
                    "target_type": 'in',
                    "base_loss": base_loss,
                })

        if args.opposite:
            for _ix in parsed_ix_values:
                for dir_idx in direction_ids:
                    task_queue.put({
                        "dir_idx": dir_idx,
                        "target_type": 'opposite',
                        "base_loss": base_loss,
                        "ix": _ix,
                    })
        else:
            for _ix in parsed_ix_values:
                for dir_idx in direction_ids:
                    task_queue.put({
                        "dir_idx": dir_idx,
                        "target_type": 'over',
                        "base_loss": base_loss,
                        "ix": _ix,
                    })
        print(f"Total tasks to be processed: {num_tasks} (ix values: {args.ix_values})")
                
        for _ in range(num_tasks):
            worker_results.append(result_queue.get())
        
    for _ in range(len(available_gpus)):
        task_queue.put(None)
        
    for p in workers:
        p.join()
        
    dir_results = {i: {"direction": "original" if (i == lora_rank and args.lora_model != "random") else f"dir{i}"} for i in direction_ids}
    dir_points = {i: [] for i in direction_ids}
    
    for res in worker_results:
        idx = res["dir_idx"]
        target_type = res["target_type"]
        
        for s, l in zip(res["scales"], res["losses"]):
            if tuple([s, l]) not in dir_points[idx]:
                dir_points[idx].append(tuple([s, l]))
                
        if target_type == 'in':
            dir_results[idx]["min_scale"] = res["final_scale"]
            dir_results[idx]["min_loss"] = res["final_loss"]
        elif target_type in ['over', 'opposite']:
            dir_results[idx]["max_scale"] = res["final_scale"]
            dir_results[idx]["max_loss_target"] = res["final_loss"]
        elif target_type == 'base':
            dir_results[idx]["base_scale"] = res["final_scale"]
            dir_results[idx]["base_loss"] = res["final_loss"]
            
    for idx in direction_ids:
        cache_dict = dict(global_cache[idx])
        for p in dir_points[idx]:
            if p[0] not in cache_dict:
                cache_dict[p[0]] = p[1]
                
        pts = sorted([(k, v) for k, v in cache_dict.items()], key=lambda x: x[0])
        plot_data[idx]["scales"] = [p[0] for p in pts]
        plot_data[idx]["losses"] = [p[1] for p in pts]
        results.append(dir_results[idx])

    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "search_results.csv"), index=False)

    search_logs = {}
    for idx in direction_ids:
        direction_name = "original" if (idx == lora_rank and args.lora_model != "random") else f"dir{idx}"
        search_logs[direction_name] = {
            "scales": plot_data[idx]["scales"],
            "losses": plot_data[idx]["losses"],
            "summary": dir_results[idx]
        }
    with open(os.path.join(args.output_dir, "search_logs.json"), "w") as f:
        json.dump(search_logs, f, indent=4)

    def plot_trajectories(filename, title_suffix, xscale='linear', xlim=None):
        plt.figure(figsize=(10, 6))
        for dir_idx in direction_ids:
            lbl = "original" if (dir_idx == lora_rank and args.lora_model != "random") else f"dir{dir_idx}"
            plt.plot(plot_data[dir_idx]["scales"], plot_data[dir_idx]["losses"], label=lbl, marker='o')
        plt.axvline(x=0, color='r', linestyle='--', label='Base Model')
        
        plt.xscale(xscale)
        if xlim is not None:
            plt.xlim(xlim)
            
        plt.xlabel(f"Perturbation Scale" + (f" ({xscale})" if xscale != 'linear' else ""))
        plt.ylabel("Loss")
        plt.title(f"Loss Trajectories for Lora Orthogonal Directions {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, filename))
        plt.close()

    plot_trajectories("loss_trajectories_original.png", "(Original Scale)")
    
    plot_trajectories("loss_trajectories_log.png", "(Log Scale)", xscale='symlog')
    
    dir1_xlim = None
    if 1 in plot_data and len(plot_data[1]["scales"]) > 0:
        dir1_xlim = (min(plot_data[1]["scales"]), max(plot_data[1]["scales"]))
    plot_trajectories("loss_trajectories_zoomed.png", "(Dir1 Range)", xlim=dir1_xlim)
    
    if args.source_eval != "none":
        raise ValueError("source evaluation is disabled; use --source-eval none")
    print("\nDone! Saved checkpoints and trajectory plots.")

if __name__ == "__main__":
    main()
