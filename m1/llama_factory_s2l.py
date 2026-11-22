#!/usr/bin/env python3
"""
LLaMA-Factory based S2L training script.
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def extract_name_from_npy_path(npy_path):
    """
    Extract meaningful name from .npy file path
    Similar logic to the original 23k_baseline.py
    """
    path_parts = Path(npy_path)
    basename = path_parts.stem  # filename without extension

    if basename == "labeled_idx":
        # If basename is labeled_idx.npy, use grandparent directory name
        # e.g., /res/most-learnable-in-5k/data/labeled_idx.npy -> most-learnable-in-5k
        grandparent_dir = path_parts.parent.parent.name
        return grandparent_dir
    else:
        # Otherwise, combine parent directory and basename
        # e.g., /res/most-learnable/prd-0chunk-2000char-rd.npy -> most-learnable-prd-0chunk-2000char-rd
        parent_dir = path_parts.parent.name
        return f"{parent_dir}-{basename}"


def create_subset_dataset(npy_path, source_jsonl, output_dir, name, dataset_type='m23k'):
    """
    Create a subset dataset by selecting specific indices from the source JSONL

    Args:
        npy_path: Path to .npy file containing indices (or None for full data)
        source_jsonl: Path to source train.jsonl file
        output_dir: Output directory for the subset
        name: Name for the dataset
        dataset_type: Type of dataset ('m23k' or 'openthoughts') for correct formatting
    """
    log(f"Creating subset dataset: {name}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load indices if provided, otherwise use all
    if npy_path:
        log(f"Loading indices from: {npy_path}")
        indices = np.load(npy_path)
        log(f"Loaded {len(indices)} indices")
        indices_set = set(indices.tolist())
    else:
        log("No subset file provided, using full dataset")
        indices_set = None

    # Read source JSONL and filter by indices
    log(f"Reading source data from: {source_jsonl}")
    output_jsonl = output_path / "train.jsonl"

    # Drop system turns for SFT: some teachers (e.g. MedGemma) keep a generation-only
    # system prompt in conversations; without system_tag (or with mismatched templates)
    # LLaMA-Factory can yield an empty train set (num_samples=0).
    selected_count = 0
    with open(source_jsonl, 'r') as f_in, open(output_jsonl, 'w') as f_out:
        for idx, line in enumerate(f_in):
            if indices_set is None or idx in indices_set:
                row = json.loads(line)
                conversations = row.get("conversations")
                if isinstance(conversations, list):
                    cleaned = [
                        turn
                        for turn in conversations
                        if not (
                            isinstance(turn, dict)
                            and turn.get("role") == "system"
                        )
                    ]
                    if len(cleaned) != len(conversations):
                        row["conversations"] = cleaned
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                selected_count += 1

    log(f"Selected {selected_count} samples, saved to: {output_jsonl}")

    # Create dataset_info.json (same format for all dataset types)
    dataset_info = {
        name: {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            }
        }
    }

    dataset_info_path = output_path / "dataset_info.json"
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    log(f"Created dataset_info.json at: {dataset_info_path}")

    return str(output_path), name


def get_gpu_count():
    """
    Get the number of GPUs to use based on CUDA_VISIBLE_DEVICES
    Returns the count and the actual device IDs
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if cuda_visible is None:
        # Not set, use all available GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_count = len(result.stdout.strip().split('\n'))
            log(f"CUDA_VISIBLE_DEVICES not set, using all {gpu_count} GPUs")
            return gpu_count
        except Exception as e:
            log(f"Error detecting GPUs: {e}, defaulting to 1 GPU")
            return 1
    else:
        # Parse CUDA_VISIBLE_DEVICES
        gpu_ids = [x.strip() for x in cuda_visible.split(',') if x.strip()]
        gpu_count = len(gpu_ids)
        log(f"CUDA_VISIBLE_DEVICES={cuda_visible}, using {gpu_count} GPU(s)")
        return gpu_count


def calculate_batch_size(gpu_count, dataset_type='m23k'):
    """
    Calculate per_device_batch_size based on GPU count and dataset type

    For m23k: choose values that keep gradient_accumulation_steps=1 when possible.
    In particular:
      - 2 GPUs -> per_device_batch_size=8
      - 4 GPUs -> per_device_batch_size=4
    For openthoughts: Always 1
    """
    # if dataset_type == 'openthoughts':
    #     return 1

    if gpu_count == 2:
        return 8 if dataset_type == 'm23k' else 1

    if gpu_count == 4:
        return 4

    if gpu_count == 1:
        return 8

    assert 16%gpu_count == 0, f"16 must be divisible by gpu_count ({gpu_count})"

    return 2

def calculate_gradient_accumulation_steps(gpu_count, per_device_batch_size, target_batch_size=16):
    """
    Calculate gradient_accumulation_steps to maintain target effective batch size

    Effective batch size = gpu_count * per_device_batch_size * gradient_accumulation_steps
    We want effective_batch_size = target_batch_size
    So: gradient_accumulation_steps = target_batch_size / (gpu_count * per_device_batch_size)
    """
    effective_per_step = gpu_count * per_device_batch_size
    grad_accum_steps = target_batch_size // effective_per_step
    assert grad_accum_steps >= 1, f"Gradient accumulation steps must be >= 1, got {grad_accum_steps}"
    return grad_accum_steps


def launch_training(
    config_path,
    dataset_dir,
    dataset_name,
    output_dir,
    gpu_count,
    run_name,
    dataset_type='m23k',
    target_batch_size=16,
    model_name_or_path=None,
    seed=None,
    cutoff_len=None,
):
    """
    Launch LLaMA-Factory training with the specified configuration
    """
    # Calculate batch size and gradient accumulation steps
    batch_size = calculate_batch_size(gpu_count, dataset_type)
    grad_accum_steps = calculate_gradient_accumulation_steps(gpu_count, batch_size, target_batch_size)
    effective_batch_size = gpu_count * batch_size * grad_accum_steps
    
    log(f"Calculated per_device_train_batch_size: {batch_size}")
    log(f"Calculated gradient_accumulation_steps: {grad_accum_steps}")
    log(f"Effective batch size: {effective_batch_size}")
    assert effective_batch_size == target_batch_size, f"Effective batch size {effective_batch_size} != target {target_batch_size}"
    
    # Build command
    cmd = [
        "llamafactory-cli", "train",
        config_path,
        f"dataset_dir={dataset_dir}",
        f"dataset={dataset_name}",
        f"output_dir={output_dir}",
        f"per_device_train_batch_size={batch_size}",
        f"gradient_accumulation_steps={grad_accum_steps}",
        f"run_name={run_name}",
    ]
    
    # Override model if specified
    if model_name_or_path:
        cmd.append(f"model_name_or_path={model_name_or_path}")
        if "llama" in model_name_or_path.lower():
            cmd.append("template=llama3")
    
    # Override seed if specified
    if seed is not None:
        cmd.append(f"seed={seed}")

    if cutoff_len is not None:
        cmd.append(f"cutoff_len={int(cutoff_len)}")
    
    log("=" * 60)
    log("Launching LLaMA-Factory training")
    log("=" * 60)
    log(f"Command: {' '.join(cmd)}")
    log(f"Config: {config_path}")
    log(f"Dataset: {dataset_name} (from {dataset_dir})")
    log(f"Output: {output_dir}")
    log(f"GPUs: {gpu_count}")
    log(f"Per-device batch size: {batch_size}")
    log(f"Gradient accumulation steps: {grad_accum_steps}")
    log(f"Effective batch size: {effective_batch_size}")
    log(f"Run name: {run_name}")
    if model_name_or_path:
        log(f"Model: {model_name_or_path}")
    if seed is not None:
        log(f"Seed: {seed}")
    if cutoff_len is not None:
        log(f"Cutoff length: {int(cutoff_len)}")
    log("=" * 60)
    
    # Set environment variables to reduce logging verbosity from all sources
    env = os.environ.copy()
    env['TRANSFORMERS_VERBOSITY'] = 'warning'
    env['DATASETS_VERBOSITY'] = 'warning'
    env['LLAMAFACTORY_VERBOSITY'] = 'warning'
    env['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (if used)
    env['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
    
    # Configure Python logging to suppress INFO from various libraries
    import logging
    logging.getLogger('deepspeed').setLevel(logging.WARNING)
    logging.getLogger('torch.distributed').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, env=env)
        log("Training completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        log(f"Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        log("Training interrupted by user")
        return 130
    except Exception as e:
        log(f"Error during training: {e}")
        return 1


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLaMA-Factory S2L Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Train on full m23k dataset
  %(prog)s --subset_file /path/to/indices.npy  # Train on m23k subset
  %(prog)s --dataset_type openthoughts        # Train on OpenThoughts dataset
        '''
    )
    parser.add_argument('--subset_file', type=str, default=None,
                        help='Path to .npy file containing subset indices (0-indexed). If not provided, uses full dataset.')
    parser.add_argument('--dataset_type', type=str, default='m23k', choices=['m23k', 'openthoughts'],
                        help='Dataset type to train on (default: m23k)')
    parser.add_argument('--llama', action='store_true',
                        help='Use meta-llama/Llama-3.1-8B-Instruct instead of default Qwen model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for training (default: uses value from config, typically 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory for model checkpoints. If not provided, uses default location based on dataset type.')
    parser.add_argument('--source_data_file', type=str, default=None,
                        help='Source JSONL file to use for creating subset. If not provided, uses default source based on dataset_type.')
    parser.add_argument('--cutoff_len', type=int, default=None,
                        help='Override LLaMA-Factory cutoff_len. If not provided, uses the selected YAML config default.')
    parser.add_argument('--model_name_or_path', type=str, default=None,
                        help='Override the model name or path. If provided, this overrides the --llama flag and YAML default.')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    S2L_ROOT = Path(os.environ.get("S2L_ROOT", Path(__file__).resolve().parents[1]))

    # Dataset-specific configuration
    MODEL_SIZE = "7b"
    
    # Determine model based on explicit override or --llama flag
    if args.model_name_or_path:
        MODEL_NAME = args.model_name_or_path
        # Derive simple prefix/tag from model name if possible
        model_name_lower = MODEL_NAME.lower()
        if "llama" in model_name_lower:
            MODEL_PREFIX = "llama"
            MODEL_TAG = "-llama"
        elif "qwen" in model_name_lower:
            MODEL_PREFIX = "qwen"
            MODEL_TAG = ""
        else:
            # Fallback generically
            parts = MODEL_NAME.split('/')[-1].split('-')
            MODEL_PREFIX = parts[0].lower() if parts else "custom"
            MODEL_TAG = f"-{MODEL_PREFIX}"
    elif args.llama:
        MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        MODEL_PREFIX = "llama"
        MODEL_TAG = "-llama"
    else:
        MODEL_NAME = None  # Will use default from config (Qwen)
        MODEL_PREFIX = "qwen"
        MODEL_TAG = ""

    if args.dataset_type == 'm23k':
        DATA_TYPE = "prd"
        SOURCE_JSONL = str(S2L_ROOT / "dataset" / "m23k-prd-sharegpt" / "train.jsonl")
        if MODEL_PREFIX == "llama":
            CONFIG_YAML = str(S2L_ROOT / "m1" / "src" / "train" / "llama_factory_s2l_config_llama.yaml")
        else:
            CONFIG_YAML = str(S2L_ROOT / "m1" / "src" / "train" / "llama_factory_s2l_config.yaml")
        SUBSET_BASE_DIR = str(S2L_ROOT / "dataset" / "23k_subset")
        TARGET_BATCH_SIZE = 16
    elif args.dataset_type == 'openthoughts':
        DATA_TYPE = "openthoughts"
        # SOURCE_JSONL = str(S2L_ROOT / "dataset" / "OpenThoughts-114k-template" / "train.jsonl")
        # CONFIG_YAML = str(S2L_ROOT / "m1" / "src" / "train" / "llama_factory_openthoughts_config.yaml")
        # SUBSET_BASE_DIR = str(S2L_ROOT / "dataset" / "openthoughts_subset")
        
        SOURCE_JSONL = str(S2L_ROOT / "dataset" / "op114kmath-prd-sharegpt" / "train.jsonl")
        CONFIG_YAML = str(S2L_ROOT / "m1" / "src" / "train" / "llama_factory_openthoughts_config.yaml")
        SUBSET_BASE_DIR = str(S2L_ROOT / "dataset" / "openthoughts-math_subset")
        
        # TARGET_BATCH_SIZE = 96
        TARGET_BATCH_SIZE = 16
        
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")
    
    # Override SOURCE_JSONL if --source_data_file is provided
    if args.source_data_file:
        SOURCE_JSONL = args.source_data_file
        log(f"Using custom source data file: {SOURCE_JSONL}")

    log("=" * 60)
    log("LLaMA-Factory S2L Training Manager Started")
    log("=" * 60)
    log(f"Dataset type: {args.dataset_type}")
    log(f"Model: {MODEL_NAME if MODEL_NAME else 'Qwen2.5-7B-Instruct (default)'}")
    log(f"Model size: {MODEL_SIZE}")
    log(f"Data type: {DATA_TYPE}")
    log(f"Source data: {SOURCE_JSONL}")
    log(f"Config: {CONFIG_YAML}")
    log(f"Target batch size: {TARGET_BATCH_SIZE}")
    if args.cutoff_len is not None:
        if args.cutoff_len <= 0:
            raise ValueError("--cutoff_len must be a positive integer when provided.")
        log(f"Cutoff length override: {args.cutoff_len}")
    log("=" * 60)
    
    # Determine dataset name and create subset if needed
    if args.subset_file:
        name = extract_name_from_npy_path(args.subset_file)
        dataset_dir, dataset_name = create_subset_dataset(
            args.subset_file,
            SOURCE_JSONL,
            os.path.join(SUBSET_BASE_DIR, name),
            name,
            dataset_type=args.dataset_type
        )
        # Determine output directory
        if args.output_dir:
            output_dir = os.path.expanduser(args.output_dir)
        elif args.dataset_type == 'm23k':
            local_cache_base = os.environ.get("S2L_LOCAL_CACHE", os.path.expanduser("~/local-cache"))
            output_dir = os.path.join(local_cache_base, f"s2l_23k-1k-llamafactory{MODEL_TAG}/{name}")
        else:  # openthoughts
            local_cache_base = os.environ.get("S2L_LOCAL_CACHE", os.path.expanduser("~/local-cache"))
            output_dir = os.path.join(local_cache_base, f"s2l_openthoughts-llamafactory{MODEL_TAG}/{name}")
        
        # Determine run name
        if args.dataset_type == 'm23k':
            run_name = f"{MODEL_PREFIX}_{MODEL_SIZE}_s2l_1k_{name}"
        else:  # openthoughts
            run_name = f"{MODEL_PREFIX}_{MODEL_SIZE}_openthoughts_{name}"
    else:
        if args.source_data_file and Path(args.source_data_file).parent.name:
            name = f"{Path(args.source_data_file).parent.name}-full"
        else:
            name = "full"
            
        dataset_dir, dataset_name = create_subset_dataset(
            None,
            SOURCE_JSONL,
            os.path.join(SUBSET_BASE_DIR, name),
            name,
            dataset_type=args.dataset_type
        )
        # Determine output directory
        if args.output_dir:
            output_dir = os.path.expanduser(args.output_dir)
        elif args.dataset_type == 'm23k':
            local_cache_base = os.environ.get("S2L_LOCAL_CACHE", os.path.expanduser("~/local-cache"))
            output_dir = os.path.join(local_cache_base, f"s2l_23k-1k-llamafactory{MODEL_TAG}/{name}")
        else:  # openthoughts
            local_cache_base = os.environ.get("S2L_LOCAL_CACHE", os.path.expanduser("~/local-cache"))
            output_dir = os.path.join(local_cache_base, f"s2l_openthoughts-llamafactory{MODEL_TAG}/{name}")
        
        # Determine run name
        if args.dataset_type == 'm23k':
            run_name = f"{MODEL_PREFIX}_{MODEL_SIZE}_s2l_{name}"
        else:  # openthoughts
            run_name = f"{MODEL_PREFIX}_{MODEL_SIZE}_openthoughts_{name}"

    log(f"Dataset name: {dataset_name}")
    log(f"Output directory: {output_dir}")
    log(f"Run name: {run_name}")

    # Get GPU count
    gpu_count = get_gpu_count()

    # Launch training
    exit_code = launch_training(
        CONFIG_YAML,
        dataset_dir,
        dataset_name,
        output_dir,
        gpu_count,
        run_name,
        dataset_type=args.dataset_type,
        target_batch_size=TARGET_BATCH_SIZE,
        model_name_or_path=MODEL_NAME,
        seed=args.seed,
        cutoff_len=args.cutoff_len,
    )
    
    log(f"Script exited with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
