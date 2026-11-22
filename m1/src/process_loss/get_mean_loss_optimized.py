import subprocess
import os
import sys
import argparse
import torch
from datasets import load_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from gpu_utils import get_available_gpus, resolve_physical_gpu


PERTURBATION_SURFACES = {"state_dict", "named_parameters_and_buffers"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--reasoning_marker", type=str, default=None)
    parser.add_argument("--strip-prefix", action="store_true", dest="strip_prefix",
                        help="Find the common prefix of all answers and exclude it from loss evaluation. "
                             "The excluded prefix does not count towards max_tokens.")
    parser.add_argument("--gpu_threshold", "--gpu-threshold", type=float, default=10.0,
                        help="Memory utilization threshold (%%) for GPU availability.")
    parser.add_argument("--perturbation_surface", type=str, choices=sorted(PERTURBATION_SURFACES), default=None,
                        help="Optional in-memory Step 2 perturbation surface to apply in each worker.")
    parser.add_argument("--perturbation_seed", type=int, default=None,
                        help="Seed for the optional in-memory Step 2 perturbation overlay.")
    parser.add_argument("--perturbation_scale", type=float, default=None,
                        help="Scale for the optional in-memory Step 2 perturbation overlay.")
    parser.add_argument("--perturbation_overlay_descriptor", type=str, default=None,
                        help="Optional descriptor JSON for an in-memory Step 2 perturbation overlay.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    overlay_enabled = args.perturbation_surface is not None or args.perturbation_overlay_descriptor is not None
    if args.perturbation_overlay_descriptor is None and args.perturbation_surface is not None and (
        args.perturbation_seed is None or args.perturbation_scale is None
    ):
        raise ValueError(
            "--perturbation_seed and --perturbation_scale are required with "
            "--perturbation_surface"
        )
    if not overlay_enabled and (args.perturbation_seed is not None or args.perturbation_scale is not None):
        raise ValueError(
            "--perturbation_surface is required when specifying perturbation seed or scale"
        )
    
    launcher_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script_path = os.path.join(launcher_dir, "_get_mean_loss_worker.py")
    abs_model_path = os.path.abspath(args.model_path) if os.path.exists(args.model_path) else args.model_path
    
    if "json" in args.dataset_path:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")
    
    if args.debug:
        dataset = dataset.select(range(99))
    dataset_size = len(dataset)

    common_prefix = None
    if args.strip_prefix:
        def _get_responses(row):
            if "conversations" in row:
                for msg in row["conversations"]:
                    if msg["role"] == "assistant":
                        return msg.get("value") or msg.get("content") or ""
                return ""
            elif "completion" in row:
                comp = row["completion"]
                if isinstance(comp, list):
                    return comp[0]["content"]
                return comp
            return ""

        responses = [_get_responses(row) for row in dataset]
        if responses:
            prefix = responses[0]
            for r in responses[1:]:
                max_len = min(len(prefix), len(r))
                cut = 0
                while cut < max_len and prefix[cut] == r[cut]:
                    cut += 1
                prefix = prefix[:cut]
            common_prefix = prefix if prefix else None
        if common_prefix:
            print(f"[strip-prefix] Detected common prefix ({len(common_prefix)} chars): {repr(common_prefix)}")
        else:
            print("[strip-prefix] No common prefix found; proceeding without stripping.")
    
    if args.output_name:
        final_output_path = args.output_name
        output_dir = os.path.dirname(os.path.abspath(args.output_name))
    else:
        output_dir = os.getcwd()
        final_output_path = os.path.join(output_dir, "mean_losses.pt")
    if os.path.exists(final_output_path):
        print(f"✅ Found existing mean losses at: {final_output_path}")
        sys.exit(0)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    idle_gpus = get_available_gpus(threshold=args.gpu_threshold)
    if not idle_gpus:
        print("No idle GPUs found among allowed devices.")
        sys.exit(1)
        
    num_processes = len(idle_gpus)
    import numpy as np
    
    indices = np.arange(dataset_size)
    np.random.seed(42)
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_processes)
        
    processes = []
    partial_files = []
    expected_sizes = {}
    
    for worker_idx, (gpu_idx, worker_indices) in enumerate(zip(idle_gpus, splits)):
        if len(worker_indices) == 0: continue
        
        worker_id = worker_idx + 1
        expected_sizes[worker_id] = len(worker_indices)
        
        partial_output = os.path.join(output_dir, f"loss-part-{worker_id}.pt")
        partial_files.append((worker_id, partial_output))
        
        indices_filename = os.path.join(output_dir, f"indices-worker-{worker_id}.npy")
        np.save(indices_filename, worker_indices)
        
        cmd = [
            sys.executable, 
            worker_script_path, 
            "--gpu_id", "0", 
            "--model_path", abs_model_path, 
            "--output_file", partial_output, 
            "--dataset_path", args.dataset_path, 
            "--max_tokens", str(args.max_tokens), 
            "--indices_file", indices_filename,
            "--process_id", str(worker_id), 
            "--total_processes", str(num_processes)
        ]
        
        if getattr(args, "reasoning_marker", None) is not None:
            cmd.extend(["--reasoning_marker", args.reasoning_marker])

        if overlay_enabled:
            if args.perturbation_overlay_descriptor is not None:
                cmd.extend(["--perturbation_overlay_descriptor", args.perturbation_overlay_descriptor])
            if args.perturbation_surface is not None:
                cmd.extend(["--perturbation_surface", args.perturbation_surface])
            if args.perturbation_seed is not None:
                cmd.extend(["--perturbation_seed", str(args.perturbation_seed)])
            if args.perturbation_scale is not None:
                cmd.extend(["--perturbation_scale", str(args.perturbation_scale)])
        
        if common_prefix is not None:
            cmd.extend(["--strip_prefix", common_prefix])
        
        if args.debug:
            cmd.append("--debug")
            
        physical_gpu_idx = resolve_physical_gpu(gpu_idx)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_idx)

        proc = subprocess.Popen(cmd, env=env)
        processes.append((worker_idx, proc))
        
    all_success = True
    for worker_idx, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            all_success = False
            
    if not all_success:
        print("One or more workers failed.")
        sys.exit(1)
    
    print("\n[Aggregation] Validating partial worker files...")
    all_results = []
    loaded_workers = set()
    
    for worker_id, partial_file in partial_files:
        if not os.path.exists(partial_file):
            print(f"❌ ERROR: Missing partial file for Worker {worker_id}: {partial_file}")
            print(f"   Expected {expected_sizes.get(worker_id, '?')} samples.")
            sys.exit(1)
        
        try:
            worker_results = torch.load(partial_file, map_location='cpu')
            
            actual_count = len(worker_results)
            expected_count = expected_sizes[worker_id]
            
            if actual_count != expected_count:
                print(f"❌ ERROR: Worker {worker_id} produced {actual_count} samples, "
                      f"but was assigned {expected_count} samples.")
                print(f"   This indicates the worker did not process all assigned samples.")
                print(f"   Partial file: {partial_file}")
                sys.exit(1)
            
            print(f"✅ Worker {worker_id}: {actual_count}/{expected_count} samples validated")
            all_results.extend(worker_results)
            loaded_workers.add(worker_id)
            os.remove(partial_file)
            
        except Exception as e:
            print(f"❌ ERROR: Failed to load partial file from Worker {worker_id}: {e}")
            sys.exit(1)
    
    if len(loaded_workers) != len(expected_sizes):
        missing_workers = set(expected_sizes.keys()) - loaded_workers
        print(f"❌ ERROR: Missing workers: {sorted(missing_workers)}")
        sys.exit(1)
    
    total_expected = sum(expected_sizes.values())
    total_actual = len(all_results)
    if total_actual != total_expected:
        print(f"❌ ERROR: Total samples mismatch: got {total_actual}, expected {total_expected}")
        sys.exit(1)
    
    print(f"✅ [Aggregation] All {len(loaded_workers)} workers validated. "
          f"Total samples: {total_actual}")
        
    all_results.sort(key=lambda x: x['index'])
    
    for i in range(num_processes):
        idx_file = os.path.join(output_dir, f"indices-worker-{i+1}.npy")
        if os.path.exists(idx_file):
            os.remove(idx_file)
            
    final_losses = [r['mean_loss'] for r in all_results]
            
    final_tensor = torch.tensor(final_losses, dtype=torch.float32)
    torch.save(final_tensor, final_output_path)
    print(f"✅ Saved final mean losses to: {final_output_path} (Shape: {final_tensor.shape})")

if __name__ == "__main__":
    main()
