#!/usr/bin/env python3
"""
Persistent vLLM generation server for resample workflow.

Three modes:
  serve  - Start a persistent vLLM server (blocks until stopped)
  index  - Send a batch of indices to the running server for generation
  stop   - Stop the running server

The server uses `vllm serve` under the hood with data parallelism + tensor parallelism.
Communication uses the OpenAI-compatible /v1/completions endpoint.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import requests


DEFAULT_VLLM_LOG_STATS_INTERVAL_SECONDS = 60
DEFAULT_MAX_NUM_SEQS = 128


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_jsonl(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def wait_for_health(port: int, timeout: int = 300):
    """Poll the /health endpoint until the server is ready."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"Server healthy on port {port}")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout}s on port {port}")


def get_num_gpus() -> int:
    """Detect number of available GPUs."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return len([x for x in cuda_visible.split(",") if x.strip()])
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True, text=True, check=True
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return 1

def get_physical_num_gpus() -> int:
    """Detect total number of physical GPUs on the node."""
    try:
        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" in env:
            del env["CUDA_VISIBLE_DEVICES"]
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, check=True, env=env
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return get_num_gpus()



# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args):
    """Start persistent vLLM server as a subprocess."""
    workdir = args.workdir

    pid_file = os.path.join(workdir, "generate_server.pid")
    port_file = os.path.join(workdir, "generate_server.port")

    if os.path.exists(pid_file):
        old_pid = int(open(pid_file).read().strip())
        # Check if still alive
        try:
            os.kill(old_pid, 0)
            print(f"Server already running (PID {old_pid}). Use 'stop' first.")
            sys.exit(1)
        except ProcessLookupError:
            print(f"Stale PID file for PID {old_pid}, cleaning up.")
            os.remove(pid_file)
            if os.path.exists(port_file):
                os.remove(port_file)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_name,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--data-parallel-size", str(args.data_parallel_size),
        "--port", str(args.port),
        "--max-num-seqs", str(args.max_num_seqs),
        "--served-model-name", "teacher",
        "--generation-config", "vllm",
        "--max-model-len", "32768",
    ]
    if getattr(args, "tokenizer", None):
        cmd.extend(["--tokenizer", str(args.tokenizer)])
    if getattr(args, "trust_remote_code", False):
        cmd.append("--trust-remote-code")

    print(f"Starting vLLM server: {' '.join(cmd)}")
    env = os.environ.copy()
    env["VLLM_LOG_STATS_INTERVAL"] = str(args.vllm_log_stats_interval)
    print(
        "vLLM stats logging interval: "
        f"{args.vllm_log_stats_interval:g}s"
    )
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    # Write PID and port
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))
    with open(port_file, "w") as f:
        f.write(str(args.port))

    # Wait for health
    try:
        wait_for_health(args.port, timeout=args.timeout)
    except TimeoutError:
        print("Server failed to start. Killing process.")
        proc.kill()
        os.remove(pid_file)
        os.remove(port_file)
        sys.exit(1)

    print(f"Server ready on port {args.port}, PID {proc.pid}")

    # Block until server exits (or Ctrl+C)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("Caught Ctrl+C, shutting down server...")
        proc.terminate()
        proc.wait(timeout=30)
    finally:
        # Clean up
        if os.path.exists(pid_file):
            os.remove(pid_file)
        if os.path.exists(port_file):
            os.remove(port_file)
        print("Server stopped.")


# ---------------------------------------------------------------------------
# index (generate batch)
# ---------------------------------------------------------------------------


def cmd_index(args):
    """Send generation requests for a batch of indices to the running server."""
    workdir = args.workdir

    port_file = os.path.join(workdir, "generate_server.port")
    if not os.path.exists(port_file):
        print("ERROR: No running server found (missing port file). Start one with 'serve' first.")
        sys.exit(1)

    port = int(open(port_file).read().strip())

    # Load indices
    indices = np.load(args.index).tolist()
    print(f"Loaded {len(indices)} indices from {args.index}")

    # Load input data
    input_data = load_jsonl(args.input_file)
    print(f"Loaded {len(input_data)} items from {args.input_file}")

    # Validate indices
    invalid = [i for i in indices if i < 0 or i >= len(input_data)]
    if invalid:
        raise ValueError(f"Invalid indices (out of range [0, {len(input_data)})): {invalid[:10]}...")

    # Load tokenizer for chat template application (match vLLM tokenizer path).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from teachers import TEACHERS, load_teacher_tokenizer, resolve_teacher_llm_kwargs

    teacher_config = dict(TEACHERS.get(args.model_name, {}) or {})
    llm_kwargs = resolve_teacher_llm_kwargs(args.model_name, teacher_config)
    if getattr(args, "tokenizer", None):
        llm_kwargs = dict(llm_kwargs)
        llm_kwargs["tokenizer"] = args.tokenizer
        teacher_config = dict(teacher_config)
        teacher_config["llm_kwargs"] = llm_kwargs
        teacher_config["fix_mislabeled_llama_tokenizer"] = True
    tokenizer = load_teacher_tokenizer(
        args.model_name,
        teacher_config,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    )

    # Parse chat template args
    try:
        chat_template_args = json.loads(args.chat_template_args)
    except json.JSONDecodeError:
        chat_template_args = {}

    # Build prompts (continuation mode logic from generate.py)
    prompts = []
    meta = []

    for idx in indices:
        item = input_data[idx]
        conversations = item.get("conversations", [])

        if not conversations:
            raise ValueError(f"Item at index {idx} has no 'conversations' field")

        # Split: history = all but last, last_msg = last message
        history = conversations[:-1]
        last_msg = conversations[-1]

        if last_msg.get("role") != "assistant":
            raise ValueError(
                f"Item at index {idx}: last message role is '{last_msg.get('role')}', expected 'assistant'"
            )

        # Apply chat template to history to get the prefix
        prefix = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_args,
        )

        # Append the raw partial content
        text = prefix + str(last_msg.get("content", ""))

        prompts.append(text)
        meta.append({"original_index": int(idx), "item": item})

    print(f"Constructed {len(prompts)} continuation prompts")

    # Single /v1/completions request for all prompts (no client-side batching)
    print(f"  Sending all {len(prompts)} prompts in one request...")

    results = []
    payload = {
        "model": "teacher",
        "prompt": prompts,
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "skip_special_tokens": False,
        "repetition_penalty": 1.0,
    }
    try:
        gen_args = json.loads(args.generation_args)
        payload.update(gen_args)
    except json.JSONDecodeError:
        pass
    generation_settings = {
        key: payload[key]
        for key in sorted(payload)
        if key not in {"model", "prompt"}
    }
    print(
        "  vLLM generation payload settings: "
        + json.dumps(generation_settings, sort_keys=True)
    )

    resp = requests.post(
        f"http://localhost:{port}/v1/completions",
        json=payload,
        timeout=None,
    )
    resp.raise_for_status()
    resp_data = resp.json()

    choices = resp_data["choices"]
    choices_sorted = sorted(choices, key=lambda c: c["index"])

    for j, choice in enumerate(choices_sorted):
        generated_text = choice["text"]
        original_row_idx = meta[j]["original_index"]
        original_item = meta[j]["item"]
        orig_idx = original_item.get("original_index", original_row_idx)
        if orig_idx is None:
            orig_idx = original_row_idx

        conversations = [dict(msg) for msg in original_item["conversations"]]
        conversations[-1]["content"] = str(conversations[-1].get("content", "")) + generated_text

        generated_token_count = len(tokenizer.encode(generated_text, add_special_tokens=False))
        original_max_tokens = original_item.get("max_tokens", 0)
        if not isinstance(original_max_tokens, int):
            original_max_tokens = 0
        final_max_tokens = original_max_tokens + generated_token_count

        result = {
            "prompt": original_item.get("prompt"),
            "source": original_item.get("source"),
            "conversations": conversations,
            "max_tokens": final_max_tokens,
            "original_index": orig_idx,
        }
        if "id" in original_item:
            result["id"] = original_item["id"]

        results.append(result)

    # Append to output file
    print(f"Appending {len(results)} results to {args.output_file}")
    with open(args.output_file, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. {len(results)} generations completed.")


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

def cmd_stop(args):
    """Stop the running vLLM server."""
    workdir = args.workdir

    pid_file = os.path.join(workdir, "generate_server.pid")
    port_file = os.path.join(workdir, "generate_server.port")

    if not os.path.exists(pid_file):
        print("No server PID file found. Server may not be running.")
        return

    pid = int(open(pid_file).read().strip())

    # Check if process exists
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"Process {pid} not found. Cleaning up stale files.")
        os.remove(pid_file)
        if os.path.exists(port_file):
            os.remove(port_file)
        return

    print(f"Sending SIGTERM to server (PID {pid})...")
    os.kill(pid, signal.SIGTERM)

    # Wait for graceful shutdown
    for i in range(30):
        try:
            os.kill(pid, 0)
            time.sleep(1)
        except ProcessLookupError:
            print(f"Server stopped (took {i+1}s).")
            break
    else:
        print(f"Server did not stop after 30s, sending SIGKILL...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # Clean up files
    if os.path.exists(pid_file):
        os.remove(pid_file)
    if os.path.exists(port_file):
        os.remove(port_file)
    print("Cleanup done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Persistent vLLM generation server for resample workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- serve ---
    serve_p = subparsers.add_parser("serve", help="Start persistent vLLM server")
    serve_p.add_argument("--model_name", required=True, help="HuggingFace model path")
    serve_p.add_argument("--tensor_parallel_size", type=int, default=None)
    serve_p.add_argument("--data_parallel_size", type=int, default=None)
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument(
        "--vllm_log_stats_interval",
        type=float,
        default=DEFAULT_VLLM_LOG_STATS_INTERVAL_SECONDS,
        help="Seconds between vLLM stats log lines",
    )
    serve_p.add_argument(
        "--max_num_seqs",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help="Forwarded to vLLM ``--max-num-seqs`` (max concurrent sequences per engine)",
    )

    serve_p.add_argument("--workdir", required=True, help="Working directory for PID/port files")
    serve_p.add_argument("--timeout", type=int, default=300, help="Health check timeout in seconds")
    serve_p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Forwarded to vLLM --tokenizer (override HF tokenizer path)",
    )
    serve_p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Forwarded to vLLM --trust-remote-code",
    )

    # --- index ---
    index_p = subparsers.add_parser("index", help="Generate for a batch of indices")
    index_p.add_argument("--index", required=True, help="Path to .npy file of indices")
    index_p.add_argument("--input_file", required=True, help="Path to input JSONL (e.g., step4)")
    index_p.add_argument("--output_file", required=True, help="Path to output JSONL (append mode)")
    index_p.add_argument("--model_name", required=True, help="Model path (for tokenizer)")
    index_p.add_argument("--workdir", required=True, help="Working directory")
    index_p.add_argument("--chat_template_args", type=str, default="{}", help="JSON string")
    index_p.add_argument("--max_tokens", type=int, default=16384, help="Max tokens to generate")
    index_p.add_argument("--generation_args", type=str, default="{}", help="JSON string of generation parameters")
    index_p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF tokenizer path matching vLLM --tokenizer",
    )
    index_p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code when loading the HF tokenizer",
    )

    # --- stop ---
    stop_p = subparsers.add_parser("stop", help="Stop the running server")
    stop_p.add_argument("--workdir", required=True, help="Working directory")

    args = parser.parse_args()

    if args.command == "serve":
        # Dynamic Defaults Logic
        if args.tensor_parallel_size is None or args.data_parallel_size is None:
            num_gpus = get_num_gpus()
            physical_gpus = get_physical_num_gpus()
            
            base_dp = 4
            if physical_gpus > 0:
                ratio = num_gpus / physical_gpus
                calculated_dp = int(base_dp * ratio)
            else:
                calculated_dp = 1
            
            default_dp = max(1, calculated_dp)
            if default_dp > num_gpus:
                default_dp = num_gpus
                
            default_tp = max(1, num_gpus // default_dp)
            
            if args.data_parallel_size is None:
                args.data_parallel_size = default_dp
            if args.tensor_parallel_size is None:
                # If TP not provided, infer from DP or default logic
                # Ideally TP * DP <= Num GPUs
                # Use the calculated TP if not provided
                args.tensor_parallel_size = default_tp
            
            print(f"Dynamic Config: Visible={num_gpus}, Physical={physical_gpus} -> DP={args.data_parallel_size}, TP={args.tensor_parallel_size}")

        cmd_serve(args)

    elif args.command == "index":
        cmd_index(args)
    elif args.command == "stop":
        cmd_stop(args)


if __name__ == "__main__":
    main()
