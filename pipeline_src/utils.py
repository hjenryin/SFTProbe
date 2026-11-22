""""""

import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict


BASE_DIR = Path(os.environ.get("S2L_ROOT", Path(__file__).resolve().parents[1]))
WORKDIR_BASE = BASE_DIR / "pipeline-result"
LOCAL_CACHE = Path(os.environ.get("S2L_LOCAL_CACHE", Path.home() / "local-cache"))

HOSTNAME = socket.gethostname()
SFT_CACHE_BASE = Path(os.environ.get("SFT_CACHE_BASE", Path.home() / f"local-cache-{HOSTNAME}"))

DEFAULTS = {
    "difficulty_n_components": 2,
    "clustering_tokens": 1000,
    "n_sample_per_cluster": 4,
    "softmax_t": 1.0,
    "chat_template_args": {"reasoning_effort": "high"},
}

DEFAULT_STEP7_MAX_TOKENS = 16384

STEP7_INDEXED_TRACES = "step7_full_traces.jsonl"
STEP7_SELECTED_TRACES = "step7_selected_traces.jsonl"

APPROVAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"


def log(message: str):
    """"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = True, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """"""
    log(f"Running: {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, env=env)


def check_output_exists(path: Path, description: str) -> bool:
    """"""
    if path.exists():
        log(f"[SKIP] {description}: Output exists at {path}")
        return True
    return False


def get_num_gpus() -> int:
    """"""
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
    """"""
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


def find_free_port() -> int:
    """"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _normalize_ckpt_entry(entry) -> str:
    """"""
    s = str(entry).strip()
    if not s:
        return s

    if s.startswith("checkpoint-"):
        return s

    if any(c.isalpha() and c != "e" for c in s):
        return s

    try:
        return f"checkpoint-{int(s)}"
    except ValueError:
        return s


def get_num_token_for_difficulty(args_like) -> int:
    """"""
    if isinstance(args_like, dict):
        value = args_like.get("num_token_for_difficulty", 100)
    else:
        value = getattr(args_like, "num_token_for_difficulty", 100)
    return int(value)


def get_step7_max_tokens(args_like) -> int:
    """"""
    if isinstance(args_like, dict):
        value = args_like.get("step7_max_tokens", DEFAULT_STEP7_MAX_TOKENS)
    else:
        value = getattr(args_like, "step7_max_tokens", DEFAULT_STEP7_MAX_TOKENS)
    if value is None:
        value = DEFAULT_STEP7_MAX_TOKENS
    return int(value)


def inputs_from_run_config(run_config: dict) -> dict:
    """"""
    if "inputs" in run_config:
        return dict(run_config["inputs"])

    merged = dict(run_config.get("task_config", {}))
    for k, v in run_config.get("args", {}).items():
        if v is None and k in merged:
            continue
        merged[k] = v

    return merged


def setup_model_symlinks(cache_path: Path, workdir_ckpt: Path):
    """"""
    workdir_ckpt.mkdir(parents=True, exist_ok=True)

    for pattern in ["*.safetensors", "*.bin", "*.json", "*.model", "*.jinja", "tokenizer*"]:
        for f in cache_path.glob(pattern):
            link = workdir_ckpt / f.name
            if not link.exists():
                link.symlink_to(f)
                log(f"  Symlinked: {f.name}")


def load_jsonl(path):
    """"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def _derive_reasoning_marker(args):
    """"""
    if not args.reasoning_loss_only:
        return None
    teacher = args.teacher
    if "qwq" in teacher.lower():
        return "</think>"
    elif teacher.lower() == "original":
        return "<|im_start|>answer\n"
    else:
        return "<|end|><|start|>assistant<|channel|>final<|message|>"


def _derive_lora_perturb_ix_values(args):
    """"""
    ix_spec = args.lora_perturb_ix
    values = []
    for tok in str(ix_spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("infl", "half-infl"):
            values.append(tok)
        else:
            values.append(f"{float(tok):g}")
    return sorted(set(values)) if values else ["infl", "half-infl"]


_DERIVED_ATTRS = {
    "reasoning_marker": _derive_reasoning_marker,
    "lora_perturb_ix_values": _derive_lora_perturb_ix_values,
}


class StepArgs:
    """"""

    _ALWAYS_ALLOWED = frozenset({"run_name", "force_rerun"})

    __slots__ = ("_args", "_allowed", "_step_name")

    def __init__(self, args, allowed_keys, step_name: str):
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_allowed", frozenset(allowed_keys) | self._ALWAYS_ALLOWED)
        object.__setattr__(self, "_step_name", step_name)

    def __getattr__(self, name: str):
        if name in _DERIVED_ATTRS:
            return _DERIVED_ATTRS[name](self)

        if name in object.__getattribute__(self, "_allowed"):
            return getattr(object.__getattribute__(self, "_args"), name)

        step = object.__getattribute__(self, "_step_name")
        raise AttributeError(
            f"[StepArgs] Step '{step}': parameter '{name}' is not registered."
        )

    def __repr__(self):
        step = object.__getattribute__(self, "_step_name")
        allowed = object.__getattribute__(self, "_allowed")
        return f"StepArgs(step={step!r}, allowed={sorted(allowed)})"
