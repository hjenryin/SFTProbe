#!/usr/bin/env bash
# Usage: bash install.sh [environment-name] [--with-flash-attn]
set -euo pipefail

if [[ ${1:-} == --help ]]; then
  echo "Usage: bash install.sh [environment-name] [--with-flash-attn]"
  exit 0
fi

if (( $# > 2 )); then
  echo "Usage: bash install.sh [environment-name] [--with-flash-attn]" >&2
  exit 1
fi

sftprobe_env_name="${1:-sftprobe}"
sftprobe_flash_flag="${2:-}"
if [[ ! "$sftprobe_env_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] ||
   [[ -n "$sftprobe_flash_flag" && "$sftprobe_flash_flag" != --with-flash-attn ]]; then
  echo "Usage: bash install.sh [environment-name] [--with-flash-attn]" >&2
  exit 1
fi

sftprobe_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list --json | "$CONDA_PYTHON_EXE" -c \
  'import json, pathlib, sys; sys.exit(0 if any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]) else 1)' \
  "$sftprobe_env_name"; then
  echo "Environment '$sftprobe_env_name' already exists; choose a new name." >&2
  exit 1
fi

conda create -y -n "$sftprobe_env_name" python=3.11 pip
conda activate "$sftprobe_env_name"
python -m pip install 'setuptools>=77,<81' wheel ninja packaging
python -m pip install 'torch==2.9.0' 'torchvision==0.24.0' 'torchaudio==2.9.0'
DS_BUILD_OPS=0 python -m pip install -r "$sftprobe_repo_dir/requirements.txt"

if [[ "$sftprobe_flash_flag" == --with-flash-attn ]]; then
  # Requires a compiler and CUDA 12.8 toolkit matching PyTorch's CUDA runtime.
  python -m pip install --no-build-isolation 'flash-attn==2.8.3'
fi

python -m pip check
printf '\nInstalled environment: %s\nActivate with: conda activate %s\n' \
  "$sftprobe_env_name" "$sftprobe_env_name"
printf 'From the repository root, run: export S2L_ROOT="$PWD"\n'
