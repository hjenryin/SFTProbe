#!/usr/bin/env bash
# Medical teacher-selection (student loss) pipeline:
# teachers generate with the teacher prompt for --num-tokens (default 120),
# then the student scores mean NLL on the normalized completions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# --num-tokens is the student loss window. Teachers generate num_tokens+50
# (default 100+50=150) to cover tokenizer length mismatch.
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/m1/output/teacher-selection/medical-100tok}"
DATA="${DATA:-${REPO_ROOT}/dataset/m23k-prd-sharegpt/train.jsonl}"
SEED="${SEED:-0}"
NUM_EXAMPLES="${NUM_EXAMPLES:-100}"
NUM_TOKENS="${NUM_TOKENS:-100}"
CONDA_EXE="${CONDA_EXE:-conda}"

TEACHERS=(
  openai/gpt-oss-120b
  google/gemma-4-31B-it
  google/gemma-4-26B-A4B-it
  Qwen/Qwen3.6-27B
  microsoft/Phi-4-reasoning-plus
  Qwen/QwQ-32B
)

cd "${REPO_ROOT}"
exec "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV:-sftprobe}" python \
  "${REPO_ROOT}/m1/src/generation/teacher_selection.py" \
  --data "${DATA}" \
  --num-examples "${NUM_EXAMPLES}" \
  --num-tokens "${NUM_TOKENS}" \
  --seed "${SEED}" \
  --output-dir "${OUT_DIR}" \
  --answer-type choice \
  --prompt-policy end \
  --teachers "${TEACHERS[@]}" \
  "$@"
