#!/bin/bash
S2L_ROOT="${S2L_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
if [ -n "${CONDA_ACTIVATE:-}" ] && [ -f "${CONDA_ACTIVATE}" ]; then
    source "${CONDA_ACTIVATE}" "${S2L_CONDA_ENV:-sftprobe}"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${S2L_CONDA_ENV:-sftprobe}"
fi

dataset_name=${1}
shift  # Remove first argument

use_llama=false
output_dir_override=""
model_path_arg=""
tokenizer_path_arg=""
temperature=0.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --llama)
            use_llama=true
            shift
            ;;
        --model_path)
            model_path_arg="$2"
            shift 2
            ;;
        --tokenizer_path)
            tokenizer_path_arg="$2"
            shift 2
            ;;
        --temperature)
            temperature="$2"
            shift 2
            ;;
        --output_dir)
            output_dir_override="$2"
            shift 2
            ;;
        *)
            if [[ "$1" == --* ]]; then
                echo "Unknown option: $1"
                shift
            elif [ -z "$output_dir_override" ]; then
                output_dir_override="$1"
                shift
            else
                echo "Unexpected positional argument: $1"
                shift
            fi
            ;;
    esac
done

REMOTE_PATH_PREFIX="${S2L_OUTPUTS:-${S2L_ROOT}/m1/outputs}/"
LOCAL_PATH_PREFIX="${S2L_LOCAL_CACHE:-$HOME/local-cache}/"

if [ "$use_llama" = true ]; then
    SPLIT_PATH=s2l_23k-1k-llamafactory-llama
else
    SPLIT_PATH=s2l_23k-1k-llamafactory
fi

MODEL_PATH=$SPLIT_PATH/$dataset_name/

mkdir -p ${REMOTE_PATH_PREFIX}/${MODEL_PATH}
echo "mkdir -p ${REMOTE_PATH_PREFIX}/${MODEL_PATH}"

if [ -n "$model_path_arg" ]; then
    FINAL_MODEL_PATH="$model_path_arg"
else
    CANDIDATE_MODEL_PATHS=(
        "${LOCAL_PATH_PREFIX}/${MODEL_PATH}"
        "${REMOTE_PATH_PREFIX}/${MODEL_PATH}"
        "${LOCAL_PATH_PREFIX}/s2l_23k-1k-llamafactory/${dataset_name}/"
        "${REMOTE_PATH_PREFIX}/s2l_23k-1k-llamafactory/${dataset_name}/"
        "${LOCAL_PATH_PREFIX}/s2l_23k-1k-llamafactory-llama/${dataset_name}/"
        "${REMOTE_PATH_PREFIX}/s2l_23k-1k-llamafactory-llama/${dataset_name}/"
    )

    FINAL_MODEL_PATH=""
    for candidate in "${CANDIDATE_MODEL_PATHS[@]}"; do
        expanded_candidate="${candidate/#\~/$HOME}"
        if [ -d "$expanded_candidate" ]; then
            FINAL_MODEL_PATH="$candidate"
            break
        fi
    done

    if [ -z "$FINAL_MODEL_PATH" ]; then
        FINAL_MODEL_PATH="${LOCAL_PATH_PREFIX}/${MODEL_PATH}"
    fi
fi

echo "Resolved model path: ${FINAL_MODEL_PATH}"

if [ -z "$tokenizer_path_arg" ] && [ "$use_llama" = true ]; then
    tokenizer_path_arg="meta-llama/Llama-3.1-8B-Instruct"
fi
if [ -n "$tokenizer_path_arg" ]; then
    echo "Resolved tokenizer path: ${tokenizer_path_arg}"
fi

pgrep -f "sglang" | xargs -r kill -9

if [ -n "$output_dir_override" ]; then
    EVAL_OUTPUT_DIR=$output_dir_override
elif [ "$use_llama" = true ]; then
    EVAL_OUTPUT_DIR="${S2L_ROOT}/m1/eval/s2l_23k-1k-llama-factory-llama/"
else
    EVAL_OUTPUT_DIR="${S2L_ROOT}/m1/eval/s2l_23k-1k-llama-factory/"
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "CUDA_VISIBLE_DEVICES is set: $CUDA_VISIBLE_DEVICES"
    echo "Using $NUM_GPUS GPUs from CUDA_VISIBLE_DEVICES"
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "CUDA_VISIBLE_DEVICES not set, detected $NUM_GPUS GPUs"
fi

if [ "$NUM_GPUS" -eq 8 ]; then
    TP=1
    DP=8
elif [ "$NUM_GPUS" -eq 6 ]; then
    TP=1
    DP=6
elif [ "$NUM_GPUS" -eq 4 ]; then
    TP=1
    DP=4
else
    TP=1
    DP="$NUM_GPUS"
fi

MAKE_ARGS=(
  -f exp/250318-eval-medical_llm/template.makefile
  "model_path=${FINAL_MODEL_PATH}"
  "tp=$TP" "dp=$DP"
  "seed=42"
  "temperature=${temperature}"
  "limit=-1"
  "overwrite=True"
  "exp_name=$dataset_name"
  "output_dir=$EVAL_OUTPUT_DIR"
  "max_new_tokens=16384"
  "port=13146"
)
if [ -n "$tokenizer_path_arg" ]; then
  MAKE_ARGS+=("tokenizer_path=${tokenizer_path_arg}")
fi
make "${MAKE_ARGS[@]}" eval_llm
