# SFTProbe

Code for [**From Weeks to Hours: Fast and Principled SFT Curation for LLM**](https://openreview.net/forum?id=4jOHUJJVCg).


## Environment setup

The curation and training stack uses Python 3.11, PyTorch 2.9.0 (CUDA 12.8), vLLM 0.13.0, Transformers 4.57.1, and LLaMA-Factory 0.9.4.

On Linux with Conda and an NVIDIA GPU driver supporting CUDA 12.8, run from the repository root:

```bash
bash install.sh sftprobe
conda activate sftprobe
export S2L_ROOT="$PWD"
```

[`install.sh`](install.sh) creates a fresh environment and installs the pinned dependencies in [`requirements.txt`](requirements.txt), including LLaMA-Factory, DeepSpeed, Liger Kernel, and FAISS. Choose a different environment name if `sftprobe` already exists.

The training YAMLs use FlashAttention 2. To install it too, use `bash install.sh sftprobe --with-flash-attn` as the initial installation command. This builds FlashAttention 2.8.3 after installing PyTorch and requires a C++ compiler and a CUDA 12.8 toolkit; the base installation and vLLM serving do not require that toolkit.

The training YAMLs under [`m1/src/train/`](m1/src/train/) reference DeepSpeed JSON configurations that must be supplied separately. Benchmark evaluation also needs its datasets and backends: a sibling `evalchemy-dev` checkout for math/physics, and the medical evaluation assets under `m1/misc/` (plus a compatible SGLang server for the SGLang evaluation path).

### Check vLLM serving

The serving command below is a smoke test for the installed environment. It checks inference; full training and benchmark evaluation require the additional assets described above.

With the environment active, start a small model:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 --port 18765 --dtype bfloat16 \
  --max-model-len 2048 --max-num-seqs 8 \
  --gpu-memory-utilization 0.1 --enforce-eager
```

The memory fraction above is for a shared H200 GPU. On a smaller GPU, increase it (for example, to `0.8`) so the model and KV cache fit. The first run downloads the model if it is not cached. Once the server reports that it is ready, run in another terminal:

```bash
curl --fail http://127.0.0.1:18765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Reply with the word ready."}],"temperature":0,"max_tokens":32}'
```

## Prepare the data

Datasets are local JSONL files under `dataset/`. Task names and input paths are defined in [`pipeline_src/configs.py`](pipeline_src/configs.py):

| Task | Input file |
| --- | --- |
| `m23k` | `dataset/m23k-prd-sharegpt/train.jsonl` |
| `openthoughts-math` | `dataset/op114kmath-prd-sharegpt/train.jsonl` |
| `numina` | `dataset/numina-math-4-sources/train.jsonl` |
| `physics` | `dataset/PhysicsMix/train.jsonl` |
| `m38k` / `m192k` | `dataset/m38k/train.jsonl` / `dataset/m192k/train.jsonl` |

For new generation, export one problem per line with `prompt` (question text), `source` (the source dataset used for grouping), and `solution` (the reference answer used for correctness checks). Preserve medical answer fields such as `answer_letter` when available. To select from existing responses with `--teacher original`, include `conversations` using `role`/`content` turns:

```json
{"prompt":"What is 2 + 2?","source":"example-math","solution":"4","conversations":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"Adding two and two gives 4."}]}
```

Convert your source datasets to this schema and place them at the corresponding paths above, or change `prompt_source` in `pipeline_src/configs.py`. Keep row order unchanged after computing losses or selecting indices: saved `.npy` indices refer to the original JSONL line numbers. Data, model weights, and generated outputs are not bundled with this code.

## Paper to code

| Part of the paper | Main files |
| --- | --- |
| Complete curation pipeline (Algorithms 2 and 5) | [`data-curation-pipeline.py`](data-curation-pipeline.py), [`pipeline_src/step_registry.py`](pipeline_src/step_registry.py) |
| Short reasoning prefixes (§3) | [`pipeline_src/step1_generate_100_tokens.py`](pipeline_src/step1_generate_100_tokens.py), [`pipeline_src/step4_continue_to_1k_tokens.py`](pipeline_src/step4_continue_to_1k_tokens.py) |
| LoRA directions and basin boundaries (§3.1.1; Algorithms 1, 3, and 4) | [`pipeline_src/step4_7_train_lora.py`](pipeline_src/step4_7_train_lora.py), [`pipeline_src/lora_decomp.py`](pipeline_src/lora_decomp.py), [`pipeline_src/step4_8_perturb_lora_inflection.py`](pipeline_src/step4_8_perturb_lora_inflection.py), [`pipeline_src/inflection.py`](pipeline_src/inflection.py) |
| Difficulty scores and filtering (§3.2) | [`pipeline_src/step2_perturb_difficulty_model.py`](pipeline_src/step2_perturb_difficulty_model.py), [`pipeline_src/step3_difficulty_filtering.py`](pipeline_src/step3_difficulty_filtering.py) |
| Loss signatures, diversity clustering, and source budgets (§3.1.3, §3.2.1) | [`pipeline_src/step5_compute_losses.py`](pipeline_src/step5_compute_losses.py), [`pipeline_src/step6_select.py`](pipeline_src/step6_select.py), [`schedules/s2l.py`](schedules/s2l.py) |
| Teacher selection (§4) | [`m1/src/generation/teacher_selection.py`](m1/src/generation/teacher_selection.py) |
| Full trace generation and correctness-based resampling | [`pipeline_src/step7_full_generation.py`](pipeline_src/step7_full_generation.py), [`pipeline_src/step6_7_resample.py`](pipeline_src/step6_7_resample.py) |
| Student fine-tuning and evaluation (§5) | [`pipeline_src/step8_sft_training.py`](pipeline_src/step8_sft_training.py), [`m1/llama_factory_s2l.py`](m1/llama_factory_s2l.py), [`pipeline_src/step9_evaluation.py`](pipeline_src/step9_evaluation.py) |

See available pipeline and selection options with:

```bash
python data-curation-pipeline.py --help
```

## Citation

```bibtex
@inproceedings{
jin2026from,
title={From Weeks to Hours: Fast and Principled {SFT} Data Generation for Reasoning {LLM}s},
author={Hongyi Henry Jin and Wenhan Yang and Meysam Ghaffari and Carlos Morato and Baharan Mirzasoleiman},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=4jOHUJJVCg}
}
```
