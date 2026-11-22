# HiPhO vLLM Text-Only Evaluation

This directory runs the HiPhO benchmark locally against a vLLM OpenAI-compatible API server. The evaluator only supports text-only questions; problems that reference `image_question` assets are skipped.

The implementation uses:

- HiPhO prompt format with `<think>` and `<answer>` tags;
- 1 run per problem by default, with `temperature=0.6`;
- pass `--runs 8` for 8-run averaging setting;
- separate CLI settings for the evaluated model and judge model;
- final-answer full credit when boxed answers match or the local judge marks them correct;
- local step-level grading against the official marking criteria when the final answer is not correct;
- exam-score aggregation and medal thresholds.

All generation and judge calls go through local vLLM API server subprocesses. No hosted LLM judge is used.

## Data

```bash
${CONDA_EXE:-conda} run -n sftprobe \
  python m1/src/eval-HiPHO/fetch_hipho.py
```

The default set is the 13 exams. `CPhO_2025` is available via `--include-current-extra` if desired.

The fetcher downloads exam JSON files only. It records referenced image counts in `manifest.json`, but does not download image files because this evaluator is text-only.

## Smoke Test

```bash
${CONDA_EXE:-conda} run -n sftprobe \
  python m1/src/eval-HiPHO/hipho_eval.py \
  --model-id google/gemma-4-31B-it \
  --judge-model-id google/gemma-4-31B-it \
  --output-dir pipeline-result/hipho/google-gemma-4-31B-it-vllm-smoke \
  --limit 1 \
  --runs 1 \
  --max-new-tokens 1024
```

Use `--disable-model-judge` for a cheaper smoke test that only performs exact/normalized answer scoring.

## Paper-Style Run

```bash
${CONDA_EXE:-conda} run -n sftprobe \
  python m1/src/eval-HiPHO/hipho_eval.py \
  --model-id google/gemma-4-31B-it \
  --judge-model-id google/gemma-4-31B-it \
  --output-dir pipeline-result/hipho/google-gemma-4-31B-it-vllm-text-only \
  --runs 8 \
  --temperature 0.6 \
  --max-new-tokens 8192
```

`--model-id` and `--judge-model-id` are independent settings. They may point to the same model, but changing `--model-id` does not implicitly change the judge.

## Thinking Mode

The evaluator enables thinking mode for solution generation by default and disables it for judge calls. It passes the thinking toggle through vLLM chat-template kwargs.

For Qwen3-style templates, the default is:

```bash
--thinking-template-key enable_thinking
```

For model families whose template expects a different key, override it, for example:

```bash
--thinking-template-key thinking
```

Disable thinking for evaluated generations with `--disable-thinking`. If your vLLM/model combination supports a reasoning budget, set `--thinking-token-budget`.

## vLLM Runtime

The evaluator starts `python -m vllm.entrypoints.openai.api_server`, waits on `/health`, and sends each run's pending prompts to `/v1/completions`. It does not implement manual shards, tensor parallelism, or a manual batch-size knob. By default, `--data-parallel-size 0` auto-selects GPUs with `gpu_utils.get_available_gpus()` and sets vLLM DP to the number of available GPUs. vLLM handles batching internally. Tune the availability filter with:

```bash
--gpu-threshold 10.0 \
--max-gpus 0
```

Use `--max-gpus 0` for all available GPUs, or set a positive value to cap DP. You can still force a fixed DP size:

```bash
--data-parallel-size 4 \
--gpu-memory-utilization 0.9 \
--dtype auto
```

`--max-model-len` defaults to `32768`.

The evaluated model server uses `--port` (default `8000`). If `--judge-model-id` points to a different model, the judge server uses `--judge-port` (default `8001`). When both IDs match, the evaluator reuses one server.

Outputs:

- `results.jsonl`: one row per problem/run, including response and score details;
- `summary.json`: machine-readable problem, exam, and medal summary;
- `evaluation_summary.txt`: compact table for quick inspection.
