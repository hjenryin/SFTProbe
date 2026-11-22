""""""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline_src.utils import (
        BASE_DIR,
        find_free_port,
        get_num_gpus,
        log,
        run_cmd,
    )
else:
    from .utils import (
        BASE_DIR,
        find_free_port,
        get_num_gpus,
        log,
        run_cmd,
    )

DEFAULT_EVAL_MAX_TOKENS = 27768
MATH_EVAL_MAX_TOKENS = 27768
PHYSICS_EVAL_RUNS = 3
EVAL_MAX_MODEL_LEN = 32768
EVALCHEMY_DIR = Path(os.environ.get("EVALCHEMY_DIR", BASE_DIR.parent / "evalchemy-dev")).resolve()

def aggregate_evaluation_results(output_dir: Path) -> None:
    """"""
    log("Aggregating evaluation results...")

    results_map = {}

    gpu_dirs = list(output_dir.glob("gpu_*"))
    log(f"Found {len(gpu_dirs)} GPU output directories")

    for gpu_dir in gpu_dirs:
        json_files = list(gpu_dir.rglob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if 'results' in data:
                    for benchmark, metrics in data['results'].items():
                        if not isinstance(metrics, dict):
                            continue

                        if isinstance(metrics, dict):
                            for diff in ['easy', 'medium', 'hard']:
                                key = f'accuracy_{diff}_avg'
                                if key in metrics:
                                    results_map.setdefault(f"{benchmark} - {diff}", []).append(metrics[key])

                        if 'run_stats' in metrics and isinstance(metrics['run_stats'], list):
                            found_run_stats = False
                            for run in metrics['run_stats']:
                                score = None
                                for key in ['accuracy', 'score', 'exact_match']:
                                    if key in run:
                                        score = run[key]
                                        break
                                if score is not None:
                                    results_map.setdefault(benchmark, []).append(score)
                                    found_run_stats = True

                            if found_run_stats:
                                continue

                        score = None
                        if isinstance(metrics, dict):
                            for key in ['accuracy', 'accuracy_avg', 'score']:
                                if key in metrics:
                                    score = metrics[key]
                                    break

                        if score is not None:
                            results_map.setdefault(benchmark, []).append(score)

            except Exception as e:
                log(f"Error reading {json_file}: {e}")

    summary_lines = ["Benchmark\tMean\tStd\tCount\tScores"]

    header2 = []
    values2 = []

    def get_sort_key(b):
        if b.endswith(" - easy"): return (b.replace(" - easy", ""), 1)
        if b.endswith(" - medium"): return (b.replace(" - medium", ""), 2)
        if b.endswith(" - hard"): return (b.replace(" - hard", ""), 3)
        return (b, 0)

    target_benchmarks = ["AMC23", "AIME24", "AIME25", "MATH500"]
    available_targets = [b for b in target_benchmarks if b in results_map and results_map[b]]
    if not available_targets:
        available_targets = [b for b in results_map if results_map[b]]
        available_targets = sorted(available_targets, key=get_sort_key)

    if available_targets:
        max_runs = max(len(results_map[b]) for b in available_targets)
        for b in available_targets:
            scores = results_map[b]
            if len(scores) < max_runs:
                padded = [scores[i % len(scores)] for i in range(max_runs)]
                results_map[b] = padded

    benchmark_stats = {}
    for benchmark in sorted(results_map.keys(), key=get_sort_key):
        scores = results_map[benchmark]
        if not scores:
            continue

        mean_val = statistics.mean(scores) * 100
        if len(scores) > 1:
            std_val = statistics.stdev(scores) * 100
            std_str = f"{std_val:.2f}%"
        else:
            std_val = None
            std_str = "N/A"

        benchmark_stats[benchmark] = {
            "mean": mean_val,
            "std_str": std_str,
            "count": len(scores),
            "scores": scores
        }

        summary_lines.append(f"{benchmark}\t{mean_val:.2f}%\t{std_str}\t{len(scores)}\t{scores}")

    if available_targets:
        max_runs = len(benchmark_stats[available_targets[0]]["scores"])
        run_avgs = []
        for i in range(max_runs):
            run_avg = statistics.mean([benchmark_stats[b]["scores"][i] for b in available_targets])
            run_avgs.append(run_avg)

        grand_avg = statistics.mean(run_avgs) * 100
        if len(run_avgs) > 1:
            grand_std = statistics.stdev(run_avgs) * 100
            grand_std_str = f"{grand_std:.2f}%"
        else:
            grand_std_str = "N/A"

        header2.append("Avg.")
        values2.append(f"{grand_avg:.2f}% ± {grand_std_str}")

    for b in target_benchmarks:
        if b in benchmark_stats:
            header2.append(b)
            values2.append(f"{benchmark_stats[b]['mean']:.2f}% ± {benchmark_stats[b]['std_str']}")

    for b in sorted(benchmark_stats.keys(), key=get_sort_key):
        if b not in target_benchmarks:
            header2.append(b)
            values2.append(f"{benchmark_stats[b]['mean']:.2f}% ± {benchmark_stats[b]['std_str']}")

    summary_lines.append("")
    summary_lines.append("\t".join(header2))
    summary_lines.append("\t".join(values2))

    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    log(f"Aggregation complete. Summary saved to {summary_path}")


def _make_job(
    gpus: List[int],
    tasks: List[str],
    tp: int = 1,
    dp: int = 1,
    runs: int = 1,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    job = {
        "gpus": gpus,
        "tasks": tasks,
        "tp": tp,
        "dp": dp,
        "runs": runs,
    }
    if max_tokens is not None:
        job["max_tokens"] = max_tokens
    return job


def _build_openthoughts_math_jobs(num_gpus: int) -> List[Dict[str, Any]]:
    """"""
    if num_gpus != 4:
        raise RuntimeError(
            f"openthoughts-math evaluation requires exactly 4 GPUs, got {num_gpus}."
        )
    return [
        _make_job(gpus=[0], tasks=["MATH500"], tp=1, dp=1, runs=1, max_tokens=MATH_EVAL_MAX_TOKENS),
        _make_job(gpus=[1], tasks=["AMC23"], tp=1, dp=1, runs=1, max_tokens=MATH_EVAL_MAX_TOKENS),
        _make_job(gpus=[2], tasks=["AIME24"], tp=1, dp=1, runs=1, max_tokens=MATH_EVAL_MAX_TOKENS),
        _make_job(gpus=[3], tasks=["AIME25"], tp=1, dp=1, runs=1, max_tokens=MATH_EVAL_MAX_TOKENS),
    ]


def _build_distributed_eval_jobs(eval_type: str, num_gpus: int) -> List[Dict[str, Any]]:
    """"""
    planners = {
        "openthoughts-math": _build_openthoughts_math_jobs,
    }

    planner = planners.get(eval_type)
    if planner is None:
        raise ValueError(f"Unsupported distributed eval type: {eval_type}")

    return planner(num_gpus)


def _resolve_eval_model_path(workdir: Path, args) -> Union[Path, str]:
    model_path = workdir / "step8_sft_model"
    if model_path.exists():
        return model_path

    log(f"WARNING: Model path {model_path} not found. Using teacher {args.teacher} for testing.")
    return args.teacher


def _find_final_aggregated_result(output_dir: Path, eval_type: str) -> Optional[Path]:
    """"""
    filename = "metrics.csv" if eval_type == "m23k" else "evaluation_summary.txt"
    matches = sorted(output_dir.rglob(filename))
    return matches[0] if matches else None


def _build_eval_cmd(
    model_path: Union[Path, str],
    tasks: List[str],
    tp: int,
    dp: int,
    gpu_output: Path,
    max_tokens: int = DEFAULT_EVAL_MAX_TOKENS,
) -> List[str]:
    return [
        "python", "-m", "eval.eval",
        "--model", "vllm",
        "--tasks", ",".join(tasks),
        "--model_args", (
            f"pretrained={model_path},"
            f"tensor_parallel_size={tp},"
            f"data_parallel_size={dp},"
            f"max_model_len={EVAL_MAX_MODEL_LEN}"
        ),
        "--batch_size", "2048",
        "--output_path", str(gpu_output),
        "--max_tokens", str(max_tokens),
    ]


def _write_multi_run_script(script_path: Path, cmd: List[str], runs: int) -> None:
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\nset -e\n")
        cmd_str = shlex.join(cmd)
        for r in range(runs):
            f.write(f"echo 'Starting run {r+1}/{runs}'\n")
            f.write(f"{cmd_str}\n")
    os.chmod(script_path, 0o755)


def _launch_eval_jobs(
    jobs: List[Dict[str, Any]],
    output_dir: Path,
    model_path: Union[Path, str],
    num_gpus: int,
    cwd: Optional[Path] = None,
) -> List[Tuple[str, subprocess.Popen]]:
    from gpu_utils import resolve_physical_gpu

    procs: List[Tuple[str, subprocess.Popen]] = []

    for job in jobs:
        gpus = job["gpus"]
        tasks = job["tasks"]
        tp = job["tp"]
        dp = job["dp"]
        runs = job.get("runs", 1)
        max_tokens = job.get("max_tokens", DEFAULT_EVAL_MAX_TOKENS)

        if max(gpus) >= num_gpus:
            log(f"Skipping job on GPUs {gpus} (only {num_gpus} available)")
            continue

        gpu_str = "_".join(map(str, gpus))
        gpu_output = output_dir / f"gpu_{gpu_str}"
        gpu_output.mkdir(exist_ok=True)

        cmd = _build_eval_cmd(model_path, tasks, tp, dp, gpu_output, max_tokens=max_tokens)

        physical_gpus = [resolve_physical_gpu(g) for g in gpus]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical_gpus))
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        log(
            f"Launching on logical GPU(s) {gpu_str} "
            f"(physical {','.join(map(str, physical_gpus))}): "
            f"{','.join(tasks)} (runs: {runs})"
        )

        if runs > 1:
            script_path = gpu_output / "run.sh"
            _write_multi_run_script(script_path, cmd, runs)
            launch_cmd = ["bash", str(script_path)]
        else:
            launch_cmd = cmd

        proc = subprocess.Popen(
            launch_cmd,
            cwd=cwd,
            env=env,
            stdout=open(gpu_output / "stdout.log", "w"),
            stderr=open(gpu_output / "stderr.log", "w"),
        )
        procs.append((gpu_str, proc))

    return procs


def _wait_for_jobs_and_aggregate(procs: List[Tuple[str, subprocess.Popen]], output_dir: Path) -> None:
    failed = False

    for job_id, proc in procs:
        if proc.wait() != 0:
            log(f"Job {job_id} evaluation failed! Check logs in {output_dir}/gpu_{job_id}")
            failed = True
        else:
            log(f"Job {job_id} finished successfully")

    if failed:
        log("Evaluation failed on some GPUs. Attempting to aggregate available results...")

    aggregate_evaluation_results(output_dir)


def _run_physics_eval(
    model_path: Union[Path, str],
    output_dir: Path,
    *,
    max_tokens: Optional[int] = None,
) -> None:
    eval_tokens = max_tokens if max_tokens is not None else MATH_EVAL_MAX_TOKENS
    port = find_free_port()
    cmd = [
        sys.executable,
        str(BASE_DIR / "m1" / "src" / "eval-HiPHO" / "hipho_eval.py"),
        "--output-dir", str(output_dir),
        "--model-id", str(model_path),
        "--judge-model-id", str(model_path),
        "--runs", str(PHYSICS_EVAL_RUNS),
        "--max-new-tokens", str(eval_tokens),
        "--port", str(port),
    ]
    run_cmd(cmd, cwd=BASE_DIR)


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    """"""
    output_dir = workdir / "step9_eval"

    log("=" * 60)
    log("STEP 9: Evaluation")
    log("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    from .step8_sft_training import prepare_sft_symlink
    prepare_sft_symlink(workdir, args.run_name)

    eval_type = config.get("eval_type", config["dataset_type"])

    final_result = _find_final_aggregated_result(output_dir, eval_type)
    if final_result is not None:
        log(f"[SKIP] Step 9 (Evaluation): Output exists at {final_result}")
        return output_dir, True

    if eval_type == "m23k":
        model_path = workdir / "step8_sft_model"
        step9_temp = float(getattr(args, "step9_eval_temperature", 0.0))
        cmd = [
            "bash",
            str(BASE_DIR / "m1" / "eval-23k-llama-factory.sh"),
            args.run_name,
            str(output_dir),
            "--model_path",
            str(model_path),
            "--temperature",
            str(step9_temp),
        ]
        tokenizer_path = getattr(args, "original_model_name", None)
        if tokenizer_path:
            cmd.extend(["--tokenizer_path", str(tokenizer_path)])
            if "llama" in str(tokenizer_path).lower():
                cmd.append("--llama")
        run_cmd(cmd, cwd=BASE_DIR / "m1")

    elif eval_type == "openthoughts-math":
        log(f"Starting distributed evaluation for {eval_type}...")

        num_gpus = get_num_gpus()
        log(f"Detected {num_gpus} GPUs")

        jobs = _build_distributed_eval_jobs(eval_type, num_gpus)

        model_path = _resolve_eval_model_path(workdir, args)

        log(f"Evaluating model: {model_path}")

        procs = _launch_eval_jobs(jobs, output_dir, model_path, num_gpus, cwd=EVALCHEMY_DIR)
        _wait_for_jobs_and_aggregate(procs, output_dir)

    elif eval_type == "physics":
        model_path = _resolve_eval_model_path(workdir, args)
        log(f"Starting HiPhO physics evaluation for {eval_type}...")
        log(f"Evaluating model: {model_path}")
        _run_physics_eval(model_path, output_dir)

    else:
        log(f"Unknown dataset type: {config['dataset_type']}")

    return output_dir, False


def run_standalone_evaluation(
    model_path: Union[Path, str],
    output_dir: Path,
    eval_type: str,
    num_gpus: Optional[int] = None,
    max_tokens: Optional[int] = None,
    *,
    step9_eval_temperature: float = 0.0,
    tokenizer_path: Optional[str] = None,
) -> None:
    """"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if eval_type == "m23k":
        step9_temp = float(step9_eval_temperature)
        cmd = [
            "bash",
            str(BASE_DIR / "m1" / "eval-23k-llama-factory.sh"),
            eval_type,
            str(output_dir),
            "--model_path",
            str(model_path),
            "--temperature",
            str(step9_temp),
        ]
        if tokenizer_path:
            cmd.extend(["--tokenizer_path", str(tokenizer_path)])
            if "llama" in str(tokenizer_path).lower():
                cmd.append("--llama")
        run_cmd(cmd, cwd=BASE_DIR / "m1")
        return

    if eval_type == "physics":
        _run_physics_eval(model_path, output_dir, max_tokens=max_tokens)
        return

    if eval_type != "openthoughts-math":
        raise ValueError(f"Unsupported standalone eval type: {eval_type}")

    if num_gpus is None:
        num_gpus = get_num_gpus()

    log(f"Starting standalone distributed evaluation for {eval_type}...")
    log(f"Evaluating model: {model_path}")
    log(f"Output dir: {output_dir}")
    log(f"Detected {num_gpus} GPUs")

    jobs = _build_distributed_eval_jobs(eval_type, num_gpus)
    if max_tokens is not None:
        for job in jobs:
            job["max_tokens"] = max_tokens

    procs = _launch_eval_jobs(jobs, output_dir, model_path, num_gpus, cwd=EVALCHEMY_DIR)
    _wait_for_jobs_and_aggregate(procs, output_dir)


def _parse_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run step 9 evaluation directly from a model path and output directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path or Hugging Face model id to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where evaluation outputs will be written.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["openthoughts-math", "m23k", "physics"],
        required=True,
        help="Evaluation suite to run.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Override detected GPU count.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Override generated-token cap for all standalone jobs. Defaults to each eval suite's configured cap.",
    )
    parser.add_argument(
        "--step9_eval_temperature",
        type=float,
        default=0.0,
        help="For --eval_type m23k only: sampling temperature for Huatuo eval. Default 0.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="For --eval_type m23k SGLang: load this tokenizer instead of the SFT dir's.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = _parse_main_args()

    log("=" * 60)
    log("STEP 9 STANDALONE EVALUATION")
    log("=" * 60)
    log(f"Eval Type:  {cli_args.eval_type}")
    log(f"Model Path: {cli_args.model_path}")
    log(f"Output Dir: {cli_args.output_dir}")
    log("=" * 60)

    if cli_args.step9_eval_temperature != 0.0 and cli_args.eval_type != "m23k":
        raise SystemExit("--step9_eval_temperature is only valid with --eval_type m23k")

    run_standalone_evaluation(
        model_path=cli_args.model_path,
        output_dir=cli_args.output_dir.expanduser().resolve(),
        eval_type=cli_args.eval_type,
        num_gpus=cli_args.num_gpus,
        max_tokens=cli_args.max_tokens,
        step9_eval_temperature=cli_args.step9_eval_temperature,
        tokenizer_path=cli_args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
