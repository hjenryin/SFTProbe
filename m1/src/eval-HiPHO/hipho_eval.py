#!/usr/bin/env python3
""""""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATE_SERVER_SCRIPT = REPO_ROOT / "m1" / "src" / "generation" / "generate-server.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_utils import get_available_gpus, resolve_physical_gpu


ENGLISH_PROMPT = """You are participating in a high school physics Olympiad exam. Please read the following question carefully and provide a clear, step-by-step solution with full reasoning.
Instructions:
1. Use LaTeX to format all variables, equations, and calculations.
2. Enclose your full reasoning process within <think></think> tags.
3. Provide the final answer within <answer></answer> tags, using the format of [\\boxed{{answer}}]. Do not include units inside the box.
4. For multiple sub-questions, list the answers in order using the format: [\\boxed{{answer1}}, \\boxed{{answer2}}, ...].
5. For multiple-choice questions, provide the final selected option(s) in the boxed answer instead of the calculation result (e.g., [\\boxed{{A}}]).

Useful information (formulas, constants, units, if applicable):
{information}

Context (if applicable):
{context}

Question (Answer only the question stated below):
{question}"""


CHINESE_PROMPT = """你正在参加高中物理竞赛。请仔细阅读下列题目，结合上下文信息，详细推导并给出清晰、有条理的解题步骤与完整的逻辑推理过程。
作答要求：
1. 所有物理量、公式和计算过程须使用 LaTeX 格式书写。
2. 将完整的推理过程用 <think>和 </think>标签括起来。
3. 将最终答案置于 <answer>和 </answer>标签中，答案格式为 [\\boxed{{答案}}]，方框内不包含单位。
4. 对于包含多个小问的题目，按顺序列出所有答案，格式为：[\\boxed{{答案1}}, \\boxed{{答案2}}, ...]。
5. 对于选择题，请在答案的方框中给出最终选择的选项，而不是计算结果（例如：[\\boxed{{A}}]）。

可用信息（如物理公式、常数、单位等）：
{information}

背景信息（如有）：
{context}

题目内容（仅回答以下问题）：
{question}"""


ANSWER_JUDGE_PROMPT = """You are a diligent and precise assistant tasked with evaluating the correctness of responses.
You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer.

Special considerations:
1. If formats differ but the answers are mathematically equivalent, respond with [Correct].
2. If values match despite harmless unit formatting differences, respond with [Correct].
3. If the question provides explicit candidate answers, the output is correct if it clearly indicates the correct option code or option content.
4. Otherwise, the output must align with the correct answer in content and meaning.

Question: {problem}
Output sentence: {given_answer}
Correct answer: {ground_truth}

Final Instruction: You must respond with exactly one of the following: [Correct] or [Incorrect]. Do NOT include any explanation, reasoning, or additional text.
Judgement:"""


STEP_JUDGE_PROMPT = """You are an expert physics competition grader. Evaluate the student’s solution against the specific grading criterion.

Physics Problem:
{question}

Student’s Solution:
{prediction}

Grading Criterion:
{criterion}

Instructions:
1. Analyze the student’s solution for physics concepts, mathematical derivations, and calculations.
2. Award points strictly according to the criterion.
3. Consider both conceptual understanding and technical accuracy.

Critical:
1. You MUST respond with ONLY a single number (e.g., 1.0, 0.5, 0.0).
2. NO explanations, NO text, NO reasoning - JUST THE NUMBER.
3. If you provide any text other than the number, your response will be invalid.

Score:"""


MEDAL_THRESHOLDS = {
    "IPhO_2025": {"gold": 19.7, "silver": 12.1, "bronze": 7.2, "model_full_mark": 29.4},
    "IPhO_2024": {"gold": 20.8, "silver": 11.1, "bronze": 3.6, "model_full_mark": 29.3},
    "APhO_2025": {"gold": 23.3, "silver": 18.7, "bronze": 13.1, "model_full_mark": 30.0},
    "EuPhO_2025": {"gold": 16.5, "silver": 9.8, "bronze": 5.8, "model_full_mark": 29.0},
    "EuPhO_2024": {"gold": 20.4, "silver": 14.2, "bronze": 8.9, "model_full_mark": 28.0},
    "NBPhO_2025": {"gold": 28.6, "silver": 20.1, "bronze": 15.2, "model_full_mark": 43.5},
    "NBPhO_2024": {"gold": 26.5, "silver": 19.4, "bronze": 13.5, "model_full_mark": 50.0},
    "PanPhO_2025": {"gold": 41.5, "silver": 28.5, "bronze": 14.5, "model_full_mark": 100.0},
    "PanPhO_2024": {"gold": 52.0, "silver": 37.5, "bronze": 16.0, "model_full_mark": 98.0},
    "PanMechanics_2025": {"gold": 52.0, "silver": 36.0, "bronze": 20.0, "model_full_mark": 100.0},
    "PanMechanics_2024": {"gold": 51.0, "silver": 26.0, "bronze": 12.0, "model_full_mark": 100.0},
    "F=MA_2025": {"gold": 15.0, "silver": 11.0, "bronze": 9.0, "model_full_mark": 25.0},
    "F=MA_2024": {"gold": 14.0, "silver": 12.0, "bronze": 10.0, "model_full_mark": 25.0},
}


@dataclass
class HiPHOItem:
    raw: dict[str, Any]
    information: str


def build_thinking_kwargs(template_key: str, enabled: bool | None) -> dict[str, bool] | None:
    if enabled is None or not template_key:
        return None
    return {template_key: enabled}


def wait_for_health(port: int, timeout: int) -> None:
    from urllib.error import URLError
    from urllib.request import urlopen

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlopen(url, timeout=5) as response:
                status = response.status
            if status == 200:
                print(f"vLLM server healthy on port {port}", flush=True)
                return
        except (OSError, URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server did not become healthy within {timeout}s on port {port}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class GenerateServerRunner:
    def __init__(
        self,
        model_id: str,
        *,
        served_model_name: str,
        port: int,
        data_parallel_size: int = 1,
        workdir: Path,
        startup_timeout: int = 300,
        vllm_log_stats_interval: float = 60.0,
    ) -> None:
        self.model_id = model_id
        self.served_model_name = served_model_name
        self.port = port
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(GENERATE_SERVER_SCRIPT),
            "serve",
            "--model_name",
            model_id,
            "--tensor_parallel_size",
            "1",
            "--data_parallel_size",
            str(data_parallel_size),
            "--port",
            str(port),
            "--workdir",
            str(self.workdir),
            "--timeout",
            str(startup_timeout),
            "--vllm_log_stats_interval",
            str(vllm_log_stats_interval),
        ]
        print(f"Starting generate-server: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=os.environ.copy())
        try:
            wait_for_health(port, timeout=startup_timeout)
        except Exception:
            self.stop()
            raise

    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        enable_thinking: bool | None = None,
        thinking_template_key: str = "enable_thinking",
    ) -> list[str]:
        if not prompts:
            return []

        import numpy as np

        chat_template_kwargs = build_thinking_kwargs(thinking_template_key, enable_thinking)
        with tempfile.TemporaryDirectory(prefix="hipho_generate_", dir=str(self.workdir)) as tmp:
            tmpdir = Path(tmp)
            input_file = tmpdir / "input.jsonl"
            index_file = tmpdir / "indices.npy"
            output_file = tmpdir / "output.jsonl"
            rows = [
                {
                    "id": str(idx),
                    "prompt": prompt,
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ""},
                    ],
                    "max_tokens": 0,
                    "original_index": idx,
                }
                for idx, prompt in enumerate(prompts)
            ]
            write_jsonl(rows, input_file)
            np.save(index_file, np.arange(len(rows), dtype=int))

            cmd = [
                sys.executable,
                str(GENERATE_SERVER_SCRIPT),
                "index",
                "--index",
                str(index_file),
                "--input_file",
                str(input_file),
                "--output_file",
                str(output_file),
                "--model_name",
                self.model_id,
                "--workdir",
                str(self.workdir),
                "--chat_template_args",
                json.dumps(chat_template_kwargs or {}),
                "--max_tokens",
                str(max_new_tokens),
            ]
            subprocess.run(cmd, check=True)

            generated_rows = load_jsonl(output_file)

        generated_rows.sort(key=lambda row: int(row.get("original_index", 0)))
        if len(generated_rows) != len(prompts):
            raise RuntimeError(f"Expected {len(prompts)} generations, got {len(generated_rows)}")
        return [
            str(row.get("conversations", [{}, {"content": ""}])[-1].get("content", ""))
            for row in generated_rows
        ]

    def stop(self) -> None:
        subprocess.run(
            [
                sys.executable,
                str(GENERATE_SERVER_SCRIPT),
                "stop",
                "--workdir",
                str(self.workdir),
            ],
            check=False,
        )
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)


def source_language(source: str) -> str:
    return "zh" if source.startswith("PanMechanics") else "en"


def build_prompt(item: HiPHOItem) -> str:
    template = CHINESE_PROMPT if source_language(item.raw["source"]) == "zh" else ENGLISH_PROMPT
    return template.format(
        information=item.information or "",
        context=item.raw.get("context") or "",
        question=item.raw.get("question") or "",
    )


def load_items(data_dir: Path, sources: set[str] | None = None) -> list[HiPHOItem]:
    items: list[HiPHOItem] = []
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = json.loads(path.read_text())
        information = ""
        for obj in data:
            if "information" in obj and "id" not in obj:
                information = obj.get("information") or ""
                continue
            source = obj.get("source") or path.stem
            if sources and source not in sources:
                continue
            if obj.get("image_question"):
                continue
            items.append(HiPHOItem(raw=obj, information=information))
    return items


def extract_answer_region(text: str) -> str:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE))
    if matches:
        return matches[-1].group(1)
    return text


def extract_boxed(text: str) -> list[str]:
    region = extract_answer_region(text)
    answers: list[str] = []
    for token in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = region.find(token, start)
            if idx < 0:
                break
            brace = region.find("{", idx + len(token))
            if brace < 0:
                break
            depth = 0
            end = None
            for pos in range(brace, len(region)):
                if region[pos] == "{":
                    depth += 1
                elif region[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        end = pos
                        break
            if end is None:
                break
            answers.append(region[brace + 1 : end].strip())
            start = end + 1
    return answers


def strip_box(text: str) -> str:
    boxed = extract_boxed(text)
    return boxed[-1] if boxed else text


def normalize_answer(text: Any) -> str:
    value = strip_box(str(text))
    value = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\text\{([^{}]+)\}", r"\1", value)
    value = value.replace("$", "")
    value = value.replace("\\,", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\s+", "", value)
    return value.lower()


def exact_answer_matches(predicted: list[str], gold: list[str]) -> bool:
    if not gold:
        return False
    if not predicted:
        return False
    selected = predicted[-len(gold) :]
    return len(selected) == len(gold) and all(
        normalize_answer(p) == normalize_answer(g) for p, g in zip(selected, gold)
    )


def parse_criterion_points(criterion: str) -> float:
    match = re.search(r"Award\s+([0-9.]+)\s*pt", criterion)
    if match:
        return float(match.group(1))
    match = re.search(r"([0-9.]+)\s*pt", criterion)
    if match:
        return float(match.group(1))
    return 0.0


def parse_numeric_score(text: str, max_points: float) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    value = float(match.group(0))
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(max_points, value))


def normalize_marking_schemes(markings: Any) -> list[list[str]]:
    if not markings:
        return []
    if not isinstance(markings, list):
        raise TypeError(f"Expected marking to be a list, got {type(markings).__name__}")
    if all(isinstance(entry, str) for entry in markings):
        scheme = [entry for entry in markings if entry.strip()]
        return [scheme] if scheme else []

    schemes: list[list[str]] = []
    for scheme in markings:
        if isinstance(scheme, str):
            if scheme.strip():
                schemes.append([scheme])
            continue
        if not isinstance(scheme, list):
            raise TypeError(f"Expected marking scheme to be a list, got {type(scheme).__name__}")
        criteria = []
        for criterion in scheme:
            if not isinstance(criterion, str):
                raise TypeError(f"Expected marking criterion to be a string, got {type(criterion).__name__}")
            if criterion.strip():
                criteria.append(criterion)
        if criteria:
            schemes.append(criteria)
    return schemes


def full_answer_score(
    total_points: float,
    predicted_answers: list[str],
    *,
    exact_correct: bool,
    answer_judge_correct: bool,
) -> dict[str, Any]:
    return {
        "score": total_points,
        "max_score": total_points,
        "score_mode": "answer_full",
        "extracted_answers": predicted_answers,
        "exact_correct": exact_correct,
        "answer_judge_correct": answer_judge_correct,
        "step_details": [],
    }


def score_predictions_batched(
    records: list[tuple[HiPHOItem, str]],
    judge_runner: GenerateServerRunner | None,
    judge_max_new_tokens: int,
    thinking_template_key: str,
    use_model_judge: bool,
) -> list[dict[str, Any]]:
    can_use_judge = use_model_judge and judge_runner is not None
    contexts: list[dict[str, Any]] = []
    answer_prompts: list[str] = []
    answer_refs: list[tuple[int, int]] = []

    for item, prediction in records:
        gold_answers = item.raw.get("answer") or []
        answer_points = item.raw.get("points") or []
        predicted_answers = extract_boxed(prediction)
        selected = predicted_answers[-len(gold_answers) :] if gold_answers else []
        context = {
            "item": item,
            "prediction": prediction,
            "gold_answers": gold_answers,
            "answer_points": answer_points,
            "predicted_answers": predicted_answers,
            "selected": selected,
            "total_points": float(sum(answer_points)),
            "problem": f"{item.raw.get('context', '')}\n\n{item.raw.get('question', '')}".strip(),
            "exact_correct": exact_answer_matches(predicted_answers, gold_answers),
            "answer_judgements": [False for _ in gold_answers],
            "result": None,
        }
        contexts.append(context)

        if (
            can_use_judge
            and not context["exact_correct"]
            and predicted_answers
            and gold_answers
            and len(selected) == len(gold_answers)
        ):
            for answer_idx, (pred, gold) in enumerate(zip(selected, gold_answers)):
                if not pred:
                    continue
                answer_prompts.append(
                    ANSWER_JUDGE_PROMPT.format(
                        problem=context["problem"],
                        given_answer=pred,
                        ground_truth=gold,
                    )
                )
                answer_refs.append((len(contexts) - 1, answer_idx))

    if answer_prompts and judge_runner is not None:
        answers = judge_runner.generate(
            answer_prompts,
            max_new_tokens=judge_max_new_tokens,
            enable_thinking=False,
            thinking_template_key=thinking_template_key,
        )
        for (context_idx, answer_idx), raw in zip(answer_refs, answers):
            contexts[context_idx]["answer_judgements"][answer_idx] = "[correct]" in raw.lower()

    step_prompts: list[str] = []
    step_refs: list[tuple[int, int, str, float]] = []
    for context_idx, context in enumerate(contexts):
        if context["exact_correct"]:
            context["result"] = full_answer_score(
                context["total_points"],
                context["predicted_answers"],
                exact_correct=True,
                answer_judge_correct=False,
            )
            continue

        answer_judge_correct_all = bool(
            can_use_judge
            and context["gold_answers"]
            and len(context["selected"]) == len(context["gold_answers"])
            and all(context["answer_judgements"])
        )
        if answer_judge_correct_all:
            context["result"] = full_answer_score(
                context["total_points"],
                context["predicted_answers"],
                exact_correct=False,
                answer_judge_correct=True,
            )
            continue

        markings = normalize_marking_schemes(context["item"].raw.get("marking"))
        if markings and can_use_judge:
            for scheme_idx, scheme in enumerate(markings):
                for criterion in scheme:
                    criterion_points = parse_criterion_points(criterion)
                    step_prompts.append(
                        STEP_JUDGE_PROMPT.format(
                            question=context["problem"],
                            prediction=context["prediction"],
                            criterion=criterion,
                        )
                    )
                    step_refs.append((context_idx, scheme_idx, criterion, criterion_points))

    step_raws = []
    if step_prompts and judge_runner is not None:
        step_raws = judge_runner.generate(
            step_prompts,
            max_new_tokens=judge_max_new_tokens,
            enable_thinking=False,
            thinking_template_key=thinking_template_key,
        )

    step_scores_by_context: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for (context_idx, scheme_idx, criterion, criterion_points), raw in zip(step_refs, step_raws):
        awarded = parse_numeric_score(raw, criterion_points)
        step_scores_by_context.setdefault(context_idx, {}).setdefault(scheme_idx, []).append(
            {
                "criterion": criterion,
                "criterion_points": criterion_points,
                "judge_raw": raw,
                "fraction": awarded / criterion_points if criterion_points > 0 else 0.0,
                "awarded": awarded,
            }
        )

    for context_idx, context in enumerate(contexts):
        if context["result"] is not None:
            continue

        if context_idx in step_scores_by_context:
            scheme_scores = []
            for scheme_idx, details in step_scores_by_context[context_idx].items():
                score = sum(float(detail["awarded"]) for detail in details)
                scheme_scores.append({"scheme_idx": scheme_idx, "score": score, "details": details})
            best = max(scheme_scores, key=lambda row: row["score"])
            context["result"] = {
                "score": min(float(best["score"]), context["total_points"]),
                "max_score": context["total_points"],
                "score_mode": "step",
                "extracted_answers": context["predicted_answers"],
                "exact_correct": False,
                "answer_judge_correct": False,
                "step_details": best["details"],
                "all_scheme_scores": [
                    {"scheme_idx": s["scheme_idx"], "score": s["score"]} for s in scheme_scores
                ],
            }
            continue

        partial = 0.0
        answer_details = []
        for answer_idx, gold in enumerate(context["gold_answers"]):
            pred = context["selected"][answer_idx] if answer_idx < len(context["selected"]) else ""
            correct = bool(pred) and normalize_answer(pred) == normalize_answer(gold)
            if not correct and can_use_judge and pred and answer_idx < len(context["answer_judgements"]):
                correct = bool(context["answer_judgements"][answer_idx])
            points = float(context["answer_points"][answer_idx]) if answer_idx < len(context["answer_points"]) else 0.0
            if correct:
                partial += points
            answer_details.append({"predicted": pred, "gold": gold, "points": points, "correct": correct})
        context["result"] = {
            "score": partial,
            "max_score": context["total_points"],
            "score_mode": "answer_partial",
            "extracted_answers": context["predicted_answers"],
            "exact_correct": False,
            "answer_judge_correct": False,
            "answer_details": answer_details,
            "step_details": [],
        }

    return [context["result"] for context in contexts]


def load_existing(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            existing[(row["id"], int(row["run_idx"]))] = row
    return existing


def medal(score: float, thresholds: dict[str, float] | None) -> str:
    if not thresholds:
        return "n/a"
    if score >= thresholds["gold"]:
        return "gold"
    if score >= thresholds["silver"]:
        return "silver"
    if score >= thresholds["bronze"]:
        return "bronze"
    return "none"


def summarize(results_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = []
    with results_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    by_problem: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_problem.setdefault(row["id"], []).append(row)
        metadata[row["id"]] = {
            "source": row["source"],
            "max_score": row["max_score"],
            "field": row.get("field"),
            "modality": row.get("modality"),
        }

    problem_scores = []
    for problem_id, problem_rows in sorted(by_problem.items()):
        run_scores = {str(r["run_idx"]): float(r["score"]) for r in problem_rows}
        score_values = [float(r["score"]) for r in problem_rows]
        avg_score = sum(score_values) / len(score_values)
        score_std = statistics.stdev(score_values) if len(score_values) > 1 else None
        meta = metadata[problem_id]
        problem_scores.append(
            {
                "id": problem_id,
                "source": meta["source"],
                "field": meta["field"],
                "modality": meta["modality"],
                "runs": len(problem_rows),
                "avg_score": avg_score,
                "score_std": score_std,
                "run_scores": run_scores,
                "max_score": meta["max_score"],
            }
        )

    source_scores: dict[str, dict[str, Any]] = {}
    for row in problem_scores:
        source = row["source"]
        source_scores.setdefault(source, {"score": 0.0, "max_score": 0.0, "num_problems": 0})
        source_scores[source]["score"] += row["avg_score"]
        source_scores[source]["max_score"] += row["max_score"]
        source_scores[source]["num_problems"] += 1

    source_run_scores: dict[str, dict[str, float]] = {}
    run_total_scores: dict[str, float] = {}
    for row in rows:
        source = row["source"]
        run_idx = str(row["run_idx"])
        score = float(row["score"])
        source_run_scores.setdefault(source, {})
        source_run_scores[source][run_idx] = source_run_scores[source].get(run_idx, 0.0) + score
        run_total_scores[run_idx] = run_total_scores.get(run_idx, 0.0) + score

    medal_counts = {"gold": 0, "silver": 0, "bronze": 0, "none": 0, "n/a": 0}
    for source, stats in source_scores.items():
        run_scores = dict(sorted(source_run_scores.get(source, {}).items(), key=lambda kv: int(kv[0])))
        run_values = list(run_scores.values())
        stats["run_scores"] = run_scores
        stats["run_score_mean"] = statistics.mean(run_values) if run_values else 0.0
        stats["run_score_std"] = statistics.stdev(run_values) if len(run_values) > 1 else None
        thresholds = MEDAL_THRESHOLDS.get(source)
        stats["medal"] = medal(stats["score"], thresholds)
        stats["thresholds"] = thresholds
        medal_counts[stats["medal"]] += 1

    source_score_values = [float(stats["score"]) for stats in source_scores.values()]
    mean_score = (
        statistics.mean(source_score_values)
        if source_score_values
        else 0.0
    )
    source_keys_sorted = sorted(source_scores.keys())
    sorted_run_indices = sorted(run_total_scores.keys(), key=int)
    per_run_mean_source_scores: list[float] = []
    for ridx in sorted_run_indices:
        rkey = str(ridx)
        if source_keys_sorted:
            per_run_mean_source_scores.append(
                statistics.mean(
                    source_run_scores.get(src, {}).get(rkey, 0.0) for src in source_keys_sorted
                )
            )
    mean_score_std_across_runs = (
        statistics.stdev(per_run_mean_source_scores) if len(per_run_mean_source_scores) > 1 else None
    )
    sorted_run_total_scores = dict(sorted(run_total_scores.items(), key=lambda kv: int(kv[0])))
    run_total_values = list(sorted_run_total_scores.values())
    run_total_score_mean = statistics.mean(run_total_values) if run_total_values else 0.0
    run_total_score_std = statistics.stdev(run_total_values) if len(run_total_values) > 1 else None

    summary = {
        "generated_at": datetime.now().isoformat(),
        "results_path": str(results_path),
        "num_problem_runs": len(rows),
        "num_unique_problems": len(problem_scores),
        "mean_score": mean_score,
        "mean_score_description": (
            "Mean across source scores after each problem score is averaged across runs. "
            "Std is across evaluation runs of the per-run mean of per-source totals."
        ),
        "mean_score_std_across_runs": mean_score_std_across_runs,
        "run_total_scores": sorted_run_total_scores,
        "run_total_score_mean": run_total_score_mean,
        "run_total_score_std": run_total_score_std,
        "source_scores": dict(sorted(source_scores.items())),
        "medal_counts": medal_counts,
        "problem_scores": problem_scores,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_summary_txt(summary, output_dir / "evaluation_summary.txt")
    return summary


def write_summary_txt(summary: dict[str, Any], path: Path) -> None:
    lines = ["HiPhO Evaluation Summary", ""]
    lines.append(f"Problem runs: {summary['num_problem_runs']}")
    lines.append(f"Unique problems: {summary['num_unique_problems']}")
    mean_src_std_runs = summary.get("mean_score_std_across_runs")
    mean_src_std_runs_text = "N/A" if mean_src_std_runs is None else f"{float(mean_src_std_runs):.3f}"
    run_total_std = summary.get("run_total_score_std")
    run_total_std_text = "N/A" if run_total_std is None else f"{float(run_total_std):.3f}"
    lines.append(
        f"Mean source score: {summary['mean_score']:.3f} (std across runs: {mean_src_std_runs_text})"
    )
    lines.append(
        "Run-total score: "
        f"{summary.get('run_total_score_mean', 0.0):.3f} "
        f"(std across runs: {run_total_std_text})"
    )
    lines.append("Scores below average each problem across runs before summing by source.")
    lines.append("")
    lines.append("Source\tScore\tRunStd\tMax\tMedal\tProblems")
    for source, stats in summary["source_scores"].items():
        run_std = stats.get("run_score_std")
        run_std_text = "N/A" if run_std is None else f"{float(run_std):.3f}"
        lines.append(
            f"{source}\t{stats['score']:.3f}\t{run_std_text}\t{stats['max_score']:.3f}\t"
            f"{stats['medal']}\t{stats['num_problems']}"
        )
    lines.append("")

    grand_avg = summary['mean_score']
    grand_std = summary.get('mean_score_std_across_runs')
    grand_std_str = f"{float(grand_std):.3f}" if grand_std is not None else "N/A"
    lines.append(f"Avg.\t{grand_avg:.3f}\t{grand_std_str}\tN/A\tN/A\tN/A")
    lines.append("")

    counts = summary["medal_counts"]
    lines.append(
        f"Medal counts: gold={counts['gold']}, silver={counts['silver']}, bronze={counts['bronze']}, none={counts['none']}, n/a={counts['n/a']}"
    )
    path.write_text("\n".join(lines) + "\n")


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=8010, help="vLLM server port for the evaluated model. Pass 0 to dynamically allocate a free port.")
    parser.add_argument("--server-timeout", type=int, default=300, help="Seconds to wait for vLLM server startup.")
    parser.add_argument("--vllm-log-stats-interval", type=float, default=60.0)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=0,
        help="Number of vLLM data-parallel workers. Use 0 to auto-select available GPUs with gpu_utils.",
    )
    parser.add_argument(
        "--gpu-threshold",
        type=float,
        default=10.0,
        help="GPU memory-utilization threshold used when --data-parallel-size=0.",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=0,
        help="Maximum GPUs to use for automatic DP selection. Use 0 for all available GPUs.",
    )


def configure_data_parallel(args: argparse.Namespace) -> None:
    if args.data_parallel_size > 0:
        print(
            f"Using explicit vLLM parallelism: tensor_parallel_size=1, data_parallel_size={args.data_parallel_size}",
            flush=True,
        )
        return

    max_gpu = args.max_gpus if args.max_gpus > 0 else None
    available_logical = get_available_gpus(threshold=args.gpu_threshold, max_gpu=max_gpu)
    if not available_logical:
        raise RuntimeError(
            f"No GPUs available below {args.gpu_threshold:.1f}% memory utilization. "
            "Adjust --gpu-threshold or set --data-parallel-size explicitly."
        )

    physical_gpus = [resolve_physical_gpu(gpu_id) for gpu_id in available_logical]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in physical_gpus)
    args.data_parallel_size = len(physical_gpus)
    print(
        "Auto-selected GPUs for vLLM DP: "
        f"logical={available_logical}, physical={physical_gpus}, "
        f"tensor_parallel_size=1, data_parallel_size={args.data_parallel_size}",
        flush=True,
    )


def make_runner(
    model_id: str,
    args: argparse.Namespace,
    *,
    served_model_name: str,
    port: int,
) -> GenerateServerRunner:
    return GenerateServerRunner(
        model_id,
        served_model_name=served_model_name,
        port=port,
        data_parallel_size=args.data_parallel_size,
        workdir=args.output_dir / "generate_server" / served_model_name,
        startup_timeout=args.server_timeout,
        vllm_log_stats_interval=args.vllm_log_stats_interval,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("m1/src/eval-HiPHO/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline-result/hipho/google-gemma-4-31B-it-vllm-text-only"))
    parser.add_argument("--model-id", default="google/gemma-4-31B-it")
    parser.add_argument("--judge-model-id", default="google/gemma-4-31B-it")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--sources", default="", help="Comma-separated source names, e.g. IPhO_2025,APhO_2025")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--judge-max-new-tokens", type=int, default=32)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--thinking-template-key", default="enable_thinking")
    parser.add_argument("--disable-model-judge", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    add_vllm_args(parser)
    args = parser.parse_args()

    if args.port <= 0:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        _, resolved_port = s.getsockname()
        s.close()
        args.port = int(resolved_port)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"

    if args.summary_only:
        summarize(results_path, args.output_dir)
        return

    if args.max_gpus < 0:
        raise ValueError("--max-gpus must be >= 0")
    if args.data_parallel_size < 0:
        raise ValueError("--data-parallel-size must be >= 0")
    configure_data_parallel(args)

    source_filter = {x.strip() for x in args.sources.split(",") if x.strip()} or None
    items = load_items(args.data_dir, source_filter)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise RuntimeError(f"No text-only HiPhO items found in {args.data_dir}")

    config = vars(args).copy()
    config["started_at"] = datetime.now().isoformat()
    config["backend"] = "vllm"
    config["text_only"] = True
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    existing = load_existing(results_path)
    pending_results: list[tuple[HiPHOItem, int, str, float, str]] = []
    model_runner = make_runner(
        args.model_id,
        args,
        served_model_name="hipho-generator",
        port=args.port,
    )
    judge_runner = None
    tasks = [(item, run_idx) for run_idx in range(args.runs) for item in items]
    total_tasks = len(tasks)

    try:
        pbar = tqdm(total=total_tasks, desc="HiPhO generation")
        pbar.update(len(existing))
        for run_idx in range(args.runs):
            run_tasks = []
            for item in items:
                key = (item.raw["id"], run_idx)
                if key in existing:
                    continue
                run_tasks.append((item, run_idx, build_prompt(item), time.time()))
            if not run_tasks:
                continue
            responses = model_runner.generate(
                [prompt for _, _, prompt, _ in run_tasks],
                max_new_tokens=args.max_new_tokens,
                enable_thinking=not args.disable_thinking,
                thinking_template_key=args.thinking_template_key,
            )
            finished_at = time.time()
            for (item, item_run_idx, prompt, start), response in zip(run_tasks, responses):
                pending_results.append((item, item_run_idx, prompt, finished_at - start, response))
                pbar.update(1)
        pbar.close()

        if args.disable_model_judge or args.judge_model_id != args.model_id:
            model_runner.stop()
            model_runner = None
        else:
            judge_runner = model_runner

        if pending_results and not args.disable_model_judge and judge_runner is None:
            judge_runner = make_runner(
                args.judge_model_id,
                args,
                served_model_name="hipho-judge",
                port=args.port,
            )

        with results_path.open("a") as out:
            pbar = tqdm(total=len(pending_results), desc="HiPhO scoring")
            scored_results = score_predictions_batched(
                [(item, response) for item, _run_idx, _prompt, _generation_elapsed, response in pending_results],
                judge_runner=judge_runner,
                judge_max_new_tokens=args.judge_max_new_tokens,
                thinking_template_key=args.thinking_template_key,
                use_model_judge=not args.disable_model_judge,
            )

            def write_result(
                item: HiPHOItem,
                run_idx: int,
                prompt: str,
                generation_elapsed: float,
                response: str,
                scored: dict[str, Any],
            ) -> None:
                row = {
                    "id": item.raw["id"],
                    "source": item.raw["source"],
                    "field": item.raw.get("field"),
                    "modality": item.raw.get("modality"),
                    "answer_type": item.raw.get("answer_type"),
                    "unit": item.raw.get("unit"),
                    "run_idx": run_idx,
                    "prompt": prompt,
                    "response": response,
                    "gold_answers": item.raw.get("answer"),
                    "points": item.raw.get("points"),
                    "elapsed_sec": generation_elapsed,
                    "model_id": args.model_id,
                    "judge_model_id": None if args.disable_model_judge else args.judge_model_id,
                    "backend": "vllm",
                    **scored,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                pbar.update(1)

            for (item, item_run_idx, prompt, generation_elapsed, response), scored in zip(
                pending_results,
                scored_results,
            ):
                write_result(item, item_run_idx, prompt, generation_elapsed, response, scored)
            pbar.close()
    finally:
        if judge_runner is not None:
            judge_runner.stop()
        if model_runner is not None and model_runner is not judge_runner:
            model_runner.stop()

    summary = summarize(results_path, args.output_dir)
    print(json.dumps(summary["source_scores"], indent=2))
    print(summary["medal_counts"])


if __name__ == "__main__":
    main()
