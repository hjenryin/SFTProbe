"""
LLM-as-a-Judge: Compare generated final answers from step7_full_traces.jsonl
against ground truth solutions from train.jsonl using gpt-oss-120b.

Uses vLLM with data parallelism following the same pattern as generate.py.
"""

import argparse
import json
import math
import multiprocessing
import os
import re
import numpy as np
from typing import List, Dict, Optional


JUDGE_SYSTEM_PROMPT_THINK = (
    "You are a math answer equivalence judge. "
    "Given a math question, a generated answer, and a ground truth solution, "
    "determine if the generated answer is mathematically equivalent and correct "
    "compared to the ground truth.\n\n"
    "In the thought channel, give a short analysis (100~500 characters, or 20~100 words). Then in your final response channel, reply with ONLY the single word \"Correct\" or \"Incorrect\"."
)

JUDGE_SYSTEM_PROMPT_NO_THINK = (
    "You are a math answer equivalence judge. "
    "Given a math question, a generated answer, and a ground truth solution, "
    "determine if the generated answer is mathematically equivalent and correct "
    "compared to the ground truth.\n\n"
    "Your entire response must be exactly one word: \"Correct\" or \"Incorrect\". "
    "Do not explain, do not add punctuation, and do not output any other text."
)

JUDGE_USER_TEMPLATE = (
    "## Question\n{question}\n\n"
    "## Generated Answer\n{generated_answer}\n\n"
    "## Ground Truth Solution\n{gt_solution}\n\n"
    "Is the generated answer correct?"
)


def extract_final_answer(assistant_content: str, template: str = "gpt") -> Optional[str]:
    """Extract the answer from assistant content based on the template.
    
    For 'gpt': Look for channel delimiters.
    For 'qwq': Look for </think> and return content after it.
    
    Returns None if the specific delimiter is not found.
    """
    if template == "gpt":
        delimiter = "<|end|><|start|>assistant<|channel|>final<|message|>"
        if delimiter in assistant_content:
            return assistant_content.split(delimiter)[-1]
    elif template == "qwq":
        delimiter = "</think>"
        if delimiter in assistant_content:
            return assistant_content.split(delimiter)[-1]
    elif template == "gemma":
        delimiter = "<channel|>"
        if delimiter in assistant_content:
            return assistant_content.split(delimiter)[-1]
    elif template == "none":
        return assistant_content
    
    return None


def parse_judge_response(response_text: str) -> Dict:
    """Parse the judge model's response to extract thinking and verdict.
    
    QwQ style output format:
    <think>
    ... thinking process ...
    </think>
    Correct/Incorrect
    
    If no <think> tags found, treat entire response as the verdict.
    """
    thinking = ""
    verdict = response_text.strip()
    
    gemma_match = re.search(
        r"<\|channel>(?:thought|analysis|reasoning)\s*(.*?)(?:<channel\|>|$)",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if gemma_match:
        thinking = gemma_match.group(1).strip()
        verdict = response_text[gemma_match.end():].strip()
    else:
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking = think_match.group(1).strip()
            verdict = response_text[think_match.end():].strip()
        elif "</think>" in response_text:
            parts = response_text.split("</think>")
            thinking = parts[0].strip()
            if thinking.startswith("<think>"):
                thinking = thinking[7:].strip()
            verdict = parts[-1].strip()
            
    # Clean up standard QwQ/DeepSeek prefixes in verdict if any
    verdict = re.sub(r"<\|?channel\|?>[a-zA-Z_]+(?:<\|message\|>)?", "", verdict).strip()
    
    return {"thinking": thinking, "verdict": verdict, "original_response": response_text}


def strict_judge_verdict(response_text: str) -> Dict:
    """Return a parsed response with verdict restricted to Correct/Incorrect/failed."""
    parsed = parse_judge_response(response_text)
    verdict_normalized = parsed["verdict"].strip().lower()
    if verdict_normalized == "correct":
        final_verdict = "Correct"
    elif verdict_normalized == "incorrect":
        final_verdict = "Incorrect"
    else:
        final_verdict = "failed"
    return {**parsed, "final_verdict": final_verdict}


def judge_allowed_token_ids(tokenizer) -> List[int]:
    correct_ids = tokenizer.encode("Correct", add_special_tokens=False)
    incorrect_ids = tokenizer.encode("Incorrect", add_special_tokens=False)
    if len(correct_ids) != 1 or len(incorrect_ids) != 1:
        raise RuntimeError(
            "Judge verdict token enforcement requires single-token "
            f"Correct/Incorrect labels, got {correct_ids}/{incorrect_ids}"
        )
    return [int(correct_ids[0]), int(incorrect_ids[0])]


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def worker_judge(
    gpu_ids: List[int],
    judge_items: List[Dict],  # Each has: question, generated_answer, gt_solution
    model_name: str,
    tensor_parallel_size: int,
    queue: multiprocessing.Queue,
    think: bool = False,
    debug: bool = False,
):
    """Worker process that runs the judge on a chunk of items using vLLM."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        worker_id = gpu_ids[0]

        # Suppress vLLM logging
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        import logging
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)

        from vllm import LLM, SamplingParams

        print(f"[Worker {worker_id}] Initializing judge model on GPUs {gpu_ids} (TP={tensor_parallel_size})...")
        
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=128,
        )
        tokenizer = llm.get_tokenizer()

        # Format prompts
        formatted_prompts = []
        for item in judge_items:
            user_content = JUDGE_USER_TEMPLATE.format(
                question=item["question"],
                generated_answer=item["generated_answer"],
                gt_solution=item["gt_solution"],
            )
            system_prompt = JUDGE_SYSTEM_PROMPT_THINK if think else JUDGE_SYSTEM_PROMPT_NO_THINK
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if "gemma" in model_name.lower() or "qwen" in model_name.lower() or "glm" in model_name.lower():
                template_kwargs["enable_thinking"] = think

            text = tokenizer.apply_chat_template(
                messages,
                **template_kwargs
            )
            formatted_prompts.append(text)

        sampling_kwargs = {
            "max_tokens": 2048 if think else 1,
            "temperature": 0.0,
            "skip_special_tokens": False,
        }
        if not think:
            sampling_kwargs["allowed_token_ids"] = judge_allowed_token_ids(tokenizer)
        sampling_params = SamplingParams(**sampling_kwargs)

        print(f"[Worker {worker_id}] Judging {len(formatted_prompts)} items...")
        outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)

        results = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text
            if debug:
                print(f"[Worker {worker_id}] Raw output [{i}]: {repr(response_text)}")
            parsed = parse_judge_response(response_text)
            results.append(parsed)

        queue.put((worker_id, results))

    except Exception as e:
        import traceback
        traceback.print_exc()
        queue.put((worker_id, e))


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for answer comparison")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-26B-A4B-it",
                        help="Model name for vLLM")
    parser.add_argument("--step7_file", type=str,
                        default="pipeline-result/openthoughts-math/step7_full_traces.jsonl",
                        help="Path to step7_full_traces.jsonl")
    parser.add_argument("--train_file", type=str,
                        default="dataset/op114kmath-prd-sharegpt/train.jsonl",
                        help="Path to train.jsonl with ground truth solutions")
    parser.add_argument("--idx_file", type=str,
                        default="pipeline-result/openthoughts-math/step6_labeled_idx.npy",
                        help="Path to step6_labeled_idx.npy")
    parser.add_argument("--output_file", type=str,
                        default="final_answer_judge.json",
                        help="Output filename (saved in same dir as step7_file)")
    parser.add_argument("--num_gpus", type=int, default=4, help="Total number of GPUs")
    parser.add_argument("--data_parallel", type=int, default=4, help="Number of data parallel workers")
    parser.add_argument("--debug", action="store_true", help="Print raw judge model output before parsing")
    parser.add_argument("--complete_answer", action=argparse.BooleanOptionalAction, default=True,
                        help="If true (default), use the complete generation as generated_answer. "
                             "If --no_complete_answer, extract only the final channel answer.")
    parser.add_argument("--reasoning_template", type=str, default="gpt", choices=["gpt", "qwq", "gemma", "none"],
                        help="Template for parsing reasoning/answer. 'gpt' uses channel tags, 'qwq' uses </think>. 'gemma' uses <channel|>. 'none' returns the full content.")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    # 1. Load indices
    print(f"Loading indices from {args.idx_file}...")
    selected_indices = np.load(args.idx_file).tolist()
    selected_indices_set = set(selected_indices)
    print(f"Loaded {len(selected_indices)} indices")

    # 2. Load both JSONL files
    print(f"Loading step7 data from {args.step7_file}...")
    step7_data = load_jsonl(args.step7_file)
    print(f"Loading train data from {args.train_file}...")
    train_data = load_jsonl(args.train_file)

    # 3. Extract judge items (only those with a final channel answer)
    judge_items = []  # Each: {index, question, generated_answer, gt_solution}
    skipped_count = 0

    # Logic to handle different step7 file types (full vs subset)
    is_subset = len(step7_data) == len(selected_indices)
    is_full = len(step7_data) == len(train_data)

    if is_subset:
        print(f"Detected subset file (len={len(step7_data)} matches selected_indices count)")
        # Iterate through the subset file directly
        for i, step7_entry in enumerate(step7_data):
            # Check for explicit original_index, otherwise assume ordered correspondence
            if "original_index" in step7_entry:
                idx = step7_entry["original_index"]
            else:
                idx = selected_indices[i]
            
            train_entry = train_data[idx]
            
            assistant_content = step7_entry["conversations"][1]["content"]

            # Always skip entries without a final channel (degenerated)
            final_answer = extract_final_answer(assistant_content, args.reasoning_template)
            if final_answer is None:
                skipped_count += 1
                continue

            if args.complete_answer:
                generated_answer = assistant_content
            else:
                generated_answer = final_answer

            gt_solution = train_entry.get("solution", "")
            question = step7_entry.get("prompt", "")

            judge_items.append({
                "index": idx,
                "question": question,
                "generated_answer": generated_answer,
                "final_answer": final_answer,
                "gt_solution": gt_solution,
            })

    elif is_full:
        print(f"Detected full file (len={len(step7_data)} matches train_data count)")
        for idx in selected_indices:
            step7_entry = step7_data[idx]
            train_entry = train_data[idx]

            assistant_content = step7_entry["conversations"][1]["content"]

            # Always skip entries without a final channel (degenerated)
            final_answer = extract_final_answer(assistant_content, args.reasoning_template)
            if final_answer is None:
                skipped_count += 1
                continue

            if args.complete_answer:
                # Use the complete generation as-is
                generated_answer = assistant_content
            else:
                # Use only the final channel answer
                generated_answer = final_answer

            # Extract ground truth
            gt_solution = train_entry.get("solution", "")

            # Extract question (prompt)
            question = step7_entry.get("prompt", "")

            judge_items.append({
                "index": idx,
                "question": question,
                "generated_answer": generated_answer,
                "final_answer": final_answer,
                "gt_solution": gt_solution,
            })
    else:
        raise ValueError(
            f"File length mismatch: step7={len(step7_data)}, train={len(train_data)}, "
            f"selected={len(selected_indices)}. Expected step7 to match either train or selected."
        )

    print(f"Total selected/in-file: {len(step7_data) if is_subset else len(selected_indices)}")
    print(f"Skipped (no final channel): {skipped_count}")
    print(f"Items to judge: {len(judge_items)}")

    # 4. Distribute work across GPU workers
    multiprocessing.set_start_method('spawn', force=True)

    tensor_parallel_size = args.num_gpus // args.data_parallel
    if tensor_parallel_size < 1:
        tensor_parallel_size = 1

    print(f"Using {args.data_parallel} data parallel workers, each with TP={tensor_parallel_size}")

    chunk_size = math.ceil(len(judge_items) / args.data_parallel)
    chunks = []
    for i in range(args.data_parallel):
        start = i * chunk_size
        end = min(start + chunk_size, len(judge_items))
        if start < len(judge_items):
            chunks.append(judge_items[start:end])
        else:
            chunks.append([])

    # 5. Launch workers
    queue = multiprocessing.Queue()
    processes = []

    for i in range(args.data_parallel):
        if not chunks[i]:
            continue
        gpu_ids = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))
        p = multiprocessing.Process(
            target=worker_judge,
            args=(
                gpu_ids,
                chunks[i],
                args.model_name,
                tensor_parallel_size,
                queue,
                args.think,
                args.debug,
            ),
        )
        p.start()
        processes.append(p)

    # 6. Collect results
    results_map = {}
    active_processes = len([c for c in chunks if c])

    for _ in range(active_processes):
        item = queue.get()
        if isinstance(item[1], Exception):
            for p in processes:
                p.terminate()
            raise item[1]
        results_map[item[0]] = item[1]

    for p in processes:
        p.join()

    # Reassemble in order
    all_responses = []
    for i in range(args.num_gpus):
        if i in results_map:
            all_responses.extend(results_map[i])

    assert len(all_responses) == len(judge_items), (
        f"Response count mismatch: {len(all_responses)} vs {len(judge_items)}"
    )

    # 7. Parse judge responses and build output
    details = []
    correct_count = 0

    for item, response in zip(judge_items, all_responses):
        final_verdict = strict_judge_verdict(response["original_response"])["final_verdict"]

        if final_verdict == "Correct":
            is_correct = True
        elif final_verdict == "Incorrect":
            is_correct = False
        else:
            is_correct = False

        if is_correct:
            correct_count += 1

        detail = {
            "index": item["index"],
            "generated_answer": item["final_answer"],
            "gt_answer": item["gt_solution"],
            "correct?": is_correct,
            "verdict": final_verdict,
            "judge_thinking": response["thinking"],
        }
        
        if not response["thinking"] or final_verdict == "failed":
            detail["original_response"] = response.get("original_response", "")
            
        details.append(detail)

    total_count = len(details)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    output = {
        "accuracy": accuracy,
        "total_count": total_count,
        "details": details,
    }

    # 8. Save
    output_dir = os.path.dirname(args.step7_file)
    output_path = os.path.join(output_dir, args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wrong_count = total_count - correct_count
    
    print(f"\n{'='*50}")
    print(f"Results saved to {args.output_file}")
    print(f"Total attempted: {total_count + skipped_count}")
    print(f"  - Correct: {correct_count}")
    print(f"  - Wrong: {wrong_count}")
    print(f"  - Incomplete (missing final answer): {skipped_count}")
    print(f"Accuracy (Correct / Total attempted): {correct_count / (total_count + skipped_count) if (total_count + skipped_count) > 0 else 0.0:.4f}")
    print(f"Accuracy (Correct / Total judged): {accuracy:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
