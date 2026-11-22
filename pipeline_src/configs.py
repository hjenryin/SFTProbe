""""""

from .utils import BASE_DIR

PHYSICS_GENERATION_PROMPT = """You are participating in a high school physics Olympiad exam. Please read the following question carefully and provide a clear, step-by-step solution with full reasoning.
Instructions:
1. Use LaTeX to format all variables, equations, and calculations.
2. Enclose your full reasoning process within <think></think> tags.
3. Provide the final answer within <answer></answer> tags, using the format of [\\boxed{{answer}}]. Do not include units inside the box.
4. For multiple sub-questions, list the answers in order using the format: [\\boxed{{answer1}}, \\boxed{{answer2}}, ...].
5. For multiple-choice questions, provide the final selected option(s) in the boxed answer instead of the calculation result (e.g., [\\boxed{{A}}]).



Question (Answer only the question stated below):
{question}"""


TASK_CONFIGS = {
    "openthoughts-math": {
        "prompt_source": str(BASE_DIR / "dataset" / "op114kmath-prd-sharegpt" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "openthoughts-math",
        "box_prompt": "begin",
        "answer_type": "others",
    },
    "numina": {
        "prompt_source": str(BASE_DIR / "dataset" / "numina-math-4-sources" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "openthoughts-math",
        "box_prompt": "begin",
        "answer_type": "others",
    },
    "numina-dup16": {
        "prompt_source": str(BASE_DIR / "dataset" / "numina-dup16" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "openthoughts-math",
        "box_prompt": "begin",
        "answer_type": "others",
    },
    "physics-dup1": {
        "prompt_source": str(BASE_DIR / "dataset" / "PhysicsMix-dup1" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "physics",
        "box_prompt": "none",
        "answer_type": "others",
        "generation_prompt_template": PHYSICS_GENERATION_PROMPT,
    },
    "physics-dup2": {
        "prompt_source": str(BASE_DIR / "dataset" / "PhysicsMix-dup2" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "physics",
        "box_prompt": "none",
        "answer_type": "others",
        "generation_prompt_template": PHYSICS_GENERATION_PROMPT,
    },
    "physics-dup4": {
        "prompt_source": str(BASE_DIR / "dataset" / "PhysicsMix-dup4" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "physics",
        "box_prompt": "none",
        "answer_type": "others",
        "generation_prompt_template": PHYSICS_GENERATION_PROMPT,
    },
    "physics-dup8": {
        "prompt_source": str(BASE_DIR / "dataset" / "PhysicsMix-dup8" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "physics",
        "box_prompt": "none",
        "answer_type": "others",
        "generation_prompt_template": PHYSICS_GENERATION_PROMPT,
    },
    "physics-dup16": {
        "prompt_source": str(BASE_DIR / "dataset" / "PhysicsMix-dup16" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "physics",
        "box_prompt": "none",
        "answer_type": "others",
        "generation_prompt_template": PHYSICS_GENERATION_PROMPT,
    },
    "physics": {
        "prompt_source": str(BASE_DIR / "dataset" / "PhysicsMix" / "train.jsonl"),
        "dataset_type": "openthoughts",
        "eval_type": "physics",
        "box_prompt": "none",
        "answer_type": "others",
        "generation_prompt_template": PHYSICS_GENERATION_PROMPT,
    },
    "m23k": {
        "prompt_source": str(BASE_DIR / "dataset" / "m23k-prd-sharegpt" / "train.jsonl"),
        "dataset_type": "m23k",
        "eval_type": "m23k",
        "box_prompt": "end",
        "answer_type": "choice",
    },
    "m192k": {
        "prompt_source": str(BASE_DIR / "dataset" / "m192k" / "train.jsonl"),
        "dataset_type": "m23k",
        "eval_type": "m23k",
        "box_prompt": "end",
        "answer_type": "choice",
        "default_teacher": "Qwen/QwQ-32B",
        "default_step7_max_tokens": 6400,
    },
    "m38k": {
        "prompt_source": str(BASE_DIR / "dataset" / "m38k" / "train.jsonl"),
        "dataset_type": "m23k",
        "eval_type": "m23k",
        "box_prompt": "end",
        "answer_type": "choice",
        "default_teacher": "Qwen/QwQ-32B",
        "default_step7_max_tokens": 6400,
    },
    "m192k-dup16": {
        "prompt_source": str(BASE_DIR / "dataset" / "m192k-dup16" / "train.jsonl"),
        "dataset_type": "m23k",
        "eval_type": "m23k",
        "box_prompt": "end",
        "answer_type": "choice",
        "default_teacher": "Qwen/QwQ-32B",
        "default_step7_max_tokens": 6400,
    },
    "m38k-dup4": {
        "prompt_source": str(BASE_DIR / "dataset" / "m38k-dup4" / "train.jsonl"),
        "dataset_type": "m23k",
        "eval_type": "m23k",
        "box_prompt": "end",
        "answer_type": "choice",
        "default_teacher": "Qwen/QwQ-32B",
        "default_step7_max_tokens": 6400,
    },
    "m38k-dup8": {
        "prompt_source": str(BASE_DIR / "dataset" / "m38k-dup8" / "train.jsonl"),
        "dataset_type": "m23k",
        "eval_type": "m23k",
        "box_prompt": "end",
        "answer_type": "choice",
        "default_teacher": "Qwen/QwQ-32B",
        "default_step7_max_tokens": 6400,
    },
    "m38k-dup16": {
        "prompt_source": str(BASE_DIR / "dataset" / "m38k-dup16" / "train.jsonl"),
        "dataset_type": "m23k",
        "eval_type": "m23k",
        "box_prompt": "end",
        "answer_type": "choice",
        "default_teacher": "Qwen/QwQ-32B",
        "default_step7_max_tokens": 6400,
    },
}
