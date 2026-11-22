""""""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

from .utils import (
    BASE_DIR,
    check_output_exists,
    get_num_gpus,
    get_num_token_for_difficulty,
    get_physical_num_gpus,
    log,
    run_cmd,
    DEFAULTS,
)


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    """"""
    output_file = workdir / "step1_generation_100tokens.jsonl"

    if check_output_exists(output_file, "Step 1 (100-token generation)"):
        return output_file, True

    log("=" * 60)
    log("STEP 1: Generate 100 Tokens")
    log("=" * 60)

    if args.teacher.lower() == "original":
        log("[BYPASS] Using original dataset instead of generation")
        dataset_path = config["prompt_source"]
        with open(dataset_path, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if line.strip():
                    item = json.loads(line)
                    out_item = {
                        "original_index": i,
                        "prompt": item.get("prompt", ""),
                        "conversations": item.get("conversations", [])
                    }
                    fout.write(json.dumps(out_item) + "\n")
        return output_file, False

    num_gpus = get_num_gpus()
    physical_gpus = get_physical_num_gpus()

    base_dp = 4
    if physical_gpus > 0:
        ratio = num_gpus / physical_gpus
        calculated_dp = int(base_dp * ratio)
    else:
        calculated_dp = 1

    data_parallel = max(1, calculated_dp)

    if data_parallel > num_gpus:
        data_parallel = num_gpus

    log(f"GPU Config: Visible={num_gpus}, Physical={physical_gpus}, DP={data_parallel} (TP={num_gpus//data_parallel})")

    difficulty_tokens = get_num_token_for_difficulty(args)
    generation_tokens = difficulty_tokens + 20

    cmd = [
        sys.executable,
        str(BASE_DIR / "m1" / "src" / "generation" / "generate.py"),
        "--input_file", config["prompt_source"],
        "--output_dir", str(workdir),
        "--model_name", args.teacher,
        "--max_tokens", str(generation_tokens),
        "--num_gpus", str(num_gpus),
        "--data_parallel", str(data_parallel),
        "--seed", str(args.seed),
        "--mode", "generation",
        "--chat_template_args", json.dumps(DEFAULTS["chat_template_args"]),
        "--box_prompt", config.get("box_prompt", "none"),
        "--answer_type", config.get("answer_type", "others"),
    ]

    if config.get("generation_prompt_template"):
        cmd.extend(["--prompt_template", config["generation_prompt_template"]])

    run_cmd(cmd)

    generated_file = workdir / "generation_results.jsonl"
    if generated_file.exists():
        generated_file.rename(output_file)

    return output_file, False
