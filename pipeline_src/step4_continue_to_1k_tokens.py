""""""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

from .utils import (
    BASE_DIR,
    check_output_exists,
    get_num_gpus,
    get_physical_num_gpus,
    log,
    run_cmd,
    DEFAULTS,
)


def run(config: Dict, workdir: Path, args) -> Tuple[Path, bool]:
    """"""
    output_file = workdir / "step4_continued_1k.jsonl"

    if check_output_exists(output_file, "Step 4 (Continuation to 1k)"):
        return output_file, True

    log("=" * 60)
    log("STEP 4: Continue Generation to 1k Tokens")
    log("=" * 60)

    if args.teacher.lower() == "original":
        log("[BYPASS] Copying original dataset instead of continuation")
        input_file = workdir / "step1_generation_100tokens.jsonl"
        shutil.copy(input_file, output_file)
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

    cmd = [
        sys.executable,
        str(BASE_DIR / "m1" / "src" / "generation" / "generate.py"),
        "--input_file", str(workdir / "step1_generation_100tokens.jsonl"),
        "--output_dir", str(workdir),
        "--model_name", args.teacher,
        "--max_tokens", "900",
        "--num_gpus", str(num_gpus),
        "--data_parallel", str(data_parallel),
        "--seed", str(args.seed),
        "--mode", "continuation",
        "--continuation_indices", str(workdir / "step3_difficult_samples.npy"),
        "--chat_template_args", json.dumps(DEFAULTS["chat_template_args"]),
    ]

    run_cmd(cmd)

    generated_file = workdir / "continuation_results.jsonl"
    if generated_file.exists():
        generated_file.rename(output_file)

    return output_file, False
