""""""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .loss_plan import LossJob
from .step_registry import RESUME_COMMON, _STEP4_8_KEYS
from .utils import get_num_token_for_difficulty, inputs_from_run_config, log


_LOSS_TOKEN_PATTERN = re.compile(r"^.*?(\d+)token.*\.pt$")

_STEP_ORDER = {
    "step1": 1.0,
    "step2": 2.0,
    "step3": 3.0,
    "step4": 4.0,
    "step4_5": 4.5,
    "step4_7": 4.7,
    "step4_8": 4.8,
    "step5": 5.0,
    "step6": 6.0,
    "step6_7": 6.5,
    "step7": 7.0,
    "step8": 8.0,
    "step9": 9.0,
}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except FileNotFoundError:
        return a.absolute() == b.absolute()


def _step5_trusted(source_inputs: Dict[str, Any]) -> bool:
    override_steps = source_inputs.get("manual_override_steps", []) or []
    for step_name in override_steps:
        if _STEP_ORDER.get(str(step_name), float("inf")) <= _STEP_ORDER["step5"]:
            return False
    return True


def _input_value(inputs: Dict[str, Any], key: str) -> Any:
    if key == "num_token_for_difficulty":
        return get_num_token_for_difficulty(inputs)
    if key == "reasoning_loss_only":
        return bool(inputs.get(key, False))
    return inputs.get(key)


def _inputs_compatible(
    *,
    current_inputs: Dict[str, Any],
    source_inputs: Dict[str, Any],
    keys: Sequence[str],
    allow_richer_lora_rank: bool = False,
) -> bool:
    for key in keys:
        if key == "num_token_for_difficulty":
            if get_num_token_for_difficulty(source_inputs) != get_num_token_for_difficulty(current_inputs):
                return False
            continue

        if key not in source_inputs:
            continue

        source_val = _input_value(source_inputs, key)
        current_val = _input_value(current_inputs, key)

        if allow_richer_lora_rank and key == "lora_train_rank":
            try:
                if int(source_val) < int(current_val):
                    return False
            except (TypeError, ValueError):
                return False
            continue

        if source_val != current_val:
            return False
    return True


def _load_step5_jobs(source_workdir: Path) -> Dict[str, Dict[str, Any]]:
    metadata = _load_step5_metadata(source_workdir)
    return _jobs_from_step5_metadata(metadata)


def _load_step5_metadata(source_workdir: Path) -> Dict[str, Any]:
    return _load_json(source_workdir / "step5_metadata.json") or {}


def _jobs_from_step5_metadata(metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    jobs = metadata.get("jobs", [])
    if not isinstance(jobs, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in jobs:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not name:
            continue
        out[str(name)] = row
    return out


def _source_is_hf_mean_loss(source_method: Any) -> bool:
    return str(source_method or "hf") == "hf"


def _checkpoint_same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    a_path = Path(str(a)).expanduser()
    b_path = Path(str(b)).expanduser()
    try:
        return a_path.resolve() == b_path.resolve()
    except FileNotFoundError:
        return str(a_path) == str(b_path)


def _overlay_metadata_for_compare(value: Any) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    if not isinstance(value, dict):
        return {"value": value}
    fingerprint = value.get("fingerprint")
    return {
        "base_model": value.get("base_model"),
        "perturbation_surface": value.get("perturbation_surface"),
        "perturbation_surface_version": value.get("perturbation_surface_version"),
        "seed": value.get("seed"),
        "scale": value.get("scale"),
        "fingerprint_digest": (
            fingerprint.get("digest") if isinstance(fingerprint, dict) else value.get("fingerprint_digest")
        ),
        "fingerprint_algorithm": (
            fingerprint.get("algorithm") if isinstance(fingerprint, dict) else value.get("fingerprint_algorithm")
        ),
        "fingerprint_samples_per_tensor": (
            fingerprint.get("samples_per_tensor")
            if isinstance(fingerprint, dict)
            else value.get("fingerprint_samples_per_tensor")
        ),
    }


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_pair(value: Any) -> Optional[Tuple[Optional[float], Optional[float]]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (_as_float(value[0]), _as_float(value[1]))


def _step2_inflection_identity(workdir: Path) -> Optional[Dict[str, Any]]:
    """"""
    step2_dir = workdir / "step2_loss_100tokens"
    metadata = _load_json(step2_dir / "inflection_metadata.json")
    if not isinstance(metadata, dict):
        return None

    search = metadata.get("search") if isinstance(metadata.get("search"), dict) else {}
    selected = metadata.get("selected") if isinstance(metadata.get("selected"), dict) else {}
    ck_inflection = step2_dir / "checkpoint-inflection"
    overlay = _load_json(ck_inflection / "perturbation_overlay.json")
    if overlay is None:
        overlay = _load_json(ck_inflection / "_perturbation_overlay.json")

    selected_scale = selected.get("scale")
    if selected_scale is None:
        selected_scale = search.get("x_opt")

    return {
        "base_model": metadata.get("base_model"),
        "perturbation_surface": metadata.get("perturbation_surface"),
        "perturbation_surface_version": _as_int(metadata.get("perturbation_surface_version")),
        "seed": _as_int(metadata.get("seed")),
        "difficulty_tokens": _as_int(metadata.get("difficulty_tokens")),
        "scan_range": _float_pair(metadata.get("scan_range") or search.get("scan_range")),
        "ytol": _as_float(search.get("ytol", metadata.get("tolerance_y_axis"))),
        "max_iter": _as_int(search.get("max_iter", metadata.get("max_iter"))),
        "curvature_ratio": _as_float(search.get("curvature_ratio", metadata.get("curvature_ratio"))),
        "selected_checkpoint": selected.get("checkpoint"),
        "selected_scale": _as_float(selected_scale),
        "overlay": _overlay_metadata_for_compare(overlay),
    }


def _source_job_metadata_compatible(
    job: LossJob,
    source_job: Optional[Dict[str, Any]],
    *,
    compare_checkpoint: bool = True,
) -> bool:
    if source_job is None:
        return False
    try:
        if int(source_job.get("max_tokens")) != int(job.max_tokens):
            return False
    except (TypeError, ValueError):
        return False
    if bool(source_job.get("is_adapter")) != bool(job.is_adapter):
        return False
    if str(source_job.get("loss_file")) != str(job.loss_file):
        return False
    if compare_checkpoint and str(source_job.get("checkpoint")) != str(job.checkpoint):
        return False
    if _overlay_metadata_for_compare(source_job.get("perturbation_overlay")) != _overlay_metadata_for_compare(
        job.perturbation_overlay
    ):
        return False
    return True


def _is_step48_adapter_job(job: LossJob) -> bool:
    return bool(job.is_adapter) and job.name.startswith("dir") and "_over_" in job.name


def _difficulty_inflection_compatible(
    *,
    job: LossJob,
    current_workdir: Path,
    current_inputs: Dict[str, Any],
    source_workdir: Path,
    source_inputs: Dict[str, Any],
    source_job: Optional[Dict[str, Any]],
    source_loss_method: Any,
) -> bool:
    if not _source_is_hf_mean_loss(source_loss_method):
        return False
    if not _inputs_compatible(
        current_inputs=current_inputs,
        source_inputs=source_inputs,
        keys=tuple(RESUME_COMMON),
    ):
        return False
    if not _source_job_metadata_compatible(job, source_job, compare_checkpoint=False):
        return False

    current_identity = _step2_inflection_identity(current_workdir)
    source_identity = _step2_inflection_identity(source_workdir)
    if current_identity is None or source_identity is None:
        return False
    return current_identity == source_identity


def _job_inputs_compatible(
    *,
    job: LossJob,
    current_workdir: Path,
    current_inputs: Dict[str, Any],
    source_workdir: Path,
    source_inputs: Dict[str, Any],
    source_job: Optional[Dict[str, Any]],
    source_loss_method: Any,
) -> bool:
    if not _source_is_hf_mean_loss(source_loss_method):
        return False

    if job.name == "base_unperturbed":
        if not _inputs_compatible(
            current_inputs=current_inputs,
            source_inputs=source_inputs,
            keys=tuple(RESUME_COMMON),
        ):
            return False
        if source_job is not None and not _source_job_metadata_compatible(job, source_job):
            return False
        return True

    if job.name == "difficulty_inflection":
        return _difficulty_inflection_compatible(
            job=job,
            current_workdir=current_workdir,
            current_inputs=current_inputs,
            source_workdir=source_workdir,
            source_inputs=source_inputs,
            source_job=source_job,
            source_loss_method=source_loss_method,
        )

    if job.name.startswith("sort_ckpt_"):
        return (
            _inputs_compatible(
                current_inputs=current_inputs,
                source_inputs=source_inputs,
                keys=tuple(RESUME_COMMON),
            )
            and _source_job_metadata_compatible(job, source_job)
            and _checkpoint_same(source_job.get("checkpoint") if source_job else None, job.checkpoint)
        )

    if _is_step48_adapter_job(job):
        return _inputs_compatible(
            current_inputs=current_inputs,
            source_inputs=source_inputs,
            keys=tuple(_STEP4_8_KEYS),
            allow_richer_lora_rank=True,
        )

    return (
        _inputs_compatible(
            current_inputs=current_inputs,
            source_inputs=source_inputs,
            keys=tuple(RESUME_COMMON),
        )
        and _source_job_metadata_compatible(job, source_job)
        and _checkpoint_same(source_job.get("checkpoint") if source_job else None, job.checkpoint)
    )


def _find_source_loss_file(
    *,
    source_step5: Path,
    job: LossJob,
    source_job: Optional[Dict[str, Any]],
) -> Optional[Path]:
    src_dir = source_step5 / job.name
    if not src_dir.is_dir():
        return None

    preferred_names = [job.loss_file]
    if source_job is not None and source_job.get("loss_file"):
        preferred_names.append(str(source_job["loss_file"]))

    for name in preferred_names:
        candidate = src_dir / name
        if candidate.exists():
            return candidate

    matches: List[Path] = []
    for pt_file in src_dir.glob("*.pt"):
        m = _LOSS_TOKEN_PATTERN.match(pt_file.name)
        if m and int(m.group(1)) == int(job.max_tokens):
            matches.append(pt_file)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        log(
            f"[step5-link] Multiple {job.max_tokens}-token source losses in "
            f"{src_dir}: {[p.name for p in matches]}; leaving {job.name} pending"
        )
    return None


def _iter_candidate_runs(scan_root: Optional[Path], current_workdir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    if scan_root is None or not scan_root.exists():
        return []

    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    for config_path in scan_root.rglob("pipeline_config.json"):
        cand_dir = config_path.parent.absolute()
        if cand_dir == current_workdir.absolute():
            continue
        cfg = _load_json(config_path)
        if cfg is None:
            continue
        candidates.append((cand_dir, cfg))
    return candidates


def _link_loss_file(src_file: Path, dst_file: Path) -> bool:
    if dst_file.exists() or dst_file.is_symlink():
        return False
    if _same_path(src_file, dst_file):
        return False

    dst_dir = dst_file.parent
    if dst_dir.is_symlink():
        dst_dir.unlink()
    elif dst_dir.exists() and not dst_dir.is_dir():
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file.symlink_to(src_file.absolute())
    return True


def link_reusable_step5_jobs(
    *,
    workdir: Path,
    current_inputs: Dict[str, Any],
    jobs: Sequence[LossJob],
    scan_root: Optional[Path],
) -> int:
    """"""
    if not jobs:
        return 0

    candidates = _iter_candidate_runs(scan_root, workdir)
    if not candidates:
        return 0

    output_dir = workdir / "step5_loss_1k_tokens"
    pending = {
        job.name: job
        for job in jobs
        if not (output_dir / job.name / job.loss_file).exists()
        and not (output_dir / job.name / job.loss_file).is_symlink()
    }
    if not pending:
        return 0

    linked = 0
    for cand_dir, cand_cfg in candidates:
        if not pending:
            break

        source_inputs = inputs_from_run_config(cand_cfg)
        if not _step5_trusted(source_inputs):
            continue

        source_step5 = cand_dir / "step5_loss_1k_tokens"
        if not source_step5.is_dir():
            continue

        source_metadata = _load_step5_metadata(cand_dir)
        source_jobs = _jobs_from_step5_metadata(source_metadata)
        source_loss_method = source_metadata.get("loss_method")
        for job_name in list(pending.keys()):
            job = pending[job_name]
            source_job = source_jobs.get(job.name)
            if not _job_inputs_compatible(
                job=job,
                current_workdir=workdir,
                current_inputs=current_inputs,
                    source_workdir=cand_dir,
                source_inputs=source_inputs,
                source_job=source_job,
                source_loss_method=source_loss_method,
            ):
                continue

            src_file = _find_source_loss_file(
                source_step5=source_step5,
                job=job,
                source_job=source_job,
            )
            if src_file is None:
                continue

            dst_file = output_dir / job.name / job.loss_file
            if _link_loss_file(src_file, dst_file):
                linked += 1
                pending.pop(job_name, None)
                log(
                    "  -> Smart-linked prior step5 loss: "
                    f"{dst_file} -> {src_file} "
                    f"(source run folder: {cand_dir})"
                )

    if linked:
        log(f"[step5-link] Linked {linked} reusable Step 5 loss file(s)")
    return linked
