""""""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .reuse_adapters import (
    materialize_step4_5_partial_y,
    materialize_step4_8_rank,
)
from .utils import (
    log,
    get_num_token_for_difficulty,
    get_step7_max_tokens,
    inputs_from_run_config,
)


@dataclass(frozen=True)
class ReuseDecision:
    """"""

    mode: str
    reason: str = ""


class SkipAndLink:
    """"""

    STEP_ORDER = {
        "step1": 1.0, "step2": 2.0, "step3": 3.0, "step4": 4.0,
        "step4_3": 4.3,
        "step4_5": 4.5, "step4_7": 4.7, "step4_8": 4.8,
        "step5": 5.0, "step6": 6.0, "step6_7": 6.5, "step7": 7.0,
        "step8": 8.0, "step9": 9.0,
    }

    def __init__(self, current_workdir: Path, inputs: Dict, scan_root: Optional[Path] = None, force_rerun: bool = False):
        self.current_workdir = current_workdir.absolute()
        self.inputs = inputs
        self.config_file = self.current_workdir / "pipeline_config.json"
        self.config_bak_file = self.current_workdir / "pipeline_config.json.bak"
        self.scan_root = scan_root
        self.force_rerun = force_rerun

        self._linked_steps: set = set()
        self._candidate_cache: Optional[List] = None
        self._active_candidates: Optional[List[Tuple[Path, Dict]]] = None
        self._filtered_steps: set = set()
        self._first_computed_order: Optional[float] = None

        self._current_run_config = self._load_json(self.config_file)
        self._previous_run_config = self._load_json(self.config_bak_file)
        self._bak_removed = False

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _remove_bak_if_needed(self) -> None:
        if self._bak_removed:
            return
        if self.config_bak_file.exists():
            self.config_bak_file.unlink()
            log(f"[RESUME] Removed backup config after reaching first pending step: {self.config_bak_file.name}")
        self._bak_removed = True

    def _is_step_complete(self, base_dir: Path, step_name: str, targets: List[str]) -> bool:
        """"""
        if step_name == "step9":
            eval_type = self.inputs.get("eval_type", self.inputs.get("dataset_type"))
            output_dir = base_dir / "step9_eval"
            filename = "metrics.csv" if eval_type == "m23k" else "evaluation_summary.txt"
            return any(output_dir.rglob(filename))
            
        elif step_name == "step2":
            for target in targets:
                target_path = base_dir / target
                if not target_path.exists():
                    return False
                subfolders = [p for p in target_path.iterdir() if p.is_dir()]
                if not subfolders:
                    return False
                for subfolder in subfolders:
                    if not any(subfolder.glob("*.pt")):
                        return False
            return True

        elif step_name in {"step7", "step6_7"}:
            return any((base_dir / target).exists() for target in targets)
            
        return all((base_dir / target).exists() for target in targets)

    def _process_manual_override(self, step_name: str, targets: List[str]) -> bool:
        """"""
        override_val = self.inputs.get(step_name)
        if not override_val:
            return False

        override_path = Path(override_val).expanduser()

        is_up_to = False
        for k, v in self.inputs.items():
            if k.startswith("up_to_") and v is not None:
                if str(Path(v).expanduser()) == str(override_path):
                    is_up_to = True
                    break

        if not is_up_to and len(targets) > 1:
            raise ValueError(f"Cannot manually override step '{step_name}' using --step because it has multiple output targets. Use --up_to_step instead.")

        log(f"  [{step_name}] Manual override with path: {override_path} (is_up_to={is_up_to})")
        
        linked_any = False
        for target in targets:
            dst_path = self.current_workdir / target
            if dst_path.exists() or dst_path.is_symlink():
                if dst_path.is_symlink() or dst_path.is_file():
                    dst_path.unlink()
                else:
                    shutil.rmtree(dst_path)
            dst_path.parent.mkdir(parents=True, exist_ok=True)        
            
            if is_up_to:
                dst_path.symlink_to((override_path / target).absolute())
                log(f"  -> Linked {target} from {override_path / target}")
            else:
                dst_path.symlink_to(override_path.absolute())
                log(f"  -> Linked {target} directly from {override_path}")
            linked_any = True

        if linked_any:
            self._linked_steps.add(step_name)
            return True
        return False

    def _compat_decision(
        self,
        current_params: Dict[str, Any],
        candidate_params: Dict[str, Any],
        step_name: str,
    ) -> ReuseDecision:
        """"""
        if not current_params:
            return ReuseDecision("exact", "")

        mode_rank = {"exact": 0, "provider": 1, "materialize": 2}
        chosen_mode = "exact"
        reasons: List[str] = []
        for key, current_val in current_params.items():
            cand_val = candidate_params.get(key)

            if key not in candidate_params:
                if key == "num_token_for_difficulty":
                    if current_val <= 100:
                        continue
                    return ReuseDecision(
                        "incompatible",
                        f"Key '{key}' absent in candidate (expected <= 100), current={current_val}",
                    )
                if key == "step7_max_tokens":
                    if current_val == get_step7_max_tokens({}):
                        continue
                    return ReuseDecision(
                        "incompatible",
                        f"Key '{key}' absent in candidate (expected {get_step7_max_tokens({})}), current={current_val}",
                    )
                if key == "step8_cutoff_len":
                    if current_val is None:
                        continue
                    return ReuseDecision(
                        "incompatible",
                        f"Key '{key}' absent in candidate (expected YAML default), current={current_val}",
                    )
                if key == "step9_eval_temperature":
                    try:
                        cur = float(current_val)
                    except (TypeError, ValueError):
                        cur = float("nan")
                    if cur == 0.0:
                        continue
                    return ReuseDecision(
                        "incompatible",
                        f"Key '{key}' absent in candidate (expected 0.0), current={current_val}",
                    )
                if key == "max_repetition":
                    if current_val == 1:
                        continue
                    return ReuseDecision(
                        "incompatible",
                        f"Key '{key}' absent in candidate (expected 1), current={current_val}",
                    )
                if key == "prompt_source_nrows":
                    return ReuseDecision(
                        "incompatible",
                        f"Key '{key}' missing in prior run config; current corpus has {current_val} rows "
                        "(refuse reuse after possible prompt_source edits).",
                    )
                continue

            if step_name == "step1" and key == "num_token_for_difficulty":
                if cand_val < current_val:
                    return ReuseDecision(
                        "incompatible",
                        f"step1 '{key}' in candidate ({cand_val}) is less than current ({current_val})",
                    )
                continue

            if step_name == "step4_5" and key == "partial_y":
                if cand_val < current_val:
                    return ReuseDecision(
                        "incompatible",
                        f"step4_5 '{key}' in candidate ({cand_val}) is less than current ({current_val})",
                    )
                if cand_val > current_val:
                    decision = ReuseDecision(
                        "materialize",
                        f"step4_5 '{key}' can be clipped from candidate={cand_val} to current={current_val}",
                    )
                    if mode_rank[decision.mode] > mode_rank[chosen_mode]:
                        chosen_mode = decision.mode
                    reasons.append(decision.reason)
                continue

            if key == "lora_train_rank" and step_name in {"step4_7", "step4_8"}:
                if cand_val < current_val:
                    return ReuseDecision(
                        "incompatible",
                        f"{step_name} '{key}' in candidate ({cand_val}) is less than current ({current_val})",
                    )
                if cand_val > current_val:
                    mode = "provider" if step_name == "step4_7" else "materialize"
                    verb = "reuse" if mode == "provider" else "narrow"
                    decision = ReuseDecision(
                        mode,
                        f"{step_name} '{key}' can {verb} candidate={cand_val} for current={current_val}",
                    )
                    if mode_rank[decision.mode] > mode_rank[chosen_mode]:
                        chosen_mode = decision.mode
                    reasons.append(decision.reason)
                continue

            if current_val != cand_val:
                return ReuseDecision(
                    "incompatible",
                    f"Mismatch for '{key}': current={current_val}, backup={cand_val}",
                )

        return ReuseDecision(chosen_mode, "; ".join(reasons))

    def _param_mismatch_details(
        self,
        current_params: Dict[str, Any],
        candidate_params: Dict[str, Any],
        step_name: str,
    ) -> List[Tuple[str, Any, Any, str]]:
        """"""
        mismatches: List[Tuple[str, Any, Any, str]] = []
        for key, here_val in current_params.items():
            if key not in candidate_params:
                if key == "num_token_for_difficulty":
                    if here_val is None:
                        continue
                    if here_val <= 100:
                        continue
                    mismatches.append(
                        (key, here_val, 100, "absent-in-candidate (expected 100)")
                    )
                elif key == "step7_max_tokens":
                    implicit = get_step7_max_tokens({})
                    if here_val == implicit:
                        continue
                    mismatches.append(
                        (key, here_val, implicit, f"absent-in-candidate (expected {implicit})")
                    )
                elif key == "step8_cutoff_len":
                    if here_val is None:
                        continue
                    mismatches.append(
                        (key, here_val, None, "absent-in-candidate (expected YAML default)")
                    )
                elif key == "max_repetition":
                    if here_val == 1:
                        continue
                    mismatches.append(
                        (key, here_val, 1, "absent-in-candidate (expected 1)")
                    )
                continue

            there_val = candidate_params.get(key)

            decision = self._compat_decision({key: here_val}, {key: there_val}, step_name)
            if decision.mode == "incompatible":
                mismatches.append((key, here_val, there_val, decision.reason or "incompatible"))

        return mismatches

    def _inputs_from_run_config(self, run_config: Dict) -> Dict[str, Any]:
        return inputs_from_run_config(run_config)

    def _extract_candidate_params(
        self,
        run_config: Dict,
        param_keys: List[str],
        step_name: str,
    ) -> Dict[str, Any]:
        """"""
        candidate_inputs = self._inputs_from_run_config(run_config)
        result: Dict[str, Any] = {}
        for key in param_keys:
            val = candidate_inputs.get(key)
            if key == "num_token_for_difficulty" and val is None:
                val = get_num_token_for_difficulty(candidate_inputs)
            elif key == "step7_max_tokens" and val is None:
                val = get_step7_max_tokens(candidate_inputs)
            result[key] = val
        return result

    def register_step(
        self,
        step_name: str,
        params: Dict[str, Any],
        targets: List[str],
        *,
        side_branch: bool = False,
        side_branch_cascade_anchor: Optional[str] = None,
    ) -> bool:
        """"""
        if side_branch:
            return self._register_side_branch_step(
                step_name, params, targets,
                cascade_anchor=side_branch_cascade_anchor,
            )

        self._filter_active_candidates_for_step(step_name, params, targets)

        if self._process_manual_override(step_name, targets):
            return True

        step_order = self._step_name_to_order(step_name)
        if self._first_computed_order is not None and step_order >= self._first_computed_order:
            log(f"  [{step_name}] Must recompute: depends on freshly computed step (order {self._first_computed_order})")
            self._remove_bak_if_needed()
            return False

        output_complete = self._is_step_complete(self.current_workdir, step_name, targets)

        if output_complete:
            if self._previous_run_config is None:
                self._linked_steps.add(step_name)
                log(f"  [{step_name}] Output already present; skipping")
                return True

            old_inputs = self._inputs_from_run_config(self._previous_run_config)
            old_params = self._extract_candidate_params(
                self._previous_run_config,
                list(params.keys()),
                step_name,
            )
            decision = self._compat_decision(params, old_params, step_name)
            if decision.mode == "incompatible":
                raise ValueError(
                    f"[{step_name}] Existing outputs found but config changed in an incompatible way. "
                    f"Details: {decision.reason}. "
                    f"Current run: {self.current_workdir}. "
                    "This step must remain config-compatible once outputs exist."
                )
            if decision.mode == "materialize":
                self._materialize_step_outputs(
                    step_name=step_name,
                    source_workdir=self.current_workdir,
                    source_inputs=old_inputs,
                )
                self._mark_step_as_computed(step_name)
                self._linked_steps.add(step_name)
                log(f"  [{step_name}] Linked compatible output from existing artifact")
                return True

            self._linked_steps.add(step_name)
            compat_kind = "provider-compatible" if decision.mode == "provider" else "compatible"
            suffix = f" ({decision.reason})" if decision.reason else ""
            log(f"  [{step_name}] Output already present and {compat_kind}; skipping{suffix}")
            return True

        if self._try_auto_discover(step_name, params, targets):
            return True

        self._mark_step_as_computed(step_name)
        return False

    def _register_side_branch_step(
        self,
        step_name: str,
        params: Dict[str, Any],
        targets: List[str],
        *,
        cascade_anchor: Optional[str],
    ) -> bool:
        """"""
        if self._process_manual_override(step_name, targets):
            return True

        step_order = self._step_name_to_order(step_name)
        if self._first_computed_order is not None and step_order >= self._first_computed_order:
            log(
                f"  [{step_name}] (side branch) Must recompute: depends on freshly "
                f"computed step (order {self._first_computed_order})"
            )
            self._remove_bak_if_needed()
            self._mark_side_branch_computed(step_name, cascade_anchor)
            return False

        output_complete = self._is_step_complete(self.current_workdir, step_name, targets)
        if output_complete:
            if self._previous_run_config is None:
                self._linked_steps.add(step_name)
                log(f"  [{step_name}] (side branch) Output already present; skipping")
                return True

            old_params = self._extract_candidate_params(
                self._previous_run_config,
                list(params.keys()),
                step_name,
            )
            decision = self._compat_decision(params, old_params, step_name)
            if decision.mode == "incompatible":
                raise ValueError(
                    f"[{step_name}] (side branch) Existing outputs found but config changed "
                    f"in an incompatible way. Details: {decision.reason}. "
                    f"Current run: {self.current_workdir}. "
                    "This step must remain config-compatible once outputs exist."
                )
            self._linked_steps.add(step_name)
            compat_kind = "provider-compatible" if decision.mode == "provider" else "compatible"
            suffix = f" ({decision.reason})" if decision.reason else ""
            log(f"  [{step_name}] (side branch) Output already present and {compat_kind}; skipping{suffix}")
            return True

        local_candidates: List[Tuple[Path, Dict]] = []
        if not self.force_rerun:
            source = self._get_active_candidates()
            if source:
                kept, (trust_reject, target_reject, param_reject) = self._compute_filtered_candidates(
                    step_name, params, targets, source,
                    log_label="AUTO-DISCOVER[side-branch]",
                )
                log(
                    f"[AUTO-DISCOVER][side-branch] {step_name}: local pool "
                    f"{len(source)} -> {len(kept)} "
                    f"(trust_reject={trust_reject}, target_reject={target_reject}, param_reject={param_reject})"
                )
                local_candidates = kept

        if local_candidates and self._try_auto_discover(
            step_name, params, targets, candidates=local_candidates
        ):
            return True

        self._mark_side_branch_computed(step_name, cascade_anchor)
        return False

    def link_step5_checkpoint(self, out_name: str, entry_path: str) -> bool:
        """"""
        step5_dst = self.current_workdir / "step5_loss_1k_tokens" / out_name
        return step5_dst.exists()

    def _extract_step_number(self, step_name: str) -> Optional[int]:
        """"""
        if step_name.startswith("step") and step_name[4:5].isdigit():
            try:
                return int(step_name[4])
            except ValueError:
                pass
        return None


    def _step_name_to_order(self, step_name: str) -> float:
        """"""
        return self.STEP_ORDER.get(step_name, float("inf"))

    def _scan_candidate_runs(self) -> List[Tuple[Path, Dict]]:
        """"""
        if self._candidate_cache is not None:
            return self._candidate_cache

        if self.scan_root is None:
            self._candidate_cache = []
            return self._candidate_cache

        log(f"[AUTO-DISCOVER] Scanning for prior runs under {self.scan_root} ...")
        candidates = []
        for config_path in self.scan_root.rglob("pipeline_config.json"):
            if config_path.name != "pipeline_config.json":
                continue
            candidate_dir = config_path.parent.absolute()
            if candidate_dir == self.current_workdir:
                continue
            cfg = self._load_json(config_path)
            if cfg is not None:
                candidates.append((candidate_dir, cfg))

        self._candidate_cache = candidates
        log(f"[AUTO-DISCOVER] Found {len(candidates)} candidate run(s)")
        return candidates

    def _get_active_candidates(self) -> List[Tuple[Path, Dict]]:
        """"""
        if self._active_candidates is None:
            self._active_candidates = list(self._scan_candidate_runs())
        return self._active_candidates

    def _compute_filtered_candidates(
        self,
        step_name: str,
        params: Dict[str, Any],
        targets: List[str],
        candidates: List[Tuple[Path, Dict]],
        *,
        log_label: str = "AUTO-DISCOVER",
    ) -> Tuple[List[Tuple[Path, Dict]], Tuple[int, int, int]]:
        """"""
        kept: List[Tuple[Path, Dict]] = []
        trust_reject = 0
        target_reject = 0
        param_reject = 0

        for cand_dir, cand_cfg in candidates:
            if not self._is_step_trusted(step_name, cand_cfg):
                trust_reject += 1
                continue

            if not self._is_step_complete(cand_dir, step_name, targets):
                target_reject += 1
                continue

            cand_params = self._extract_candidate_params(
                cand_cfg,
                list(params.keys()),
                step_name,
            )
            decision = self._compat_decision(params, cand_params, step_name)
            if decision.mode == "incompatible":
                param_reject += 1
                mismatches = self._param_mismatch_details(params, cand_params, step_name)
                if mismatches:
                    for key, here_val, there_val, note in mismatches:
                        log(
                            f"[{log_label}][param_reject] step={step_name} run={cand_dir} "
                            f"key={key} there={there_val!r} here={here_val!r} ({note})"
                        )
                else:
                    log(
                        f"[{log_label}][param_reject] step={step_name} run={cand_dir} "
                        "reason=unknown (params considered incompatible)"
                    )
                continue

            kept.append((cand_dir, cand_cfg))

        return kept, (trust_reject, target_reject, param_reject)

    def _filter_active_candidates_for_step(
        self,
        step_name: str,
        params: Dict[str, Any],
        targets: List[str],
    ) -> None:
        """"""
        if self.force_rerun:
            return
        if step_name in self._filtered_steps:
            return

        candidates = self._get_active_candidates()
        if not candidates:
            self._filtered_steps.add(step_name)
            return

        kept, (trust_reject, target_reject, param_reject) = self._compute_filtered_candidates(
            step_name, params, targets, candidates, log_label="AUTO-DISCOVER"
        )

        before = len(candidates)
        after = len(kept)
        self._active_candidates = kept
        self._filtered_steps.add(step_name)
        log(
            f"[AUTO-DISCOVER] {step_name}: candidate pool {before} -> {after} "
            f"(trust_reject={trust_reject}, target_reject={target_reject}, param_reject={param_reject})"
        )

    def _is_step_trusted(self, step_name: str, candidate_config: Dict) -> bool:
        """"""
        candidate_inputs = self._inputs_from_run_config(candidate_config)
        override_steps = candidate_inputs.get("manual_override_steps", []) or []
        if not override_steps:
            return True

        min_override_order = min(
            self._step_name_to_order(s) for s in override_steps
        )
        return self._step_name_to_order(step_name) < min_override_order

    def _try_auto_discover(
        self,
        step_name: str,
        params: Dict[str, Any],
        targets: List[str],
        *,
        candidates: Optional[List[Tuple[Path, Dict]]] = None,
    ) -> bool:
        """"""
        if self.force_rerun:
            return False
        if step_name in {"step6_7", "step7"}:
            log(f"  [{step_name}] Auto-discovery handled inside Step 7 token-budget reuse")
            return False

        if candidates is None:
            candidates = self._get_active_candidates()
        if not candidates:
            return False

        best: Optional[Tuple[Tuple[int, int], Path, Dict, ReuseDecision]] = None

        for cand_dir, cand_cfg in candidates:
            if not self._is_step_complete(cand_dir, step_name, targets):
                continue

            cand_params = self._extract_candidate_params(
                cand_cfg,
                list(params.keys()),
                step_name,
            )
            decision = self._compat_decision(params, cand_params, step_name)
            if decision.mode == "incompatible":
                continue

            mode_order = {"exact": 0, "provider": 1, "materialize": 2}[decision.mode]
            symlink_penalty = 1 if any((cand_dir / t).is_symlink() for t in targets) else 0
            score = (mode_order, symlink_penalty)
            if best is None or score < best[0]:
                best = (score, cand_dir, cand_cfg, decision)

        if best is None:
            return False

        _, chosen, chosen_cfg, decision = best

        if decision.mode == "materialize":
            self._materialize_step_outputs(
                step_name=step_name,
                source_workdir=chosen,
                source_inputs=self._inputs_from_run_config(chosen_cfg),
            )
            self._mark_step_as_computed(step_name)
            self._linked_steps.add(step_name)
            log(
                f"  [{step_name}] Auto-discovered richer output and materialized "
                f"a local artifact from: {chosen}"
            )
            return True

        for target in targets:
            dst = self.current_workdir / target
            src = chosen / target
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            dst.symlink_to(src)

        kind = "symlinked" if any((chosen / t).is_symlink() for t in targets) else "real"
        self._linked_steps.add(step_name)
        compat_kind = "provider-compatible" if decision.mode == "provider" else "compatible"
        suffix = f" ({decision.reason})" if decision.reason else ""
        log(f"  [{step_name}] Auto-discovered {compat_kind} output ({kind}) from: {chosen}{suffix}")
        return True

    def _mark_step_as_computed(self, step_name: str) -> None:
        step_order = self._step_name_to_order(step_name)
        if self._first_computed_order is None:
            self._first_computed_order = step_order
        self._remove_bak_if_needed()

    def _mark_side_branch_computed(
        self,
        step_name: str,
        cascade_anchor: Optional[str],
    ) -> None:
        """"""
        if cascade_anchor is not None:
            anchor_order = self._step_name_to_order(cascade_anchor)
            if self._first_computed_order is None or anchor_order < self._first_computed_order:
                self._first_computed_order = anchor_order
        self._remove_bak_if_needed()

    def _materialize_step_outputs(
        self,
        *,
        step_name: str,
        source_workdir: Path,
        source_inputs: Dict[str, Any],
    ) -> None:
        if step_name == "step4_5":
            materialize_step4_5_partial_y(
                source_workdir=source_workdir,
                target_workdir=self.current_workdir,
                current_inputs=self.inputs,
                source_inputs=source_inputs,
            )
            return
        if step_name == "step4_8":
            materialize_step4_8_rank(
                source_workdir=source_workdir,
                target_workdir=self.current_workdir,
                current_inputs=self.inputs,
                source_inputs=source_inputs,
            )
            return
        raise ValueError(f"No materializer registered for step {step_name!r}")
