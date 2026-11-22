""""""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils import WORKDIR_BASE, get_step7_max_tokens


APPROVED = "approved"
DENIED = "denied"
INCOMPLETE = "incomplete"
PENDING = "pending"
STATUSES = {APPROVED, DENIED, INCOMPLETE, PENDING}
SCHEMA_VERSION = 3


class PrefixMismatchError(ValueError):
    """"""


def canonical_teacher_name(teacher: str) -> str:
    """"""
    leaf = str(teacher).rstrip("/").split("/")[-1] or str(teacher)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", leaf)


def generation_store_path(task: str, teacher: str, base_dir: Path = WORKDIR_BASE) -> Path:
    return base_dir / f"{task}-{canonical_teacher_name(teacher)}" / "generation_store.sqlite"


def assistant_message(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    conversations = row.get("conversations") or []
    for msg in reversed(conversations):
        role = str(msg.get("role", msg.get("from", ""))).lower()
        if role == "assistant":
            return msg
    return None


def assistant_content(row: Mapping[str, Any]) -> str:
    msg = assistant_message(row)
    if msg is None:
        return ""
    return str(msg.get("content", msg.get("value", "")))


def step4_prefix_hash(step4_row: Mapping[str, Any]) -> str:
    return text_hash(assistant_content(step4_row))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trace_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text_hash(payload)


def _json_dumps(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _status_rank(status: str) -> int:
    return {PENDING: 0, INCOMPLETE: 1, DENIED: 2, APPROVED: 3}[status]


def _chunked(values: Sequence[int], size: int = 500) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


@dataclass(frozen=True)
class InsertedAttempt:
    attempt_id: int
    inserted: bool
    status: str


class GenerationStore:
    """"""

    def __init__(
        self,
        path: Path,
        *,
        task: str,
        teacher: str,
        prompt_source: str,
        create: bool = True,
        busy_timeout_ms: int = 30000,
    ):
        self.path = Path(path)
        self.task = str(task)
        self.teacher = str(teacher)
        self.teacher_name = canonical_teacher_name(teacher)
        self.prompt_source = str(prompt_source)
        self.busy_timeout_ms = int(busy_timeout_ms)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=max(1.0, busy_timeout_ms / 1000.0))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if create:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()

    @classmethod
    def for_inputs(cls, inputs: Mapping[str, Any], *, base_dir: Path = WORKDIR_BASE) -> "GenerationStore":
        task = str(inputs["task"])
        teacher = str(inputs["teacher"])
        prompt_source = str(inputs["prompt_source"])
        return cls(
            generation_store_path(task, teacher, base_dir=base_dir),
            task=task,
            teacher=teacher,
            prompt_source=prompt_source,
        )

    @classmethod
    def for_run(cls, config: Mapping[str, Any], args, *, base_dir: Path = WORKDIR_BASE) -> "GenerationStore":
        inputs = dict(config)
        inputs.setdefault("task", getattr(args, "task"))
        inputs.setdefault("teacher", getattr(args, "teacher"))
        inputs.setdefault("prompt_source", config.get("prompt_source"))
        return cls.for_inputs(inputs, base_dir=base_dir)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "GenerationStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """"""
            )
            self._ensure_generated_continuation_hash_column()
            self._ensure_status_schema()
            self._ensure_generation_attempt_indexes()
            self.conn.execute(
                "INSERT OR REPLACE INTO store_metadata(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO store_metadata(key, value) VALUES (?, ?)",
                ("task", self.task),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO store_metadata(key, value) VALUES (?, ?)",
                ("teacher", self.teacher),
            )

    def _ensure_status_schema(self) -> None:
        row = self.conn.execute(
            """"""
        ).fetchone()
        if row is None or "incomplete" in str(row["sql"]):
            return

        self.conn.executescript(
            """"""
        )

    def _ensure_generated_continuation_hash_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(generation_attempts)")
        }
        if "generated_continuation_hash" not in columns:
            self.conn.execute(
                "ALTER TABLE generation_attempts ADD COLUMN generated_continuation_hash TEXT"
            )
        rows = self.conn.execute(
            """"""
        ).fetchall()
        for row in rows:
            self.conn.execute(
                """""",
                (text_hash(str(row["generated_continuation"])), int(row["attempt_id"])),
            )

    def _ensure_generation_attempt_indexes(self) -> None:
        self.conn.execute("DROP INDEX IF EXISTS idx_generation_attempt_identity")
        self.conn.execute(
            """"""
        )

    def validate_trace_extends_step4(
        self,
        trace_row: Mapping[str, Any],
        step4_row: Mapping[str, Any],
    ) -> Tuple[str, str]:
        prefix = assistant_content(step4_row)
        full_content = assistant_content(trace_row)
        if not prefix:
            raise PrefixMismatchError("Step 4 row has no assistant prefix")
        if not full_content.startswith(prefix):
            raise PrefixMismatchError("Trace assistant content does not extend Step 4 prefix")
        return prefix, full_content[len(prefix):]

    def insert_attempt_from_trace(
        self,
        *,
        trace_row: Mapping[str, Any],
        step4_row: Mapping[str, Any],
        global_index: Optional[int] = None,
        requested_max_new_tokens: Optional[int] = None,
        status: str = PENDING,
        source_run: str = "",
        source_path: str = "",
        generated_tokens: Optional[int] = None,
    ) -> InsertedAttempt:
        if status not in STATUSES:
            raise ValueError(f"Invalid generation status: {status!r}")

        idx = trace_row.get("original_index", global_index)
        if idx is None:
            raise ValueError("Trace row has no original_index and no fallback global_index")
        idx = int(idx)

        prefix, generated_continuation = self.validate_trace_extends_step4(trace_row, step4_row)
        prefix_hash = text_hash(prefix)
        generated_continuation_hash = text_hash(generated_continuation)
        row_trace_hash = trace_hash(trace_row)
        trace_json = _json_dumps(trace_row)
        requested = (
            int(requested_max_new_tokens)
            if requested_max_new_tokens is not None
            else get_step7_max_tokens({"step7_max_tokens": None})
        )
        if generated_tokens is None:
            step4_tokens = step4_row.get("max_tokens", 0)
            trace_tokens = trace_row.get("max_tokens", 0)
            if not isinstance(step4_tokens, int):
                step4_tokens = 0
            if not isinstance(trace_tokens, int):
                trace_tokens = 0
            generated_tokens = max(0, int(trace_tokens) - int(step4_tokens))

        prompt = trace_row.get("prompt") or step4_row.get("prompt")
        source_run = str(source_run or "")
        source_path = str(source_path or "")

        columns = (
            "task, teacher, teacher_name, prompt_source, global_index, prompt, "
            "step4_prefix, step4_prefix_hash, generated_continuation, generated_continuation_hash, generated_tokens, "
            "trace_json, trace_hash, requested_max_new_tokens, status, source_run, source_path"
        )
        values = (
            self.task,
            self.teacher,
            self.teacher_name,
            self.prompt_source,
            idx,
            prompt,
            prefix,
            prefix_hash,
            generated_continuation,
            generated_continuation_hash,
            int(generated_tokens),
            trace_json,
            row_trace_hash,
            requested,
            status,
            source_run,
            source_path,
        )
        placeholders = ", ".join("?" for _ in values)

        with self.conn:
            current = self.conn.execute(
                """""",
                (
                    self.task,
                    self.teacher,
                    self.prompt_source,
                    idx,
                    prefix_hash,
                    generated_continuation_hash,
                ),
            ).fetchone()
            if current is not None:
                update_parts = [
                    "requested_max_new_tokens = ?",
                    "source_run = ?",
                    "source_path = ?",
                    "updated_at = CURRENT_TIMESTAMP",
                ]
                update_params: List[Any] = [requested, source_run, source_path]
                if _status_rank(status) > _status_rank(str(current["status"])):
                    update_parts.insert(0, "status = ?")
                    update_params.insert(0, status)
                update_params.append(int(current["attempt_id"]))
                self.conn.execute(
                    f"""
                    UPDATE generation_attempts
                    SET {", ".join(update_parts)}
                    WHERE attempt_id = ?
                    """,
                    update_params,
                )
                attempt_id = int(current["attempt_id"])
                inserted = False
            else:
                cur = self.conn.execute(
                    f"INSERT INTO generation_attempts ({columns}) VALUES ({placeholders})",
                    values,
                )
                attempt_id = int(cur.lastrowid)
                inserted = True

        row = self.conn.execute(
            "SELECT status FROM generation_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        return InsertedAttempt(attempt_id=attempt_id, inserted=inserted, status=str(row["status"]))

    def update_status(
        self,
        attempt_id: int,
        status: str,
        *,
        source_run: str = "",
        source_path: str = "",
    ) -> None:
        if status not in STATUSES:
            raise ValueError(f"Invalid generation status: {status!r}")
        with self.conn:
            current = self.conn.execute(
                "SELECT status FROM generation_attempts WHERE attempt_id = ?",
                (int(attempt_id),),
            ).fetchone()
            if current is None:
                raise KeyError(f"No generation attempt with attempt_id={attempt_id}")

            update_parts = ["updated_at = CURRENT_TIMESTAMP"]
            params: List[Any] = []
            if _status_rank(status) > _status_rank(str(current["status"])):
                update_parts.insert(0, "status = ?")
                params.insert(0, status)
            if source_run:
                update_parts.append("source_run = ?")
                params.append(str(source_run))
            if source_path:
                update_parts.append("source_path = ?")
                params.append(str(source_path))
            params.append(int(attempt_id))
            cur = self.conn.execute(
                f"""
                UPDATE generation_attempts
                SET {", ".join(update_parts)}
                WHERE attempt_id = ?
                """,
                params,
            )
            if cur.rowcount != 1:
                raise KeyError(f"No generation attempt with attempt_id={attempt_id}")

    def fetch_attempts_for_prefixes(self, prefixes_by_idx: Mapping[int, str]) -> List[Dict[str, Any]]:
        if not prefixes_by_idx:
            return []

        wanted = {int(idx): text_hash(prefix) for idx, prefix in prefixes_by_idx.items()}
        rows: List[Dict[str, Any]] = []
        indices = sorted(wanted)
        for chunk in _chunked(indices):
            placeholders = ", ".join("?" for _ in chunk)
            query = f"""
                SELECT *
                FROM generation_attempts
                WHERE task = ? AND teacher = ? AND prompt_source = ?
                  AND global_index IN ({placeholders})
                ORDER BY global_index ASC, updated_at DESC, attempt_id DESC
            """
            params: List[Any] = [self.task, self.teacher, self.prompt_source]
            params.extend(int(idx) for idx in chunk)
            for row in self.conn.execute(query, params):
                idx = int(row["global_index"])
                if row["step4_prefix_hash"] == wanted.get(idx):
                    rows.append(dict(row))
        return rows

    def denied_counts(self, prefixes_by_idx: Mapping[int, str]) -> Dict[int, int]:
        if not prefixes_by_idx:
            return {}

        wanted = {int(idx): text_hash(prefix) for idx, prefix in prefixes_by_idx.items()}
        counts: Dict[int, int] = {}
        indices = sorted(wanted)
        for chunk in _chunked(indices):
            placeholders = ", ".join("?" for _ in chunk)
            query = f"""
                SELECT global_index, step4_prefix_hash
                FROM generation_attempts
                WHERE task = ? AND teacher = ? AND prompt_source = ?
                  AND global_index IN ({placeholders})
                  AND status = ?
            """
            params: List[Any] = [self.task, self.teacher, self.prompt_source]
            params.extend(int(idx) for idx in chunk)
            params.append(DENIED)
            for row in self.conn.execute(query, params):
                idx = int(row["global_index"])
                if row["step4_prefix_hash"] == wanted.get(idx):
                    counts[idx] = counts.get(idx, 0) + 1
        return counts

    def status_indices(self, status: str, prefixes_by_idx: Mapping[int, str]) -> set[int]:
        if status not in STATUSES:
            raise ValueError(f"Invalid generation status: {status!r}")
        return {
            int(row["global_index"])
            for row in self.fetch_attempts_for_prefixes(prefixes_by_idx)
            if row["status"] == status
        }

    def claim_indices(
        self,
        prefixes_by_idx: Mapping[int, str],
        indices: Iterable[int],
        *,
        owner: str,
        lease_seconds: int = 3600,
    ) -> Tuple[List[int], List[int]]:
        now = time.time()
        expires_at = now + int(lease_seconds)
        claimed: List[int] = []
        blocked: List[int] = []

        with self.conn:
            self.conn.execute("DELETE FROM generation_claims WHERE expires_at <= ?", (now,))
            for idx_raw in indices:
                idx = int(idx_raw)
                prefix = prefixes_by_idx.get(idx)
                if prefix is None:
                    blocked.append(idx)
                    continue
                prefix_hash = text_hash(prefix)
                self.conn.execute(
                    """""",
                    (
                        self.task,
                        self.teacher,
                        self.prompt_source,
                        idx,
                        prefix_hash,
                        owner,
                        now,
                        expires_at,
                        now,
                    ),
                )
                row = self.conn.execute(
                    """""",
                    (self.task, self.teacher, self.prompt_source, idx, prefix_hash),
                ).fetchone()
                if row is not None and row["claimed_by"] == owner:
                    claimed.append(idx)
                else:
                    blocked.append(idx)
        return claimed, blocked

    def release_claims(
        self,
        prefixes_by_idx: Mapping[int, str],
        indices: Iterable[int],
        *,
        owner: str,
    ) -> None:
        with self.conn:
            for idx_raw in indices:
                idx = int(idx_raw)
                prefix = prefixes_by_idx.get(idx)
                if prefix is None:
                    continue
                self.conn.execute(
                    """""",
                    (
                        self.task,
                        self.teacher,
                        self.prompt_source,
                        idx,
                        text_hash(prefix),
                        owner,
                    ),
                )
