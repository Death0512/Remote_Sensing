from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional, Union

from src.domain.enums import Capability, ReviewAction, ReviewState


@dataclass(frozen=True)
class ReviewAuditRecord:
    id: int
    alert_id: str
    operator_id: str
    action: ReviewAction
    previous_state: ReviewState
    new_state: ReviewState
    notes: Optional[str]
    timestamp_utc: datetime
    previous_hash: str
    entry_hash: str

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "operator_id": self.operator_id,
            "action": self.action.value if hasattr(self.action, "value") else str(self.action),
            "previous_state": self.previous_state.value if hasattr(self.previous_state, "value") else str(self.previous_state),
            "new_state": self.new_state.value if hasattr(self.new_state, "value") else str(self.new_state),
            "notes": self.notes,
            "timestamp_utc": self.timestamp_utc.isoformat() if hasattr(self.timestamp_utc, "isoformat") else str(self.timestamp_utc),
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }


def _compute_hash(
    entry_id: int,
    alert_id: str,
    operator_id: str,
    action: str,
    prev_state: str,
    new_state: str,
    notes: Optional[str],
    timestamp_str: str,
    previous_hash: str,
) -> str:
    raw = f"{entry_id}|{alert_id}|{operator_id}|{action}|{prev_state}|{new_state}|{notes or ''}|{timestamp_str}|{previous_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ReviewAuditStore:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        db_path: Union[str, Path] = "data/review_audit.db",
        journal_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.is_memory = str(db_path) == ":memory:"
        if self.is_memory:
            self.db_path = ":memory:"
            self.journal_path = Path(journal_path) if journal_path else None
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            if journal_path is not None:
                self.journal_path = Path(journal_path)
            else:
                self.journal_path = self.db_path.with_suffix(".jsonl")
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        if not self.is_memory:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS review_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    notes TEXT,
                    timestamp_utc TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS prevent_audit_update
                BEFORE UPDATE ON review_audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'Audit log is append-only: updates forbidden');
                END;
                """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
                BEFORE DELETE ON review_audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'Audit log is append-only: deletions forbidden');
                END;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_review_state (
                    alert_id TEXT PRIMARY KEY,
                    current_state TEXT NOT NULL,
                    last_updated_utc TEXT NOT NULL
                );
                """
            )

    def get_review_state(self, alert_id: str) -> ReviewState:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT current_state FROM alert_review_state WHERE alert_id = ?", (alert_id,))
            row = cur.fetchone()
            if row is None:
                return ReviewState.PENDING_REVIEW
            val = row[0]
            try:
                return ReviewState(val)
            except ValueError:
                if val == "PENDING":
                    return ReviewState.PENDING_REVIEW
                if val == "UNCERTAIN":
                    return ReviewState.FLAGGED_UNCERTAIN
                return ReviewState(val)

    def record_review(
        self,
        alert_id: str,
        action: Union[ReviewAction, str],
        operator_id: str,
        notes: Optional[str] = None,
        timestamp_utc: Optional[datetime] = None,
    ) -> ReviewAuditRecord:
        act = ReviewAction(action) if isinstance(action, str) else action
        now_utc = timestamp_utc or datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()

        if act == ReviewAction.CONFIRM:
            new_state = ReviewState.CONFIRMED
        elif act == ReviewAction.DISMISS:
            new_state = ReviewState.DISMISSED
        elif act == ReviewAction.FLAG_UNCERTAIN:
            new_state = ReviewState.FLAGGED_UNCERTAIN
        else:
            raise ValueError(f"Unsupported review action: {act}")

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                cur.execute("SELECT current_state FROM alert_review_state WHERE alert_id = ?", (alert_id,))
                state_row = cur.fetchone()
                if state_row is None:
                    current_state = ReviewState.PENDING_REVIEW
                else:
                    val = state_row[0]
                    try:
                        current_state = ReviewState(val)
                    except ValueError:
                        if val == "PENDING":
                            current_state = ReviewState.PENDING_REVIEW
                        elif val == "UNCERTAIN":
                            current_state = ReviewState.FLAGGED_UNCERTAIN
                        else:
                            current_state = ReviewState(val)

                if current_state in (ReviewState.CONFIRMED, ReviewState.DISMISSED):
                    raise ValueError(
                        f"Alert {alert_id} review is finalized ({current_state.value}); modifications forbidden"
                    )

                cur.execute("SELECT id, entry_hash FROM review_audit_log ORDER BY id DESC LIMIT 1")
                last_row = cur.fetchone()
                if last_row is None:
                    next_id = 1
                    prev_hash = "GENESIS"
                else:
                    next_id = last_row[0] + 1
                    prev_hash = last_row[1]

                entry_hash = _compute_hash(
                    entry_id=next_id,
                    alert_id=alert_id,
                    operator_id=operator_id,
                    action=act.value,
                    prev_state=current_state.value,
                    new_state=new_state.value,
                    notes=notes,
                    timestamp_str=now_iso,
                    previous_hash=prev_hash,
                )

                cur.execute(
                    """
                    INSERT INTO review_audit_log (
                        id, alert_id, operator_id, action, previous_state,
                        new_state, notes, timestamp_utc, previous_hash, entry_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        next_id,
                        alert_id,
                        operator_id,
                        act.value,
                        current_state.value,
                        new_state.value,
                        notes,
                        now_iso,
                        prev_hash,
                        entry_hash,
                    ),
                )

                cur.execute(
                    """
                    INSERT INTO alert_review_state (alert_id, current_state, last_updated_utc)
                    VALUES (?, ?, ?)
                    ON CONFLICT(alert_id) DO UPDATE SET
                        current_state = excluded.current_state,
                        last_updated_utc = excluded.last_updated_utc
                    """,
                    (alert_id, new_state.value, now_iso),
                )

                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

        record = ReviewAuditRecord(
            id=next_id,
            alert_id=alert_id,
            operator_id=operator_id,
            action=act,
            previous_state=current_state,
            new_state=new_state,
            notes=notes,
            timestamp_utc=now_utc,
            previous_hash=prev_hash,
            entry_hash=entry_hash,
        )

        if self.journal_path is not None:
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")

        return record

    def record_action(
        self,
        alert_id: str,
        action: Union[ReviewAction, str],
        operator_id: str,
        notes: Optional[str] = None,
        timestamp_utc: Optional[datetime] = None,
    ) -> ReviewAuditRecord:
        return self.record_review(
            alert_id=alert_id,
            action=action,
            operator_id=operator_id,
            notes=notes,
            timestamp_utc=timestamp_utc,
        )

    def get_review_history(self, alert_id: Optional[str] = None) -> list[ReviewAuditRecord]:
        with self._lock:
            cur = self._conn.cursor()
            if alert_id is not None:
                cur.execute(
                    """
                    SELECT id, alert_id, operator_id, action, previous_state,
                           new_state, notes, timestamp_utc, previous_hash, entry_hash
                    FROM review_audit_log
                    WHERE alert_id = ?
                    ORDER BY id ASC
                    """,
                    (alert_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, alert_id, operator_id, action, previous_state,
                           new_state, notes, timestamp_utc, previous_hash, entry_hash
                    FROM review_audit_log
                    ORDER BY id ASC
                    """
                )
            records: list[ReviewAuditRecord] = []
            for row in cur.fetchall():
                ts = datetime.fromisoformat(row[7])
                records.append(
                    ReviewAuditRecord(
                        id=row[0],
                        alert_id=row[1],
                        operator_id=row[2],
                        action=ReviewAction(row[3]),
                        previous_state=ReviewState(row[4]),
                        new_state=ReviewState(row[5]),
                        notes=row[6],
                        timestamp_utc=ts,
                        previous_hash=row[8],
                        entry_hash=row[9],
                    )
                )
            return records

    def get_history(self, alert_id: Optional[str] = None) -> list[ReviewAuditRecord]:
        return self.get_review_history(alert_id)

    def verify_integrity(self) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, alert_id, operator_id, action, previous_state,
                       new_state, notes, timestamp_utc, previous_hash, entry_hash
                FROM review_audit_log
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()
            expected_prev_hash = "GENESIS"
            expected_id = 1
            for row in rows:
                entry_id = row[0]
                alert_id = row[1]
                operator_id = row[2]
                action = row[3]
                prev_state = row[4]
                new_state = row[5]
                notes = row[6]
                ts_str = row[7]
                prev_hash = row[8]
                stored_hash = row[9]

                if entry_id != expected_id:
                    return False

                if prev_hash != expected_prev_hash:
                    return False

                recomputed = _compute_hash(
                    entry_id=entry_id,
                    alert_id=alert_id,
                    operator_id=operator_id,
                    action=action,
                    prev_state=prev_state,
                    new_state=new_state,
                    notes=notes,
                    timestamp_str=ts_str,
                    previous_hash=prev_hash,
                )
                if recomputed != stored_hash:
                    return False

                expected_prev_hash = stored_hash
                expected_id += 1

            return True

    def export_archive(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        target = Path(output_path) if output_path else self.journal_path
        if target is None:
            raise ValueError("No output path specified for archive export")
        target.parent.mkdir(parents=True, exist_ok=True)
        records = self.get_review_history()
        with open(target, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")
        return target

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ReviewAuditStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
