"""Job model, control state and persistence (Astra ``ingest_jobs`` collection).

One *job document* holds everything about a run: parameters, control state, progress, failures,
recent events, heartbeat and lease. Restart recovery, pause/resume/stop and the UI all derive from it.

Ownership of fields
-------------------
* **Worker-owned** (written only by the lease holder, guarded by ``lease_owner``): ``last_key``,
  ``counters``, ``failures``, ``failures_truncated``, ``in_flight``, lease/heartbeat fields.
* **Control-owned** (``status``, ``desired_state``, ``pause_reason``, ``control_events``, activity timers):
  written by ``JobStore.set_desired_state`` (the API, no lease needed) and by ``JobStore.transition_status`` (the worker).
* ``events`` is recorded by worker and merged with ``control_events`` for rendering.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from .astra_store import AstraStore
from .config import Settings
from .retry import RetryPolicy, with_retry

# --------------------------------------------------------------------------- #
# States and transition rules (pure data / functions)
# --------------------------------------------------------------------------- #
QUEUED, RUNNING, PAUSING, PAUSED = "QUEUED", "RUNNING", "PAUSING", "PAUSED"
STOPPING, STOPPED, COMPLETED, FAILED = "STOPPING", "STOPPED", "COMPLETED", "FAILED"

STATUSES: Tuple[str, ...] = (QUEUED, RUNNING, PAUSING, PAUSED, STOPPING, STOPPED, COMPLETED, FAILED)
TERMINAL_STATUSES = frozenset({STOPPED, COMPLETED, FAILED})
ACTIVE_STATUSES = frozenset({RUNNING, PAUSING, STOPPING})

DESIRED_RUNNING, DESIRED_PAUSED, DESIRED_STOPPED = "RUNNING", "PAUSED", "STOPPED"
DESIRED_STATES: Tuple[str, ...] = (DESIRED_RUNNING, DESIRED_PAUSED, DESIRED_STOPPED)

# status -> statuses it may move to. Terminal states have no outgoing transitions.
ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    QUEUED: frozenset({RUNNING, PAUSED, STOPPED, FAILED}),
    RUNNING: frozenset({PAUSING, STOPPING, COMPLETED, FAILED}),
    PAUSING: frozenset({PAUSED, RUNNING, STOPPING, FAILED}),
    PAUSED: frozenset({RUNNING, STOPPING, STOPPED, FAILED}),
    STOPPING: frozenset({STOPPED, FAILED}),
    STOPPED: frozenset(),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
}

# (status, desired_state) -> target status
_DESIRED_PLAN: Dict[Tuple[str, str], str] = {
    (QUEUED, DESIRED_PAUSED): PAUSED,
    (QUEUED, DESIRED_STOPPED): STOPPED,
    (QUEUED, DESIRED_RUNNING): QUEUED,
    (RUNNING, DESIRED_PAUSED): PAUSING,
    (RUNNING, DESIRED_STOPPED): STOPPING,
    (RUNNING, DESIRED_RUNNING): RUNNING,
    (PAUSING, DESIRED_RUNNING): RUNNING,
    (PAUSING, DESIRED_STOPPED): STOPPING,
    (PAUSING, DESIRED_PAUSED): PAUSING,
    (PAUSED, DESIRED_RUNNING): RUNNING,
    (PAUSED, DESIRED_STOPPED): STOPPED,
    (PAUSED, DESIRED_PAUSED): PAUSED,
    (STOPPING, DESIRED_STOPPED): STOPPING,
    (STOPPED, DESIRED_STOPPED): STOPPED,
}

_DESIRED_FOR_STATUS: Dict[str, str] = {
    QUEUED: DESIRED_RUNNING,
    RUNNING: DESIRED_RUNNING,
    PAUSING: DESIRED_PAUSED,
    PAUSED: DESIRED_PAUSED,
    STOPPING: DESIRED_STOPPED,
    STOPPED: DESIRED_STOPPED,
}

JOB_COUNTERS: Tuple[str, ...] = (
    "scanned", "ingested", "skipped", "failed", "chunks", "archived", "move_failed", "bytes",
)

GUARD_PREFIX = "guard:"
MAX_LIST_LIMIT = 100
MAX_IN_FLIGHT_STORED = 200
MAX_EVENT_CHARS = 300
MAX_REASON_CHARS = 500

_ACTION_TO_DESIRED = {
    "pause": DESIRED_PAUSED,
    "resume": DESIRED_RUNNING,
    "stop": DESIRED_STOPPED,
    DESIRED_PAUSED: DESIRED_PAUSED,
    DESIRED_RUNNING: DESIRED_RUNNING,
    DESIRED_STOPPED: DESIRED_STOPPED,
}


class JobLeaseLost(RuntimeError):
    """Raised when another worker took over or finished a job this worker was working on."""


class InvalidTransition(ValueError):
    """Raised when a status change or user action is not allowed from the job's current state."""

    def __init__(self, message: str, current: str = "", requested: str = ""):
        super().__init__(message)
        self.current = current
        self.requested = requested


class DuplicateActiveJobError(RuntimeError):
    """Raised when a non-terminal job already exists for the same (bucket, prefix)."""

    def __init__(self, existing_job_id: str, status: str, bucket: str, prefix: str):
        super().__init__(
            f"A job is already active for bucket '{bucket}' prefix '{prefix}': job {existing_job_id} ({status}). "
            "Pause or stop it, or wait for it to finish, before starting another."
        )
        self.existing_job_id = existing_job_id
        self.status = status
        self.bucket = bucket
        self.prefix = prefix


class JobStatusChanged(RuntimeError):
    """Raised when a worker's status transition CAS fails because the job status changed."""

    def __init__(self, message: str, expected: str = "", actual: str = ""):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class JobNotFound(LookupError):
    """Raised when a job id does not exist."""


class JobConflict(RuntimeError):
    """Raised when a concurrent update kept winning (caller may simply retry)."""


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def can_transition(current: str, new: str) -> bool:
    return new in ALLOWED_TRANSITIONS.get(current, frozenset())


def validate_transition(current: str, new: str) -> None:
    if current not in ALLOWED_TRANSITIONS:
        raise InvalidTransition(f"unknown job status '{current}'", current, new)
    if new not in ALLOWED_TRANSITIONS:
        raise InvalidTransition(f"unknown job status '{new}'", current, new)
    if not can_transition(current, new):
        if is_terminal(current):
            raise InvalidTransition(
                f"job is already {current} and final; cannot transition to {new}", current, new
            )
        allowed = ", ".join(sorted(ALLOWED_TRANSITIONS[current])) or "none"
        raise InvalidTransition(
            f"transition from {current} to {new} is not allowed (allowed from {current}: {allowed})",
            current, new,
        )


def sources_of(new: str) -> List[str]:
    return sorted(s for s, targets in ALLOWED_TRANSITIONS.items() if new in targets)


class DesiredPlan(NamedTuple):
    status: str
    changed: bool


def plan_desired_state_change(status: str, desired: str) -> DesiredPlan:
    if desired not in DESIRED_STATES:
        raise InvalidTransition(f"unknown desired state '{desired}'", status, desired)
    if status == STOPPED and desired == DESIRED_STOPPED:
        return DesiredPlan(STOPPED, False)
    if is_terminal(status):
        action = "pause" if desired == DESIRED_PAUSED else ("resume" if desired == DESIRED_RUNNING else "stop")
        raise InvalidTransition(f"cannot {action} a job that is already {status}", status, desired)
    if status == STOPPING and desired in (DESIRED_RUNNING, DESIRED_PAUSED):
        raise InvalidTransition("stop cannot be undone", status, desired)
    target = _DESIRED_PLAN.get((status, desired))
    if target is None:
        raise InvalidTransition(f"cannot change desired state to {desired} from {status}", status, desired)
    return DesiredPlan(target, target != status)


def allowed_actions(status: str) -> List[str]:
    if status == RUNNING:
        return ["pause", "stop"]
    if status in (PAUSING, PAUSED):
        return ["resume", "stop"]
    if status == QUEUED:
        return ["pause", "stop"]
    return []


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _parse_iso(ts_str: Optional[str]) -> Optional[float]:
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except Exception:
        return None


def guard_id(bucket: str, prefix: str) -> str:
    return GUARD_PREFIX + hashlib.sha1(f"{bucket}\0{prefix}".encode("utf-8")).hexdigest()[:32]


def _is_duplicate_id_error(exc: BaseException) -> bool:
    for descriptor in getattr(exc, "error_descriptors", None) or []:
        if getattr(descriptor, "error_code", "") == "DOCUMENT_ALREADY_EXISTS":
            return True
    return "DOCUMENT_ALREADY_EXISTS" in str(exc)


def _insert_once(collection: Any, document: Dict[str, Any]) -> bool:
    try:
        collection.insert_one(document)
        return True
    except Exception as exc:  # noqa: BLE001
        if _is_duplicate_id_error(exc):
            return False
        raise


def _event(now: float, message: str, level: str = "info") -> Dict[str, Any]:
    return {"at": _iso(now), "level": level, "message": str(message)[:MAX_EVENT_CHARS]}


# --------------------------------------------------------------------------- #
# In-memory progress tracking helper
# --------------------------------------------------------------------------- #
def record_progress(
    job: Dict[str, Any],
    settings: Settings,
    *,
    last_key: Optional[str] = None,
    counters: Optional[Dict[str, int]] = None,
    failures: Optional[List[Dict[str, Any]]] = None,
    in_flight: Optional[List[Dict[str, Any]]] = None,
    events: Optional[List[Any]] = None,
    now: Optional[Any] = None,
) -> None:
    """Update in-memory job dictionary with progress deltas and capped logs."""
    if last_key is not None:
        job["last_key"] = last_key

    if counters:
        c = job.setdefault("counters", {})
        for name, delta in counters.items():
            if name not in JOB_COUNTERS:
                raise ValueError(f"unknown job counter '{name}' (expected one of {', '.join(JOB_COUNTERS)})")
            c[name] = int(c.get(name, 0)) + int(delta)

    if failures:
        current_failures = job.setdefault("failures", [])
        cap = settings.max_job_failures
        now_ts = _iso(time.time()) if now is None else (now if isinstance(now, str) else _iso(float(now)))
        for item in failures:
            if len(current_failures) < cap:
                err_class = item.get("error_class")
                if err_class is not None:
                    err_class = str(getattr(err_class, "value", err_class))
                entry = {
                    "key": str(item.get("key", "")),
                    "reason": str(item.get("reason", ""))[:MAX_REASON_CHARS],
                    "at": str(item.get("at") or now_ts),
                }
                if err_class:
                    entry["error_class"] = err_class
                current_failures.append(entry)
            else:
                job["failures_truncated"] = int(job.get("failures_truncated", 0)) + 1

    if in_flight is not None:
        job["in_flight"] = [
            {k: item[k] for k in ("key", "size", "started_at") if k in item}
            for item in in_flight[:MAX_IN_FLIGHT_STORED]
        ]

    if events:
        event_list = job.setdefault("events", [])
        now_val = time.time() if now is None else (time.time() if isinstance(now, str) else float(now))
        for ev in events:
            if isinstance(ev, dict):
                event_list.append(ev)
            else:
                event_list.append(_event(now_val, str(ev)))
        cap = settings.max_job_events
        if len(event_list) > cap:
            job["events"] = event_list[-cap:]


# --------------------------------------------------------------------------- #
# Metrics and Rendering
# --------------------------------------------------------------------------- #
def compute_metrics(
    job: Dict[str, Any],
    total_estimate: Optional[int] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    now = time.time() if now is None else now
    c = job.get("counters") or {}
    ingested = int(c.get("ingested", 0))
    skipped = int(c.get("skipped", 0))
    failed = int(c.get("failed", 0))
    bytes_count = int(c.get("bytes", 0))
    done = ingested + skipped + failed

    status = job.get("status", "")
    started_ts = _parse_iso(job.get("started_at")) or _parse_iso(job.get("created_at"))
    finished_ts = _parse_iso(job.get("finished_at"))

    if is_terminal(status) and finished_ts and started_ts:
        elapsed = max(0.0, finished_ts - started_ts)
    elif started_ts:
        elapsed = max(0.0, now - started_ts)
    else:
        elapsed = 0.0

    files_per_sec = round(done / elapsed, 4) if elapsed > 0 and done > 0 else 0.0
    bytes_per_sec = round(bytes_count / elapsed, 2) if elapsed > 0 and bytes_count > 0 else 0.0

    out: Dict[str, Any] = {
        "elapsed_seconds": round(elapsed, 1),
        "files_processed": done,
        "files_per_sec": files_per_sec,
        "bytes_per_sec": bytes_per_sec,
        "total_estimate": total_estimate,
        "remaining_files": None,
        "percent": None,
        "eta_seconds": None,
    }

    if total_estimate is not None and total_estimate > 0:
        remaining = max(total_estimate - done, 0)
        out["remaining_files"] = remaining
        out["percent"] = round(min(100.0, done * 100.0 / total_estimate), 1)
        if files_per_sec > 0 and status == RUNNING:
            out["eta_seconds"] = 0.0 if remaining == 0 else round(remaining / files_per_sec, 1)

    return out


def render_job(
    job: Dict[str, Any],
    total_estimate: Optional[int] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    now = time.time() if now is None else now
    status = job.get("status", "")
    terminal = is_terminal(status)
    lease_until = float(job.get("lease_until") or 0.0)
    worker_alive = not terminal and lease_until > now

    counters = {name: int((job.get("counters") or {}).get(name, 0)) for name in JOB_COUNTERS}

    # Merge events and control_events, sorted by 'at'
    worker_events = list(job.get("events") or [])
    ctrl_events = list(job.get("control_events") or [])
    all_events = worker_events + ctrl_events
    all_events.sort(key=lambda e: str(e.get("at", "")))

    return {
        "id": job.get("_id"),
        "bucket": job.get("bucket"),
        "prefix": job.get("prefix", ""),
        "collection": job.get("collection"),
        "archive_bucket": job.get("archive_bucket"),
        "params": dict(job.get("params") or {}),
        "desired_state": job.get("desired_state"),
        "status": status,
        "pause_reason": job.get("pause_reason", ""),
        "error": job.get("error", ""),
        "is_terminal": terminal,
        "worker_alive": worker_alive,
        "allowed_actions": allowed_actions(status),
        "last_key": job.get("last_key", ""),
        "counters": counters,
        "failures": list(job.get("failures") or []),
        "failures_truncated": int(job.get("failures_truncated") or 0),
        "in_flight": list(job.get("in_flight") or []),
        "events": all_events,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "heartbeat_at": job.get("heartbeat_at"),
        "metrics": compute_metrics(job, total_estimate, now),
    }


# --------------------------------------------------------------------------- #
# JobStore
# --------------------------------------------------------------------------- #
_PROGRESS_KEYS = ("last_key", "counters", "failures", "failures_truncated", "in_flight", "events")
_CONTROL_KEYS = (
    "status", "desired_state", "pause_reason", "control_events",
    "started_at", "finished_at", "lease_owner", "lease_until", "error",
)
_LIST_PROJECTION = {"events": False, "failures": False, "in_flight": False, "control_events": False}


class SetDesiredResult(NamedTuple):
    job: Dict[str, Any]
    changed: bool


class JobStore:
    def __init__(self, store: AstraStore, settings: Settings, clock: Callable[[], float] = time.time):
        self._store = store
        self.store = store
        self.settings = settings
        self.policy = RetryPolicy.from_settings(settings)
        self._clock = clock
        self.clock = clock
        self._lock = threading.RLock()

    @property
    def _jobs(self) -> Any:
        return self._store.get_jobs_collection()

    def _now(self) -> float:
        return float(self._clock())

    def _call(self, action: str, func: Callable[[], Any]) -> Any:
        return with_retry(func, self.policy, action_name=action)

    def _event(self, now: float, message: str, level: str = "info") -> Dict[str, Any]:
        return _event(now, message, level)

    def _merge_control(self, job: Dict[str, Any], doc: Dict[str, Any]) -> None:
        for key in _CONTROL_KEYS:
            if key in doc:
                job[key] = doc[key]

    def create_job(
        self,
        bucket: str,
        prefix: str = "",
        *,
        collection: Optional[str] = None,
        archive_bucket: Optional[str] = None,
        concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        s = self.settings
        bucket = (bucket or "").strip()
        prefix = prefix or ""
        archive = (archive_bucket or s.archive_bucket).strip()
        if not bucket:
            raise ValueError("bucket is required")
        if archive == bucket:
            raise ValueError("archive bucket must differ from the source bucket")
        workers = s.concurrency if concurrency is None else int(concurrency)
        if workers < 1:
            raise ValueError("concurrency must be >= 1")

        job_id = uuid.uuid4().hex[:12]
        now = self._now()
        gid = guard_id(bucket, prefix)
        doc: Dict[str, Any] = {
            "_id": job_id,
            "doc_type": "job",
            "bucket": bucket,
            "prefix": prefix,
            "collection": collection or s.collection_name,
            "archive_bucket": archive,
            "params": {"concurrency": workers},
            "desired_state": DESIRED_RUNNING,
            "status": QUEUED,
            "pause_reason": "",
            "error": "",
            "last_key": "",
            "counters": {name: 0 for name in JOB_COUNTERS},
            "failures": [],
            "failures_truncated": 0,
            "in_flight": [],
            "events": [self._event(now, "Job created")],
            "control_events": [],
            "created_at": _iso(now),
            "updated_at": _iso(now),
            "started_at": None,
            "finished_at": None,
            "heartbeat_at": None,
            "lease_owner": "",
            "lease_until": 0,
            "guard_id": gid,
        }
        self._claim_guard(gid, bucket, prefix, job_id, now)
        try:
            self._call("create job", lambda: _insert_once(self._jobs, doc))
        except Exception:
            self._release_guard(gid, job_id)
            raise
        return doc

    def _claim_guard(self, gid: str, bucket: str, prefix: str, job_id: str, now: float) -> None:
        guard_doc = {
            "_id": gid,
            "doc_type": "guard",
            "bucket": bucket,
            "prefix": prefix,
            "active_job_id": job_id,
            "created_ts": now,
            "created_at": _iso(now),
        }
        for _ in range(3):
            if self._call("claim job guard", lambda: _insert_once(self._jobs, guard_doc)):
                return
            guard = self._call("read job guard", lambda: self._jobs.find_one({"_id": gid}))
            if guard is None:
                continue
            holder_id = guard.get("active_job_id") or guard.get("job_id")
            if holder_id == job_id:
                return
            holder = self.load_job(holder_id)
            if holder is not None and not is_terminal(holder.get("status", "")):
                raise DuplicateActiveJobError(holder_id, holder.get("status", ""), bucket, prefix)
            if holder is None and now - float(guard.get("created_ts") or 0.0) < self.settings.lease_seconds:
                # Creator may be in progress
                raise DuplicateActiveJobError(holder_id, "CREATING", bucket, prefix)
            # Stale guard: take over
            swapped = self._call("take over stale guard", lambda: self._jobs.find_one_and_update(
                {"_id": gid, "active_job_id": holder_id},
                {"$set": {"active_job_id": job_id, "created_ts": now, "created_at": _iso(now)}},
                return_document="after",
            ))
            if swapped is not None:
                return
        raise JobConflict(f"could not claim the job guard for bucket '{bucket}' prefix '{prefix}'; try again")

    def _release_guard(self, gid: str, job_id: str) -> None:
        if not gid:
            return
        try:
            self._call("release job guard", lambda: self._jobs.delete_one({
                "_id": gid,
                "$or": [{"active_job_id": job_id}, {"job_id": job_id}],
            }))
        except Exception:  # noqa: BLE001
            pass

    def load_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job_id = (job_id or "").strip()
        if not job_id:
            return None
        return self._call(f"load job '{job_id}'", lambda: self._jobs.find_one({"_id": job_id, "doc_type": "job"}))

    def list_jobs(self, limit: int = 50, statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        flt: Dict[str, Any] = {"doc_type": "job"}
        if statuses:
            flt["status"] = {"$in": list(statuses)}
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        return self._call("list jobs", lambda: list(self._jobs.find(
            flt, projection=_LIST_PROJECTION, sort={"created_at": -1}, limit=limit
        )))

    def find_active_job(self, bucket: str, prefix: str = "") -> Optional[Dict[str, Any]]:
        gid = guard_id(bucket, prefix or "")
        guard = self._call("read job guard", lambda: self._jobs.find_one({"_id": gid}))
        if not guard:
            return None
        job_id = guard.get("active_job_id") or guard.get("job_id")
        job = self.load_job(job_id)
        return job if job is not None and not is_terminal(job.get("status", "")) else None

    def find_recoverable_jobs(self, include_queued: bool = True, limit: int = 100) -> List[Dict[str, Any]]:
        statuses = [RUNNING, PAUSING, STOPPING] + ([QUEUED] if include_queued else [])
        now = self._now()
        flt = {
            "doc_type": "job",
            "status": {"$in": statuses},
            "$or": [
                {"lease_until": {"$exists": False}},
                {"lease_until": 0},
                {"lease_until": {"$lt": now}},
            ],
        }
        return self._call("find recoverable jobs", lambda: list(self._jobs.find(
            flt, sort={"created_at": 1}, limit=max(1, min(int(limit), MAX_LIST_LIMIT))
        )))

    def acquire_lease(self, job_id: str, owner: Optional[str] = None) -> Optional[Dict[str, Any]]:
        now = self._now()
        owner = owner or uuid.uuid4().hex
        return self._call(f"acquire lease for job '{job_id}'", lambda: self._jobs.find_one_and_update(
            {
                "_id": job_id,
                "doc_type": "job",
                "status": {"$nin": sorted(TERMINAL_STATUSES)},
                "$or": [
                    {"lease_until": {"$exists": False}},
                    {"lease_until": 0},
                    {"lease_until": {"$lt": now}},
                ],
            },
            {
                "$set": {
                    "lease_owner": owner,
                    "lease_until": now + self.settings.lease_seconds,
                    "heartbeat_at": _iso(now),
                    "updated_at": _iso(now),
                    "in_flight": [],
                },
            },
            return_document="after",
        ))

    def _lost(self, job: Dict[str, Any]) -> JobLeaseLost:
        return JobLeaseLost(
            f"job '{job.get('_id')}' was taken over by another worker or has already finished; "
            "this call's progress was not saved"
        )

    def _owned_filter(self, job: Dict[str, Any]) -> Dict[str, Any]:
        if not job.get("lease_owner"):
            raise self._lost(job)
        return {
            "_id": job["_id"],
            "doc_type": "job",
            "lease_owner": job.get("lease_owner"),
            "status": {"$nin": sorted(TERMINAL_STATUSES)},
        }

    def heartbeat(self, job: Dict[str, Any]) -> None:
        now = self._now()
        updated = self._call(f"heartbeat job '{job['_id']}'", lambda: self._jobs.find_one_and_update(
            self._owned_filter(job),
            {"$set": {"lease_until": now + self.settings.lease_seconds, "heartbeat_at": _iso(now)}},
            return_document="after",
        ))
        if updated is None:
            raise self._lost(job)
        job["lease_until"] = now + self.settings.lease_seconds
        job["heartbeat_at"] = _iso(now)
        self._merge_control(job, updated)

    def release_lease(self, job: Dict[str, Any]) -> bool:
        try:
            updated = self._call(f"release lease of job '{job['_id']}'", lambda: self._jobs.find_one_and_update(
                {"_id": job["_id"], "lease_owner": job.get("lease_owner")},
                {"$set": {"lease_owner": "", "lease_until": 0}},
                return_document="after",
            ))
        except Exception:  # noqa: BLE001
            return False
        if updated is not None:
            job["lease_owner"] = ""
            job["lease_until"] = 0
            return True
        return False

    def refresh_control(self, job: Dict[str, Any]) -> None:
        fresh = self._call(f"read job '{job['_id']}'", lambda: self._jobs.find_one({"_id": job["_id"], "doc_type": "job"}))
        if fresh is None or fresh.get("lease_owner") != job.get("lease_owner"):
            raise self._lost(job)
        self._merge_control(job, fresh)

    def checkpoint(self, job: Dict[str, Any], release: bool = False) -> None:
        now = self._now()
        progress = {key: copy.deepcopy(job.get(key)) for key in _PROGRESS_KEYS if key in job}
        # Cap events in progress payload
        if "events" in progress:
            progress["events"] = progress["events"][-self.settings.max_job_events:]
        update: Dict[str, Any] = {
            "$set": {
                **progress,
                "updated_at": _iso(now),
                "heartbeat_at": _iso(now),
                "lease_owner": "" if release else job.get("lease_owner", ""),
                "lease_until": 0 if release else now + self.settings.lease_seconds,
            }
        }
        updated = self._call(f"checkpoint job '{job['_id']}'", lambda: self._jobs.find_one_and_update(
            self._owned_filter(job), update, return_document="after"
        ))
        if updated is None:
            raise self._lost(job)
        self._merge_control(job, updated)
        if release:
            job["lease_owner"] = ""
            job["lease_until"] = 0

    def transition_status(
        self,
        job: Dict[str, Any],
        new_status: str,
        *,
        pause_reason: Optional[str] = None,
        error: str = "",
        event: Optional[str] = None,
        release: bool = False,
    ) -> None:
        if new_status not in ALLOWED_TRANSITIONS:
            raise InvalidTransition(f"unknown job status '{new_status}'", job.get("status", ""), new_status)
        now = self._now()
        current_status = job.get("status", "")
        validate_transition(current_status, new_status)

        progress = {key: copy.deepcopy(job.get(key)) for key in _PROGRESS_KEYS if key in job}
        if "events" in progress:
            progress["events"] = progress["events"][-self.settings.max_job_events:]

        sets: Dict[str, Any] = {
            **progress,
            "status": new_status,
            "updated_at": _iso(now),
            "heartbeat_at": _iso(now),
            "lease_owner": "" if (release or is_terminal(new_status)) else job.get("lease_owner", ""),
            "lease_until": 0 if (release or is_terminal(new_status)) else now + self.settings.lease_seconds,
        }
        if new_status in _DESIRED_FOR_STATUS:
            sets["desired_state"] = _DESIRED_FOR_STATUS[new_status]
        if new_status in ACTIVE_STATUSES and not job.get("started_at"):
            sets["started_at"] = _iso(now)
        if is_terminal(new_status):
            sets["finished_at"] = _iso(now)
        if new_status == RUNNING:
            sets["pause_reason"] = ""
        elif pause_reason is not None:
            sets["pause_reason"] = pause_reason
        if error:
            sets["error"] = error[:MAX_REASON_CHARS]

        ev_msg = event or f"Status {current_status} -> {new_status}"
        new_ev = self._event(now, ev_msg, "error" if new_status == FAILED else "info")
        cur_events = list(job.get("events") or []) + [new_ev]
        sets["events"] = cur_events[-self.settings.max_job_events:]

        flt = self._owned_filter(job)
        flt["status"] = current_status

        updated = self._call(f"move job '{job['_id']}' to {new_status}", lambda: self._jobs.find_one_and_update(
            flt, {"$set": sets}, return_document="after"
        ))
        if updated is None:
            current = self._call(f"read job '{job['_id']}'", lambda: self._jobs.find_one({"_id": job["_id"], "doc_type": "job"}))
            if current is None or current.get("lease_owner") != job.get("lease_owner"):
                raise self._lost(job)
            self._merge_control(job, current)
            if current.get("status") == new_status:
                if is_terminal(new_status):
                    self._release_guard(job.get("guard_id") or guard_id(job.get("bucket", ""), job.get("prefix", "")), job["_id"])
                return
            raise JobStatusChanged(
                f"job status changed from {current_status} to {current.get('status')}",
                expected=current_status,
                actual=current.get("status", ""),
            )

        self._merge_control(job, updated)
        job["events"] = sets["events"]
        if is_terminal(new_status):
            self._release_guard(job.get("guard_id") or guard_id(job.get("bucket", ""), job.get("prefix", "")), job["_id"])

    def set_desired_state(self, job_id: str, action_or_desired: str, reason: str = "") -> SetDesiredResult:
        action_or_desired = (action_or_desired or "").strip()
        desired = _ACTION_TO_DESIRED.get(action_or_desired.lower()) or _ACTION_TO_DESIRED.get(action_or_desired.upper())
        if not desired:
            raise InvalidTransition(f"unknown desired state '{action_or_desired}'", requested=action_or_desired)

        for _ in range(4):
            job = self.load_job(job_id)
            if job is None:
                raise JobNotFound(f"job '{job_id}' not found")
            now = self._now()
            current_status = job.get("status", "")
            plan = plan_desired_state_change(current_status, desired)
            if not plan.changed and job.get("desired_state") == desired:
                return SetDesiredResult(job, False)

            sets: Dict[str, Any] = {
                "status": plan.status,
                "desired_state": desired,
                "updated_at": _iso(now),
            }
            if plan.status == RUNNING:
                sets["pause_reason"] = ""
            elif desired == DESIRED_PAUSED:
                sets["pause_reason"] = reason or "Paused by user"
            elif desired == DESIRED_STOPPED and reason:
                sets["pause_reason"] = reason

            if is_terminal(plan.status):
                sets["finished_at"] = _iso(now)
                sets["lease_owner"] = ""
                sets["lease_until"] = 0

            action_name = "pause" if desired == DESIRED_PAUSED else ("resume" if desired == DESIRED_RUNNING else "stop")
            msg = f"User requested {action_name}" + (f": {reason}" if reason else "")
            new_ctrl_ev = self._event(now, msg)
            ctrl_events = list(job.get("control_events") or []) + [new_ctrl_ev]
            sets["control_events"] = ctrl_events[-self.settings.max_job_events:]

            flt = {
                "_id": job_id,
                "doc_type": "job",
                "status": current_status,
                "desired_state": job.get("desired_state"),
            }
            updated = self._call(f"set desired state on job '{job_id}'", lambda: self._jobs.find_one_and_update(
                flt, {"$set": sets}, return_document="after"
            ))
            if updated is not None:
                if is_terminal(plan.status):
                    self._release_guard(job.get("guard_id") or guard_id(job.get("bucket", ""), job.get("prefix", "")), job_id)
                return SetDesiredResult(updated, True)

        raise JobConflict(f"job '{job_id}' kept changing while applying desired state; try again")
