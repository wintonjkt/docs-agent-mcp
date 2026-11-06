"""Background job controller: execution loop, pause/resume/stop, circuit breaker, recovery."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from .astra_store import AstraStore, get_existing_doc_etags
from .config import Settings
from .cos_client import discover_cos_batch
from .jobs import (
    COMPLETED,
    DESIRED_PAUSED,
    DESIRED_RUNNING,
    DESIRED_STOPPED,
    FAILED,
    PAUSED,
    PAUSING,
    RUNNING,
    STOPPED,
    STOPPING,
    JobConflict,
    JobLeaseLost,
    JobNotFound,
    JobStatusChanged,
    JobStore,
    is_terminal,
    record_progress,
)
from .pipeline import FileOutcome, process_batch_parallel

logger = logging.getLogger("astra_ingest.controller")

DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5


class CircuitBreaker:
    """Tracks consecutive transient/throttled failures to pause before exhausting budgets."""

    def __init__(self, threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD):
        self.threshold = max(1, threshold)
        self.consecutive_failures = 0
        self.last_error = ""

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self, outcome: FileOutcome) -> None:
        if outcome.is_permanent_failure:
            # Permanent errors (empty file, unsupported format, etc.) do not trip the breaker
            return
        self.consecutive_failures += 1
        self.last_error = outcome.error or outcome.move_reason or "unknown transient error"

    def is_tripped(self) -> bool:
        return self.consecutive_failures >= self.threshold

    def trip_reason(self) -> str:
        return f"circuit breaker: {self.threshold} consecutive failures ({self.last_error})"


class JobWorker(threading.Thread):
    """Background worker thread executing one ingestion job until paused, stopped, or completed."""

    def __init__(
        self,
        job_id: str,
        job_store: JobStore,
        astra_store: AstraStore,
        cos_client: Any,
        settings: Settings,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    ):
        super().__init__(name=f"worker-{job_id}", daemon=True)
        self.job_id = job_id
        self.job_store = job_store
        self.astra_store = astra_store
        self.cos_client = cos_client
        self.settings = settings
        self.circuit_breaker = CircuitBreaker(circuit_breaker_threshold)
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        js = self.job_store
        settings = self.settings

        # 1. Acquire lease
        job = js.acquire_lease(self.job_id)
        if not job:
            logger.warning("Could not acquire lease for job %s; another worker may own it", self.job_id)
            return

        bucket = job.get("bucket", "")
        prefix = job.get("prefix", "")
        collection_name = job.get("collection") or settings.collection_name
        archive_bucket = job.get("archive_bucket")
        collection = self.astra_store.get_collection(collection_name)
        concurrency = int((job.get("params") or {}).get("concurrency") or settings.concurrency)

        try:
            if job.get("status") != RUNNING:
                js.transition_status(job, RUNNING)
        except (JobLeaseLost, JobStatusChanged, Exception) as exc:
            logger.error("Failed to start job %s: %s", self.job_id, exc)
            return

        last_heartbeat = time.time()
        heartbeat_interval = max(5.0, settings.lease_seconds / 3.0)

        # Main processing loop
        while not self._stop_requested.is_set():
            # Refresh control state to see user actions (pause/resume/stop)
            try:
                js.refresh_control(job)
            except JobLeaseLost:
                logger.warning("Lease lost for job %s; stopping worker", self.job_id)
                return

            desired = job.get("desired_state", DESIRED_RUNNING)
            if desired == DESIRED_PAUSED:
                reason = job.get("pause_reason") or "Paused by user"
                try:
                    js.transition_status(job, PAUSED, pause_reason=reason)
                except Exception as exc:
                    logger.warning("Error pausing job %s: %s", self.job_id, exc)
                return

            if desired == DESIRED_STOPPED:
                reason = job.get("pause_reason") or "Stopped by user"
                try:
                    js.transition_status(job, STOPPED, pause_reason=reason)
                except Exception as exc:
                    logger.warning("Error stopping job %s: %s", self.job_id, exc)
                return

            # Heartbeat check
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                try:
                    js.heartbeat(job)
                    last_heartbeat = now
                except JobLeaseLost:
                    logger.warning("Heartbeat failed; lease lost for job %s", self.job_id)
                    return

            # Discover next batch from COS
            checkpoint = job.get("last_key", "")
            try:
                items, next_checkpoint, has_more = discover_cos_batch(
                    self.cos_client,
                    settings,
                    bucket_name=bucket,
                    prefix=prefix,
                    checkpoint=checkpoint,
                    target_count=settings.list_page_size,
                )
            except Exception as exc:
                logger.error("Discovery failed for job %s: %s", self.job_id, exc)
                try:
                    js.transition_status(job, FAILED, error=f"Discovery failed: {exc}")
                except Exception:
                    pass
                return

            if not items and not has_more:
                # Prefix completed!
                try:
                    js.transition_status(job, COMPLETED)
                except Exception as exc:
                    logger.warning("Failed to mark job %s COMPLETED: %s", self.job_id, exc)
                return

            if items:
                # Query smart skip ETags
                existing_etags: Dict[str, str] = {}
                try:
                    source_names = [it["key"] for it in items]
                    existing_etags = get_existing_doc_etags(collection, settings, source_names, bucket_name=bucket)
                except Exception as exc:
                    logger.warning("Smart skip check failed for batch: %s", exc)

                # Set in-flight items
                now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                in_flight_list = [{"key": it["key"], "size": it.get("size", 0), "started_at": now_str} for it in items]
                record_progress(job, settings, in_flight=in_flight_list)

                # Process batch in parallel
                outcomes: List[FileOutcome] = process_batch_parallel(
                    self.cos_client,
                    collection,
                    settings,
                    items,
                    src_bucket=bucket,
                    archive_bucket=archive_bucket,
                    existing_etags=existing_etags,
                    max_workers=concurrency,
                )

                # Compute batch deltas
                deltas = {
                    "scanned": len(items),
                    "ingested": 0,
                    "skipped": 0,
                    "failed": 0,
                    "chunks": 0,
                    "archived": 0,
                    "move_failed": 0,
                    "bytes": 0,
                }
                batch_failures: List[Dict[str, Any]] = []

                for out in outcomes:
                    if out.status == "ingested":
                        deltas["ingested"] += 1
                        deltas["chunks"] += out.chunks
                        deltas["bytes"] += out.bytes_count
                        if out.moved:
                            deltas["archived"] += 1
                        elif out.move_reason:
                            deltas["move_failed"] += 1
                            batch_failures.append({
                                "key": out.key,
                                "reason": f"move: {out.move_reason}",
                                "error_class": out.error_class,
                            })
                        self.circuit_breaker.record_success()

                    elif out.status == "skipped":
                        deltas["skipped"] += 1
                        if out.moved:
                            deltas["archived"] += 1
                        elif out.move_reason:
                            deltas["move_failed"] += 1
                            batch_failures.append({
                                "key": out.key,
                                "reason": f"move: {out.move_reason}",
                                "error_class": out.error_class,
                            })
                        self.circuit_breaker.record_success()

                    else:  # failed
                        deltas["failed"] += 1
                        deltas["bytes"] += out.bytes_count
                        batch_failures.append({
                            "key": out.key,
                            "reason": out.error,
                            "error_class": out.error_class,
                        })
                        self.circuit_breaker.record_failure(out)

                last_key = next_checkpoint if has_more else items[-1]["key"]
                record_progress(
                    job,
                    settings,
                    last_key=last_key,
                    counters=deltas,
                    failures=batch_failures,
                    in_flight=[],
                )

                try:
                    js.checkpoint(job)
                except JobLeaseLost:
                    logger.warning("Checkpoint failed; lease lost for job %s", self.job_id)
                    return

                # Check circuit breaker
                if self.circuit_breaker.is_tripped():
                    reason = self.circuit_breaker.trip_reason()
                    logger.warning("Circuit breaker tripped for job %s: %s", self.job_id, reason)
                    try:
                        js.transition_status(job, PAUSING, pause_reason=reason)
                        js.transition_status(job, PAUSED, pause_reason=reason)
                    except Exception as exc:
                        logger.error("Failed to auto-pause job %s on circuit breaker trip: %s", self.job_id, exc)
                    return

            if not has_more:
                try:
                    js.transition_status(job, COMPLETED)
                except Exception as exc:
                    logger.warning("Failed to mark job %s COMPLETED: %s", self.job_id, exc)
                return


class JobController:
    """Manages active job workers and lifecycle operations (start, pause, resume, stop, recovery)."""

    def __init__(
        self,
        settings: Settings,
        job_store: JobStore,
        astra_store: AstraStore,
        cos_client: Any,
    ):
        self.settings = settings
        self.job_store = job_store
        self.astra_store = astra_store
        self.cos_client = cos_client
        self._workers: Dict[str, JobWorker] = {}
        self._lock = threading.Lock()

    def start_job(
        self,
        bucket: str,
        prefix: str = "",
        *,
        collection: Optional[str] = None,
        archive_bucket: Optional[str] = None,
        concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        job = self.job_store.create_job(
            bucket=bucket,
            prefix=prefix,
            collection=collection,
            archive_bucket=archive_bucket,
            concurrency=concurrency,
        )
        self._spawn_worker(job["_id"])
        return job

    def pause_job(self, job_id: str, reason: str = "") -> Dict[str, Any]:
        res = self.job_store.set_desired_state(job_id, "PAUSED", reason=reason)
        return res.job

    def resume_job(self, job_id: str) -> Dict[str, Any]:
        res = self.job_store.set_desired_state(job_id, "RUNNING")
        self._spawn_worker(job_id)
        return res.job

    def stop_job(self, job_id: str, reason: str = "") -> Dict[str, Any]:
        res = self.job_store.set_desired_state(job_id, "STOPPED", reason=reason)
        with self._lock:
            worker = self._workers.get(job_id)
            if worker:
                worker.request_stop()
        return res.job

    def recover_jobs(self) -> List[str]:
        """Find unowned recoverable jobs (after app restart) and spawn workers for them."""
        recoverable = self.job_store.find_recoverable_jobs(include_queued=True)
        started_ids: List[str] = []
        for job in recoverable:
            jid = job["_id"]
            if self._spawn_worker(jid):
                started_ids.append(jid)
        return started_ids

    def _spawn_worker(self, job_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(job_id)
            if worker and worker.is_alive():
                return False
            worker = JobWorker(
                job_id=job_id,
                job_store=self.job_store,
                astra_store=self.astra_store,
                cos_client=self.cos_client,
                settings=self.settings,
            )
            self._workers[job_id] = worker
            worker.start()
            return True

    def shutdown(self, timeout: float = 30.0) -> None:
        """Signal all workers to stop and wait for them to finish."""
        with self._lock:
            workers = list(self._workers.values())
        for w in workers:
            w.request_stop()
        start = time.time()
        for w in workers:
            remaining = max(0.1, timeout - (time.time() - start))
            w.join(timeout=remaining)
