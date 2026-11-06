"""Unit tests for app/controller.py (CircuitBreaker, JobWorker, JobController)."""

import time
import pytest

from app.astra_store import AstraStore
from app.controller import CircuitBreaker, JobController, JobWorker
from app.jobs import COMPLETED, FAILED, PAUSED, RUNNING, STOPPED, JobStore
from app.pipeline import FileOutcome
from app.retry import ErrorClass
from tests.fakes import FakeCollection, FakeCOS, FakeDB, FakeJobsCollection


def test_circuit_breaker_logic():
    cb = CircuitBreaker(threshold=3)

    # Transient errors increment
    cb.record_failure(FileOutcome(key="k1", status="failed", error="network error"))
    assert cb.consecutive_failures == 1
    assert not cb.is_tripped()

    # Permanent error does NOT increment
    cb.record_failure(FileOutcome(key="k2", status="failed", error="empty file", error_class=ErrorClass.PERMANENT))
    assert cb.consecutive_failures == 1

    # More transient errors trip
    cb.record_failure(FileOutcome(key="k3", status="failed", error="500 server error"))
    cb.record_failure(FileOutcome(key="k4", status="failed", error="503 throttled"))
    assert cb.consecutive_failures == 3
    assert cb.is_tripped()
    assert "3 consecutive failures" in cb.trip_reason()

    # Success resets
    cb.record_success()
    assert cb.consecutive_failures == 0
    assert not cb.is_tripped()


def test_job_worker_full_run_to_completion(settings):
    keys = ["docs/f1.txt", "docs/f2.txt"]
    cos = FakeCOS(keys=keys)
    db = FakeDB()
    jobs_col = FakeJobsCollection()
    db.cols[settings.jobs_collection] = jobs_col
    db.cols[settings.collection_name] = FakeCollection()

    astra_store = AstraStore(settings, database=db)
    job_store = JobStore(astra_store, settings)

    job = job_store.create_job("src", "docs/")
    worker = JobWorker(job["_id"], job_store, astra_store, cos, settings)
    worker.run()

    loaded = job_store.load_job(job["_id"])
    assert loaded["status"] == COMPLETED
    assert loaded["counters"]["ingested"] == 2
    assert loaded["counters"]["archived"] == 2


def test_job_worker_circuit_breaker_trips_to_paused(settings):
    keys = [f"docs/f{i}.txt" for i in range(10)]
    cos = FakeCOS(keys=keys)
    db = FakeDB()
    jobs_col = FakeJobsCollection()
    db.cols[settings.jobs_collection] = jobs_col
    ingest_col = FakeCollection()
    ingest_col.fail_insert = True       # Ingestion will fail
    db.cols[settings.collection_name] = ingest_col

    astra_store = AstraStore(settings, database=db)
    job_store = JobStore(astra_store, settings)

    job = job_store.create_job("src", "docs/")
    worker = JobWorker(job["_id"], job_store, astra_store, cos, settings, circuit_breaker_threshold=3)
    worker.run()

    loaded = job_store.load_job(job["_id"])
    assert loaded["status"] == PAUSED
    assert "circuit breaker" in loaded["pause_reason"]


def test_job_controller_lifecycle(settings):
    keys = ["docs/a.txt", "docs/b.txt"]
    cos = FakeCOS(keys=keys)
    db = FakeDB()
    jobs_col = FakeJobsCollection()
    db.cols[settings.jobs_collection] = jobs_col
    db.cols[settings.collection_name] = FakeCollection()

    astra_store = AstraStore(settings, database=db)
    job_store = JobStore(astra_store, settings)
    ctrl = JobController(settings, job_store, astra_store, cos)

    # Start job
    job = ctrl.start_job("src", "docs/")
    jid = job["_id"]

    # Pause job
    ctrl.pause_job(jid, reason="test pause")
    loaded = job_store.load_job(jid)
    assert loaded["desired_state"] == "PAUSED"

    # Stop job
    ctrl.stop_job(jid)
    loaded = job_store.load_job(jid)
    assert loaded["desired_state"] == "STOPPED"

    ctrl.shutdown(timeout=1.0)
