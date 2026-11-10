"""Unit and integration tests for the FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.astra_store import AstraStore
from app.config import Settings
from app.jobs import JOB_COUNTERS, QUEUED, RUNNING, JobStore
from app.main import create_app
from tests.fakes import FakeCollection, FakeCOS, FakeDB, FakeJobsCollection


@pytest.fixture
def api_setup(settings):
    cos = FakeCOS(keys=["docs/sample1.txt", "docs/sample2.txt"])
    db = FakeDB()
    jobs_col = FakeJobsCollection()
    db.cols[settings.jobs_collection] = jobs_col
    db.cols[settings.collection_name] = FakeCollection()

    astra_store = AstraStore(settings, database=db)
    app = create_app(settings=settings, astra_store=astra_store, cos_client=cos)
    with TestClient(app) as client:
        yield client, settings, jobs_col


def test_health_probes(api_setup):
    client, settings, _ = api_setup
    r_liveness = client.get("/healthz")
    assert r_liveness.status_code == 200
    assert r_liveness.json() == {"status": "ok"}

    r_readiness = client.get("/readyz")
    assert r_readiness.status_code == 200
    assert r_readiness.json() == {"status": "ready"}


def test_auth_protection(api_setup):
    client, settings, _ = api_setup
    # No auth header -> 401
    r = client.get("/api/jobs")
    assert r.status_code == 401

    # Wrong token -> 401
    r = client.get("/api/jobs", headers={"X-Access-Token": "wrong-token"})
    assert r.status_code == 401

    # Correct header token -> 200
    r = client.get("/api/jobs", headers={"X-Access-Token": settings.ui_access_token})
    assert r.status_code == 200

    # Correct Bearer token -> 200
    r = client.get("/api/jobs", headers={"Authorization": f"Bearer {settings.ui_access_token}"})
    assert r.status_code == 200


def test_job_create_and_list_api(api_setup):
    client, settings, _ = api_setup
    headers = {"X-Access-Token": settings.ui_access_token}

    create_payload = {
        "bucket": "src",
        "prefix": "docs/",
        "concurrency": 2,
    }
    r = client.post("/api/jobs", json=create_payload, headers=headers)
    assert r.status_code == 200
    job_data = r.json()
    job_id = job_data["id"]
    assert job_data["bucket"] == "src"
    assert job_data["prefix"] == "docs/"

    # List jobs
    r_list = client.get("/api/jobs", headers=headers)
    assert r_list.status_code == 200
    jobs_list = r_list.json()
    assert len(jobs_list) >= 1
    assert any(j["id"] == job_id for j in jobs_list)

    # Get job detail
    r_get = client.get(f"/api/jobs/{job_id}", headers=headers)
    assert r_get.status_code == 200
    assert r_get.json()["id"] == job_id


def test_job_control_api(api_setup):
    client, settings, jobs_col = api_setup
    headers = {"X-Access-Token": settings.ui_access_token}

    from app.jobs import guard_id

    # Pre-populate a running job in jobs_col
    job_id = "test-job-1"
    gid = guard_id("src", "test/")
    jobs_col.docs[job_id] = {
        "_id": job_id,
        "doc_type": "job",
        "bucket": "src",
        "prefix": "test/",
        "collection": settings.collection_name,
        "archive_bucket": settings.archive_bucket,
        "params": {"concurrency": 2},
        "desired_state": "RUNNING",
        "status": RUNNING,
        "pause_reason": "",
        "error": "",
        "last_key": "",
        "counters": {name: 0 for name in JOB_COUNTERS},
        "failures": [],
        "failures_truncated": 0,
        "in_flight": [],
        "events": [],
        "control_events": [],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": None,
        "heartbeat_at": "2026-01-01T00:00:00Z",
        "lease_owner": "worker-1",
        "lease_until": 9999999999.0,
        "guard_id": gid,
    }
    jobs_col.docs[gid] = {
        "_id": gid,
        "doc_type": "guard",
        "bucket": "src",
        "prefix": "test/",
        "active_job_id": job_id,
    }

    # 1. Pause job
    r_pause = client.post(f"/api/jobs/{job_id}/pause", json={"reason": "test operator pause"}, headers=headers)
    assert r_pause.status_code == 200
    assert r_pause.json()["desired_state"] == "PAUSED"

    # 2. Duplicate create -> 409 Conflict
    r_dup = client.post("/api/jobs", json={"bucket": "src", "prefix": "test/"}, headers=headers)
    assert r_dup.status_code == 409

    # 3. Resume job
    r_resume = client.post(f"/api/jobs/{job_id}/resume", headers=headers)
    assert r_resume.status_code == 200
    assert r_resume.json()["desired_state"] == "RUNNING"

    # 4. Stop job
    r_stop = client.post(f"/api/jobs/{job_id}/stop", json={"reason": "test operator stop"}, headers=headers)
    assert r_stop.status_code == 200
    assert r_stop.json()["desired_state"] == "STOPPED"


def test_get_nonexistent_job_returns_404(api_setup):
    client, settings, _ = api_setup
    headers = {"X-Access-Token": settings.ui_access_token}
    r = client.get("/api/jobs/missing123", headers=headers)
    assert r.status_code == 404


def test_preview_endpoint(api_setup):
    client, settings, _ = api_setup
    headers = {"X-Access-Token": settings.ui_access_token}
    r = client.get("/api/preview", params={"bucket": "src", "prefix": "docs/"}, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["bucket"] == "src"
    assert data["count"] == 2
    assert len(data["items"]) == 2


def test_prefixes_endpoint(api_setup):
    client, settings, _ = api_setup
    headers = {"X-Access-Token": settings.ui_access_token}

    r = client.get("/api/prefixes", params={"bucket": "src"}, headers=headers)
    assert r.status_code == 200
    assert r.json() == {"bucket": "src", "prefix": "", "count": 1, "truncated": False, "prefixes": ["docs/"]}

    r = client.get("/api/prefixes", params={"bucket": "src", "prefix": "docs/"}, headers=headers)
    assert r.status_code == 200
    assert r.json() == {"bucket": "src", "prefix": "docs/", "count": 0, "truncated": False, "prefixes": []}


def test_prefixes_endpoint_requires_auth(api_setup):
    client, settings, _ = api_setup
    assert client.get("/api/prefixes", params={"bucket": "src"}).status_code == 401
    r = client.get("/api/prefixes", params={"bucket": "src"}, headers={"X-Access-Token": "wrong-token"})
    assert r.status_code == 401


def test_prefixes_endpoint_error_maps_to_400(api_setup, no_sleep):
    client, settings, _ = api_setup
    headers = {"X-Access-Token": settings.ui_access_token}
    r = client.get("/api/prefixes", params={"bucket": "missing-bucket"}, headers=headers)
    assert r.status_code == 400

