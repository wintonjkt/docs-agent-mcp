"""Unit tests for app/jobs.py (job model, transitions, persistence, lease, guard, metrics). No network."""

import copy

import pytest

from app import jobs
from app.astra_store import AstraStore
from app.jobs import (
    ALLOWED_TRANSITIONS, COMPLETED, FAILED, PAUSED, PAUSING, QUEUED, RUNNING, STATUSES, STOPPED, STOPPING,
    DuplicateActiveJobError, InvalidTransition, JobConflict, JobLeaseLost, JobNotFound, JobStatusChanged,
    JobStore,
)
from tests.fakes import FakeDB, FakeJobsCollection

T0 = 1_800_000_000.0


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def col(settings):
    return FakeJobsCollection()


@pytest.fixture
def js(settings, col, clock):
    db = FakeDB()
    db.cols[settings.jobs_collection] = col
    return JobStore(AstraStore(settings, database=db), settings, clock=clock)


def make_job(js, **kw):
    kw.setdefault("bucket", "src")
    kw.setdefault("prefix", "docs/")
    return js.create_job(**kw)


def started(js, **kw):
    """A RUNNING job whose lease is held by 'w1'."""
    job = js.acquire_lease(make_job(js, **kw)["_id"], owner="w1")
    js.transition_status(job, RUNNING)
    return job


# --------------------------------------------------------------------------- #
# Transition rules (pure)
# --------------------------------------------------------------------------- #
def test_table_covers_all_statuses_and_terminals_are_final():
    assert set(ALLOWED_TRANSITIONS) == set(STATUSES)
    for status in (COMPLETED, FAILED, STOPPED):
        assert ALLOWED_TRANSITIONS[status] == frozenset()
        assert jobs.is_terminal(status)
    assert not any(jobs.is_terminal(s) for s in (QUEUED, RUNNING, PAUSING, PAUSED, STOPPING))


@pytest.mark.parametrize("current", STATUSES)
@pytest.mark.parametrize("new", STATUSES)
def test_validate_transition_matches_table(current, new):
    if new in ALLOWED_TRANSITIONS[current]:
        jobs.validate_transition(current, new)
    else:
        with pytest.raises(InvalidTransition):
            jobs.validate_transition(current, new)


def test_invalid_transition_messages_are_clear():
    with pytest.raises(InvalidTransition, match="already COMPLETED.*final"):
        jobs.validate_transition(COMPLETED, RUNNING)
    with pytest.raises(InvalidTransition, match="from RUNNING to PAUSED.*allowed from RUNNING"):
        jobs.validate_transition(RUNNING, PAUSED)
    with pytest.raises(InvalidTransition, match="unknown job status"):
        jobs.validate_transition("BOGUS", RUNNING)


def test_every_user_plan_move_is_an_allowed_transition():
    for (status, desired), new_status in jobs._DESIRED_PLAN.items():
        assert desired in jobs.DESIRED_STATES
        if new_status != status:
            assert new_status in ALLOWED_TRANSITIONS[status], (status, desired, new_status)


@pytest.mark.parametrize("status,desired,expected,changed", [
    (QUEUED, "PAUSED", PAUSED, True),
    (QUEUED, "STOPPED", STOPPED, True),
    (QUEUED, "RUNNING", QUEUED, False),
    (RUNNING, "PAUSED", PAUSING, True),
    (RUNNING, "STOPPED", STOPPING, True),
    (RUNNING, "RUNNING", RUNNING, False),
    (PAUSING, "RUNNING", RUNNING, True),
    (PAUSING, "STOPPED", STOPPING, True),
    (PAUSING, "PAUSED", PAUSING, False),
    (PAUSED, "RUNNING", RUNNING, True),
    (PAUSED, "STOPPED", STOPPED, True),
    (PAUSED, "PAUSED", PAUSED, False),
    (STOPPING, "STOPPED", STOPPING, False),
    (STOPPED, "STOPPED", STOPPED, False),
])
def test_plan_desired_state_change(status, desired, expected, changed):
    plan = jobs.plan_desired_state_change(status, desired)
    assert (plan.status, plan.changed) == (expected, changed)


@pytest.mark.parametrize("status,desired,message", [
    (COMPLETED, "PAUSED", "cannot pause a job that is already COMPLETED"),
    (FAILED, "RUNNING", "cannot resume a job that is already FAILED"),
    (STOPPED, "RUNNING", "cannot resume a job that is already STOPPED"),
    (STOPPING, "RUNNING", "stop cannot be undone"),
    (STOPPING, "PAUSED", "stop cannot be undone"),
    (RUNNING, "BOGUS", "unknown desired state"),
])
def test_plan_desired_state_change_rejects(status, desired, message):
    with pytest.raises(InvalidTransition, match=message):
        jobs.plan_desired_state_change(status, desired)


def test_allowed_actions_per_status():
    assert jobs.allowed_actions(RUNNING) == ["pause", "stop"]
    assert jobs.allowed_actions(PAUSING) == ["resume", "stop"]
    assert jobs.allowed_actions(PAUSED) == ["resume", "stop"]
    assert jobs.allowed_actions(QUEUED) == ["pause", "stop"]
    assert jobs.allowed_actions(STOPPING) == []
    for terminal in (COMPLETED, FAILED, STOPPED):
        assert jobs.allowed_actions(terminal) == []


# --------------------------------------------------------------------------- #
# create / load / list
# --------------------------------------------------------------------------- #
def test_create_job_schema_and_defaults(js, settings, col):
    job = make_job(js)
    stored = col.docs[job["_id"]]
    assert stored["status"] == QUEUED and stored["desired_state"] == "RUNNING"
    assert (stored["bucket"], stored["prefix"]) == ("src", "docs/")
    assert stored["collection"] == settings.collection_name
    assert stored["archive_bucket"] == settings.archive_bucket
    assert stored["params"] == {"concurrency": settings.concurrency}
    assert stored["pause_reason"] == "" and stored["last_key"] == ""
    assert stored["counters"] == {n: 0 for n in jobs.JOB_COUNTERS}
    assert stored["failures"] == [] and stored["failures_truncated"] == 0 and stored["in_flight"] == []
    assert stored["lease_owner"] == "" and stored["lease_until"] == 0
    assert stored["created_at"] == stored["updated_at"]
    assert stored["started_at"] is None and stored["finished_at"] is None
    assert [e["message"] for e in stored["events"]] == ["Job created"]


def test_create_job_overrides_and_validation(js):
    job = js.create_job("src", "p/", collection="other", archive_bucket="arc2", concurrency=3)
    assert (job["collection"], job["archive_bucket"], job["params"]) == ("other", "arc2", {"concurrency": 3})
    with pytest.raises(ValueError, match="archive bucket must differ"):
        js.create_job("src", "x/", archive_bucket="src")
    with pytest.raises(ValueError, match="concurrency"):
        js.create_job("src", "x/", concurrency=0)
    with pytest.raises(ValueError, match="bucket is required"):
        js.create_job("")


def test_load_job(js):
    job = make_job(js)
    assert js.load_job(job["_id"])["_id"] == job["_id"]
    assert js.load_job("nope") is None
    assert js.load_job("  ") is None


def test_load_job_does_not_return_guard_documents(js):
    job = make_job(js)
    assert js.load_job(jobs.guard_id("src", "docs/")) is None
    assert job["_id"]


def test_list_jobs_most_recent_first_with_limit(js, clock):
    ids = []
    for i in range(5):
        clock.advance(10)
        ids.append(make_job(js, prefix=f"p{i}/")["_id"])
    listed = js.list_jobs(limit=3)
    assert [j["_id"] for j in listed] == ids[::-1][:3]
    assert len(js.list_jobs(limit=500)) == 5            # clamped, not an error
    assert len(js.list_jobs(limit=0)) == 1              # clamped to at least 1
    assert all(j["doc_type"] == "job" for j in js.list_jobs())    # guards are not listed


def test_list_jobs_status_filter(js):
    a = make_job(js, prefix="a/")
    make_job(js, prefix="b/")
    js.set_desired_state(a["_id"], "STOPPED")
    assert [j["_id"] for j in js.list_jobs(statuses=[STOPPED])] == [a["_id"]]


def test_astra_calls_are_retried(js, col):
    col.fail_next = [RuntimeError("blip")]
    job = make_job(js)               # first insert_one (the guard) fails once, then succeeds
    assert job["_id"] in col.docs


# --------------------------------------------------------------------------- #
# duplicate-active-job guard
# --------------------------------------------------------------------------- #
def test_duplicate_active_job_rejected(js):
    first = make_job(js)
    with pytest.raises(DuplicateActiveJobError, match=first["_id"]) as ei:
        make_job(js)
    assert ei.value.existing_job_id == first["_id"]
    assert "docs/" in str(ei.value)


def test_duplicate_guard_is_per_bucket_and_prefix(js):
    make_job(js, prefix="a/")
    make_job(js, prefix="b/")
    make_job(js, bucket="other", prefix="a/")
    with pytest.raises(DuplicateActiveJobError):
        make_job(js, prefix="a/")


@pytest.mark.parametrize("finish", ["stop", "complete", "fail"])
def test_guard_released_when_job_terminal(js, finish):
    first = make_job(js)
    if finish == "stop":
        js.set_desired_state(first["_id"], "STOPPED")           # QUEUED -> STOPPED
    else:
        job = js.acquire_lease(first["_id"], owner="w1")
        js.transition_status(job, RUNNING)
        js.transition_status(job, COMPLETED if finish == "complete" else FAILED, error="x")
    second = make_job(js)                                       # now allowed
    assert second["_id"] != first["_id"]


def test_paused_job_still_blocks_new_job(js):
    first = make_job(js)
    js.set_desired_state(first["_id"], "PAUSED")
    with pytest.raises(DuplicateActiveJobError):
        make_job(js)


def test_stale_guard_pointing_to_missing_job_is_taken_over(js, col):
    col.docs[jobs.guard_id("src", "docs/")] = {
        "_id": jobs.guard_id("src", "docs/"), "doc_type": "guard", "active_job_id": "ghost",
    }
    job = make_job(js)
    assert col.docs[jobs.guard_id("src", "docs/")]["active_job_id"] == job["_id"]


def test_stale_guard_pointing_to_terminal_job_is_taken_over(js, col):
    first = make_job(js)
    col.docs[first["_id"]]["status"] = COMPLETED               # finished, but guard release was lost
    second = make_job(js)
    assert col.docs[jobs.guard_id("src", "docs/")]["active_job_id"] == second["_id"]


def test_guard_takeover_race_loses_to_concurrent_creator(js, col):
    """Between reading a stale guard and taking it over, another creator wins: we must be rejected."""
    gid = jobs.guard_id("src", "docs/")
    col.docs[gid] = {"_id": gid, "doc_type": "guard", "active_job_id": "ghost"}
    rival = {"_id": "rival", "doc_type": "job", "status": RUNNING, "bucket": "src", "prefix": "docs/"}

    def rival_wins(c, flt, update):
        if flt.get("_id") == gid and flt.get("active_job_id") == "ghost":
            c.docs["rival"] = copy.deepcopy(rival)
            c.docs[gid]["active_job_id"] = "rival"
            c.before_update = None

    col.before_update = rival_wins
    with pytest.raises(DuplicateActiveJobError, match="rival"):
        make_job(js)
    assert col.docs[gid]["active_job_id"] == "rival"


def test_concurrent_creates_only_one_wins(js):
    import threading

    results, errors = [], []

    def run():
        try:
            results.append(make_job(js)["_id"])
        except DuplicateActiveJobError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 1 and len(errors) == 7


# --------------------------------------------------------------------------- #
# lease
# --------------------------------------------------------------------------- #
def test_acquire_lease_on_new_job(js, settings, clock):
    job = make_job(js)
    got = js.acquire_lease(job["_id"], owner="w1")
    assert got["lease_owner"] == "w1"
    assert got["lease_until"] == clock.now + settings.lease_seconds
    assert got["heartbeat_at"]


def test_live_lease_blocks_takeover_and_expired_allows_it(js, settings, clock):
    job = make_job(js)
    assert js.acquire_lease(job["_id"], owner="w1")
    clock.advance(settings.lease_seconds - 1)
    assert js.acquire_lease(job["_id"], owner="w2") is None          # still live
    clock.advance(2)
    taken = js.acquire_lease(job["_id"], owner="w2")                 # expired
    assert taken["lease_owner"] == "w2"
    assert js.acquire_lease(job["_id"], owner="w3") is None


def test_acquire_lease_unknown_or_terminal_job_returns_none(js):
    assert js.acquire_lease("nope") is None
    job = make_job(js)
    js.set_desired_state(job["_id"], "STOPPED")
    assert js.acquire_lease(job["_id"]) is None


def test_acquire_lease_clears_stale_in_flight(js, col, settings, clock):
    job = started(js)
    job["in_flight"] = [{"key": "a.txt", "size": 1, "started_at": "t"}]
    js.checkpoint(job)
    clock.advance(settings.lease_seconds + 1)
    taken = js.acquire_lease(job["_id"], owner="w2")
    assert taken["in_flight"] == []


def test_heartbeat_extends_lease_and_detects_takeover(js, settings, clock):
    job = started(js)
    clock.advance(100)
    js.heartbeat(job)
    assert job["lease_until"] == clock.now + settings.lease_seconds
    clock.advance(settings.lease_seconds + 1)
    js.acquire_lease(job["_id"], owner="w2")
    with pytest.raises(JobLeaseLost):
        js.heartbeat(job)


def test_release_lease(js):
    job = started(js)
    assert js.release_lease(job) is True
    assert js.acquire_lease(job["_id"], owner="w2")["lease_owner"] == "w2"
    assert js.release_lease(job) is False                      # no longer the owner; does not raise


def test_release_lease_never_raises_on_astra_error(js, col):
    job = started(js)
    col.fail_next = [RuntimeError("down")] * 10
    assert js.release_lease(job) is False


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def test_checkpoint_persists_progress(js, col, settings, clock):
    job = started(js)
    clock.advance(30)
    jobs.record_progress(
        job, settings, last_key="docs/b.txt",
        counters={"scanned": 5, "ingested": 3, "chunks": 12, "bytes": 999},
        failures=[{"key": "docs/x.txt", "reason": "empty file"}],
        in_flight=[{"key": "docs/c.txt", "size": 10, "started_at": "t", "junk": 1}],
        events=["page done"],
    )
    js.checkpoint(job)
    stored = col.docs[job["_id"]]
    assert stored["last_key"] == "docs/b.txt"
    assert stored["counters"]["ingested"] == 3 and stored["counters"]["chunks"] == 12
    assert stored["failures"][0]["key"] == "docs/x.txt"
    assert stored["in_flight"] == [{"key": "docs/c.txt", "size": 10, "started_at": "t"}]
    assert stored["events"][-1]["message"] == "page done"
    assert stored["lease_until"] == clock.now + settings.lease_seconds


def test_checkpoint_raises_job_lease_lost_when_taken_over(js, col, settings, clock):
    job = started(js)
    clock.advance(settings.lease_seconds + 1)
    js.acquire_lease(job["_id"], owner="w2")
    jobs.record_progress(job, settings, last_key="stale", counters={"ingested": 99})
    with pytest.raises(JobLeaseLost):
        js.checkpoint(job)
    stored = col.docs[job["_id"]]
    assert stored["last_key"] == "" and stored["counters"]["ingested"] == 0       # nothing written


def test_checkpoint_after_job_deleted_is_lease_lost(js, col):
    job = started(js)
    del col.docs[job["_id"]]
    with pytest.raises(JobLeaseLost):
        js.checkpoint(job)


def test_checkpoint_release_gives_up_lease(js, col):
    job = started(js)
    js.checkpoint(job, release=True)
    assert col.docs[job["_id"]]["lease_owner"] == "" and col.docs[job["_id"]]["lease_until"] == 0
    assert js.acquire_lease(job["_id"], owner="w2")


def test_checkpoint_is_retried_on_transient_error(js, col):
    job = started(js)
    col.fail_next = [RuntimeError("blip")]
    js.checkpoint(job)


# --------------------------------------------------------------------------- #
# set_desired_state (API) vs worker
# --------------------------------------------------------------------------- #
def test_set_desired_state_does_not_clobber_progress(js, col, settings):
    job = started(js)
    jobs.record_progress(job, settings, last_key="docs/k1", counters={"ingested": 7, "bytes": 70},
                         failures=[{"key": "f", "reason": "r"}], in_flight=[{"key": "k2"}], events=["e1"])
    js.checkpoint(job)
    before = copy.deepcopy(col.docs[job["_id"]])

    res = js.set_desired_state(job["_id"], "PAUSED")
    assert res.changed and res.job["status"] == PAUSING

    after = col.docs[job["_id"]]
    for field in ("last_key", "counters", "failures", "failures_truncated", "in_flight", "events",
                  "lease_owner", "lease_until", "heartbeat_at", "created_at", "bucket", "prefix"):
        assert after[field] == before[field], field
    assert (after["status"], after["desired_state"]) == (PAUSING, "PAUSED")
    assert after["pause_reason"] == "Paused by user"
    assert after["control_events"][-1]["message"] == "User requested pause"


def test_worker_checkpoint_does_not_clobber_user_control_and_observes_it(js, col, settings):
    job = started(js)
    js.set_desired_state(job["_id"], "PAUSED")                  # user pauses while the worker is busy
    assert job["desired_state"] == "RUNNING"                    # worker has not seen it yet
    jobs.record_progress(job, settings, counters={"ingested": 2})
    js.checkpoint(job)
    assert (job["status"], job["desired_state"]) == (PAUSING, "PAUSED")      # observed via checkpoint
    assert job["pause_reason"] == "Paused by user"
    stored = col.docs[job["_id"]]
    assert stored["status"] == PAUSING and stored["counters"]["ingested"] == 2


def test_worker_observes_desired_state_via_heartbeat(js):
    job = started(js)
    js.set_desired_state(job["_id"], "STOPPED")
    js.heartbeat(job)
    assert (job["status"], job["desired_state"]) == (STOPPING, "STOPPED")


def test_set_desired_state_full_lifecycle(js, col):
    job = started(js)
    jid = job["_id"]
    assert js.set_desired_state(jid, "PAUSED").job["status"] == PAUSING
    assert js.set_desired_state(jid, "PAUSED").changed is False                 # idempotent
    js.checkpoint(job)
    js.transition_status(job, PAUSED, pause_reason="Paused by user")             # worker finished draining
    assert col.docs[jid]["status"] == PAUSED
    res = js.set_desired_state(jid, "RUNNING")
    assert res.job["status"] == RUNNING and res.job["pause_reason"] == ""
    res = js.set_desired_state(jid, "STOPPED")
    assert res.job["status"] == STOPPING and res.job["finished_at"] is None


def test_stop_of_paused_job_is_immediately_terminal_and_frees_guard(js):
    job = started(js)
    js.set_desired_state(job["_id"], "PAUSED")
    js.checkpoint(job)                                          # worker observes PAUSING
    js.transition_status(job, PAUSED)
    res = js.set_desired_state(job["_id"], "STOPPED")
    assert res.job["status"] == STOPPED and res.job["finished_at"]
    make_job(js)                                                # guard released


def test_set_desired_state_custom_reason(js):
    job = started(js)
    res = js.set_desired_state(job["_id"], "PAUSED", reason="maintenance window")
    assert res.job["pause_reason"] == "maintenance window"
    assert "maintenance window" in res.job["control_events"][-1]["message"]


def test_set_desired_state_errors(js):
    with pytest.raises(JobNotFound):
        js.set_desired_state("nope", "PAUSED")
    job = make_job(js)
    js.set_desired_state(job["_id"], "STOPPED")
    with pytest.raises(InvalidTransition, match="already STOPPED"):
        js.set_desired_state(job["_id"], "RUNNING")


def test_set_desired_state_cas_conflict_is_retried(js, col):
    job = started(js)
    state = {"n": 0}

    def interfere(c, flt, update):
        if "desired_state" in flt and state["n"] == 0:           # only the first API CAS
            state["n"] += 1
            c.docs[job["_id"]]["status"] = PAUSING               # someone else changed it first
            c.docs[job["_id"]]["desired_state"] = "PAUSED"

    col.before_update = interfere
    res = js.set_desired_state(job["_id"], "STOPPED")
    assert res.job["status"] == STOPPING                         # re-planned from PAUSING


def test_set_desired_state_gives_up_after_repeated_conflicts(js, col):
    job = started(js)
    flip = {"s": 0}

    def always(c, flt, update):
        if "desired_state" in flt:
            flip["s"] += 1
            c.docs[job["_id"]]["desired_state"] = "X%d" % flip["s"]

    col.before_update = always
    with pytest.raises(JobConflict):
        js.set_desired_state(job["_id"], "PAUSED")


def test_user_resume_while_draining_beats_worker_pause(js):
    job = started(js)
    js.set_desired_state(job["_id"], "PAUSED")
    js.checkpoint(job)                                           # worker sees PAUSING
    js.set_desired_state(job["_id"], "RUNNING")                  # user resumes before the drain ends
    with pytest.raises(JobStatusChanged) as ei:
        js.transition_status(job, PAUSED)                        # worker wanted PAUSING -> PAUSED
    assert ei.value.actual == RUNNING
    assert (job["status"], job["desired_state"]) == (RUNNING, "RUNNING")      # job refreshed, user wins


# --------------------------------------------------------------------------- #
# worker status transitions
# --------------------------------------------------------------------------- #
def test_transition_status_sets_timestamps_and_events(js, col, clock):
    job = js.acquire_lease(make_job(js)["_id"], owner="w1")
    clock.advance(5)
    js.transition_status(job, RUNNING)
    stored = col.docs[job["_id"]]
    assert stored["status"] == RUNNING and stored["started_at"]
    assert stored["events"][-1]["message"] == "Status QUEUED -> RUNNING"
    started_at = stored["started_at"]
    clock.advance(60)
    js.transition_status(job, COMPLETED)
    stored = col.docs[job["_id"]]
    assert stored["finished_at"] and stored["started_at"] == started_at


def test_transition_status_pause_reason_for_breaker(js, col):
    job = started(js)
    js.transition_status(job, PAUSING, pause_reason="circuit breaker: 5 consecutive failures")
    js.transition_status(job, PAUSED, pause_reason="circuit breaker: 5 consecutive failures")
    stored = col.docs[job["_id"]]
    assert stored["pause_reason"].startswith("circuit breaker")
    assert stored["desired_state"] == "PAUSED"


def test_transition_status_rejects_invalid_and_lost_lease(js, settings, clock):
    job = started(js)
    with pytest.raises(InvalidTransition):
        js.transition_status(job, PAUSED)                        # RUNNING -> PAUSED is not allowed
    clock.advance(settings.lease_seconds + 1)
    js.acquire_lease(job["_id"], owner="w2")
    with pytest.raises(JobLeaseLost):
        js.transition_status(job, PAUSING)


def test_transition_status_failed_records_error(js, col):
    job = started(js)
    js.transition_status(job, FAILED, error="listing failed: boom")
    assert col.docs[job["_id"]]["error"] == "listing failed: boom"


def test_terminal_job_cannot_transition(js):
    job = started(js)
    js.transition_status(job, COMPLETED)
    with pytest.raises(InvalidTransition):
        js.transition_status(job, RUNNING)


# --------------------------------------------------------------------------- #
# caps
# --------------------------------------------------------------------------- #
def test_failure_list_is_capped_with_truncated_count(settings, make_settings):
    s = make_settings(MAX_JOB_FAILURES=3)
    job = {"counters": {}, "failures": []}
    jobs.record_progress(job, s, counters={"failed": 5},
                         failures=[{"key": f"k{i}", "reason": "r"} for i in range(5)])
    assert [f["key"] for f in job["failures"]] == ["k0", "k1", "k2"]      # first ones are kept
    assert job["failures_truncated"] == 2
    assert job["counters"]["failed"] == 5                                 # counter still counts all
    jobs.record_progress(job, s, failures=[{"key": "k9", "reason": "r"}])
    assert job["failures_truncated"] == 3


def test_failure_entry_normalised_and_reason_capped(settings):
    from app.retry import ErrorClass

    job = {"counters": {}, "failures": []}
    jobs.record_progress(job, settings, failures=[
        {"key": "k", "reason": "x" * 5000, "error_class": ErrorClass.PERMANENT, "extra": "dropped"}
    ], now="NOW")
    f = job["failures"][0]
    assert len(f["reason"]) == jobs.MAX_REASON_CHARS
    assert f["error_class"] == "permanent" and f["at"] == "NOW" and "extra" not in f


def test_event_log_is_capped_keeping_newest(make_settings):
    s = make_settings(MAX_JOB_EVENTS=4)
    job = {"counters": {}}
    jobs.record_progress(job, s, events=[f"e{i}" for i in range(3)])
    jobs.record_progress(job, s, events=[f"e{i}" for i in range(3, 7)])
    assert [e["message"] for e in job["events"]] == ["e3", "e4", "e5", "e6"]


def test_event_cap_applies_to_persisted_events(js, col, make_settings):
    job = started(js)
    s = make_settings(MAX_JOB_EVENTS=3)
    for i in range(10):
        jobs.record_progress(job, s, events=[f"e{i}"])
    js.checkpoint(job)
    assert [e["message"] for e in col.docs[job["_id"]]["events"]] == ["e7", "e8", "e9"]


def test_control_events_are_capped_too(js, col, make_settings):
    s = make_settings(MAX_JOB_EVENTS=2)
    store = JobStore(js.store, s, clock=js.clock)
    job = make_job(js)
    for _ in range(3):
        store.set_desired_state(job["_id"], "PAUSED")
        store.set_desired_state(job["_id"], "RUNNING")
    assert len(col.docs[job["_id"]]["control_events"]) == 2


def test_record_progress_rejects_unknown_counter_and_accumulates(settings):
    job = {"counters": {"ingested": 1}}
    jobs.record_progress(job, settings, counters={"ingested": 2, "bytes": 5})
    assert job["counters"] == {"ingested": 3, "bytes": 5}
    with pytest.raises(ValueError, match="unknown job counter"):
        jobs.record_progress(job, settings, counters={"typo": 1})


def test_in_flight_replaces_and_clears(settings):
    job = {"counters": {}}
    jobs.record_progress(job, settings, in_flight=[{"key": "a"}, {"key": "b"}])
    assert [i["key"] for i in job["in_flight"]] == ["a", "b"]
    jobs.record_progress(job, settings, in_flight=[])
    assert job["in_flight"] == []


# --------------------------------------------------------------------------- #
# recoverable jobs
# --------------------------------------------------------------------------- #
def test_find_recoverable_jobs(js, col, settings, clock):
    def job_in(status, prefix, lease_until):
        j = make_job(js, prefix=prefix)
        col.docs[j["_id"]]["status"] = status
        col.docs[j["_id"]]["lease_until"] = lease_until
        return j["_id"]

    expired = clock.now - 1
    live = clock.now + 100
    running_expired = job_in(RUNNING, "a/", expired)
    pausing_expired = job_in(PAUSING, "b/", expired)
    stopping_expired = job_in(STOPPING, "c/", expired)
    queued = job_in(QUEUED, "d/", 0)
    job_in(RUNNING, "e/", live)                                  # a live worker owns it
    job_in(PAUSED, "f/", expired)                                # stays paused
    for i, terminal in enumerate((COMPLETED, FAILED, STOPPED)):
        job_in(terminal, f"t{i}/", expired)

    found = {j["_id"] for j in js.find_recoverable_jobs()}
    assert found == {running_expired, pausing_expired, stopping_expired, queued}


def test_find_recoverable_jobs_oldest_first_and_released_lease(js, col, clock):
    a = make_job(js, prefix="a/")
    clock.advance(10)
    b = make_job(js, prefix="b/")
    assert [j["_id"] for j in js.find_recoverable_jobs()] == [a["_id"], b["_id"]]
    job = js.acquire_lease(a["_id"], owner="w1")
    assert [j["_id"] for j in js.find_recoverable_jobs()] == [b["_id"]]
    js.release_lease(job)
    assert a["_id"] in [j["_id"] for j in js.find_recoverable_jobs()]


def test_resumed_paused_job_without_worker_becomes_recoverable(js):
    job = started(js)
    js.set_desired_state(job["_id"], "PAUSED")
    js.checkpoint(job)                                          # worker observes PAUSING
    js.transition_status(job, PAUSED)
    js.release_lease(job)
    assert js.find_recoverable_jobs() == []
    js.set_desired_state(job["_id"], "RUNNING")
    assert [j["_id"] for j in js.find_recoverable_jobs()] == [job["_id"]]


# --------------------------------------------------------------------------- #
# metrics / ETA / render
# --------------------------------------------------------------------------- #
def _metrics_job(**over):
    base = {
        "status": RUNNING,
        "started_at": jobs._iso(T0),
        "created_at": jobs._iso(T0 - 10),
        "counters": {"ingested": 60, "skipped": 30, "failed": 10, "bytes": 5000},
    }
    base.update(over)
    return base


def test_metrics_throughput_and_eta():
    m = jobs.compute_metrics(_metrics_job(), now=T0 + 100, total_estimate=1000)
    assert m["files_processed"] == 100
    assert m["elapsed_seconds"] == 100.0
    assert m["files_per_sec"] == 1.0 and m["bytes_per_sec"] == 50.0
    assert m["remaining_files"] == 900 and m["percent"] == 10.0
    assert m["eta_seconds"] == 900.0


def test_metrics_without_estimate_or_progress():
    m = jobs.compute_metrics(_metrics_job(counters={}), now=T0 + 50)
    assert m["files_per_sec"] == 0.0 and m["eta_seconds"] is None and m["percent"] is None
    m = jobs.compute_metrics(_metrics_job(), now=T0, total_estimate=10)       # zero elapsed
    assert m["files_per_sec"] == 0.0 and m["eta_seconds"] is None


def test_metrics_eta_only_while_running_and_clamped():
    assert jobs.compute_metrics(_metrics_job(status=PAUSED), now=T0 + 100, total_estimate=1000)["eta_seconds"] is None
    m = jobs.compute_metrics(_metrics_job(), now=T0 + 100, total_estimate=50)  # estimate already exceeded
    assert m["remaining_files"] == 0 and m["percent"] == 100.0 and m["eta_seconds"] == 0.0


def test_metrics_finished_job_uses_finished_at_and_created_at_fallback():
    m = jobs.compute_metrics(_metrics_job(status=COMPLETED, finished_at=jobs._iso(T0 + 20)), now=T0 + 9999)
    assert m["elapsed_seconds"] == 20.0 and m["files_per_sec"] == 5.0
    m = jobs.compute_metrics(_metrics_job(started_at=None), now=T0)           # falls back to created_at
    assert m["elapsed_seconds"] == 10.0


def test_render_job_is_api_safe(js, settings, clock):
    job = started(js)
    jobs.record_progress(job, settings, counters={"ingested": 4, "bytes": 40}, events=["hello"])
    js.checkpoint(job)
    js.set_desired_state(job["_id"], "PAUSED")
    view = jobs.render_job(js.load_job(job["_id"]), now=clock.now, total_estimate=100)

    assert view["id"] == job["_id"] and "_id" not in view
    for leaked in ("lease_owner", "lease_until", "doc_type", "control_events", "active_job_id"):
        assert leaked not in view
    assert view["worker_alive"] is True
    assert view["status"] == PAUSING and view["is_terminal"] is False
    assert view["allowed_actions"] == ["resume", "stop"]
    assert view["counters"]["ingested"] == 4 and set(view["counters"]) == set(jobs.JOB_COUNTERS)
    assert view["metrics"]["total_estimate"] == 100
    messages = [e["message"] for e in view["events"]]
    assert "hello" in messages and "User requested pause" in messages          # worker + API events merged
    assert [e["at"] for e in view["events"]] == sorted(e["at"] for e in view["events"])
    assert "lease" not in repr(view).lower()


def test_render_job_terminal_and_dead_worker(js, settings, clock):
    job = started(js)
    js.transition_status(job, COMPLETED)
    js.release_lease(job)
    view = jobs.render_job(js.load_job(job["_id"]), now=clock.now)
    assert view["is_terminal"] and view["allowed_actions"] == [] and view["worker_alive"] is False
    assert view["finished_at"]


def test_render_job_tolerates_sparse_document():
    view = jobs.render_job({"_id": "x", "status": "WEIRD"}, now=T0)
    assert view["id"] == "x" and view["allowed_actions"] == [] and view["counters"]["scanned"] == 0
