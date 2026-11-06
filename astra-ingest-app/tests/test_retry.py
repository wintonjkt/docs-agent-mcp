import httpx
import pytest

from app.retry import (
    ErrorClass, RetryError, RetryPolicy, classify_error, is_permanent_error,
    is_permanent_failure_reason, is_throttled_error, with_retry,
)
from tests.fakes import ClientErr


def http_error(status):
    """An httpx-style error (what astrapy's DataAPIHttpException is a subclass of)."""
    req = httpx.Request("POST", "https://astra.example/api")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(status, request=req))


# ----------------------------- classification ----------------------------- #
@pytest.mark.parametrize("exc", [
    ClientErr("AccessDenied", 403), ClientErr("NoSuchKey", 404), ClientErr("404", None),
    ClientErr("", 401), ClientErr("InvalidObjectState", 403), ClientErr("SignatureDoesNotMatch", 403),
    http_error(401), http_error(403),
])
def test_permanent(exc):
    assert classify_error(exc) is ErrorClass.PERMANENT
    assert is_permanent_error(exc) and not is_throttled_error(exc)


@pytest.mark.parametrize("exc", [
    ClientErr("", 429), ClientErr("", 503), ClientErr("SlowDown", None), ClientErr("ServiceUnavailable", 503),
    http_error(429), http_error(503),
])
def test_throttled(exc):
    assert classify_error(exc) is ErrorClass.THROTTLED
    assert is_throttled_error(exc) and not is_permanent_error(exc)


@pytest.mark.parametrize("exc", [
    Exception("connection reset"), TimeoutError("t"), ClientErr("InternalError", 500), http_error(500), http_error(502),
])
def test_transient(exc):
    assert classify_error(exc) is ErrorClass.TRANSIENT


def test_classification_follows_cause_chain_and_retryerror():
    try:
        try:
            raise ClientErr("", 429)
        except Exception as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert classify_error(outer) is ErrorClass.THROTTLED
    policy = RetryPolicy(max_retries=1, throttle_max_retries=1)
    with pytest.raises(RetryError) as ei:
        with_retry(lambda: (_ for _ in ()).throw(ClientErr("", 503)), policy)
    assert ei.value.error_class is ErrorClass.THROTTLED
    assert classify_error(ei.value) is ErrorClass.THROTTLED


def test_permanent_wins_over_throttled_in_chain():
    try:
        try:
            raise ClientErr("AccessDenied", 403)
        except Exception as inner:
            raise http_error(429) from inner
    except Exception as outer:
        assert classify_error(outer) is ErrorClass.PERMANENT


def test_classify_none_is_transient():
    assert classify_error(None) is ErrorClass.TRANSIENT


def test_permanent_failure_reason_prefixes():
    for r in ("empty file", "file too large (3 bytes)", "unreadable file (x); skipped"):
        assert is_permanent_failure_reason(r)
    assert not is_permanent_failure_reason("download failed (x)")


# --------------------------------- retry --------------------------------- #
POLICY = RetryPolicy(max_retries=3, base_delay=1.0, max_delay=8.0,
                     throttle_max_retries=5, throttle_base_delay=5.0, throttle_max_delay=60.0)


def test_success_first_try(no_sleep):
    assert with_retry(lambda: 42, POLICY) == 42
    assert no_sleep == []


def test_permanent_fails_fast(no_sleep):
    n = {"c": 0}

    def denied():
        n["c"] += 1
        raise ClientErr("AccessDenied", 403)

    with pytest.raises(RetryError) as ei:
        with_retry(denied, POLICY, action_name="x")
    assert n["c"] == 1 and no_sleep == []
    assert ei.value.error_class is ErrorClass.PERMANENT and isinstance(ei.value, RuntimeError)
    assert "Failed x after 1 attempts" in str(ei.value)


def test_transient_retried_to_limit_with_exponential_backoff(no_sleep):
    n = {"c": 0}

    def flaky():
        n["c"] += 1
        raise Exception("connection reset")

    with pytest.raises(RetryError) as ei:
        with_retry(flaky, POLICY)
    assert n["c"] == 3 and ei.value.error_class is ErrorClass.TRANSIENT
    assert len(no_sleep) == 2
    assert 1.1 <= no_sleep[0] <= 1.5 and 2.1 <= no_sleep[1] <= 2.5     # base * 2**(n-1) + jitter(0.1-0.5)


def test_transient_succeeds_after_retries():
    calls = iter([Exception("a"), Exception("b"), "ok"])

    def f():
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    assert with_retry(f, POLICY) == "ok"


def test_backoff_is_capped(no_sleep):
    policy = RetryPolicy(max_retries=10, base_delay=1.0, max_delay=3.0)
    with pytest.raises(RetryError):
        with_retry(lambda: (_ for _ in ()).throw(Exception("x")), policy)
    assert max(no_sleep) <= 3.0


def test_max_retries_override():
    n = {"c": 0}

    def f():
        n["c"] += 1
        raise Exception("x")

    with pytest.raises(RetryError):
        with_retry(f, POLICY, max_retries=2)
    assert n["c"] == 2


@pytest.mark.parametrize("status", [429, 503])
def test_throttled_uses_longer_backoff_and_larger_budget(no_sleep, status):
    n = {"c": 0}

    def throttled():
        n["c"] += 1
        raise http_error(status)

    with pytest.raises(RetryError) as ei:
        with_retry(throttled, POLICY)
    assert n["c"] == 5                                   # throttle_max_retries, not max_retries
    assert ei.value.error_class is ErrorClass.THROTTLED
    assert len(no_sleep) == 4
    assert no_sleep[0] >= 5.0 and no_sleep[1] >= 10.0 and no_sleep[2] >= 20.0   # throttle_base_delay * 2**(n-1)
    assert max(no_sleep) <= 60.0
    assert min(no_sleep) >= 5.0


def test_throttled_then_recovers(no_sleep):
    calls = iter([http_error(429), http_error(503), "done"])

    def f():
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    assert with_retry(f, POLICY) == "done"
    assert len(no_sleep) == 2 and all(d >= 5.0 for d in no_sleep)


def test_throttle_to_permanent_stops_immediately(no_sleep):
    calls = iter([http_error(429), ClientErr("AccessDenied", 403), "never"])

    def f():
        v = next(calls)
        raise v

    with pytest.raises(RetryError) as ei:
        with_retry(f, POLICY)
    assert ei.value.error_class is ErrorClass.PERMANENT and len(no_sleep) == 1


def test_policy_from_settings(make_settings):
    s = make_settings(MAX_RETRIES=4, THROTTLE_MAX_RETRIES=7, THROTTLE_BASE_DELAY=2, THROTTLE_MAX_DELAY=30)
    p = RetryPolicy.from_settings(s)
    assert (p.max_retries, p.throttle_max_retries, p.throttle_base_delay, p.throttle_max_delay) == (4, 7, 2.0, 30.0)
