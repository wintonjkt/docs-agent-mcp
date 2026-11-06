"""Retry with backoff, and classification of errors as permanent / transient / throttled.

Ported from ``_with_retry`` / ``_is_permanent_error`` in ``ingest_tools.py`` with one
new rule: HTTP 429 / 503 (rate limiting, service unavailable) are *throttled*. They
are retried with a longer backoff and a separate attempt budget, and they are neither
permanent nor a per-file problem, so the (later) circuit breaker can treat them as
"back off / pause" rather than "this file is bad".

``classify_error`` looks through the ``__cause__`` chain, so the ``RetryError`` raised by
``with_retry`` classifies the same as the underlying failure.
"""

from __future__ import annotations

import enum
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import Settings


class ErrorClass(str, enum.Enum):
    PERMANENT = "permanent"   # retrying cannot help (access denied, missing key, ...)
    TRANSIENT = "transient"   # network blips, 5xx, timeouts: retry with normal backoff
    THROTTLED = "throttled"   # HTTP 429/503: retry with long backoff; says nothing about the file


PERMANENT_ERROR_CODES = frozenset({
    "401", "403", "404", "AccessDenied", "Forbidden", "NoSuchKey", "NotFound", "NoSuchBucket",
    "InvalidObjectState", "InvalidAccessKeyId", "SignatureDoesNotMatch",
})
PERMANENT_HTTP_STATUSES = frozenset({401, 403, 404})
THROTTLED_ERROR_CODES = frozenset({"429", "503", "SlowDown", "ServiceUnavailable", "TooManyRequests"})
THROTTLED_HTTP_STATUSES = frozenset({429, 503})

# Per-file failure reasons that are permanent for that file and say nothing about Astra/COS health.
PERMANENT_FAILURE_PREFIXES = ("empty file", "file too large", "unreadable file")

_MAX_CHAIN = 5


class RetryError(RuntimeError):
    """Raised by ``with_retry`` when attempts are exhausted or the error is permanent.

    ``__cause__`` is the last underlying exception; ``error_class`` is its classification.
    """

    def __init__(self, message: str, error_class: ErrorClass):
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    throttle_max_retries: int = 5
    throttle_base_delay: float = 5.0
    throttle_max_delay: float = 60.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetryPolicy":
        return cls(
            max_retries=settings.max_retries,
            base_delay=settings.retry_base_delay,
            max_delay=settings.retry_max_delay,
            throttle_max_retries=settings.throttle_max_retries,
            throttle_base_delay=settings.throttle_base_delay,
            throttle_max_delay=settings.throttle_max_delay,
        )


def _code_and_status(exc: BaseException) -> tuple:
    """(error code string, HTTP status or None) from a boto-style or httpx-style exception."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):  # botocore ClientError
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code, status
    status = getattr(response, "status_code", None)  # httpx.HTTPStatusError / astrapy DataAPIHttpException
    return "", status if isinstance(status, int) else None


def classify_error(exc: Optional[BaseException]) -> ErrorClass:
    """Classify an exception (following ``__cause__``) as permanent, throttled or transient.

    Permanent wins over throttled when both appear in the chain.
    """
    throttled = False
    seen = 0
    while exc is not None and seen < _MAX_CHAIN:
        code, status = _code_and_status(exc)
        if code in PERMANENT_ERROR_CODES or status in PERMANENT_HTTP_STATUSES:
            return ErrorClass.PERMANENT
        if code in THROTTLED_ERROR_CODES or status in THROTTLED_HTTP_STATUSES:
            throttled = True
        exc = exc.__cause__
        seen += 1
    return ErrorClass.THROTTLED if throttled else ErrorClass.TRANSIENT


def is_permanent_error(exc: BaseException) -> bool:
    """True for errors that retrying cannot fix (access denied, missing key, archived object, ...)."""
    return classify_error(exc) is ErrorClass.PERMANENT


def is_throttled_error(exc: BaseException) -> bool:
    return classify_error(exc) is ErrorClass.THROTTLED


def is_permanent_failure_reason(reason: str) -> bool:
    """True for per-file failure reasons that are permanent for the file (not an infrastructure signal)."""
    return reason.startswith(PERMANENT_FAILURE_PREFIXES)


def with_retry(
    func: Callable[[], Any],
    policy: RetryPolicy,
    action_name: str = "operation",
    max_retries: Optional[int] = None,
) -> Any:
    """Execute ``func`` with exponential backoff and jitter.

    * permanent errors fail fast (no retry);
    * transient errors get ``max_retries`` attempts (``policy.max_retries`` unless overridden),
      delays ``base * 2**(n-1) + jitter`` capped at ``max_delay``;
    * throttled errors (HTTP 429/503) get ``policy.throttle_max_retries`` attempts with the longer
      ``throttle_*`` delays. The attempt counter is shared, the limit follows the latest error class.

    Raises ``RetryError`` (a ``RuntimeError``) chained from the last exception.
    """
    transient_limit = max_retries if max_retries is not None else policy.max_retries
    last_exc: Optional[BaseException] = None
    kind = ErrorClass.TRANSIENT
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            kind = classify_error(exc)
            if kind is ErrorClass.PERMANENT:
                break
            if kind is ErrorClass.THROTTLED:
                limit, base, cap = policy.throttle_max_retries, policy.throttle_base_delay, policy.throttle_max_delay
            else:
                limit, base, cap = transient_limit, policy.base_delay, policy.max_delay
            if attempt >= limit:
                break
            time.sleep(min(base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), cap))
    raise RetryError(f"Failed {action_name} after {attempt} attempts: {last_exc}", kind) from last_exc
