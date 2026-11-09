"""Application configuration loaded from environment variables.

All settings come from the environment so the same code runs locally and on
OpenShift (Secrets/ConfigMap injected as env vars). ``load_settings`` validates
everything up front and raises ``ConfigError`` with one message listing every
problem. Secret values are never included in error messages or ``repr``.

Tunable defaults mirror the constants block of
``unstructured-knowledge-agent/ingest_tools.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Tuple

# Required variables (no default). Values are secrets or deployment-specific.
REQUIRED_VARS: Tuple[str, ...] = (
    "ASTRA_DB_API_ENDPOINT",
    "ASTRA_DB_APPLICATION_TOKEN",
    "COS_APIKEY",
    "COS_RESOURCE_INSTANCE_ID",
    "UI_ACCESS_TOKEN",
)

DEFAULT_COS_ENDPOINT = "https://s3.jp-tok.cloud-object-storage.appdomain.cloud"
DEFAULT_SOURCE_BUCKET = "cloud-object-storage-cos-standard-vk7"
DEFAULT_ARCHIVE_BUCKET = "ww-archive"
DEFAULT_COLLECTION_NAME = "ibm_docs_rag2"
DEFAULT_JOBS_COLLECTION = "ingest_jobs"

# File types ingested by the current agent; other files stay in the source bucket.
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".csv", ".rst", ".log",
)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    # --- Astra DB (secrets are excluded from repr) ---
    astra_api_endpoint: str
    astra_token: str = field(repr=False)
    # --- IBM COS ---
    cos_apikey: str = field(repr=False)
    cos_resource_instance_id: str
    cos_endpoint: str
    # --- Buckets / collections ---
    source_bucket: str
    archive_bucket: str
    collection_name: str
    jobs_collection: str
    # --- UI protection ---
    ui_access_token: str = field(repr=False)
    # --- Concurrency / batching ---
    concurrency: int = 8              # parallel file workers (MAX_WORKERS)
    insert_batch: int = 40            # docs per insert_many call (INSERT_BATCH)
    list_page_size: int = 25          # files fetched per listing page (INGEST_PREFIX_BATCH)
    # --- Retry ---
    max_retries: int = 3              # attempts per operation (_with_retry default)
    retry_base_delay: float = 1.0     # seconds
    retry_max_delay: float = 8.0      # seconds
    # Longer backoff for throttled/unavailable responses (HTTP 429/503); see app/retry.py
    throttle_base_delay: float = 5.0  # seconds
    throttle_max_delay: float = 60.0  # seconds
    throttle_max_retries: int = 5     # attempts per operation when throttled
    # --- COS client ---
    cos_connect_timeout: int = 10     # seconds
    cos_read_timeout: int = 30        # seconds
    copy_object_max_bytes: int = 4 * 1024 ** 3  # larger objects use the managed multipart copy (COPY_OBJECT_MAX_BYTES)
    # --- Chunking / limits ---
    # Default 600 chars keeps token-dense markdown (~2 chars/token) under the 512-token embedding limit.
    chunk_size: int = 600             # characters per chunk (CHUNK_SIZE)
    chunk_overlap: int = 100          # characters shared between chunks (CHUNK_OVERLAP)
    max_file_bytes: int = 2 * 1024 ** 2   # files larger than this are skipped (MAX_FILE_BYTES)
    max_job_failures: int = 200       # cap on failures stored per job (MAX_JOB_FAILURES)
    max_job_events: int = 100         # cap on recent events kept per event log of a job (MAX_JOB_EVENTS)
    lease_seconds: int = 180          # job lease duration (LEASE_SECONDS)
    supported_extensions: Tuple[str, ...] = SUPPORTED_EXTENSIONS


def _parse_number(
    env: Mapping[str, str],
    name: str,
    default,
    cast: Callable,
    minimum,
    errors: List[str],
):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        errors.append(f"{name} must be a {cast.__name__} (got a non-numeric value)")
        return default
    if value < minimum:
        errors.append(f"{name} must be >= {minimum} (got {value})")
        return default
    return value


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build ``Settings`` from ``env`` (defaults to ``os.environ``).

    Raises ``ConfigError`` listing all missing/invalid variables. Never includes
    secret values in the message.
    """
    env = os.environ if env is None else env
    errors: List[str] = []

    def get(name: str, default: Optional[str] = None) -> str:
        value = (env.get(name) or "").strip()
        return value if value else (default or "")

    missing = [name for name in REQUIRED_VARS if not get(name)]
    if missing:
        errors.append("missing required environment variable(s): " + ", ".join(missing))

    cos_endpoint = get("COS_ENDPOINT", DEFAULT_COS_ENDPOINT).rstrip("/")
    if not cos_endpoint.startswith(("https://", "http://")):
        errors.append("COS_ENDPOINT must start with https:// (or http://)")

    astra_endpoint = get("ASTRA_DB_API_ENDPOINT")
    if astra_endpoint and not astra_endpoint.startswith(("https://", "http://")):
        errors.append("ASTRA_DB_API_ENDPOINT must start with https:// (or http://)")

    source_bucket = get("SOURCE_BUCKET", DEFAULT_SOURCE_BUCKET)
    archive_bucket = get("ARCHIVE_BUCKET", DEFAULT_ARCHIVE_BUCKET)
    if source_bucket == archive_bucket:
        errors.append("SOURCE_BUCKET and ARCHIVE_BUCKET must be different")

    concurrency = _parse_number(env, "CONCURRENCY", 8, int, 1, errors)
    insert_batch = _parse_number(env, "INSERT_BATCH", 40, int, 1, errors)
    list_page_size = _parse_number(env, "LIST_PAGE_SIZE", 25, int, 1, errors)
    max_retries = _parse_number(env, "MAX_RETRIES", 3, int, 1, errors)
    retry_base_delay = _parse_number(env, "RETRY_BASE_DELAY", 1.0, float, 0.0, errors)
    retry_max_delay = _parse_number(env, "RETRY_MAX_DELAY", 8.0, float, 0.0, errors)
    throttle_base_delay = _parse_number(env, "THROTTLE_BASE_DELAY", 5.0, float, 0.0, errors)
    throttle_max_delay = _parse_number(env, "THROTTLE_MAX_DELAY", 60.0, float, 0.0, errors)
    throttle_max_retries = _parse_number(env, "THROTTLE_MAX_RETRIES", 5, int, 1, errors)
    cos_connect_timeout = _parse_number(env, "COS_CONNECT_TIMEOUT", 10, int, 1, errors)
    cos_read_timeout = _parse_number(env, "COS_READ_TIMEOUT", 30, int, 1, errors)
    copy_object_max_bytes = _parse_number(env, "COPY_OBJECT_MAX_BYTES", 4 * 1024 ** 3, int, 1, errors)
    chunk_size = _parse_number(env, "CHUNK_SIZE", 600, int, 1, errors)
    chunk_overlap = _parse_number(env, "CHUNK_OVERLAP", 100, int, 0, errors)
    max_file_bytes = _parse_number(env, "MAX_FILE_BYTES", 2 * 1024 ** 2, int, 1, errors)
    max_job_failures = _parse_number(env, "MAX_JOB_FAILURES", 200, int, 1, errors)
    max_job_events = _parse_number(env, "MAX_JOB_EVENTS", 100, int, 1, errors)
    lease_seconds = _parse_number(env, "LEASE_SECONDS", 180, int, 1, errors)

    if chunk_overlap >= chunk_size:
        errors.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if retry_max_delay < retry_base_delay:
        errors.append("RETRY_MAX_DELAY must be >= RETRY_BASE_DELAY")
    if throttle_max_delay < throttle_base_delay:
        errors.append("THROTTLE_MAX_DELAY must be >= THROTTLE_BASE_DELAY")

    if errors:
        raise ConfigError(
            "Invalid configuration:\n  - " + "\n  - ".join(errors)
            + "\nSet these environment variables (values are never logged)."
        )

    return Settings(
        astra_api_endpoint=astra_endpoint,
        astra_token=get("ASTRA_DB_APPLICATION_TOKEN"),
        cos_apikey=get("COS_APIKEY"),
        cos_resource_instance_id=get("COS_RESOURCE_INSTANCE_ID"),
        cos_endpoint=cos_endpoint,
        source_bucket=source_bucket,
        archive_bucket=archive_bucket,
        collection_name=get("COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
        jobs_collection=get("JOBS_COLLECTION", DEFAULT_JOBS_COLLECTION),
        ui_access_token=get("UI_ACCESS_TOKEN"),
        concurrency=concurrency,
        insert_batch=insert_batch,
        list_page_size=list_page_size,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        throttle_base_delay=throttle_base_delay,
        throttle_max_delay=throttle_max_delay,
        throttle_max_retries=throttle_max_retries,
        cos_connect_timeout=cos_connect_timeout,
        cos_read_timeout=cos_read_timeout,
        copy_object_max_bytes=copy_object_max_bytes,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_file_bytes=max_file_bytes,
        max_job_failures=max_job_failures,
        max_job_events=max_job_events,
        lease_seconds=lease_seconds,
    )
