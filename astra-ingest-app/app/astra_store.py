"""Astra DB collection / jobs-collection accessors and document ingestion, ported from ``ingest_tools.py``.

astrapy 2.x notes (verified against 2.2.1):
  * ``DataAPIClient(api_options=APIOptions(timeout_options=TimeoutOptions(...)))`` then
    ``client.get_database(endpoint, token=token)`` and ``database.get_collection(name)``.
    ``get_collection`` makes no network call and does not check the collection exists.
  * ``insert_many(docs, ordered=False)`` sends chunks of 50 docs (``chunk_size``) with up to 20
    threads; with ``Settings.insert_batch`` <= 50 every call is a single HTTP request.
  * HTTP 429/503 surface as ``DataAPIHttpException`` (an ``httpx.HTTPStatusError``, so
    ``exc.response.status_code`` works with ``retry.classify_error``). Per-document errors in an
    HTTP-200 reply raise ``CollectionInsertManyException``.
  * ``delete_many`` / ``update_many`` loop internally over server pages until done.

Thread-safety: build ONE ``AstraStore`` at startup and share it. astrapy's ``DataAPIClient``,
``Database`` and ``Collection`` hold only immutable options plus a shared ``httpx.Client``
(thread-safe, pooled), and keep no per-request state, so concurrent ``find`` / ``insert_many`` /
``delete_many`` / ``update_many`` from worker threads are supported. Caveat: astrapy does not document
this guarantee explicitly; it is only exercised against fakes in the unit tests and should be
confirmed by the cluster tests in Sub-Task 8. ``AstraStore`` itself only mutates its collection
cache and the "jobs collection created" flag, both under a lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional

from astrapy import DataAPIClient
from astrapy.api_options import APIOptions, TimeoutOptions

from .chunking import chunk_document
from .config import Settings
from .retry import ErrorClass, RetryError, RetryPolicy, classify_error, with_retry


@dataclass
class Document:
    """A single text document to ingest (replaces the original ``DocInput``)."""

    name: str
    text: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    bucket: Optional[str] = None


class IngestOutcome(NamedTuple):
    """Result of ``safe_ingest_single_doc``. ``error_class`` is set only when ``ok`` is False."""

    ok: bool
    chunks: int
    message: str
    error_class: Optional[ErrorClass] = None


def new_version_id() -> int:
    """Millisecond timestamp used as ``ingest_version`` (same scheme as the original)."""
    return int(time.time() * 1000)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AstraStore:
    """Holds the Astra database handle and hands out collections. Build once, share across threads."""

    def __init__(self, settings: Settings, database: Any = None):
        self.settings = settings
        self.policy = RetryPolicy.from_settings(settings)
        if database is None:
            client = DataAPIClient(
                api_options=APIOptions(
                    timeout_options=TimeoutOptions(
                        request_timeout_ms=30_000,           # astrapy default is 10 s; vectorize calls can be slower
                        general_method_timeout_ms=120_000,   # whole insert_many/delete_many/update_many call
                    )
                )
            )
            database = client.get_database(settings.astra_api_endpoint, token=settings.astra_token)
        self.database = database
        self._lock = threading.Lock()
        self._jobs_ready = False

    def get_collection(self, name: Optional[str] = None) -> Any:
        """Return the (vectorize-enabled) ingest collection. Never creates it: it must already exist."""
        return self.database.get_collection(name or self.settings.collection_name)

    def get_jobs_collection(self) -> Any:
        """Return the plain jobs collection, creating it (without vectorize) on first use."""
        name = self.settings.jobs_collection
        with self._lock:
            if not self._jobs_ready:
                if name not in set(self.database.list_collection_names()):
                    try:
                        self.database.create_collection(name)
                    except Exception:  # noqa: BLE001
                        # Another pod may have created it concurrently; re-check before giving up.
                        if name not in set(self.database.list_collection_names()):
                            raise
                self._jobs_ready = True
        return self.database.get_collection(name)


def build_astra_store(settings: Settings) -> AstraStore:
    return AstraStore(settings)


# --------------------------------------------------------------------------- #
# Smart-skip
# --------------------------------------------------------------------------- #
def get_existing_doc_etags(
    collection: Any,
    settings: Settings,
    source_names: List[str],
    bucket_name: str = "",
) -> Dict[str, str]:
    """Query Astra for stored ETags of existing documents (for smart-skip).

    Rows ingested for a different bucket are ignored when ``bucket_name`` is given. Legacy rows
    without a 'bucket' field are treated as belonging to the requested bucket. A row that explicitly
    belongs to the bucket wins over a legacy row. Raises ``RetryError`` on Astra errors so the caller
    can report that smart-skip was unavailable.
    """
    if not source_names:
        return {}

    policy = RetryPolicy.from_settings(settings)
    existing_etags: Dict[str, str] = {}
    for i in range(0, len(source_names), 50):
        batch_sources = source_names[i : i + 50]
        rows = with_retry(
            lambda b=batch_sources: list(collection.find(
                {"source": {"$in": b}, "chunk_no": 0},
                projection={"source": 1, "etag": 1, "bucket": 1},
            )),
            policy,
            action_name="find existing ETags",
        )
        for doc in rows:
            src = doc.get("source")
            etag = doc.get("etag")
            row_bucket = doc.get("bucket")
            if bucket_name and row_bucket and row_bucket != bucket_name:
                continue
            if src and etag:
                if row_bucket or src not in existing_etags:
                    existing_etags[src] = str(etag).strip('"')
    return existing_etags


# --------------------------------------------------------------------------- #
# Safe ingest
# --------------------------------------------------------------------------- #
def safe_ingest_single_doc(
    collection: Any,
    settings: Settings,
    doc: Document,
    version_id: int,
    ingest_time: str,
) -> IngestOutcome:
    """Ingest one document with zero data loss on partial failure.

    Inserts all chunks of the new version (``$vectorize`` per chunk, ``settings.insert_batch`` per call),
    then deletes older versions of the same source (scoped to the document's bucket, plus legacy rows with
    no bucket field). If anything fails the new version's rows are removed so the old version stays intact.
    """
    chunks = chunk_document(doc.text, settings)
    if not chunks:
        return IngestOutcome(True, 0, "skipped (empty text)")

    rows = [
        {
            "source": doc.name,
            "bucket": doc.bucket or "",
            "chunk_no": n,
            "chunk_count": len(chunks),
            "char_count": len(chunk),
            "etag": doc.etag or "",
            "last_modified": doc.last_modified or "",
            "ingest_version": version_id,
            "ingest_time": ingest_time,
            "text": chunk,
            "$vectorize": chunk,
        }
        for n, chunk in enumerate(chunks)
    ]

    old_filter: Dict[str, Any] = {"source": doc.name, "ingest_version": {"$ne": version_id}}
    new_filter: Dict[str, Any] = {"source": doc.name, "ingest_version": version_id}
    if doc.bucket:
        # Only touch rows of this bucket (or legacy rows with no bucket field), never another bucket's copy.
        scope = [{"bucket": doc.bucket}, {"bucket": {"$exists": False}}]
        old_filter["$or"] = scope
        new_filter["$or"] = scope

    policy = RetryPolicy.from_settings(settings)
    batch_size = settings.insert_batch
    inserted_count = 0
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            with_retry(
                lambda b=batch: collection.insert_many(b, ordered=False),
                policy,
                action_name=f"insert_many for '{doc.name}'",
            )
            inserted_count += len(batch)

        with_retry(
            lambda: collection.delete_many(old_filter),
            policy,
            action_name=f"delete_many old versions for '{doc.name}'",
        )
        return IngestOutcome(True, inserted_count, f"{inserted_count} chunks")

    except Exception as exc:  # noqa: BLE001
        try:
            collection.delete_many(new_filter)
        except Exception:  # noqa: BLE001
            pass  # leftover new rows are removed by the next ingest (their version differs from the next one)
        kind = exc.error_class if isinstance(exc, RetryError) else classify_error(exc)
        return IngestOutcome(False, 0, f"FAILED ({exc})", kind)
