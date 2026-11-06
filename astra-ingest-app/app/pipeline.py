"""Per-file ingest lifecycle and batch execution.

Each file follows an idempotent multi-step lifecycle:
1. Smart-skip check: query Astra DB for an existing row's ETag. If identical, skip re-chunking/embedding.
   If archive_bucket is set, move the already-up-to-date source file to the archive bucket.
2. Fetch & decode: download object body from IBM COS.
3. Chunk & embed: paragraph-aware chunking with $vectorize embedding on each chunk row.
4. Idempotent versioned insertion: insert all chunks for new version, delete older versions.
5. Verified move: copy to archive bucket, verify integrity, update Astra DB rows, delete source object.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .astra_store import Document, new_version_id, now_iso, safe_ingest_single_doc
from .config import Settings
from .cos_client import fetch_document, move_object
from .retry import ErrorClass, is_permanent_error, is_permanent_failure_reason


@dataclass
class FileOutcome:
    """Outcome of processing a single file through the pipeline."""
    key: str
    status: str                          # "ingested", "skipped", "failed"
    size: int = 0
    chunks: int = 0
    bytes_count: int = 0
    error: str = ""
    error_class: Optional[ErrorClass] = None
    moved: bool = False
    move_reason: str = ""

    @property
    def is_failed(self) -> bool:
        return self.status == "failed" or bool(self.move_reason)

    @property
    def is_permanent_failure(self) -> bool:
        if self.error_class is ErrorClass.PERMANENT:
            return True
        if self.error and is_permanent_failure_reason(self.error):
            return True
        return False


def process_file(
    cos_client: Any,
    collection: Any,
    settings: Settings,
    item_info: Dict[str, Any],
    src_bucket: str,
    archive_bucket: Optional[str] = None,
    existing_etag: Optional[str] = None,
) -> FileOutcome:
    """Execute the full per-file ingest lifecycle with zero data loss on failure."""
    key = item_info.get("key", "")
    item_etag = item_info.get("etag", "").strip('"')
    size = item_info.get("size", 0)

    # 1. Smart-skip check
    if item_etag and existing_etag and item_etag == existing_etag.strip('"'):
        if archive_bucket and archive_bucket != src_bucket:
            move_res = move_object(
                cos_client, collection, settings,
                src_bucket=src_bucket, key=key,
                dst_bucket=archive_bucket, dst_key=key,
                expected_etag=item_etag,
            )
            if not move_res.moved:
                return FileOutcome(
                    key=key, status="skipped", size=size, bytes_count=0,
                    moved=False, move_reason=move_res.reason, error_class=move_res.error_class,
                )
            return FileOutcome(key=key, status="skipped", size=size, bytes_count=0, moved=True)
        return FileOutcome(key=key, status="skipped", size=size, bytes_count=0)

    # 2. Fetch document
    fetch_res = fetch_document(cos_client, settings, src_bucket, item_info)
    if fetch_res.error or fetch_res.doc is None:
        return FileOutcome(
            key=key, status="failed", size=size,
            error=fetch_res.error or "fetch failed", error_class=fetch_res.error_class,
        )

    doc: Document = fetch_res.doc
    doc_bytes = len(doc.text.encode("utf-8", errors="ignore"))

    # 3 & 4. Ingest to Astra DB
    version_id = new_version_id()
    ingest_time = now_iso()
    ingest_res = safe_ingest_single_doc(collection, settings, doc, version_id, ingest_time)
    if not ingest_res.ok:
        return FileOutcome(
            key=key, status="failed", size=size, bytes_count=doc_bytes,
            error=ingest_res.message, error_class=ingest_res.error_class,
        )

    # 5. Verified move to archive
    if archive_bucket and archive_bucket != src_bucket:
        move_res = move_object(
            cos_client, collection, settings,
            src_bucket=src_bucket, key=key,
            dst_bucket=archive_bucket, dst_key=key,
            expected_etag=(doc.etag or item_etag).strip('"'),
        )
        if not move_res.moved:
            return FileOutcome(
                key=key, status="ingested", size=size, chunks=ingest_res.chunks, bytes_count=doc_bytes,
                moved=False, move_reason=move_res.reason, error_class=move_res.error_class,
            )
        return FileOutcome(
            key=key, status="ingested", size=size, chunks=ingest_res.chunks, bytes_count=doc_bytes,
            moved=True,
        )

    return FileOutcome(
        key=key, status="ingested", size=size, chunks=ingest_res.chunks, bytes_count=doc_bytes,
    )


def process_batch_parallel(
    cos_client: Any,
    collection: Any,
    settings: Settings,
    items: List[Dict[str, Any]],
    src_bucket: str,
    archive_bucket: Optional[str] = None,
    existing_etags: Optional[Dict[str, str]] = None,
    max_workers: Optional[int] = None,
    on_file_complete: Optional[Any] = None,
) -> List[FileOutcome]:
    """Process a batch of files concurrently using a ThreadPoolExecutor."""
    if not items:
        return []

    workers = max_workers or settings.concurrency
    etags = existing_etags or {}
    outcomes: List[FileOutcome] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(
                process_file,
                cos_client, collection, settings, item,
                src_bucket, archive_bucket, etags.get(item["key"]),
            ): item
            for item in items
        }
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                outcome = future.result()
            except Exception as exc:  # noqa: BLE001
                item = future_to_item[future]
                outcome = FileOutcome(
                    key=item.get("key", ""),
                    status="failed",
                    size=item.get("size", 0),
                    error=f"Unexpected pipeline exception: {exc}",
                )
            outcomes.append(outcome)
            if on_file_complete:
                try:
                    on_file_complete(outcome)
                except Exception:  # noqa: BLE001
                    pass

    return outcomes
