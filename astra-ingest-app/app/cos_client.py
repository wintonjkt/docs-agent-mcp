"""IBM COS client construction and helpers, ported from ``ingest_tools.py``.

Thread-safety: ``build_cos_client`` returns a low-level ``ibm_boto3`` S3 client. Low-level
botocore clients are documented as thread-safe for API calls, so build ONE client at startup
and share it among all worker threads (``max_pool_connections`` is sized for that). Do not
share boto3 *resources* or *sessions* across threads; this module only uses the client.
Client construction itself goes through the default boto3 session and is not guaranteed to be
thread-safe, so call ``build_cos_client`` once from the main thread.

The functions below take the shared client plus ``Settings`` and hold no state.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import ibm_boto3
from ibm_botocore.client import Config

from .astra_store import Document
from .config import Settings
from .retry import ErrorClass, RetryError, RetryPolicy, classify_error, with_retry


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def build_cos_client(settings: Settings) -> Any:
    """Create the shared IBM COS S3 client (OAuth/IAM api key, timeouts, pooled connections)."""
    client_config = Config(
        signature_version="oauth",
        connect_timeout=settings.cos_connect_timeout,
        read_timeout=settings.cos_read_timeout,
        # one connection per worker plus listing/move headroom
        max_pool_connections=max(25, settings.concurrency * 2),
        retries={"max_attempts": 3, "mode": "standard"},
    )
    return ibm_boto3.client(
        "s3",
        ibm_api_key_id=settings.cos_apikey,
        ibm_service_instance_id=settings.cos_resource_instance_id,
        endpoint_url=settings.cos_endpoint,
        config=client_config,
    )


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def discover_cos_batch(
    cos_client: Any,
    settings: Settings,
    bucket_name: str,
    prefix: str = "",
    checkpoint: str = "",
    target_count: Optional[int] = None,
    filter_extensions: Optional[Tuple[str, ...]] = None,
    deadline: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
    """Retrieve up to ``target_count`` matching objects, resuming after a checkpoint key.

    ``target_count`` defaults to ``settings.list_page_size``; ``filter_extensions`` defaults to
    ``settings.supported_extensions`` (pass ``()`` to disable filtering).

    The checkpoint is the last object key examined, passed to S3 as ``StartAfter``. Unlike S3 page
    tokens it is exact: objects remaining in the current page after ``target_count`` is reached are
    returned by the next call instead of being skipped.

    If ``deadline`` (a ``time.time()`` value) passes between pages, the scan stops early and returns
    whatever matched so far, which may be an empty list, with ``has_more=True`` and a checkpoint.
    Callers must therefore treat "no items but has_more" as "keep going", not as "finished".

    Returns:
        (items, next_checkpoint_key, has_more). ``next_checkpoint_key`` is None when the prefix is exhausted.
        Each item is ``{key, size, etag, last_modified}``.
    """
    policy = RetryPolicy.from_settings(settings)
    if target_count is None:
        target_count = settings.list_page_size
    if filter_extensions is None:
        filter_extensions = settings.supported_extensions

    matched_items: List[Dict[str, Any]] = []
    start_after = checkpoint.strip() if checkpoint and checkpoint.strip() else None
    if start_after and prefix and not start_after.startswith(prefix):
        raise ValueError(
            f"checkpoint '{start_after}' does not belong to prefix '{prefix}' "
            "(it is stale or from another listing); start again without a checkpoint"
        )
    last_key_examined = start_after
    has_more = False
    first_page = True

    while True:
        params: Dict[str, Any] = {
            "Bucket": bucket_name,
            "Prefix": prefix,
            # Small first page for dense prefixes; once a page did not fill the batch, scan sparse
            # prefixes (many unsupported files) with full-size pages.
            "MaxKeys": min(max(target_count * 2, 50), 1000) if first_page else 1000,
        }
        if start_after:
            params["StartAfter"] = start_after

        resp = with_retry(
            lambda p=params: cos_client.list_objects_v2(**p),
            policy,
            action_name=f"list_objects_v2 in '{bucket_name}'",
        )

        contents = resp.get("Contents", [])
        is_truncated = bool(resp.get("IsTruncated", False))
        target_reached = False

        for idx, item in enumerate(contents):
            key = item["Key"]
            last_key_examined = key
            if key.endswith("/"):
                continue
            if filter_extensions and not key.lower().endswith(tuple(filter_extensions)):
                continue
            last_modified = item.get("LastModified", "")
            matched_items.append({
                "key": key,
                "size": item.get("Size", 0),
                "etag": str(item.get("ETag", "")).strip('"'),
                "last_modified": last_modified.isoformat() if hasattr(last_modified, "isoformat") else str(last_modified),
            })
            if len(matched_items) >= target_count:
                target_reached = True
                # Unexamined objects left in this page, or further pages, mean more work remains.
                has_more = idx < len(contents) - 1 or is_truncated
                break

        if target_reached:
            break
        if not is_truncated or not contents:
            has_more = False
            break
        start_after = last_key_examined
        first_page = False
        if deadline is not None and time.time() >= deadline:
            # Sparse prefix: hand back what was found (possibly nothing) with an exact checkpoint so the
            # caller can save progress and continue instead of scanning past its time budget.
            has_more = True
            break

    return matched_items, (last_key_examined if has_more else None), has_more


# --------------------------------------------------------------------------- #
# Head / destination validation
# --------------------------------------------------------------------------- #
def is_not_found(exc: BaseException) -> bool:
    """True if a COS client exception represents HTTP 404 / missing key."""
    response = getattr(exc, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in ("404", "NoSuchKey", "NotFound") or status == 404


def head_object_or_none(cos_client: Any, settings: Settings, bucket_name: str, key: str) -> Optional[Dict[str, Any]]:
    """head_object with retries on transient errors; returns None when the object does not exist."""

    def _head() -> Optional[Dict[str, Any]]:
        try:
            return cos_client.head_object(Bucket=bucket_name, Key=key)
        except Exception as exc:  # noqa: BLE001
            if is_not_found(exc):
                return None
            raise

    return with_retry(_head, RetryPolicy.from_settings(settings), action_name=f"head_object '{bucket_name}/{key}'")


def normalize_dest_prefix(destination_prefix: str) -> str:
    """Strip leading slashes and ensure a non-empty prefix ends with '/'."""
    dest = (destination_prefix or "").strip().lstrip("/")
    return dest if not dest or dest.endswith("/") else dest + "/"


def validate_destination(cos_client: Any, settings: Settings, bucket_name: str, destination_bucket: str) -> str:
    """Return an error message if the destination bucket is unusable, else an empty string."""
    if not destination_bucket:
        return ""
    if destination_bucket == bucket_name:
        return (
            "destination_bucket must differ from the source bucket: moving within one bucket could "
            "re-list moved files and delete them."
        )
    try:
        with_retry(
            lambda: cos_client.head_bucket(Bucket=destination_bucket),
            RetryPolicy.from_settings(settings),
            action_name=f"head_bucket '{destination_bucket}'",
            max_retries=2,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Destination bucket '{destination_bucket}' is not accessible: {exc}"
    return ""


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
class FetchResult(NamedTuple):
    """Exactly one of ``doc`` / ``error`` is set. ``error_class`` is set for download exceptions only."""
    doc: Optional[Document]
    error: Optional[str]
    error_class: Optional[ErrorClass] = None


def fetch_document(cos_client: Any, settings: Settings, bucket_name: str, item_info: Dict[str, Any]) -> FetchResult:
    """Fetch and decode a single document from COS preserving metadata.

    Error strings keep the original prefixes (``file too large``, ``unreadable file``, ``empty file``,
    ``download failed``); see ``retry.is_permanent_failure_reason``.
    """
    key = item_info["key"] if isinstance(item_info, dict) else str(item_info)
    etag = item_info.get("etag") if isinstance(item_info, dict) else None
    last_mod = item_info.get("last_modified") if isinstance(item_info, dict) else None
    size = item_info.get("size") if isinstance(item_info, dict) else None

    if isinstance(size, int) and size > settings.max_file_bytes:
        return FetchResult(None, f"file too large ({size:,} bytes; limit {settings.max_file_bytes:,}); not ingested")

    try:
        obj = with_retry(
            lambda: cos_client.get_object(Bucket=bucket_name, Key=key),
            RetryPolicy.from_settings(settings),
            action_name=f"get_object '{key}'",
        )
        text_data = obj["Body"].read().decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        kind = classify_error(exc)
        if kind is ErrorClass.PERMANENT:
            return FetchResult(None, f"unreadable file ({exc}); skipped", kind)
        return FetchResult(None, f"download failed ({exc})", kind)

    if not text_data.strip():
        return FetchResult(None, "empty file")

    obj_last_modified = obj.get("LastModified")
    return FetchResult(
        Document(
            name=key,
            text=text_data,
            etag=etag or str(obj.get("ETag", "")).strip('"'),
            last_modified=last_mod or (
                obj_last_modified.isoformat() if hasattr(obj_last_modified, "isoformat") else str(obj_last_modified or "")
            ),
            bucket=bucket_name,
        ),
        None,
    )


# --------------------------------------------------------------------------- #
# Verified move
# --------------------------------------------------------------------------- #
class MoveResult(NamedTuple):
    """``moved`` False carries a ``reason``. ``error_class`` is the class of the exception that caused
    the failure (None for logical refusals such as ETag mismatch, collision or verification failure)."""
    moved: bool
    reason: str = ""
    error_class: Optional[ErrorClass] = None


def _exc_class(exc: BaseException) -> ErrorClass:
    return exc.error_class if isinstance(exc, RetryError) else classify_error(exc)


def move_object(
    cos_client: Any,
    collection: Any,
    settings: Settings,
    src_bucket: str,
    key: str,
    dst_bucket: str,
    dst_key: str,
    expected_etag: str = "",
) -> MoveResult:
    """Move one object: verify source, copy, verify copy, update Astra rows, then delete the source.

    The source is deleted last and only after every earlier step succeeded, so any failure leaves the
    source intact. Never overwrites a different object at the destination. If deleting the source fails
    after the Astra rows were repointed, the rows are rolled back and a copy created by this call is removed.
    """
    policy = RetryPolicy.from_settings(settings)
    try:
        src = head_object_or_none(cos_client, settings, src_bucket, key)
        if src is None:
            return MoveResult(False, "source object no longer exists")
        src_etag = str(src.get("ETag", "")).strip('"')
        src_size = src.get("ContentLength")
        if expected_etag and src_etag != expected_etag.strip('"'):
            return MoveResult(False, "source changed since it was ingested (ETag mismatch); not moved")

        def _same_as_source(head: Dict[str, Any], just_copied: bool = False) -> bool:
            dst_etag = str(head.get("ETag", "")).strip('"')
            if head.get("ContentLength") != src_size:
                return False
            if dst_etag == src_etag:
                return True
            # Multipart ETags ("-N") are not content hashes and differ after a copy. A matching size is only
            # accepted as proof for an object this call just created, never for one that was already there.
            return just_copied and "-" in src_etag

        existing = head_object_or_none(cos_client, settings, dst_bucket, dst_key)
        created_copy = False
        if existing is not None:
            if not _same_as_source(existing):
                return MoveResult(False, (
                    f"destination '{dst_bucket}/{dst_key}' already contains an object that cannot be proven identical "
                    "to the source (different ETag/size); not overwritten, source kept"
                ))
        else:
            copy_source = {"Bucket": src_bucket, "Key": key}
            if src_size is not None and src_size > settings.copy_object_max_bytes:
                with_retry(
                    lambda: cos_client.copy(copy_source, dst_bucket, dst_key),
                    policy, action_name=f"managed copy '{key}'", max_retries=2,
                )
            else:
                with_retry(
                    lambda: cos_client.copy_object(CopySource=copy_source, Bucket=dst_bucket, Key=dst_key),
                    policy, action_name=f"copy_object '{key}'",
                )
            created_copy = True

        copied = head_object_or_none(cos_client, settings, dst_bucket, dst_key)
        if copied is None or not _same_as_source(copied, just_copied=created_copy):
            if created_copy:
                try:
                    cos_client.delete_object(Bucket=dst_bucket, Key=dst_key)
                except Exception:  # noqa: BLE001
                    pass
            return MoveResult(False, "copy verification failed (size/ETag mismatch); source kept")

        try:
            with_retry(
                lambda: collection.update_many(
                    {"source": key, "$or": [{"bucket": src_bucket}, {"bucket": {"$exists": False}}]},
                    {"$set": {"bucket": dst_bucket, "source": dst_key}},
                ),
                policy, action_name=f"update Astra rows for '{key}'",
            )
        except Exception as exc:  # noqa: BLE001
            return MoveResult(
                False, f"copied to destination but Astra records could not be updated; source kept ({exc})",
                _exc_class(exc),
            )

        try:
            with_retry(
                lambda: cos_client.delete_object(Bucket=src_bucket, Key=key),
                policy, action_name=f"delete_object '{key}'",
            )
        except Exception as exc:  # noqa: BLE001
            # Roll the Astra rows back so they keep matching the source that is still in place.
            try:
                with_retry(
                    lambda: collection.update_many(
                        {"source": dst_key, "bucket": dst_bucket},
                        {"$set": {"bucket": src_bucket, "source": key}},
                    ),
                    policy, action_name=f"roll back Astra rows for '{key}'",
                )
            except Exception as rb_exc:  # noqa: BLE001
                return MoveResult(False, (
                    f"copied to destination, but deleting the source failed ({exc}) and the Astra records could not "
                    f"be rolled back ({rb_exc}); the file exists in both buckets and its records point at "
                    f"'{dst_bucket}/{dst_key}'"
                ), _exc_class(exc))
            if created_copy:
                try:
                    cos_client.delete_object(Bucket=dst_bucket, Key=dst_key)
                except Exception:  # noqa: BLE001
                    pass
            return MoveResult(False, (
                f"copied to destination but deleting the source failed ({exc}); "
                "records rolled back, source kept"
            ), _exc_class(exc))
        return MoveResult(True)
    except Exception as exc:  # noqa: BLE001
        return MoveResult(False, str(exc), _exc_class(exc))
