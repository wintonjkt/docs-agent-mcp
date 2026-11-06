"""Unit tests for app/pipeline.py (file lifecycle, smart-skip, batch execution)."""

import pytest

from app.astra_store import Document
from app.pipeline import FileOutcome, process_batch_parallel, process_file
from app.retry import ErrorClass
from tests.fakes import FakeCollection, FakeCOS


def test_process_file_success_with_archive(settings):
    cos = FakeCOS(keys=["doc1.txt"])
    col = FakeCollection()
    item = {"key": "doc1.txt", "size": 100, "etag": FakeCOS.etag(b"text for doc1.txt")}

    outcome = process_file(cos, col, settings, item, src_bucket="src", archive_bucket="dst")

    assert outcome.status == "ingested"
    assert outcome.chunks > 0
    assert outcome.moved is True
    assert "doc1.txt" in cos.b["dst"]
    assert "doc1.txt" not in cos.b["src"]
    assert len(col.rows) > 0
    assert col.rows[0]["bucket"] == "dst"


def test_process_file_success_without_archive(settings):
    cos = FakeCOS(keys=["doc1.txt"])
    col = FakeCollection()
    item = {"key": "doc1.txt", "size": 100, "etag": FakeCOS.etag(b"text for doc1.txt")}

    outcome = process_file(cos, col, settings, item, src_bucket="src", archive_bucket=None)

    assert outcome.status == "ingested"
    assert outcome.chunks > 0
    assert outcome.moved is False
    assert "doc1.txt" in cos.b["src"]


def test_process_file_smart_skip_with_archive(settings):
    cos = FakeCOS(keys=["doc1.txt"])
    col = FakeCollection()
    etag = FakeCOS.etag(b"text for doc1.txt")
    item = {"key": "doc1.txt", "size": 100, "etag": etag}

    # Pre-populate Astra DB with matching ETag
    col.rows.append({"source": "doc1.txt", "etag": etag.strip('"'), "chunk_no": 0, "bucket": "src"})

    outcome = process_file(
        cos, col, settings, item,
        src_bucket="src", archive_bucket="dst", existing_etag=etag,
    )

    assert outcome.status == "skipped"
    assert outcome.moved is True
    assert "doc1.txt" in cos.b["dst"]
    assert "doc1.txt" not in cos.b["src"]


def test_process_file_empty_file_fails_permanently(settings):
    cos = FakeCOS()
    cos.b["src"]["empty.txt"] = b""
    col = FakeCollection()
    item = {"key": "empty.txt", "size": 0, "etag": "abc"}

    outcome = process_file(cos, col, settings, item, src_bucket="src", archive_bucket="dst")

    assert outcome.status == "failed"
    assert "empty file" in outcome.error
    assert outcome.is_permanent_failure is True


def test_process_file_too_large_fails_permanently(settings, make_settings):
    s = make_settings(MAX_FILE_BYTES=50)
    cos = FakeCOS()
    cos.b["src"]["big.txt"] = b"x" * 100
    col = FakeCollection()
    item = {"key": "big.txt", "size": 100, "etag": "abc"}

    outcome = process_file(cos, col, s, item, src_bucket="src", archive_bucket="dst")

    assert outcome.status == "failed"
    assert "file too large" in outcome.error
    assert outcome.is_permanent_failure is True


def test_process_file_ingest_failure(settings):
    cos = FakeCOS(keys=["doc1.txt"])
    col = FakeCollection()
    col.fail_insert = True
    item = {"key": "doc1.txt", "size": 100, "etag": FakeCOS.etag(b"text for doc1.txt")}

    outcome = process_file(cos, col, settings, item, src_bucket="src", archive_bucket="dst")

    assert outcome.status == "failed"
    assert "FAILED" in outcome.error
    assert outcome.is_permanent_failure is False


def test_process_batch_parallel(settings):
    keys = [f"file{i}.txt" for i in range(10)]
    cos = FakeCOS(keys=keys)
    col = FakeCollection()
    items = [{"key": k, "size": 100, "etag": FakeCOS.etag(b"text for " + k.encode())} for k in keys]

    outcomes = process_batch_parallel(cos, col, settings, items, src_bucket="src", archive_bucket="dst")

    assert len(outcomes) == 10
    assert all(o.status == "ingested" and o.moved is True for o in outcomes)
    assert len(cos.b["src"]) == 0
    assert len(cos.b["dst"]) == 10
