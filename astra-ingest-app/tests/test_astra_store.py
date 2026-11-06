import threading

import pytest

from app.astra_store import (
    AstraStore, Document, get_existing_doc_etags, new_version_id, safe_ingest_single_doc,
)
from app.retry import ErrorClass
from tests.fakes import ClientErr, FakeCollection, FakeDB


def doc(name="a.md", text="hello world", bucket="B", etag="et1"):
    return Document(name=name, text=text, etag=etag, last_modified="lm", bucket=bucket)


# ------------------------------ safe ingest ------------------------------ #
def test_ingest_inserts_rows_with_vectorize_and_metadata(settings):
    col = FakeCollection()
    out = safe_ingest_single_doc(col, settings, doc(), 5, "now")
    assert out.ok and out.chunks == 1 and out.message == "1 chunks" and out.error_class is None
    r = col.rows[0]
    assert r["$vectorize"] == r["text"] == "hello world"
    assert (r["source"], r["bucket"], r["chunk_no"], r["chunk_count"], r["etag"], r["ingest_version"]) == ("a.md", "B", 0, 1, "et1", 5)
    assert r["ingest_time"] == "now" and r["last_modified"] == "lm" and r["char_count"] == 11


def test_ingest_empty_text_is_skipped(settings):
    col = FakeCollection()
    out = safe_ingest_single_doc(col, settings, doc(text="  \n "), 1, "now")
    assert out.ok and out.chunks == 0 and "skipped" in out.message and col.rows == [] and col.delete_filters == []


def test_ingest_batch_size_comes_from_settings(make_settings):
    s = make_settings(CHUNK_SIZE=10, CHUNK_OVERLAP=2, INSERT_BATCH=3)
    col = FakeCollection()
    text = "\n\n".join(["x" * 10] * 8)             # 8 chunks
    out = safe_ingest_single_doc(col, s, doc(text=text), 1, "now")
    assert out.ok and out.chunks == 8 and col.insert_batches == [3, 3, 2]


def test_reingest_replaces_old_version_in_same_bucket_only(settings):
    col = FakeCollection()
    safe_ingest_single_doc(col, settings, doc(bucket="B", text="old text"), 1, "t1")
    safe_ingest_single_doc(col, settings, doc(bucket="C", text="other bucket"), 1, "t1")
    col.rows.append({"source": "a.md", "chunk_no": 0, "ingest_version": 0, "text": "legacy"})   # legacy row, no bucket
    safe_ingest_single_doc(col, settings, doc(bucket="B", text="new text"), 2, "t2")
    mine = [r for r in col.rows if r.get("bucket") == "B"]
    assert [r["text"] for r in mine] == ["new text"] and mine[0]["ingest_version"] == 2
    assert [r["text"] for r in col.rows if r.get("bucket") == "C"] == ["other bucket"]
    assert not [r for r in col.rows if r["text"] == "legacy"], "legacy bucket-less rows are replaced"


def test_delete_filters_are_bucket_scoped(settings):
    col = FakeCollection()
    safe_ingest_single_doc(col, settings, doc(bucket="B"), 5, "now")
    f = col.delete_filters[0]
    assert f["$or"] == [{"bucket": "B"}, {"bucket": {"$exists": False}}] and f["ingest_version"] == {"$ne": 5}
    col.delete_filters.clear()
    safe_ingest_single_doc(col, settings, doc(bucket=None), 6, "now")
    assert "$or" not in col.delete_filters[0]


def test_insert_failure_rolls_back_new_rows_and_keeps_old(settings, no_sleep):
    col = FakeCollection()
    safe_ingest_single_doc(col, settings, doc(text="old"), 1, "t1")
    col.fail_insert = True
    out = safe_ingest_single_doc(col, settings, doc(text="new"), 2, "t2")
    col.fail_insert = False
    assert not out.ok and out.chunks == 0 and out.message.startswith("FAILED") and out.error_class is ErrorClass.TRANSIENT
    assert [r["text"] for r in col.rows] == ["old"]
    f = col.delete_filters[-1]
    assert f["ingest_version"] == 2 and f["$or"][0] == {"bucket": "B"}      # rollback targets only the new version


def test_partial_insert_failure_rolls_back_inserted_batches(make_settings, no_sleep):
    s = make_settings(CHUNK_SIZE=10, CHUNK_OVERLAP=2, INSERT_BATCH=2)
    col = FakeCollection()
    real, calls = col.insert_many, {"n": 0}

    def flaky(batch, ordered=False):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise Exception("astra down")
        real(batch, ordered=ordered)

    col.insert_many = flaky
    out = safe_ingest_single_doc(col, s, doc(text="\n\n".join(["x" * 10] * 5)), 7, "t")
    assert not out.ok and col.rows == []                 # first batch was inserted, then rolled back


def test_delete_old_failure_rolls_back_new_rows(settings, no_sleep):
    col = FakeCollection()
    safe_ingest_single_doc(col, settings, doc(text="old"), 1, "t1")
    col.fail_delete_old = True
    out = safe_ingest_single_doc(col, settings, doc(text="new"), 2, "t2")
    assert not out.ok and [r["text"] for r in col.rows] == ["old"]


def test_rollback_failure_is_swallowed(settings, no_sleep):
    col = FakeCollection()
    col.fail_insert = True
    col.delete_many = lambda f: (_ for _ in ()).throw(Exception("rollback down"))
    out = safe_ingest_single_doc(col, settings, doc(), 1, "t")
    assert not out.ok and out.message.startswith("FAILED")


@pytest.mark.parametrize("code,status,kind", [
    ("", 429, ErrorClass.THROTTLED), ("AccessDenied", 403, ErrorClass.PERMANENT),
])
def test_ingest_failure_carries_error_class(settings, no_sleep, code, status, kind):
    col = FakeCollection()
    col.insert_many = lambda b, ordered=False: (_ for _ in ()).throw(ClientErr(code, status))
    out = safe_ingest_single_doc(col, settings, doc(), 1, "t")
    assert not out.ok and out.error_class is kind


def test_throttled_insert_is_retried_with_long_backoff_then_succeeds(settings, no_sleep):
    col = FakeCollection()
    real, calls = col.insert_many, {"n": 0}

    def flaky(batch, ordered=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ClientErr("", 429)
        real(batch, ordered=ordered)

    col.insert_many = flaky
    out = safe_ingest_single_doc(col, settings, doc(), 1, "t")
    assert out.ok and len(col.rows) == 1 and len(no_sleep) == 2 and all(d >= 5.0 for d in no_sleep)


def test_version_id_is_millisecond_timestamp():
    a = new_version_id()
    assert isinstance(a, int) and a > 1_600_000_000_000


# ------------------------------ smart-skip ------------------------------ #
def test_etags_bucket_scoped_and_prefers_explicit_bucket(settings):
    col = FakeCollection()
    col.rows = [
        {"source": "a.md", "chunk_no": 0, "etag": "legacy"},
        {"source": "a.md", "chunk_no": 0, "etag": '"mine"', "bucket": "B"},
        {"source": "a.md", "chunk_no": 0, "etag": "other", "bucket": "C"},
        {"source": "a.md", "chunk_no": 1, "etag": "chunk1", "bucket": "B"},
        {"source": "b.md", "chunk_no": 0, "etag": "legacy-b"},
        {"source": "c.md", "chunk_no": 0, "etag": "only-other", "bucket": "C"},
        {"source": "d.md", "chunk_no": 0, "etag": ""},
    ]
    got = get_existing_doc_etags(col, settings, ["a.md", "b.md", "c.md", "d.md", "e.md"], "B")
    assert got == {"a.md": "mine", "b.md": "legacy-b"}          # quotes stripped; other bucket ignored


def test_etags_without_bucket_filter_returns_any(settings):
    col = FakeCollection()
    col.rows = [{"source": "c.md", "chunk_no": 0, "etag": "x", "bucket": "C"}]
    assert get_existing_doc_etags(col, settings, ["c.md"]) == {"c.md": "x"}


def test_etags_empty_input_makes_no_query(settings):
    col = FakeCollection()
    col.find = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no query expected"))
    assert get_existing_doc_etags(col, settings, [], "B") == {}


def test_etags_queries_in_batches_of_50(settings):
    col = FakeCollection()
    sizes = []
    real = col.find
    col.find = lambda flt, projection=None: (sizes.append(len(flt["source"]["$in"])), real(flt, projection))[1]
    get_existing_doc_etags(col, settings, [f"f{i}.md" for i in range(120)], "B")
    assert sizes == [50, 50, 20]


def test_etags_raise_when_astra_unavailable(settings, no_sleep):
    col = FakeCollection()
    col.find = lambda *a, **k: (_ for _ in ()).throw(Exception("astra down"))
    with pytest.raises(RuntimeError):
        get_existing_doc_etags(col, settings, ["a.md"], "B")


def test_smart_skip_end_to_end(settings):
    col = FakeCollection()
    safe_ingest_single_doc(col, settings, doc(name="a.md", etag="et1", bucket="B"), 1, "t")
    assert get_existing_doc_etags(col, settings, ["a.md"], "B")["a.md"] == "et1"      # unchanged -> caller skips
    assert "a.md" not in get_existing_doc_etags(col, settings, ["a.md"], "other")     # other bucket must not skip


# ------------------------------- store ---------------------------------- #
def test_get_collection_never_creates_main_collection(settings):
    db = FakeDB()
    store = AstraStore(settings, database=db)
    assert store.get_collection() is db.cols[settings.collection_name]      # accessor only; fake registers on access
    assert db.created == []                                                 # nothing was created via create_collection


def test_jobs_collection_created_once_without_vectorize(settings):
    db = FakeDB()
    store = AstraStore(settings, database=db)
    store.get_jobs_collection()
    store.get_jobs_collection()
    store.get_jobs_collection()
    assert db.created == [(settings.jobs_collection, {})]
    assert db.list_calls == 1


def test_jobs_collection_existing_is_not_recreated(settings):
    db = FakeDB()
    db.cols[settings.jobs_collection] = FakeCollection()
    AstraStore(settings, database=db).get_jobs_collection()
    assert db.created == []


def test_jobs_collection_concurrent_create_race_tolerated(settings):
    class RacyDB(FakeDB):
        def create_collection(self, name, **kw):
            self.cols[name] = FakeCollection()               # someone else created it
            raise Exception("already exists")

    db = RacyDB()
    assert AstraStore(settings, database=db).get_jobs_collection() is db.cols[settings.jobs_collection]


def test_jobs_collection_create_failure_propagates(settings):
    class BrokenDB(FakeDB):
        def create_collection(self, name, **kw):
            raise Exception("forbidden")

    store = AstraStore(settings, database=BrokenDB())
    with pytest.raises(Exception, match="forbidden"):
        store.get_jobs_collection()
    with pytest.raises(Exception, match="forbidden"):
        store.get_jobs_collection()                           # not marked ready after a failure


def test_jobs_collection_threadsafe_single_create(settings):
    db = FakeDB()
    store = AstraStore(settings, database=db)
    threads = [threading.Thread(target=store.get_jobs_collection) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(db.created) == 1


def test_real_astrapy_client_builds_without_network(settings):
    """Constructing the real client/collection handles makes no network call (astrapy 2.x)."""
    store = AstraStore(settings)
    col = store.get_collection()
    assert type(col).__name__ == "Collection" and col.name == settings.collection_name
    opts = col.api_options.timeout_options
    assert opts.request_timeout_ms == 30_000 and opts.general_method_timeout_ms == 120_000


def test_real_cos_client_builds_without_network(make_settings):
    from app.cos_client import build_cos_client

    s = make_settings(CONCURRENCY=40, COS_READ_TIMEOUT=45)
    cos = build_cos_client(s)
    cfg = cos.meta.config
    assert cfg.max_pool_connections == 80 and cfg.read_timeout == 45 and cfg.connect_timeout == 10
    assert cos.meta.endpoint_url == s.cos_endpoint
    assert build_cos_client(make_settings()).meta.config.max_pool_connections == 25
