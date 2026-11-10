import time

import pytest

from app.cos_client import (
    discover_cos_batch, fetch_document, head_object_or_none, list_cos_prefixes, move_object,
    normalize_dest_prefix, validate_destination,
)
from app.retry import ErrorClass
from tests.fakes import ClientErr, FakeCOS, FakeCollection


# ---------------------------- prefix listing ---------------------------- #
PREFIX_KEYS = ["root.md", "docs/a.md", "docs/cp4ba/b.md", "docs/cp4ba/deep/c.md", "docs/odm/d.md", "other/e.md"]


def test_list_prefixes_root_level(settings):
    prefixes, truncated = list_cos_prefixes(FakeCOS(PREFIX_KEYS), settings, "src")
    assert prefixes == ["docs/", "other/"]
    assert truncated is False


def test_list_prefixes_nested_level_returns_full_prefixes(settings):
    cos = FakeCOS(PREFIX_KEYS)
    assert list_cos_prefixes(cos, settings, "src", "docs/") == (["docs/cp4ba/", "docs/odm/"], False)
    assert list_cos_prefixes(cos, settings, "src", "docs/cp4ba/") == (["docs/cp4ba/deep/"], False)
    assert list_cos_prefixes(cos, settings, "src", "docs/odm/") == ([], False)


def test_list_prefixes_cap_sets_truncated(settings):
    cos = FakeCOS([f"p{i:02d}/f.md" for i in range(5)])
    prefixes, truncated = list_cos_prefixes(cos, settings, "src", max_prefixes=3)
    assert prefixes == ["p00/", "p01/", "p02/"]
    assert truncated is True
    # exactly at the cap is not truncated
    assert list_cos_prefixes(cos, settings, "src", max_prefixes=5) == ([f"p{i:02d}/" for i in range(5)], False)


def test_list_prefixes_follows_continuation_tokens(settings):
    cos = FakeCOS([f"p{i:04d}/f.md" for i in range(1200)])
    prefixes, truncated = list_cos_prefixes(cos, settings, "src", max_prefixes=5000)
    assert len(prefixes) == 1200 and truncated is False
    assert cos.list_calls == 2


def test_fake_cos_without_delimiter_is_unchanged():
    resp = FakeCOS(PREFIX_KEYS).list_objects_v2("src", Prefix="docs/")
    assert [c["Key"] for c in resp["Contents"]] == ["docs/a.md", "docs/cp4ba/b.md", "docs/cp4ba/deep/c.md", "docs/odm/d.md"]
    assert "CommonPrefixes" not in resp


def test_fake_cos_delimiter_excludes_collapsed_keys_from_contents():
    resp = FakeCOS(PREFIX_KEYS).list_objects_v2("src", Prefix="docs/", Delimiter="/")
    assert [c["Key"] for c in resp["Contents"]] == ["docs/a.md"]
    assert [p["Prefix"] for p in resp["CommonPrefixes"]] == ["docs/cp4ba/", "docs/odm/"]


# ------------------------------- listing -------------------------------- #
def test_exact_checkpoint_listing_covers_every_file_once(settings):
    keys = [f"d/f{i:03d}.md" for i in range(130)] + ["d/skip.bin"]
    cos = FakeCOS(keys)
    for target in (20, 50, 7):
        seen, tok, n = [], "", 0
        while True:
            items, tok, more = discover_cos_batch(cos, settings, "src", "d/", tok or "", target)
            assert len(items) <= target
            seen += [i["key"] for i in items]
            n += 1
            if not more:
                assert tok is None
                break
            assert tok
        assert seen == [k for k in keys if k.endswith(".md")], target


def test_item_shape_and_defaults_from_settings(make_settings):
    s = make_settings(LIST_PAGE_SIZE=3)
    cos = FakeCOS([f"a{i}.md" for i in range(10)] + ["dir/"])
    items, tok, more = discover_cos_batch(cos, s, "src")
    assert [i["key"] for i in items] == ["a0.md", "a1.md", "a2.md"] and more and tok == "a2.md"
    assert set(items[0]) == {"key", "size", "etag", "last_modified"} and '"' not in items[0]["etag"]


def test_checkpoint_resumes_strictly_after_key(settings):
    cos = FakeCOS([f"d/f{i}.md" for i in range(5)])
    items, tok, more = discover_cos_batch(cos, settings, "src", "d/", "d/f2.md", 10)
    assert [i["key"] for i in items] == ["d/f3.md", "d/f4.md"] and not more and tok is None


def test_stale_checkpoint_rejected(settings):
    with pytest.raises(ValueError, match="does not belong to prefix"):
        discover_cos_batch(FakeCOS(["docs/a.md"]), settings, "src", "docs/", "other/x.md")


def test_sparse_prefix_uses_full_pages_and_stays_bounded(settings):
    cos = FakeCOS()
    cos.b["src"] = {f"d/img{i:05d}.png": b"x" for i in range(5000)}
    cos.b["src"].update({f"d/z{i}.md": b"text" for i in range(3)})
    items, tok, more = discover_cos_batch(cos, settings, "src", "d/", "", 25)
    assert [i["key"] for i in items] == [f"d/z{i}.md" for i in range(3)] and not more
    assert cos.list_calls <= 7 and cos.max_keys_seen == 1000


def test_deadline_returns_empty_with_exact_checkpoint_and_progress(settings):
    cos = FakeCOS()
    cos.b["src"] = {f"d/img{i:05d}.png": b"x" for i in range(5000)}
    cos.b["src"].update({f"d/z{i}.md": b"text" for i in range(3)})
    items, tok, more = discover_cos_batch(cos, settings, "src", "d/", "", 25, deadline=time.time() - 1)
    assert items == [] and more and tok and cos.list_calls == 1
    # keep going from each checkpoint until done: nothing is lost, "empty but has_more" never means finished
    found, passes = [], 1
    while more:
        items, tok, more = discover_cos_batch(cos, settings, "src", "d/", tok, 25, deadline=time.time() - 1)
        found += [i["key"] for i in items]
        passes += 1
        assert passes < 300
    # each pass examines one 50-key first page (5003 keys) -> ~100 passes, most of them empty
    assert found == [f"d/z{i}.md" for i in range(3)] and passes >= 50 and tok is None


def test_listing_retries_throttled_then_succeeds(settings, no_sleep):
    cos = FakeCOS(["a.md"])
    real = cos.list_objects_v2
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ClientErr("", 503)
        return real(**kw)

    cos.list_objects_v2 = flaky
    items, _, _ = discover_cos_batch(cos, settings, "src")
    assert [i["key"] for i in items] == ["a.md"] and no_sleep and no_sleep[0] >= 5.0


# --------------------------- head / destination --------------------------- #
def test_head_object_or_none(settings):
    cos = FakeCOS(["a.md"])
    assert head_object_or_none(cos, settings, "src", "a.md")["ContentLength"] > 0
    assert head_object_or_none(cos, settings, "src", "missing.md") is None


def test_head_object_propagates_other_errors(settings):
    cos = FakeCOS(["a.md"])
    cos.head_object = lambda **kw: (_ for _ in ()).throw(ClientErr("AccessDenied", 403))
    with pytest.raises(RuntimeError):
        head_object_or_none(cos, settings, "src", "a.md")


def test_validate_destination(settings):
    cos = FakeCOS()
    assert validate_destination(cos, settings, "src", "") == ""
    assert "must differ" in validate_destination(cos, settings, "src", "src")
    assert "not accessible" in validate_destination(cos, settings, "src", "nope")
    assert validate_destination(cos, settings, "src", "dst") == ""


def test_normalize_dest_prefix():
    assert normalize_dest_prefix("") == "" and normalize_dest_prefix("/a") == "a/" and normalize_dest_prefix("a/b/") == "a/b/"


# --------------------------------- fetch ---------------------------------- #
def item(key, size=None, etag="e", lm="2024-01-01T00:00:00"):
    return {"key": key, "size": size, "etag": etag, "last_modified": lm}


def test_fetch_ok_preserves_metadata(settings):
    cos = FakeCOS(["a.md"])
    r = fetch_document(cos, settings, "src", item("a.md", size=5))
    assert r.error is None and r.doc.name == "a.md" and r.doc.bucket == "src"
    assert r.doc.etag == "e" and r.doc.last_modified == "2024-01-01T00:00:00" and r.doc.text == "text for a.md"


def test_fetch_falls_back_to_object_etag(settings):
    cos = FakeCOS(["a.md"])
    r = fetch_document(cos, settings, "src", {"key": "a.md"})
    assert r.doc.etag == cos.etag(cos.b["src"]["a.md"]).strip('"')


def test_fetch_too_large_not_downloaded(make_settings):
    s = make_settings(MAX_FILE_BYTES=10)
    cos = FakeCOS(["a.md"])
    r = fetch_document(cos, s, "src", item("a.md", size=11))
    assert r.doc is None and r.error.startswith("file too large") and cos.get_calls == []


def test_fetch_empty_file(settings):
    cos = FakeCOS()
    cos.b["src"]["e.md"] = b"  \n "
    r = fetch_document(cos, settings, "src", item("e.md", size=3))
    assert r.doc is None and r.error == "empty file"


def test_fetch_invalid_utf8_is_decoded_leniently(settings):
    cos = FakeCOS()
    cos.b["src"]["b.md"] = b"ok \xff\xfe text"
    r = fetch_document(cos, settings, "src", item("b.md", size=9))
    assert r.error is None and "ok" in r.doc.text and "text" in r.doc.text


def test_fetch_permanent_error_is_unreadable_and_not_retried(settings):
    cos = FakeCOS(["a.md"])
    n = {"c": 0}

    def denied(**kw):
        n["c"] += 1
        raise ClientErr("AccessDenied", 403)

    cos.get_object = denied
    r = fetch_document(cos, settings, "src", item("a.md", size=3))
    assert r.error.startswith("unreadable file") and n["c"] == 1 and r.error_class is ErrorClass.PERMANENT


def test_fetch_transient_error_is_download_failed(settings, no_sleep):
    cos = FakeCOS(["a.md"])
    cos.get_object = lambda **kw: (_ for _ in ()).throw(Exception("reset"))
    r = fetch_document(cos, settings, "src", item("a.md", size=3))
    assert r.error.startswith("download failed") and r.error_class is ErrorClass.TRANSIENT and len(no_sleep) == 2


def test_fetch_throttled_error_class(settings, no_sleep):
    cos = FakeCOS(["a.md"])
    cos.get_object = lambda **kw: (_ for _ in ()).throw(ClientErr("", 503))
    r = fetch_document(cos, settings, "src", item("a.md", size=3))
    assert r.error.startswith("download failed") and r.error_class is ErrorClass.THROTTLED


# ---------------------------- verified move ------------------------------ #
def row(source, bucket="src", chunk_no=0, **kw):
    d = {"source": source, "chunk_no": chunk_no, "etag": "e", "ingest_version": 1}
    if bucket is not None:
        d["bucket"] = bucket
    d.update(kw)
    return d


def test_move_happy_path_repoints_rows_and_deletes_source_last(settings):
    cos = FakeCOS(["d/a.md"])
    col = FakeCollection()
    col.rows = [row("d/a.md"), row("d/a.md", chunk_no=1), row("d/a.md", bucket="other")]
    order = []
    od, ou = cos.delete_object, col.update_many
    cos.delete_object = lambda **kw: (order.append(("del", kw["Bucket"])), od(**kw))[1]
    col.update_many = lambda f, u: (order.append("update"), ou(f, u))[1]
    r = move_object(cos, col, settings, "src", "d/a.md", "dst", "archive/d/a.md")
    assert r.moved and r.reason == "" and r.error_class is None
    assert "d/a.md" not in cos.b["src"] and "archive/d/a.md" in cos.b["dst"]
    assert order == ["update", ("del", "src")]
    assert [(x["source"], x["bucket"]) for x in col.rows if x["bucket"] == "dst"] == [("archive/d/a.md", "dst")] * 2
    assert [x for x in col.rows if x["bucket"] == "other"][0]["source"] == "d/a.md"   # other bucket untouched


def test_move_legacy_rows_without_bucket_are_repointed(settings):
    cos = FakeCOS(["d/a.md"])
    col = FakeCollection()
    col.rows = [row("d/a.md", bucket=None)]
    assert move_object(cos, col, settings, "src", "d/a.md", "dst", "x/d/a.md").moved
    assert col.rows[0]["source"] == "x/d/a.md" and col.rows[0]["bucket"] == "dst"


def test_move_source_missing(settings):
    r = move_object(FakeCOS(), FakeCollection(), settings, "src", "nope.md", "dst", "nope.md")
    assert not r.moved and "no longer exists" in r.reason


def test_move_etag_mismatch_keeps_source(settings):
    cos = FakeCOS(["a.md"])
    r = move_object(cos, FakeCollection(), settings, "src", "a.md", "dst", "m/a.md", expected_etag="deadbeef")
    assert not r.moved and "ETag mismatch" in r.reason and "a.md" in cos.b["src"] and cos.b["dst"] == {}


def test_move_expected_etag_match_moves(settings):
    cos = FakeCOS(["a.md"])
    et = cos.etag(cos.b["src"]["a.md"]).strip('"')
    assert move_object(cos, FakeCollection(), settings, "src", "a.md", "dst", "a.md", expected_etag=et).moved


def test_move_destination_collision_never_overwrites(settings):
    cos = FakeCOS(["a.md"])
    cos.b["dst"]["m/a.md"] = b"DIFFERENT"
    r = move_object(cos, FakeCollection(), settings, "src", "a.md", "dst", "m/a.md")
    assert not r.moved and "cannot be proven identical" in r.reason
    assert cos.b["dst"]["m/a.md"] == b"DIFFERENT" and "a.md" in cos.b["src"]


def test_move_copy_denied_keeps_source(settings, no_sleep):
    cos = FakeCOS(["a.md"])
    cos.fail_copy = {"m/a.md"}
    col = FakeCollection()
    col.rows = [row("a.md")]
    r = move_object(cos, col, settings, "src", "a.md", "dst", "m/a.md")
    assert not r.moved and "copy denied" in r.reason and "a.md" in cos.b["src"] and col.rows[0]["bucket"] == "src"


def test_move_corrupt_copy_is_removed_and_source_kept(settings):
    cos = FakeCOS(["a.md"])
    cos.corrupt = {"m/a.md"}
    col = FakeCollection()
    col.rows = [row("a.md")]
    r = move_object(cos, col, settings, "src", "a.md", "dst", "m/a.md")
    assert not r.moved and "verification failed" in r.reason
    assert "m/a.md" not in cos.b["dst"] and "a.md" in cos.b["src"] and col.rows[0]["source"] == "a.md"


def test_move_astra_update_failure_keeps_source(settings, no_sleep):
    cos = FakeCOS(["a.md"])
    col = FakeCollection()
    col.rows = [row("a.md")]
    col.fail_update = True
    r = move_object(cos, col, settings, "src", "a.md", "dst", "m/a.md")
    assert not r.moved and "Astra records could not be updated" in r.reason
    assert "a.md" in cos.b["src"] and r.error_class is ErrorClass.TRANSIENT
    assert col.rows[0]["source"] == "a.md"


def test_move_astra_throttled_update_is_classified_throttled(settings, no_sleep):
    cos = FakeCOS(["a.md"])
    col = FakeCollection()
    col.update_many = lambda f, u: (_ for _ in ()).throw(ClientErr("", 429))
    r = move_object(cos, col, settings, "src", "a.md", "dst", "m/a.md")
    assert not r.moved and r.error_class is ErrorClass.THROTTLED and "a.md" in cos.b["src"]


def test_move_delete_failure_rolls_back_rows_and_removes_new_copy(settings, no_sleep):
    cos = FakeCOS(["d/a.md"])
    cos.fail_delete = {"d/a.md"}
    col = FakeCollection()
    col.rows = [row("d/a.md")]
    r = move_object(cos, col, settings, "src", "d/a.md", "dst", "m/d/a.md")
    assert not r.moved and "records rolled back" in r.reason
    assert col.rows[0]["source"] == "d/a.md" and col.rows[0]["bucket"] == "src"
    assert "d/a.md" in cos.b["src"] and "m/d/a.md" not in cos.b["dst"]


def test_move_delete_failure_keeps_preexisting_identical_destination(settings, no_sleep):
    cos = FakeCOS(["a.md"])
    cos.b["dst"]["m/a.md"] = cos.b["src"]["a.md"]          # identical object already archived (crash re-run)
    cos.fail_delete = {"a.md"}
    col = FakeCollection()
    col.rows = [row("a.md")]
    r = move_object(cos, col, settings, "src", "a.md", "dst", "m/a.md")
    assert not r.moved and "m/a.md" in cos.b["dst"]        # not created by this call, so not removed


def test_move_delete_failure_and_rollback_failure_reports_inconsistent_state(settings, no_sleep):
    cos = FakeCOS(["a.md"])
    cos.fail_delete = {"a.md"}
    col = FakeCollection()
    col.rows = [row("a.md")]
    col.fail_after_n_updates = 1                            # first update ok, rollback fails
    r = move_object(cos, col, settings, "src", "a.md", "dst", "m/a.md")
    assert not r.moved and "could not be rolled back" in r.reason and "both buckets" in r.reason


def test_move_idempotent_rerun_with_identical_destination(settings):
    cos = FakeCOS(["a.md"])
    cos.b["dst"]["m/a.md"] = cos.b["src"]["a.md"]
    col = FakeCollection()
    col.rows = [row("a.md")]
    r = move_object(cos, col, settings, "src", "a.md", "dst", "m/a.md")
    assert r.moved and "a.md" not in cos.b["src"] and col.rows[0]["source"] == "m/a.md"


class MultipartCOS(FakeCOS):
    """Source ETag looks multipart; a fresh copy gets a different multipart ETag (as real COS does)."""

    def __init__(self, keys):
        super().__init__(keys)
        self.mp_src = set(keys)
        self.mp_copied = set()

    def head_object(self, Bucket, Key):
        h = super().head_object(Bucket, Key)
        if Bucket == "src" and Key in self.mp_src:
            h["ETag"] = '"abc123-3"'
        elif Bucket == "dst" and Key in self.mp_copied:
            h["ETag"] = '"zzz999-3"'
        return h

    def copy_object(self, CopySource, Bucket, Key):
        super().copy_object(CopySource, Bucket, Key)
        if CopySource["Key"] in self.mp_src:
            self.mp_copied.add(Key)


def test_move_multipart_etag_lookalike_refused_but_fresh_copy_accepted(settings):
    cos = MultipartCOS(["d/mp.md", "d/mp2.md"])
    col = FakeCollection()
    cos.b["dst"]["d/mp.md"] = b"X" * len(cos.b["src"]["d/mp.md"])      # different object, same size
    r = move_object(cos, col, settings, "src", "d/mp.md", "dst", "d/mp.md", "abc123-3")
    assert not r.moved and "cannot be proven identical" in r.reason and "d/mp.md" in cos.b["src"]
    r = move_object(cos, col, settings, "src", "d/mp2.md", "dst", "d/mp2.md", "abc123-3")
    assert r.moved and "d/mp2.md" not in cos.b["src"] and "d/mp2.md" in cos.b["dst"]


def test_move_large_object_uses_managed_copy(make_settings):
    s = make_settings(COPY_OBJECT_MAX_BYTES=5)
    cos = FakeCOS(["a.md"])
    used = []
    cos.copy = lambda src, bucket, key: (used.append(key), cos.copy_object(CopySource=src, Bucket=bucket, Key=key))[1]
    assert move_object(cos, FakeCollection(), s, "src", "a.md", "dst", "m/a.md").moved and used == ["m/a.md"]
