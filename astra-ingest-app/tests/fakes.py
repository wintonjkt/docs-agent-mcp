"""In-memory fakes for COS and Astra collections (no network, no credentials)."""

import copy
import hashlib
import threading


class Body:
    def __init__(self, data: bytes):
        self.data = data

    def read(self):
        return self.data


class NotFound(Exception):
    def __init__(self):
        super().__init__("404 not found")
        self.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}


class ClientErr(Exception):
    """boto-style error with a code / HTTP status."""

    def __init__(self, code="", status=None, msg="boom"):
        super().__init__(msg)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeCOS:
    """Two-bucket in-memory COS with list/head/get/copy/delete and failure injection."""

    def __init__(self, keys=(), buckets=("src", "dst")):
        self.b = {name: {} for name in buckets}
        for k in keys:
            self.b["src"][k] = b"text for " + k.encode()
        self.fail_copy = set()
        self.corrupt = set()
        self.fail_delete = set()
        self.list_calls = 0
        self.max_keys_seen = 0
        self.get_calls = []

    @staticmethod
    def etag(data):
        return '"' + hashlib.md5(data).hexdigest() + '"'

    def head_bucket(self, Bucket):
        if Bucket not in self.b:
            raise NotFound()

    def head_object(self, Bucket, Key):
        d = self.b[Bucket]
        if Key not in d:
            raise NotFound()
        return {"ETag": self.etag(d[Key]), "ContentLength": len(d[Key])}

    def copy_object(self, CopySource, Bucket, Key):
        if Key in self.fail_copy:
            raise Exception("copy denied")
        data = self.b[CopySource["Bucket"]][CopySource["Key"]]
        self.b[Bucket][Key] = data[:-1] if Key in self.corrupt else data

    def delete_object(self, Bucket, Key):
        if Bucket == "src" and Key in self.fail_delete:
            raise Exception("delete denied")
        self.b[Bucket].pop(Key, None)

    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=1000, StartAfter=None, Delimiter=None, ContinuationToken=None):
        self.list_calls += 1
        self.max_keys_seen = max(self.max_keys_seen, MaxKeys)
        if Delimiter is None and ContinuationToken is None:
            ks = sorted(k for k in self.b[Bucket] if k.startswith(Prefix) and (not StartAfter or k > StartAfter))
            page = ks[:MaxKeys]
            return {
                "Contents": [{"Key": k, "Size": len(self.b[Bucket][k]), "ETag": self.etag(self.b[Bucket][k])} for k in page],
                "IsTruncated": len(ks) > MaxKeys,
            }

        # Delimiter / continuation mode: keys containing the delimiter after Prefix collapse into one
        # CommonPrefixes entry; both kinds count toward MaxKeys. The token is the last entry name returned.
        after = ContinuationToken or StartAfter
        names = set()
        for k in self.b[Bucket]:
            if not k.startswith(Prefix):
                continue
            rest = k[len(Prefix):]
            if Delimiter and Delimiter in rest:
                names.add(Prefix + rest.split(Delimiter)[0] + Delimiter)
            else:
                names.add(k)
        entries = sorted(n for n in names if not after or n > after)
        page = entries[:MaxKeys]
        truncated = len(entries) > MaxKeys
        resp = {
            "Contents": [
                {"Key": n, "Size": len(self.b[Bucket][n]), "ETag": self.etag(self.b[Bucket][n])}
                for n in page if n in self.b[Bucket]
            ],
            "CommonPrefixes": [{"Prefix": n} for n in page if n not in self.b[Bucket]],
            "IsTruncated": truncated,
        }
        if truncated:
            resp["NextContinuationToken"] = page[-1]
        return resp

    def get_object(self, Bucket, Key):
        self.get_calls.append(Key)
        d = self.b[Bucket][Key]
        return {"Body": Body(d), "ETag": self.etag(d)}


class FakeCollection:
    """Astra collection fake covering find / insert_many / delete_many / update_many with the filters the app uses."""

    def __init__(self):
        self.rows = []
        self.fail_insert = False
        self.fail_delete_old = False   # makes the "delete old versions" call fail (ingest_version $ne)
        self.fail_update = False
        self.fail_after_n_updates = None
        self.update_calls = 0
        self.delete_filters = []
        self.insert_batches = []

    @staticmethod
    def _bucket_ok(row, scope):
        return any(
            (c["bucket"] == row.get("bucket") if "$exists" not in str(c["bucket"]) else "bucket" not in row)
            for c in scope
        )

    def find(self, flt, projection=None):
        srcs = flt["source"]["$in"]
        return [dict(r) for r in self.rows if r["source"] in srcs and r["chunk_no"] == flt["chunk_no"]]

    def insert_many(self, batch, ordered=False):
        self.insert_batches.append(len(batch))
        if self.fail_insert:
            raise Exception("astra down")
        self.rows += [dict(r) for r in batch]

    def delete_many(self, flt):
        self.delete_filters.append(flt)
        is_old = isinstance(flt["ingest_version"], dict)
        if is_old and self.fail_delete_old:
            raise Exception("astra delete down")

        def match(r):
            if r["source"] != flt["source"]:
                return False
            if "$or" in flt and not self._bucket_ok(r, flt["$or"]):
                return False
            v = r.get("ingest_version")
            return v != flt["ingest_version"]["$ne"] if is_old else v == flt["ingest_version"]

        self.rows = [r for r in self.rows if not match(r)]

    def update_many(self, flt, upd):
        self.update_calls += 1
        if self.fail_update:
            raise Exception("astra update down")
        if self.fail_after_n_updates is not None and self.update_calls > self.fail_after_n_updates:
            raise Exception("astra down")
        for r in self.rows:
            if r["source"] != flt["source"]:
                continue
            if "$or" in flt:
                ok = self._bucket_ok(r, flt["$or"])
            else:
                ok = r.get("bucket") == flt["bucket"]
            if ok:
                r.update(upd["$set"])


class FakeDB:
    def __init__(self):
        self.cols = {}
        self.created = []
        self.list_calls = 0

    def list_collection_names(self):
        self.list_calls += 1
        return list(self.cols)

    def create_collection(self, name, **kw):
        self.cols[name] = FakeCollection()
        self.created.append((name, kw))

    def get_collection(self, name):
        return self.cols.setdefault(name, FakeCollection()) if name != "__missing__" else None


class DuplicateKey(Exception):
    """Mimics astrapy's DataAPIResponseException for DOCUMENT_ALREADY_EXISTS."""

    def __init__(self):
        super().__init__("Failed to insert document with _id: Document already exists with the given _id")
        self.error_descriptors = [type("D", (), {"error_code": "DOCUMENT_ALREADY_EXISTS"})()]


def _get_path(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _match_cond(doc, field, cond):
    exists, value = _get_path(doc, field)
    if isinstance(cond, dict) and cond and all(k.startswith("$") for k in cond):
        for op, arg in cond.items():
            if op == "$exists":
                if exists != bool(arg):
                    return False
            elif op == "$in":
                if not exists or value not in arg:
                    return False
            elif op == "$nin":
                if exists and value in arg:
                    return False
            elif op == "$ne":
                if exists and value == arg:
                    return False
            elif op == "$size":
                if not exists or not isinstance(value, list) or len(value) != arg:
                    return False
            elif op == "$lt":
                if not exists or value is None or not value < arg:
                    return False
            elif op == "$gt":
                if not exists or value is None or not value > arg:
                    return False
            else:
                raise NotImplementedError(op)
        return True
    return exists and value == cond


def match_filter(doc, flt):
    for key, cond in (flt or {}).items():
        if key == "$or":
            if not any(match_filter(doc, sub) for sub in cond):
                return False
        elif key == "$and":
            if not all(match_filter(doc, sub) for sub in cond):
                return False
        elif not _match_cond(doc, key, cond):
            return False
    return True


class FakeJobsCollection:
    """In-memory Astra collection for the jobs layer.

    Supports find_one / find (sort, limit) / insert_one (duplicate ``_id`` raises) / find_one_and_update
    with ``$set`` and the filter operators the jobs layer uses. ``before_update`` is a hook called just
    before every ``find_one_and_update`` is applied, to simulate concurrent writers.
    """

    def __init__(self):
        self.docs = {}
        self.fail_next = []            # exceptions raised by the next operations (any method)
        self.before_update = None      # callable(collection, filter, update)
        self.calls = []                # (method, filter) log
        self.update_filters = []
        self.updates = []              # update documents passed to find_one_and_update
        self.fail_insert_if = None     # callable(doc) -> exception or None, checked on insert_one
        self._lock = threading.Lock()  # makes insert_one atomic like Astra's unique _id

    def _maybe_fail(self):
        if self.fail_next:
            raise self.fail_next.pop(0)

    def insert_one(self, doc):
        self.calls.append(("insert_one", doc.get("_id")))
        self._maybe_fail()
        if self.fail_insert_if:
            exc = self.fail_insert_if(doc)
            if exc is not None:
                raise exc
        with self._lock:
            if doc["_id"] in self.docs:
                raise DuplicateKey()
            self.docs[doc["_id"]] = copy.deepcopy(doc)

    def find_one(self, flt=None, **kw):
        self.calls.append(("find_one", flt))
        self._maybe_fail()
        for doc in self.docs.values():
            if match_filter(doc, flt):
                return copy.deepcopy(doc)
        return None

    def find(self, flt=None, sort=None, limit=None, projection=None, **kw):
        self.calls.append(("find", flt))
        self._maybe_fail()
        out = [copy.deepcopy(d) for d in self.docs.values() if match_filter(d, flt)]
        for key, direction in reversed(list((sort or {}).items())):
            out.sort(key=lambda d, k=key: str(d.get(k, "")), reverse=direction < 0)
        out = out[:limit] if limit else out
        excluded = [k for k, v in (projection or {}).items() if not v]
        for d in out:
            for k in excluded:
                d.pop(k, None)
        return out

    def delete_one(self, flt):
        self.calls.append(("delete_one", flt))
        self._maybe_fail()
        for doc_id, doc in list(self.docs.items()):
            if match_filter(doc, flt):
                del self.docs[doc_id]
                return

    def find_one_and_update(self, flt, update, return_document="before", **kw):
        self.calls.append(("find_one_and_update", flt))
        self.update_filters.append(flt)
        self.updates.append(update)
        self._maybe_fail()
        if self.before_update:
            self.before_update(self, flt, update)
        for doc in self.docs.values():
            if match_filter(doc, flt):
                before = copy.deepcopy(doc)
                for k, v in update.get("$set", {}).items():
                    doc[k] = copy.deepcopy(v)
                for k, spec in update.get("$push", {}).items():
                    items = spec["$each"] if isinstance(spec, dict) and "$each" in spec else [spec]
                    doc.setdefault(k, []).extend(copy.deepcopy(items))
                return copy.deepcopy(doc) if return_document == "after" else before
        return None
