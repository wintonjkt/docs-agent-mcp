import pytest

from app.config import ConfigError, load_settings

SECRETS = {
    "ASTRA_DB_API_ENDPOINT": "https://example-region.apps.astra.datastax.com",
    "ASTRA_DB_APPLICATION_TOKEN": "AstraCS:super-secret-token",
    "COS_APIKEY": "super-secret-cos-key",
    "COS_RESOURCE_INSTANCE_ID": "crn:v1:bluemix:public:cloud-object-storage:global:a/x::",
    "UI_ACCESS_TOKEN": "super-secret-ui-token",
}


def test_defaults_applied():
    s = load_settings(dict(SECRETS))
    assert s.cos_endpoint == "https://s3.jp-tok.cloud-object-storage.appdomain.cloud"
    assert s.source_bucket == "cloud-object-storage-cos-standard-vk7"
    assert s.archive_bucket == "ww-archive"
    assert s.collection_name == "ibm_docs_rag2"
    assert s.jobs_collection == "ingest_jobs"
    assert s.concurrency == 8
    assert s.insert_batch == 40
    assert s.chunk_size == 1000 and s.chunk_overlap == 150
    assert s.max_file_bytes == 2 * 1024 ** 2
    assert s.max_job_failures == 200
    assert s.max_job_events == 100
    assert ".md" in s.supported_extensions


def test_overrides_applied():
    env = dict(SECRETS, SOURCE_BUCKET="src", ARCHIVE_BUCKET="arc", COLLECTION_NAME="c",
               JOBS_COLLECTION="j", CONCURRENCY="3", INSERT_BATCH="10",
               COS_ENDPOINT="https://s3.example.com/")
    s = load_settings(env)
    assert (s.source_bucket, s.archive_bucket, s.collection_name, s.jobs_collection) == ("src", "arc", "c", "j")
    assert s.concurrency == 3 and s.insert_batch == 10
    assert load_settings(dict(env, MAX_JOB_EVENTS="7")).max_job_events == 7
    assert s.cos_endpoint == "https://s3.example.com"


def test_missing_required_lists_all_and_hides_values():
    env = {"COS_APIKEY": "super-secret-cos-key"}
    with pytest.raises(ConfigError) as exc:
        load_settings(env)
    msg = str(exc.value)
    for name in ("ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN",
                 "COS_RESOURCE_INSTANCE_ID", "UI_ACCESS_TOKEN"):
        assert name in msg
    assert "COS_APIKEY," not in msg and "super-secret" not in msg


def test_blank_required_counts_as_missing():
    with pytest.raises(ConfigError, match="UI_ACCESS_TOKEN"):
        load_settings(dict(SECRETS, UI_ACCESS_TOKEN="   "))


def test_invalid_values_reported_without_secrets():
    env = dict(SECRETS, CONCURRENCY="abc", INSERT_BATCH="0", ARCHIVE_BUCKET="same",
               SOURCE_BUCKET="same", CHUNK_OVERLAP="2000")
    with pytest.raises(ConfigError) as exc:
        load_settings(env)
    msg = str(exc.value)
    assert "CONCURRENCY" in msg and "INSERT_BATCH" in msg
    assert "SOURCE_BUCKET and ARCHIVE_BUCKET" in msg
    assert "CHUNK_OVERLAP" in msg
    assert "super-secret" not in msg


def test_repr_does_not_leak_secrets():
    text = repr(load_settings(dict(SECRETS)))
    assert "super-secret" not in text


def test_throttle_and_cos_tunables():
    s = load_settings(dict(SECRETS))
    assert (s.throttle_base_delay, s.throttle_max_delay, s.throttle_max_retries) == (5.0, 60.0, 5)
    assert (s.cos_connect_timeout, s.cos_read_timeout, s.copy_object_max_bytes) == (10, 30, 4 * 1024 ** 3)
    s = load_settings(dict(SECRETS, THROTTLE_BASE_DELAY="2", THROTTLE_MAX_DELAY="9", THROTTLE_MAX_RETRIES="7"))
    assert (s.throttle_base_delay, s.throttle_max_delay, s.throttle_max_retries) == (2.0, 9.0, 7)
    with pytest.raises(ConfigError, match="THROTTLE_MAX_DELAY"):
        load_settings(dict(SECRETS, THROTTLE_BASE_DELAY="30", THROTTLE_MAX_DELAY="10"))
