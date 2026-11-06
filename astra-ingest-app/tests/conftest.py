import pytest

from app.config import load_settings

BASE_ENV = {
    "ASTRA_DB_API_ENDPOINT": "https://example-region.apps.astra.datastax.com",
    "ASTRA_DB_APPLICATION_TOKEN": "AstraCS:fake",
    "COS_APIKEY": "fake-cos-key",
    "COS_RESOURCE_INSTANCE_ID": "crn:v1:fake",
    "UI_ACCESS_TOKEN": "fake-ui-token",
    "SOURCE_BUCKET": "src",
    "ARCHIVE_BUCKET": "dst",
}


@pytest.fixture
def make_settings():
    def _make(**overrides):
        env = dict(BASE_ENV)
        env.update({k: str(v) for k, v in overrides.items()})
        return load_settings(env)

    return _make


@pytest.fixture
def settings(make_settings):
    return make_settings()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make retry backoff instant; records the requested delays in ``no_sleep.delays``."""
    delays = []
    monkeypatch.setattr("app.retry.time.sleep", lambda s: delays.append(s))
    return delays
