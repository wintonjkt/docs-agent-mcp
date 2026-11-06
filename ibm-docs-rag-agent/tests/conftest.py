"""Test configuration and in-memory mock Astra DB fixtures for ibm-docs-rag-agent."""

import sys
import types
import pytest
from typing import Any, Dict, List, Optional

# Ensure ibm_watsonx_orchestrate tool decorator is passthrough in unit tests
if "ibm_watsonx_orchestrate" not in sys.modules:
    for name in [
        "ibm_watsonx_orchestrate",
        "ibm_watsonx_orchestrate.agent_builder",
        "ibm_watsonx_orchestrate.agent_builder.tools",
        "ibm_watsonx_orchestrate.run",
        "ibm_watsonx_orchestrate.run.connections",
    ]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["ibm_watsonx_orchestrate.agent_builder.tools"].tool = lambda *a, **k: (lambda f: f)
    c = sys.modules["ibm_watsonx_orchestrate.run.connections"]
    c.ConnectionType = types.SimpleNamespace(KEY_VALUE="key_value")
    c.key_value = lambda a: {}
    sys.modules["ibm_watsonx_orchestrate.run"].connections = c
else:
    # If installed, mock the tool decorator so test calls return raw string results directly
    sys.modules["ibm_watsonx_orchestrate.agent_builder.tools"].tool = lambda *a, **k: (lambda f: f)


class FakeCollection:
    """In-memory mock Astra DB Collection for testing vector searches and document queries."""

    def __init__(self, sample_docs: Optional[List[Dict[str, Any]]] = None):
        self.docs = list(sample_docs or [])
        self.last_find_filter: Optional[Dict[str, Any]] = None
        self.last_find_sort: Optional[Dict[str, Any]] = None

    def find(
        self,
        filter_dict: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        projection: Optional[Dict[str, Any]] = None,
        include_similarity: bool = False,
    ) -> List[Dict[str, Any]]:
        self.last_find_filter = filter_dict
        self.last_find_sort = sort

        matched = []
        for doc in self.docs:
            # Check filter
            match = True
            if filter_dict:
                for k, v in filter_dict.items():
                    if k == "source" and isinstance(v, dict) and "$regex" in v:
                        import re
                        if not re.search(v["$regex"], doc.get("source", "")):
                            match = False
                            break
                    elif k == "chunk_no" and doc.get("chunk_no") != v:
                        match = False
                        break
                    elif k in doc and doc[k] != v:
                        match = False
                        break
            if match:
                res_doc = dict(doc)
                if include_similarity and "$similarity" not in res_doc:
                    res_doc["$similarity"] = doc.get("similarity", 0.85)
                matched.append(res_doc)

        # Sort by similarity if present
        if include_similarity:
            matched.sort(key=lambda d: d.get("$similarity", 0.0), reverse=True)

        return matched[:limit]


@pytest.fixture
def sample_ibm_docs():
    """Sample IBM documentation dataset with varying similarity scores for testing."""
    return [
        {
            "_id": "doc1",
            "source": "cloud-paks/cp-biz-automation/26.0.0__topic=administering-configuring-security.md",
            "chunk_no": 0,
            "chunk_count": 2,
            "text": "CP4BA 26.0.0 Security Configuration: Configure Keycloak SSO and LDAP user registry.",
            "$similarity": 0.89,
            "last_modified": "2026-11-05T15:05:03Z",
        },
        {
            "_id": "doc2",
            "source": "cloud-paks/cp-biz-automation/26.0.0__topic=deployment-troubleshooting.md",
            "chunk_no": 0,
            "chunk_count": 3,
            "text": "CP4BA 26.0.0 Troubleshooting: Check operator pod logs with oc logs -n cp4ba.",
            "$similarity": 0.78,
            "last_modified": "2026-11-05T15:04:26Z",
        },
        {
            "_id": "doc3",
            "source": "cloud-paks/cp-biz-automation/24.0.1__topic=deployment-troubleshooting.md",
            "chunk_no": 0,
            "chunk_count": 2,
            "text": "CP4BA 24.0.1 Troubleshooting: Check ICP4ACluster CR status and IBM MQ connectivity.",
            "$similarity": 0.72,  # Below 0.75 threshold
            "last_modified": "2026-11-05T15:04:26Z",
        },
        {
            "_id": "doc4",
            "source": "general/generic-overview.md",
            "chunk_no": 0,
            "chunk_count": 1,
            "text": "General overview of distributed software deployments.",
            "$similarity": 0.45,  # Far below threshold
            "last_modified": "2026-11-01T10:00:00Z",
        },
    ]


@pytest.fixture
def mock_collection(monkeypatch, sample_ibm_docs):
    """Fixture providing a mock Astra DB collection and monkeypatching _get_collection."""
    fake_col = FakeCollection(sample_ibm_docs)
    monkeypatch.setattr("rag_tools._get_collection", lambda coll_name="ibm_docs_rag2": fake_col)
    return fake_col
