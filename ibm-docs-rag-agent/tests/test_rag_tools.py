"""Comprehensive unit tests for IBM Product RAG Agent tools."""

import sys
from pathlib import Path
import pytest

# Ensure ibm-docs-rag-agent is in python path
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from rag_tools import (
    search_ibm_knowledge,
    search_with_query_expansion,
    search_by_product_version,
    compare_product_versions,
    generate_mermaid_diagram,
    get_ibm_doc_content,
    list_ibm_knowledge_sources,
    _generate_query_variations,
    _sanitize_id,
    _sanitize_label,
    _with_retry,
    _is_permanent_error,
)
from tests.conftest import FakeCollection


# --------------------------------------------------------------------------- #
# Tool 1: search_ibm_knowledge tests
# --------------------------------------------------------------------------- #
def test_search_ibm_knowledge_strict_confidence_filtering(mock_collection):
    """Verify hits with score >= 0.75 are returned and hits < 0.75 are excluded."""
    result = search_ibm_knowledge(
        query="CP4BA security and troubleshooting",
        collection_name="ibm_docs_rag2",
        min_confidence=0.75,
    )

    # doc1 (0.89) and doc2 (0.78) meet threshold
    assert "0.890" in result
    assert "0.780" in result
    assert "administering-configuring-security.md" in result
    assert "deployment-troubleshooting.md" in result

    # doc3 (0.72) and doc4 (0.45) must be excluded
    assert "0.720" not in result
    assert "0.450" not in result
    assert "generic-overview.md" not in result


def test_search_ibm_knowledge_no_confident_hits(monkeypatch):
    """Verify that when all hits are below min_confidence, a clear threshold notice is returned."""
    low_confidence_docs = [
        {
            "source": "some/doc.md",
            "chunk_no": 0,
            "text": "Uncertain text",
            "$similarity": 0.68,
        }
    ]
    fake_col = FakeCollection(low_confidence_docs)
    monkeypatch.setattr("rag_tools._get_collection", lambda coll="ibm_docs_rag2": fake_col)

    result = search_ibm_knowledge(query="rare custom setting", min_confidence=0.75)

    assert "Confidence Threshold Notice" in result
    assert "0.75" in result
    assert "0.680" in result
    assert "Recommended Refinements" in result


def test_search_ibm_knowledge_empty_query():
    """Verify empty or whitespace query returns a validation message."""
    assert "Please provide a valid search query" in search_ibm_knowledge("")
    assert "Please provide a valid search query" in search_ibm_knowledge("   ")


def test_search_ibm_knowledge_empty_collection(monkeypatch):
    """Verify empty search results return a friendly not-found message."""
    fake_col = FakeCollection([])
    monkeypatch.setattr("rag_tools._get_collection", lambda coll="ibm_docs_rag2": fake_col)

    result = search_ibm_knowledge("any query")
    assert "No documentation found in collection" in result


# --------------------------------------------------------------------------- #
# Tool 2: search_with_query_expansion tests
# --------------------------------------------------------------------------- #
def test_generate_query_variations():
    """Verify domain abbreviations and synonyms are properly expanded."""
    variations = _generate_query_variations("How to configure CP4BA BAW SSO")
    assert any("Cloud Pak for Business Automation" in v for v in variations)
    assert any("Business Automation Workflow" in v for v in variations)
    assert any("Single Sign-On" in v or "SAML" in v for v in variations)


def test_search_with_query_expansion_execution(mock_collection):
    """Verify multi-query execution merges and deduplicates confident results."""
    result = search_with_query_expansion(
        query="CP4BA BAW SSO security",
        collection_name="ibm_docs_rag2",
        top_k=5,
        min_confidence=0.75,
    )
    assert "Multi-Query Expanded Search Results" in result
    assert "Expansions Evaluated" in result
    assert "0.890" in result


# --------------------------------------------------------------------------- #
# Tool 3 & 4: Version Filtering & Comparison tests
# --------------------------------------------------------------------------- #
def test_search_by_product_version(mock_collection):
    """Verify version-scoped query execution."""
    result = search_by_product_version(
        query="security configuration",
        product_or_topic="cp-biz-automation",
        version="26.0.0",
        collection_name="ibm_docs_rag2",
        min_confidence=0.75,
    )
    assert "administering-configuring-security.md" in result
    assert "0.890" in result


def test_compare_product_versions(mock_collection):
    """Verify dual-version comparison output."""
    result = compare_product_versions(
        topic="deployment troubleshooting",
        version_a="24.0.1",
        version_b="26.0.0",
        collection_name="ibm_docs_rag2",
        min_confidence=0.70,  # lower threshold to include 24.0.1 doc (0.72)
    )
    assert "IBM Product Version Comparison" in result
    assert "Version 24.0.1 Context" in result
    assert "Version 26.0.0 Context" in result


# --------------------------------------------------------------------------- #
# Tool 5: generate_mermaid_diagram tests
# --------------------------------------------------------------------------- #
def test_generate_mermaid_flowchart():
    """Verify flowchart generation with nodes, shapes, and edge labels."""
    nodes = [
        {"id": "client", "label": "Web Browser", "shape": "round"},
        {"id": "keycloak", "label": "Keycloak SSO", "shape": "rectangle"},
        {"id": "db", "label": "Db2 Database", "shape": "database"},
    ]
    edges = [
        {"from": "client", "to": "keycloak", "label": "Authenticate"},
        {"from": "keycloak", "to": "db", "label": "User Query", "style": "thick"},
    ]
    diagram = generate_mermaid_diagram(
        title="CP4BA Security Flow",
        diagram_type="flowchart",
        direction="LR",
        nodes=nodes,
        edges=edges,
    )
    assert "```mermaid" in diagram
    assert "flowchart LR" in diagram
    assert "client(Web Browser)" in diagram
    assert "db[(Db2 Database)]" in diagram
    assert "client -->|Authenticate| keycloak" in diagram
    assert "keycloak == User Query ==> db" in diagram


def test_generate_mermaid_sequence_and_state():
    """Verify sequence and state diagrams."""
    seq = generate_mermaid_diagram(
        title="Auth Sequence",
        diagram_type="sequence",
        edges=[{"from": "User", "to": "App", "label": "Login Request"}],
    )
    assert "sequenceDiagram" in seq
    assert "User->>App: Login Request" in seq

    state = generate_mermaid_diagram(
        title="Deployment State",
        diagram_type="state",
        edges=[{"from": "Pending", "to": "Running", "label": "Pods Ready"}],
    )
    assert "stateDiagram-v2" in state
    assert "Pending --> Running: Pods Ready" in state


def test_sanitize_helpers():
    """Verify label and ID sanitizers prevent syntax errors."""
    assert _sanitize_id("node-1 with spaces & symbols!") == "node_1_with_spaces___symbols_"
    assert _sanitize_label('Label with "quotes" and [brackets]') == "Label with 'quotes' and (brackets)"


# --------------------------------------------------------------------------- #
# Tool 6 & 7: Doc Content & Sources List tests
# --------------------------------------------------------------------------- #
def test_get_ibm_doc_content(monkeypatch):
    """Verify multi-chunk reassembly."""
    multi_chunks = [
        {"chunk_no": 1, "chunk_count": 2, "text": "Part 2 of document.", "source": "test.md"},
        {"chunk_no": 0, "chunk_count": 2, "text": "Part 1 of document.", "source": "test.md"},
    ]
    fake_col = FakeCollection(multi_chunks)
    monkeypatch.setattr("rag_tools._get_collection", lambda coll="ibm_docs_rag2": fake_col)

    content = get_ibm_doc_content("test.md")
    assert "Document: `test.md`" in content
    assert "--- Chunk 1/2 ---" in content
    assert "Part 1 of document." in content
    assert "--- Chunk 2/2 ---" in content
    assert "Part 2 of document." in content


def test_list_ibm_knowledge_sources(mock_collection):
    """Verify source document inventory listing and in-memory TTL caching."""
    from rag_tools import _clear_sources_cache
    _clear_sources_cache()

    sources = list_ibm_knowledge_sources(collection_name="ibm_docs_rag2")
    assert "Indexed IBM Documentation Sources" in sources
    assert "cloud-paks/cp-biz-automation/26.0.0__topic=administering-configuring-security.md" in sources

    # Calling again hits the cache without re-querying collection
    cached_sources = list_ibm_knowledge_sources(collection_name="ibm_docs_rag2")
    assert cached_sources == sources


# --------------------------------------------------------------------------- #
# Retry & Error Handling tests
# --------------------------------------------------------------------------- #
def test_retry_on_transient_error():
    """Verify function retries on transient network errors and succeeds."""
    attempts = 0

    def _flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("Transient network glitch")
        return "SUCCESS"

    result = _with_retry(_flaky, max_retries=3, base_delay=0.01)
    assert result == "SUCCESS"
    assert attempts == 3


def test_is_permanent_error():
    """Verify permanent error detection."""
    assert _is_permanent_error(ValueError("401 Unauthorized access")) is True
    assert _is_permanent_error(RuntimeError("404 Not Found")) is True
    assert _is_permanent_error(ConnectionError("Timeout")) is False
