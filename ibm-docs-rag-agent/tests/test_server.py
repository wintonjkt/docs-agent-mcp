"""Tests for IBM Documentation RAG FastMCP Server (server.py)."""

import pytest
import sys
import os

# Add parent directory to sys.path so server can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import (
    mcp,
    search_ibm_knowledge,
    search_with_query_expansion,
    search_by_product_version,
    compare_product_versions,
    generate_mermaid_diagram,
    get_ibm_doc_content,
    list_ibm_knowledge_sources,
    get_catalog_resource,
    get_document_resource,
    ask_ibm_doc_prompt,
    compare_ibm_versions_prompt,
    troubleshoot_ibm_issue_prompt,
)


def test_mcp_server_metadata():
    """Verify FastMCP server initialization and identity."""
    assert mcp.name == "ibm-docs-rag-agent"
    assert "Astra DB" in mcp.instructions


@pytest.mark.asyncio
async def test_mcp_registered_tools_list():
    """Verify all 7 RAG tools are registered on the FastMCP server."""
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    expected_tools = {
        "search_ibm_knowledge",
        "search_with_query_expansion",
        "search_by_product_version",
        "compare_product_versions",
        "generate_mermaid_diagram",
        "get_ibm_doc_content",
        "list_ibm_knowledge_sources",
    }
    assert expected_tools.issubset(tool_names)


@pytest.mark.asyncio
async def test_mcp_registered_resources_list():
    """Verify MCP resource catalog and template registration."""
    resources = await mcp.list_resources()
    resource_uris = {str(r.uri) for r in resources}
    assert "ibm-docs://catalog" in resource_uris

    templates = await mcp.list_resource_templates()
    template_uris = {str(tp.uriTemplate) for tp in templates}
    assert "ibm-docs://doc/{source_path}" in template_uris


@pytest.mark.asyncio
async def test_mcp_registered_prompts_list():
    """Verify MCP prompt templates registration."""
    prompts = await mcp.list_prompts()
    prompt_names = {p.name for p in prompts}
    expected_prompts = {"ask_ibm_doc", "compare_ibm_versions", "troubleshoot_ibm_issue"}
    assert expected_prompts.issubset(prompt_names)


def test_search_ibm_knowledge_via_mcp_tool(mock_collection):
    """Test search_ibm_knowledge tool execution and confidence thresholding."""
    result = search_ibm_knowledge(query="Keycloak SSO configuration", min_confidence=0.75)
    assert "2 matching chunks with confidence ≥ 0.75" in result
    assert "0.890" in result or "0.89" in result
    assert "0.780" in result or "0.78" in result
    assert "0.720" not in result and "0.72" not in result  # Filtered out (0.72 < 0.75)


def test_search_with_query_expansion_via_mcp_tool(mock_collection):
    """Test search_with_query_expansion tool execution."""
    result = search_with_query_expansion(query="CP4BA SSO security")
    assert "0.89" in result
    assert "Keycloak SSO" in result


def test_search_by_product_version_via_mcp_tool(mock_collection):
    """Test search_by_product_version tool execution."""
    result = search_by_product_version(
        query="security",
        product_or_topic="cp-biz-automation",
        version="26.0.0",
        min_confidence=0.75,
    )
    assert "26.0.0" in result
    assert "0.89" in result


def test_compare_product_versions_via_mcp_tool(mock_collection):
    """Test compare_product_versions tool execution."""
    result = compare_product_versions(
        topic="troubleshooting",
        version_a="26.0.0",
        version_b="24.0.1",
        min_confidence=0.70,
    )
    assert "26.0.0" in result
    assert "24.0.1" in result


def test_generate_mermaid_diagram_via_mcp_tool():
    """Test generate_mermaid_diagram tool without database credentials."""
    result = generate_mermaid_diagram(
        title="CP4BA Authentication Flow",
        diagram_type="flowchart",
        direction="LR",
        nodes=[
            {"id": "user", "label": "User Browser"},
            {"id": "keycloak", "label": "Keycloak SSO"},
            {"id": "baw", "label": "BAW Server"},
        ],
        edges=[
            {"from": "user", "to": "keycloak", "label": "Authenticate"},
            {"from": "keycloak", "to": "baw", "label": "SAML Token"},
        ],
    )
    assert "```mermaid" in result
    assert "flowchart LR" in result
    assert "User Browser" in result
    assert "Keycloak SSO" in result


def test_get_ibm_doc_content_via_mcp_tool(mock_collection):
    """Test get_ibm_doc_content tool execution."""
    result = get_ibm_doc_content(
        source_name="cloud-paks/cp-biz-automation/26.0.0__topic=administering-configuring-security.md"
    )
    assert "Document:" in result
    assert "Chunk 1/2" in result
    assert "Keycloak SSO" in result


def test_list_ibm_knowledge_sources_via_mcp_tool(mock_collection):
    """Test list_ibm_knowledge_sources tool execution."""
    result = list_ibm_knowledge_sources()
    assert "cloud-paks/cp-biz-automation/26.0.0__topic=administering-configuring-security.md" in result


def test_mcp_catalog_resource(mock_collection):
    """Test reading the ibm-docs://catalog resource."""
    result = get_catalog_resource()
    assert "Indexed IBM Documentation Sources" in result
    assert "cloud-paks/cp-biz-automation/26.0.0" in result


def test_mcp_document_resource(mock_collection):
    """Test reading the ibm-docs://doc/{source_path} resource."""
    result = get_document_resource(
        source_path="cloud-paks/cp-biz-automation/26.0.0__topic=administering-configuring-security.md"
    )
    assert "Document:" in result
    assert "Chunk 1/2" in result
    assert "Keycloak SSO" in result


def test_mcp_prompts():
    """Test generation of all 3 MCP prompt templates."""
    # Test ask_ibm_doc prompt
    ask_p = ask_ibm_doc_prompt(question="How to configure LDAP in CP4BA?", product_version="26.0.0")
    assert "How to configure LDAP in CP4BA?" in ask_p
    assert "26.0.0" in ask_p
    assert "0.75" in ask_p

    # Test compare_ibm_versions prompt
    comp_p = compare_ibm_versions_prompt(topic="SSO configuration", version_a="24.0.1", version_b="26.0.0")
    assert "SSO configuration" in comp_p
    assert "24.0.1" in comp_p
    assert "26.0.0" in comp_p

    # Test troubleshoot_ibm_issue prompt
    trouble_p = troubleshoot_ibm_issue_prompt(symptom_or_error="CrashLoopBackOff on baw-operator", product="CP4BA")
    assert "CrashLoopBackOff on baw-operator" in trouble_p
    assert "CP4BA" in trouble_p
    assert "generate_mermaid_diagram" in trouble_p
