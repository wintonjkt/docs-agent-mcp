"""MCP Server for IBM Documentation RAG Agent.

Exposes tools, resources, and prompts over Model Context Protocol (MCP) using FastMCP (stdio).
Connects to DataStax Astra DB for vector semantic search across enterprise IBM documentation.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from rag_tools import (
    DEFAULT_COLLECTION,
    DEFAULT_MIN_CONFIDENCE,
    _compare_product_versions_impl,
    _generate_mermaid_diagram_impl,
    _get_ibm_doc_content_impl,
    _list_ibm_knowledge_sources_impl,
    _search_by_product_version_impl,
    _search_ibm_knowledge_impl,
    _search_with_query_expansion_impl,
)

# Configure logging strictly to stderr to prevent stdio protocol channel corruption
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("ibm-docs-rag-mcp")

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from mcp.server.fastmcp.server import TransportSecuritySettings

# Initialize FastMCP Server
mcp = FastMCP(
    name="ibm-docs-rag-agent",
    instructions=(
        "Enterprise IBM Documentation RAG MCP Server. Provides high-precision semantic search, "
        "query expansion, version filtering/comparison, full document retrieval, and Mermaid "
        "architecture diagram generation grounded in Astra DB vector storage."
    ),
    host="0.0.0.0",
    port=8080,
)

# Disable DNS rebinding checks to allow OpenShift Route hostnames and proxy domains
mcp.settings.transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False,
    allowed_hosts=["*"],
    allowed_origins=["*"],
)


@mcp.custom_route("/health", methods=["GET"])
@mcp.custom_route("/", methods=["GET"])
async def health_check(request: Any) -> JSONResponse:
    """Kubernetes / OpenShift liveness and readiness probe endpoint."""
    return JSONResponse({
        "status": "ok",
        "service": "ibm-docs-rag-mcp",
        "version": "1.0.0",
        "mcp_endpoints": {
            "streamable_http": "/mcp",
        },
    })


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool(
    name="search_ibm_knowledge",
    description=(
        "Search IBM product documentation (e.g. CP4BA, Business Automation Workflow, "
        "FileNet, ODM, Cloud Pak Operator) in Astra DB vector database. "
        "Filters results with strict confidence score threshold (>= min_confidence)."
    ),
)
def search_ibm_knowledge(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    product_filter: Optional[str] = None,
) -> str:
    """Execute vector search over IBM documentation.

    Args:
        query: Technical question, symptom, command, or topic.
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        top_k: Maximum number of confident chunks to return (max: 20).
        min_confidence: Minimum similarity score threshold (default: 0.75).
        product_filter: Optional substring filter on document path (e.g. "cp-biz-automation").

    Returns:
        Formatted markdown containing confidence scores, source citations, and content.
    """
    logger.info("Executing search_ibm_knowledge: query=%r, top_k=%d", query, top_k)
    return _search_ibm_knowledge_impl(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        min_confidence=min_confidence,
        product_filter=product_filter,
    )


@mcp.tool(
    name="search_with_query_expansion",
    description=(
        "Search IBM documentation using automated domain-specific synonym and acronym expansion "
        "(e.g. CP4BA, BAW, ODM, FNCM, SSO, LDAP, TLS, CR). Queries Astra DB in parallel, "
        "deduplicates chunks, and re-ranks by similarity."
    ),
)
def search_with_query_expansion(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Execute query expansion and multi-query semantic search.

    Args:
        query: Broad or multi-faceted search query with acronyms or technical concepts.
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        top_k: Maximum number of re-ranked unique chunks to return (default: 5).
        min_confidence: Minimum similarity score threshold (default: 0.75).

    Returns:
        Formatted markdown with deduplicated, top-ranked document chunks.
    """
    logger.info("Executing search_with_query_expansion: query=%r", query)
    return _search_with_query_expansion_impl(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        min_confidence=min_confidence,
    )


@mcp.tool(
    name="search_by_product_version",
    description=(
        "Search documentation strictly scoped to a specific IBM product release "
        "(e.g. product 'cp-biz-automation', version '24.0.1' or '26.0.0')."
    ),
)
def search_by_product_version(
    query: str,
    product_or_topic: str,
    version: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Execute version-scoped semantic search.

    Args:
        query: Technical topic or question (e.g. "Security configuration steps").
        product_or_topic: Product name or keyword (e.g. "cp-biz-automation", "baw").
        version: Target product version string (e.g. "24.0.1", "26.0.0").
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        top_k: Maximum number of results to return (default: 5).
        min_confidence: Minimum similarity threshold (default: 0.75).

    Returns:
        Version-filtered documentation results with confidence scores.
    """
    logger.info("Executing search_by_product_version: product=%r, version=%r", product_or_topic, version)
    return _search_by_product_version_impl(
        query=query,
        product_or_topic=product_or_topic,
        version=version,
        collection_name=collection_name,
        top_k=top_k,
        min_confidence=min_confidence,
    )


@mcp.tool(
    name="compare_product_versions",
    description=(
        "Retrieve and compare documentation across two different IBM product releases "
        "(e.g. comparing features, prerequisites, or security between 24.0.1 and 26.0.0)."
    ),
)
def compare_product_versions(
    topic: str,
    version_a: str,
    version_b: str,
    collection_name: str = DEFAULT_COLLECTION,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Retrieve parallel evidence across two versions for side-by-side comparison.

    Args:
        topic: Feature, component, or configuration topic to compare.
        version_a: First product release version (e.g. "24.0.1").
        version_b: Second product release version (e.g. "26.0.0").
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        min_confidence: Minimum similarity score threshold (default: 0.75).

    Returns:
        Dual-version evidence package formatted for side-by-side analysis.
    """
    logger.info("Executing compare_product_versions: topic=%r, vA=%r, vB=%r", topic, version_a, version_b)
    return _compare_product_versions_impl(
        topic=topic,
        version_a=version_a,
        version_b=version_b,
        collection_name=collection_name,
        min_confidence=min_confidence,
    )


@mcp.tool(
    name="generate_mermaid_diagram",
    description=(
        "Generate clean, syntax-validated Mermaid diagram code blocks for IBM Cloud Pak architectures, "
        "component interactions, deployment topologies, or troubleshooting workflows. Does not require database credentials."
    ),
)
def generate_mermaid_diagram(
    title: str,
    diagram_type: str = "flowchart",
    direction: str = "TD",
    nodes: Optional[List[Dict[str, str]]] = None,
    edges: Optional[List[Dict[str, str]]] = None,
    raw_content: Optional[str] = None,
) -> str:
    """Construct sanitized Mermaid diagram markdown code block.

    Args:
        title: Title and caption for the diagram.
        diagram_type: Diagram category: "flowchart", "architecture", "sequence", "state" (default: "flowchart").
        direction: Flowchart layout direction: "TD", "TB", "LR", "RL" (default: "TD").
        nodes: Structured node definitions with 'id', 'label', and optional 'shape'.
        edges: Structured connection definitions with 'from', 'to', and optional 'label'/'style'.
        raw_content: Optional raw Mermaid body to sanitize and format.

    Returns:
        Sanitized Mermaid markdown code block.
    """
    logger.info("Generating Mermaid diagram: title=%r, type=%r", title, diagram_type)
    return _generate_mermaid_diagram_impl(
        title=title,
        diagram_type=diagram_type,
        direction=direction,
        nodes=nodes,
        edges=edges,
        raw_content=raw_content,
    )


@mcp.tool(
    name="get_ibm_doc_content",
    description=(
        "Fetch and reassemble all chunks of a specific IBM documentation file from Astra DB "
        "using its exact source file path."
    ),
)
def get_ibm_doc_content(
    source_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    max_chunks: int = 15,
) -> str:
    """Fetch and reassemble consecutive chunks of a documentation file.

    Args:
        source_name: Exact document path (e.g. "cloud-paks/cp-biz-automation/26.0.0__topic=administering-configuring-security.md").
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        max_chunks: Maximum chunks to retrieve (default: 15, max: 50).

    Returns:
        Full reassembled markdown content of the requested document.
    """
    logger.info("Fetching doc content: source=%r", source_name)
    return _get_ibm_doc_content_impl(
        source_name=source_name,
        collection_name=collection_name,
        max_chunks=max_chunks,
    )


@mcp.tool(
    name="list_ibm_knowledge_sources",
    description=(
        "List all indexed IBM documentation sources and chunk counts currently stored in Astra DB."
    ),
)
def list_ibm_knowledge_sources(
    collection_name: str = DEFAULT_COLLECTION,
    prefix: str = "",
    limit: int = 50,
) -> str:
    """Discover available IBM knowledge base sources.

    Args:
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        prefix: Optional prefix filter for source paths (e.g. "cloud-paks/cp-biz-automation").
        limit: Maximum number of sources to return (default: 50, max: 100).

    Returns:
        Markdown catalog listing indexed document paths and total chunk counts.
    """
    logger.info("Listing knowledge sources: prefix=%r, limit=%d", prefix, limit)
    return _list_ibm_knowledge_sources_impl(
        collection_name=collection_name,
        prefix=prefix,
        limit=limit,
    )


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource(
    uri="ibm-docs://catalog",
    name="IBM Documentation Knowledge Catalog",
    description="Catalog of all indexed IBM documentation sources and chunk counts stored in Astra DB.",
    mime_type="text/markdown",
)
def get_catalog_resource() -> str:
    """Read the full catalog of indexed documentation sources."""
    logger.info("Reading MCP resource: ibm-docs://catalog")
    return _list_ibm_knowledge_sources_impl(
        collection_name=DEFAULT_COLLECTION,
        prefix="",
        limit=100,
    )


@mcp.resource(
    uri="ibm-docs://doc/{source_path}",
    name="IBM Documentation Document Content",
    description="Full reassembled text content for a specific documentation file in Astra DB.",
    mime_type="text/markdown",
)
def get_document_resource(source_path: str) -> str:
    """Read full content of a specific documentation file by source path."""
    logger.info("Reading MCP resource: ibm-docs://doc/%s", source_path)
    return _get_ibm_doc_content_impl(
        source_name=source_path,
        collection_name=DEFAULT_COLLECTION,
        max_chunks=50,
    )


# ============================================================================
# MCP Prompts
# ============================================================================


@mcp.prompt(
    name="ask_ibm_doc",
    description="Structured prompt for answering IBM product questions grounded in Astra DB documentation.",
)
def ask_ibm_doc_prompt(question: str, product_version: Optional[str] = None) -> str:
    """Generate prompt template for answering technical IBM product questions."""
    version_clause = f" specifically for version `{product_version}`" if product_version else ""
    return (
        f"You are an expert IBM Product Architect and Technical Support Specialist.\n\n"
        f"The user has the following question{version_clause}:\n"
        f"\"{question}\"\n\n"
        "Instructions:\n"
        "1. Use `search_ibm_knowledge` or `search_with_query_expansion` (or `search_by_product_version` if version specified) "
        "to retrieve documentation from Astra DB with a confidence threshold >= 0.75.\n"
        "2. Ground all answers strictly in retrieved evidence. Never fabricate parameters, paths, or commands.\n"
        "3. If no documents meet the 0.75 threshold, state clearly that no confident documentation was found.\n"
        "4. Structure your response with:\n"
        "   - 📌 Direct Answer / Executive Summary\n"
        "   - ⚙️ Key Details & Procedure Steps\n"
        "   - 📐 Mermaid Diagram (if architectural or workflow related)\n"
        "   - 📚 Source References & Confidence Scores\n"
        "   - 💡 Related Topics / Next Steps"
    )


@mcp.prompt(
    name="compare_ibm_versions",
    description="Structured prompt for comparing technical features, prerequisites, or architecture between two IBM product releases.",
)
def compare_ibm_versions_prompt(topic: str, version_a: str, version_b: str) -> str:
    """Generate prompt template for cross-version product comparisons."""
    return (
        f"You are an expert IBM Product Architect.\n\n"
        f"The user wants to compare \"{topic}\" between IBM product version `{version_a}` and `{version_b}`.\n\n"
        "Instructions:\n"
        "1. Call `compare_product_versions` to fetch parallel documentation evidence for both versions.\n"
        "2. Structure your comparison with:\n"
        "   - 📌 Summary of Key Differences\n"
        "   - 📊 Comparative Table (Feature/Parameter | Version A | Version B | Impact)\n"
        "   - ⚠️ Migration & Compatibility Considerations\n"
        "   - 📚 Source Citations & Confidence Scores"
    )


@mcp.prompt(
    name="troubleshoot_ibm_issue",
    description="Diagnostic troubleshooting prompt for IBM Cloud Pak / product errors and symptoms.",
)
def troubleshoot_ibm_issue_prompt(symptom_or_error: str, product: Optional[str] = None) -> str:
    """Generate prompt template for diagnosing and resolving IBM technical issues."""
    prod_str = f" in {product}" if product else ""
    return (
        f"You are an expert IBM Technical Support Specialist.\n\n"
        f"The user is experiencing the following issue{prod_str}:\n"
        f"\"{symptom_or_error}\"\n\n"
        "Instructions:\n"
        "1. Use `search_with_query_expansion` or `search_ibm_knowledge` to search troubleshooting, logs, "
        "and error resolution documentation in Astra DB.\n"
        "2. Structure your diagnosis with:\n"
        "   - 🔍 Root Cause Analysis & Likely Triggers\n"
        "   - 🛠️ Step-by-Step Resolution / Remediation\n"
        "   - 🧪 Verification & Health Check Steps\n"
        "   - 📐 Troubleshooting Flowchart (using `generate_mermaid_diagram`)\n"
        "   - 📚 Source Citations"
    )


# ============================================================================
# ASGI Application for Streamable HTTP Container Deployments (e.g., Uvicorn)
# ============================================================================
app = mcp.streamable_http_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # If MCP_TRANSPORT=streamable-http or '--http' argument is given, run Streamable HTTP; otherwise default to stdio
    transport_mode = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport_mode in ("streamable-http", "http") or "--http" in sys.argv:
        logger.info("Starting ibm-docs-rag-agent MCP Server over Streamable HTTP transport on 0.0.0.0:8080/mcp...")
        mcp.run(transport="streamable-http")
    else:
        logger.info("Starting ibm-docs-rag-agent MCP Server over stdio transport...")
        mcp.run(transport="stdio")
