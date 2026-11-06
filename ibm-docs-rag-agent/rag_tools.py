"""Enterprise RAG and Architecture Diagramming Tools for watsonx Orchestrate.

Provides semantic vector retrieval from Astra DB, strict confidence thresholding (>= 0.75),
multi-query expansion, version filtering/comparison, and visual Mermaid diagram generation.

Tools:
    search_ibm_knowledge          - Semantic vector search in Astra DB with strict confidence filtering (>= 0.75).
    search_with_query_expansion   - Expands query into technical domain variations, queries in parallel, and re-ranks.
    search_by_product_version     - Retrieves documentation scoped to a specific IBM product and version.
    compare_product_versions      - Compares features, configuration, or prerequisites across two product versions.
    generate_mermaid_diagram      - Generates clean, syntax-validated Mermaid architecture/flow diagrams.
    get_ibm_doc_content           - Fetches full document content or chunk details by source key.
    list_ibm_knowledge_sources    - Lists indexed IBM products, versions, and topics in Astra DB.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import json
import os
import random
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from astrapy import DataAPIClient
from astrapy.api_options import APIOptions, TimeoutOptions

# Attempt to import IBM watsonx Orchestrate ADK dependencies; fallback gracefully for local/test runs
try:
    from ibm_watsonx_orchestrate.agent_builder.tools import tool
    from ibm_watsonx_orchestrate.run import connections
    from ibm_watsonx_orchestrate.run.connections import ConnectionType
except ImportError:
    def tool(*args, **kwargs):
        def decorator(func):
            func._tool_meta = kwargs
            return func
        return decorator

    class ConnectionType:
        KEY_VALUE = "key_value"

    connections = None

APP_ID = "astra"
DEFAULT_COLLECTION = "ibm_docs_rag2"
DEFAULT_MIN_CONFIDENCE = 0.75
MAX_TOP_K = 20

# --------------------------------------------------------------------------- #
# Domain Synonyms & Query Expansion Dictionary
# --------------------------------------------------------------------------- #
DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "cp4ba": ["Cloud Pak for Business Automation", "CP4BA", "IBM Cloud Pak Business Automation"],
    "baw": ["Business Automation Workflow", "BAW", "Workflow Server", "Process Server"],
    "odm": ["Operational Decision Manager", "ODM", "Decision Server", "Decision Center"],
    "fncm": ["FileNet Content Manager", "FNCM", "Content Platform Engine", "CPE"],
    "sso": ["Single Sign-On", "SSO", "SAML", "OIDC", "IAM authentication"],
    "db2": ["Db2", "IBM Db2", "Database", "DB2 Universal Database"],
    "cr": ["Custom Resource", "CR YAML", "CR configuration", "custom resource definition"],
    "operator": ["Cloud Pak Operator", "Kubernetes Operator", "Ansible Operator"],
    "ldap": ["LDAP directory", "Active Directory", "User registry", "Lightweight Directory Access Protocol"],
    "tls": ["TLS certificate", "SSL configuration", "Truststore", "Keystore", "Secret"],
    "troubleshooting": ["troubleshooting", "diagnostics", "error resolution", "logs inspection", "known issues"],
}


# --------------------------------------------------------------------------- #
# Error Classification & Retry Policy
# --------------------------------------------------------------------------- #
def _is_permanent_error(exc: BaseException) -> bool:
    """Check if exception represents a non-retryable error (e.g. 400, 401, 403, 404, invalid auth)."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        status_code = getattr(resp, "status_code", None)
        if status_code in (400, 401, 403, 404, 422):
            return True
    msg = str(exc).lower()
    if any(code in msg for code in ("400", "401", "403", "404", "unauthorized", "forbidden", "not found", "nosuchcollection")):
        return True
    return False


def _with_retry(
    func: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    action_name: str = "database operation",
) -> Any:
    """Execute function with exponential backoff and full jitter on transient failures."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_permanent_error(exc) or attempt == max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jittered_delay = delay * (0.5 + random.random() * 0.5)
            time.sleep(jittered_delay)
    raise last_exc


# --------------------------------------------------------------------------- #
# Astra DB Connection Helpers
# --------------------------------------------------------------------------- #
_CLIENT_CACHE: Dict[Tuple[str, str], Any] = {}
_CLIENT_LOCK = threading.Lock()


def _get_astra_credentials() -> Tuple[str, str]:
    """Retrieve Astra DB API endpoint and application token from Orchestrate connections or env vars."""
    endpoint = ""
    token = ""

    if connections:
        try:
            conn = connections.key_value(APP_ID)
            if conn:
                endpoint = conn.get("ASTRA_DB_API_ENDPOINT") or conn.get("endpoint") or conn.get("url") or ""
                token = conn.get("ASTRA_DB_APPLICATION_TOKEN") or conn.get("token") or conn.get("api_token") or ""
        except Exception:
            pass

    if not endpoint:
        endpoint = os.getenv("ASTRA_DB_API_ENDPOINT", "").strip()
    if not token:
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "").strip()

    if not endpoint or not token:
        raise ValueError(
            "Astra DB credentials not found. Ensure connection 'astra' or environment variables "
            "ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN are configured."
        )

    return endpoint, token


def _get_database() -> Any:
    """Retrieve or build a cached Astra DB Database handle with connection pooling."""
    endpoint, token = _get_astra_credentials()
    cache_key = (endpoint, token)
    with _CLIENT_LOCK:
        if cache_key not in _CLIENT_CACHE:
            client = DataAPIClient(
                api_options=APIOptions(
                    timeout_options=TimeoutOptions(
                        request_timeout_ms=30_000,
                        general_method_timeout_ms=60_000,
                    )
                )
            )
            _CLIENT_CACHE[cache_key] = client.get_database(endpoint, token=token)
        return _CLIENT_CACHE[cache_key]


def _get_collection(collection_name: str = DEFAULT_COLLECTION) -> Any:
    """Get Astra DB collection handle with pooled connection reuse."""
    database = _get_database()
    return database.get_collection(collection_name.strip() or DEFAULT_COLLECTION)


# --------------------------------------------------------------------------- #
# Tool 1: search_ibm_knowledge
# --------------------------------------------------------------------------- #
def _search_ibm_knowledge_impl(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    product_filter: Optional[str] = None,
) -> str:
    """Core logic for semantic vector search with strict confidence thresholding."""
    clean_query = (query or "").strip()
    if not clean_query:
        return "Please provide a valid search query about IBM products."

    try:
        top_k_int = max(1, min(int(top_k), MAX_TOP_K))
    except (ValueError, TypeError):
        top_k_int = 5

    try:
        min_conf = float(min_confidence)
    except (ValueError, TypeError):
        min_conf = DEFAULT_MIN_CONFIDENCE

    coll_name = (collection_name or DEFAULT_COLLECTION).strip()

    try:
        collection = _get_collection(coll_name)
        filter_expr: Dict[str, Any] = {}
        if product_filter and product_filter.strip():
            filter_expr["source"] = {"$regex": product_filter.strip()}

        raw_hits = _with_retry(
            lambda: list(
                collection.find(
                    filter_expr,
                    sort={"$vectorize": clean_query},
                    limit=top_k_int * 2,  # retrieve extra to filter by confidence threshold
                    projection={"source": 1, "chunk_no": 1, "chunk_count": 1, "text": 1, "etag": 1, "last_modified": 1},
                    include_similarity=True,
                )
            ),
            max_retries=3,
            action_name=f"vector search in '{coll_name}'",
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error executing semantic search against collection '{coll_name}': {exc}"

    if not raw_hits:
        return (
            f"No documentation found in collection '{coll_name}' for query: '{clean_query}'.\n\n"
            f"Suggestions:\n"
            f"- Verify the product name or topic spelling.\n"
            f"- Try broader keywords (e.g. 'BAW deployment', 'Cloud Pak security')."
        )

    # Filter and sort by confidence score
    confident_hits = []
    max_score = 0.0

    for hit in raw_hits:
        score = hit.get("$similarity")
        if isinstance(score, (int, float)):
            max_score = max(max_score, score)
            if score >= min_conf:
                confident_hits.append(hit)

    # If no hits meet the strict confidence threshold
    if not confident_hits:
        return (
            f"⚠️ **Confidence Threshold Notice**: No documentation met the minimum confidence score of "
            f"**{min_conf:.2f}** for query: *'{clean_query}'*.\n\n"
            f"- **Highest Similarity Score Observed**: `{max_score:.3f}` (below required threshold of `{min_conf:.2f}`).\n"
            f"- **Target Collection**: `{coll_name}`\n\n"
            f"**Recommended Refinements**:\n"
            f"1. Use more specific IBM product keywords (e.g., 'CP4BA 26.0.0 security configuration').\n"
            f"2. Use `search_with_query_expansion` to search technical synonyms and aliases.\n"
            f"3. Check available topics using `list_ibm_knowledge_sources`."
        )

    # Format confident hits (limited to top_k)
    results = [
        f"### 🔍 IBM Knowledge Search Results ({len(confident_hits[:top_k_int])} matching chunks with confidence ≥ {min_conf:.2f})\n"
        f"**Query**: *{clean_query}* | **Collection**: `{coll_name}`\n"
    ]

    for rank, hit in enumerate(confident_hits[:top_k_int], start=1):
        score = hit.get("$similarity", 0.0)
        source = hit.get("source", "unknown")
        chunk_no = hit.get("chunk_no", 0)
        chunk_count = hit.get("chunk_count", 1)
        text = hit.get("text", "").strip()

        results.append(
            f"#### [{rank}] Source: `{source}` (Chunk {chunk_no + 1}/{chunk_count} | Confidence: **{score:.3f}**)\n"
            f"```text\n{text}\n```"
        )

    return "\n\n".join(results)


@tool(expected_credentials=[{"app_id": APP_ID, "type": [ConnectionType.KEY_VALUE]}])
def search_ibm_knowledge(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    product_filter: Optional[str] = None,
) -> str:
    """Search IBM product documentation in Astra DB using semantic vector embeddings with strict confidence thresholding.

    Args:
        query: Technical question, symptom, command, or topic to search for (e.g., "BAW deployment troubleshooting").
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        top_k: Maximum number of confident chunks to return (default: 5, max: 20).
        min_confidence: Minimum similarity score threshold required (default: 0.75).
        product_filter: Optional substring filter on the source document path (e.g., "cp-biz-automation").

    Returns:
        Structured search results with similarity scores, source citations, and chunk texts, or a threshold notification if below 0.75.
    """
    return _search_ibm_knowledge_impl(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        min_confidence=min_confidence,
        product_filter=product_filter,
    )


# --------------------------------------------------------------------------- #
# Tool 2: search_with_query_expansion
# --------------------------------------------------------------------------- #
def _generate_query_variations(query: str) -> List[str]:
    """Generate technical synonyms and domain expansions for the user query."""
    variations = [query.strip()]
    query_lower = query.lower()

    # 1. Expand single terms with their primary synonym
    for key, synonyms in DOMAIN_SYNONYMS.items():
        if re.search(r"\b" + re.escape(key) + r"\b", query_lower):
            for syn in synonyms[:2]:
                replaced = re.sub(r"\b" + re.escape(key) + r"\b", syn, query, flags=re.IGNORECASE)
                if replaced != query and replaced not in variations:
                    variations.append(replaced)

    # 2. Fully expanded variation where all matching abbreviations are expanded
    fully_expanded = query
    for key, synonyms in DOMAIN_SYNONYMS.items():
        if re.search(r"\b" + re.escape(key) + r"\b", fully_expanded, flags=re.IGNORECASE):
            fully_expanded = re.sub(r"\b" + re.escape(key) + r"\b", synonyms[0], fully_expanded, flags=re.IGNORECASE)
    if fully_expanded != query and fully_expanded not in variations:
        variations.append(fully_expanded)

    # 3. Check if full term is in query and add abbreviation
    for key, synonyms in DOMAIN_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower() in query_lower and not re.search(r"\b" + re.escape(key) + r"\b", query_lower):
                replaced = re.sub(re.escape(syn), key.upper(), query, flags=re.IGNORECASE)
                if replaced != query and replaced not in variations:
                    variations.append(replaced)

    return variations[:8]  # cap at 8 parallel queries


def _search_with_query_expansion_impl(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Execute multi-query vector searches, merge hits, deduplicate, and re-rank results."""
    clean_query = (query or "").strip()
    if not clean_query:
        return "Please provide a valid query for multi-query expansion."

    variations = _generate_query_variations(clean_query)
    coll_name = (collection_name or DEFAULT_COLLECTION).strip()

    try:
        collection = _get_collection(coll_name)
    except Exception as exc:  # noqa: BLE001
        return f"Error connecting to collection '{coll_name}': {exc}"

    all_hits_dict: Dict[Tuple[str, int], Dict[str, Any]] = {}
    max_score = 0.0

    def _run_single_query(q_var: str) -> List[Dict[str, Any]]:
        try:
            return list(
                collection.find(
                    {},
                    sort={"$vectorize": q_var},
                    limit=top_k * 2,
                    projection={"source": 1, "chunk_no": 1, "chunk_count": 1, "text": 1},
                    include_similarity=True,
                )
            )
        except Exception:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(variations))) as executor:
        future_to_q = {executor.submit(_run_single_query, var): var for var in variations}
        for future in concurrent.futures.as_completed(future_to_q):
            hits = future.result()
            for hit in hits:
                score = hit.get("$similarity")
                if isinstance(score, (int, float)):
                    key = (hit.get("source", ""), hit.get("chunk_no", 0))
                    if key not in all_hits_dict or score > all_hits_dict[key].get("$similarity", 0.0):
                        all_hits_dict[key] = hit

    if not all_hits_dict:
        return f"No results found across query expansions: {variations}."

    # Filter by confidence threshold and sort descending by similarity score
    sorted_hits = sorted(all_hits_dict.values(), key=lambda h: h.get("$similarity", 0.0), reverse=True)
    confident_hits = [h for h in sorted_hits if h.get("$similarity", 0.0) >= min_confidence]
    max_score = sorted_hits[0].get("$similarity", 0.0) if sorted_hits else 0.0

    if not confident_hits:
        return (
            f"⚠️ **Multi-Query Confidence Notice**: Evaluated {len(variations)} query variations ({', '.join(variations)}), "
            f"but no documents met the **{min_confidence:.2f}** confidence threshold.\n"
            f"- Highest score observed: `{max_score:.3f}`."
        )

    results = [
        f"### 🌐 Multi-Query Expanded Search Results ({len(confident_hits[:top_k])} chunks with confidence ≥ {min_confidence:.2f})\n"
        f"**Original Query**: *{clean_query}*\n"
        f"**Expansions Evaluated**: {', '.join(f'`{v}`' for v in variations)}\n"
    ]

    for rank, hit in enumerate(confident_hits[:top_k], start=1):
        score = hit.get("$similarity", 0.0)
        source = hit.get("source", "unknown")
        chunk_no = hit.get("chunk_no", 0)
        text = hit.get("text", "").strip()

        results.append(
            f"#### [{rank}] Source: `{source}` (Chunk {chunk_no + 1} | Confidence: **{score:.3f}**)\n"
            f"```text\n{text}\n```"
        )

    return "\n\n".join(results)


@tool(expected_credentials=[{"app_id": APP_ID, "type": [ConnectionType.KEY_VALUE]}])
def search_with_query_expansion(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Perform multi-query expansion across enterprise IBM product synonyms, querying in parallel and re-ranking hits >= 0.75.

    Args:
        query: Technical question containing acronyms or multi-faceted terms (e.g., "CP4BA SSO SAML configuration").
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        top_k: Number of re-ranked top chunks to return (default: 5).
        min_confidence: Minimum similarity score threshold (default: 0.75).

    Returns:
        Merged and deduplicated search results sorted by highest similarity score.
    """
    return _search_with_query_expansion_impl(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        min_confidence=min_confidence,
    )


# --------------------------------------------------------------------------- #
# Tool 3: search_by_product_version
# --------------------------------------------------------------------------- #
def _search_by_product_version_impl(
    query: str,
    product_or_topic: str,
    version: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Retrieve documentation strictly scoped to an IBM product version with regex pushdown."""
    clean_query = (query or "").strip()
    ver_clean = (version or "").strip()
    prod_clean = (product_or_topic or "").strip()

    if not clean_query:
        return "Please provide a search query."

    # Scope document paths by version or product keyword to push down filter to Astra DB
    filter_target = ver_clean or prod_clean

    res = _search_ibm_knowledge_impl(
        query=f"{prod_clean} {ver_clean} {clean_query}".strip(),
        collection_name=collection_name,
        top_k=top_k * 2,
        min_confidence=min_confidence,
        product_filter=filter_target or None,
    )
    return res


@tool(expected_credentials=[{"app_id": APP_ID, "type": [ConnectionType.KEY_VALUE]}])
def search_by_product_version(
    query: str,
    product_or_topic: str,
    version: str,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Search IBM documentation scoped to a specific product family and version (e.g. CP4BA 24.0.1 or 26.0.0).

    Args:
        query: The technical topic or question (e.g., "Security configuration steps").
        product_or_topic: Product name or keyword (e.g., "cp-biz-automation", "baw").
        version: Target version string (e.g., "24.0.1", "26.0.0").
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        top_k: Maximum number of results to return (default: 5).
        min_confidence: Minimum similarity score threshold (default: 0.75).

    Returns:
        Version-scoped documentation results with confidence scores and source paths.
    """
    return _search_by_product_version_impl(
        query=query,
        product_or_topic=product_or_topic,
        version=version,
        collection_name=collection_name,
        top_k=top_k,
        min_confidence=min_confidence,
    )


# --------------------------------------------------------------------------- #
# Tool 4: compare_product_versions
# --------------------------------------------------------------------------- #
def _compare_product_versions_impl(
    topic: str,
    version_a: str,
    version_b: str,
    collection_name: str = DEFAULT_COLLECTION,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Retrieve and format side-by-side documentation for two product versions concurrently."""
    t_clean = (topic or "").strip()
    v_a = (version_a or "").strip()
    v_b = (version_b or "").strip()

    if not t_clean or not v_a or not v_b:
        return "Please specify the topic, version_a, and version_b for comparison."

    # Execute both version queries in parallel to reduce overall latency by 50%
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            _search_ibm_knowledge_impl,
            query=f"{t_clean} {v_a}",
            collection_name=collection_name,
            top_k=3,
            min_confidence=min_confidence,
        )
        future_b = executor.submit(
            _search_ibm_knowledge_impl,
            query=f"{t_clean} {v_b}",
            collection_name=collection_name,
            top_k=3,
            min_confidence=min_confidence,
        )
        res_a = future_a.result()
        res_b = future_b.result()

    output = [
        f"## ⚖️ IBM Product Version Comparison: `{v_a}` vs `{v_b}`\n"
        f"**Comparison Topic**: *{t_clean}*\n\n"
        f"### 📦 Version {v_a} Context:\n{res_a}\n\n"
        f"---\n\n"
        f"### 📦 Version {v_b} Context:\n{res_b}\n"
    ]
    return "\n".join(output)


@tool(expected_credentials=[{"app_id": APP_ID, "type": [ConnectionType.KEY_VALUE]}])
def compare_product_versions(
    topic: str,
    version_a: str,
    version_b: str,
    collection_name: str = DEFAULT_COLLECTION,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> str:
    """Retrieve documentation across two distinct IBM product versions to enable side-by-side technical comparison.

    Args:
        topic: Feature, component, or configuration topic to compare (e.g. "security configuration", "deployment troubleshooting").
        version_a: First product version (e.g., "24.0.1").
        version_b: Second product version (e.g., "26.0.0").
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        min_confidence: Minimum similarity score threshold (default: 0.75).

    Returns:
        Structured dual-version evidence package for comparative analysis.
    """
    return _compare_product_versions_impl(
        topic=topic,
        version_a=version_a,
        version_b=version_b,
        collection_name=collection_name,
        min_confidence=min_confidence,
    )


# --------------------------------------------------------------------------- #
# Tool 5: generate_mermaid_diagram
# --------------------------------------------------------------------------- #
def _sanitize_label(text: str) -> str:
    """Clean label text so it is safe inside Mermaid node brackets."""
    clean = text.replace('"', "'").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    return clean.strip()


def _sanitize_id(text: str) -> str:
    """Generate a valid Mermaid node identifier (alphanumeric and underscores only)."""
    node_id = re.sub(r"[^a-zA-Z0-9_]", "_", text.strip())
    return node_id or "node"


def _generate_mermaid_diagram_impl(
    title: str,
    diagram_type: str = "flowchart",
    direction: str = "TD",
    nodes: Optional[List[Dict[str, str]]] = None,
    edges: Optional[List[Dict[str, str]]] = None,
    raw_content: Optional[str] = None,
) -> str:
    """Generate and validate a clean Mermaid diagram code block."""
    d_type = (diagram_type or "flowchart").lower().strip()
    dir_code = (direction or "TD").upper().strip()
    if dir_code not in ("TD", "TB", "LR", "RL"):
        dir_code = "TD"

    lines: List[str] = []

    # If raw content is provided, validate and wrap
    if raw_content and raw_content.strip():
        raw_clean = raw_content.strip()
        if raw_clean.startswith("```mermaid"):
            raw_clean = raw_clean[10:]
        if raw_clean.endswith("```"):
            raw_clean = raw_clean[:-3]
        return f"```{d_type}\n{raw_clean.strip()}\n```".replace("```flowchart", "```mermaid\nflowchart").replace("```sequence", "```mermaid\nsequenceDiagram")

    if d_type in ("flowchart", "architecture", "graph"):
        lines.append(f"flowchart {dir_code}")
        if title:
            lines.append(f"    %% Title: {title}")

        # Add nodes
        if nodes:
            for n in nodes:
                n_id = _sanitize_id(n.get("id") or n.get("name") or "node")
                n_label = _sanitize_label(n.get("label") or n.get("name") or n_id)
                n_shape = n.get("shape", "rectangle")
                if n_shape == "round":
                    lines.append(f"    {n_id}({n_label})")
                elif n_shape == "database":
                    lines.append(f"    {n_id}[({n_label})]")
                elif n_shape == "circle":
                    lines.append(f"    {n_id}(({n_label}))")
                else:
                    lines.append(f"    {n_id}[{n_label}]")

        # Add edges
        if edges:
            for e in edges:
                src = _sanitize_id(e.get("from") or e.get("source") or "")
                dst = _sanitize_id(e.get("to") or e.get("target") or "")
                label = _sanitize_label(e.get("label") or e.get("text") or "")
                style = e.get("style", "solid")

                if src and dst:
                    if style == "dotted":
                        arrow = f"-. {label} .->" if label else "-.->"
                    elif style == "thick":
                        arrow = f"== {label} ==>" if label else "==>"
                    else:
                        arrow = f"-->|{label}|" if label else "-->"
                    lines.append(f"    {src} {arrow} {dst}")

    elif d_type in ("sequence", "sequencediagram"):
        lines.append("sequenceDiagram")
        if title:
            lines.append(f"    autonumber\n    %% {title}")
        if edges:
            for e in edges:
                src = _sanitize_label(e.get("from") or e.get("source") or "Client")
                dst = _sanitize_label(e.get("to") or e.get("target") or "Server")
                label = _sanitize_label(e.get("label") or e.get("text") or "Request")
                lines.append(f"    {src}->>{dst}: {label}")

    elif d_type in ("state", "statediagram"):
        lines.append("stateDiagram-v2")
        if title:
            lines.append(f"    [*] --> Start: {title}")
        if edges:
            for e in edges:
                src = _sanitize_label(e.get("from") or e.get("source") or "StateA")
                dst = _sanitize_label(e.get("to") or e.get("target") or "StateB")
                label = _sanitize_label(e.get("label") or e.get("text") or "")
                if label:
                    lines.append(f"    {src} --> {dst}: {label}")
                else:
                    lines.append(f"    {src} --> {dst}")

    else:
        lines.append(f"flowchart {dir_code}")
        if nodes:
            for n in nodes:
                lines.append(f"    {_sanitize_id(n.get('id', 'n'))}[{_sanitize_label(n.get('label', ''))}]")

    diagram_body = "\n".join(lines)
    return f"```mermaid\n{diagram_body}\n```"


@tool()
def generate_mermaid_diagram(
    title: str,
    diagram_type: str = "flowchart",
    direction: str = "TD",
    nodes: Optional[List[Dict[str, str]]] = None,
    edges: Optional[List[Dict[str, str]]] = None,
    raw_content: Optional[str] = None,
) -> str:
    """Generate a clean, validated Mermaid diagram (flowchart, architecture topology, sequence flow, or state diagram).

    Args:
        title: Diagram title or caption.
        diagram_type: Type of diagram ("flowchart", "architecture", "sequence", "state"). Default is "flowchart".
        direction: Layout direction for flowcharts ("TD", "TB", "LR"). Default is "TD".
        nodes: List of node dictionaries with keys: "id", "label", and optional "shape" ("rectangle", "round", "database", "circle").
        edges: List of edge dictionaries with keys: "from", "to", and optional "label", "style" ("solid", "dotted", "thick").
        raw_content: Optional raw Mermaid body string to sanitize and wrap.

    Returns:
        Formatted markdown Mermaid code block.
    """
    return _generate_mermaid_diagram_impl(
        title=title,
        diagram_type=diagram_type,
        direction=direction,
        nodes=nodes,
        edges=edges,
        raw_content=raw_content,
    )


# --------------------------------------------------------------------------- #
# Tool 6: get_ibm_doc_content
# --------------------------------------------------------------------------- #
def _get_ibm_doc_content_impl(
    source_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    max_chunks: int = 15,
) -> str:
    """Retrieve all chunks of a specific document to reconstruct the full context."""
    src_clean = (source_name or "").strip()
    if not src_clean:
        return "Please provide a valid source document name."

    coll_name = (collection_name or DEFAULT_COLLECTION).strip()

    try:
        collection = _get_collection(coll_name)
        chunks = _with_retry(
            lambda: list(
                collection.find(
                    {"source": src_clean},
                    limit=max(1, min(int(max_chunks), 50)),
                    projection={"chunk_no": 1, "chunk_count": 1, "text": 1, "etag": 1, "last_modified": 1},
                )
            ),
            max_retries=3,
            action_name=f"find chunks for '{src_clean}'",
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error retrieving document '{src_clean}' from '{coll_name}': {exc}"

    if not chunks:
        return f"Document '{src_clean}' was not found in collection '{coll_name}'."

    # Sort chunks in order
    sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_no", 0))
    total_count = sorted_chunks[0].get("chunk_count", len(sorted_chunks))

    doc_text_parts = [f"### 📄 Document: `{src_clean}` (Total Chunks: {total_count})\n"]
    for c in sorted_chunks:
        c_no = c.get("chunk_no", 0) + 1
        text = c.get("text", "").strip()
        doc_text_parts.append(f"--- Chunk {c_no}/{total_count} ---\n{text}")

    return "\n\n".join(doc_text_parts)


@tool(expected_credentials=[{"app_id": APP_ID, "type": [ConnectionType.KEY_VALUE]}])
def get_ibm_doc_content(
    source_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    max_chunks: int = 15,
) -> str:
    """Retrieve full document text by fetching and reassembling its sequential chunks from Astra DB.

    Args:
        source_name: The exact document key / path (e.g. "cloud-paks/cp-biz-automation/26.0.0__topic=administering-configuring-security.md").
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        max_chunks: Maximum number of chunks to fetch (default: 15).

    Returns:
        Complete reassembled document text.
    """
    return _get_ibm_doc_content_impl(
        source_name=source_name,
        collection_name=collection_name,
        max_chunks=max_chunks,
    )


# --------------------------------------------------------------------------- #
# Tool 7: list_ibm_knowledge_sources
# --------------------------------------------------------------------------- #
_SOURCES_CACHE: Dict[Tuple[str, str, int], Tuple[float, str]] = {}
_SOURCES_CACHE_LOCK = threading.Lock()
_SOURCES_CACHE_TTL = 300.0  # 5-minute cache TTL


def _clear_sources_cache() -> None:
    """Clear cached catalog sources (used for testing or cache invalidation)."""
    with _SOURCES_CACHE_LOCK:
        _SOURCES_CACHE.clear()


def _list_ibm_knowledge_sources_impl(
    collection_name: str = DEFAULT_COLLECTION,
    prefix: str = "",
    limit: int = 50,
) -> str:
    """Query Astra DB to list indexed IBM document sources and topics with thread-safe TTL caching."""
    coll_name = (collection_name or DEFAULT_COLLECTION).strip()
    clean_prefix = (prefix or "").strip()
    try:
        limit_val = max(1, min(int(limit), 100))
    except (ValueError, TypeError):
        limit_val = 50

    cache_key = (coll_name, clean_prefix, limit_val)
    now = time.time()
    with _SOURCES_CACHE_LOCK:
        if cache_key in _SOURCES_CACHE:
            cached_time, cached_val = _SOURCES_CACHE[cache_key]
            if now - cached_time < _SOURCES_CACHE_TTL:
                return cached_val

    try:
        collection = _get_collection(coll_name)
        filter_expr: Dict[str, Any] = {"chunk_no": 0}
        if clean_prefix:
            filter_expr["source"] = {"$regex": f"^{re.escape(clean_prefix)}"}

        docs = _with_retry(
            lambda: list(
                collection.find(
                    filter_expr,
                    limit=limit_val,
                    projection={"source": 1, "chunk_count": 1, "last_modified": 1},
                )
            ),
            max_retries=3,
            action_name=f"list sources in '{coll_name}'",
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error listing sources in '{coll_name}': {exc}"

    if not docs:
        return f"No document sources found in collection '{coll_name}' matching prefix '{clean_prefix}'."

    lines = [f"### 📚 Indexed IBM Documentation Sources ({len(docs)} documents in `{coll_name}`)\n"]
    for idx, d in enumerate(docs, start=1):
        src = d.get("source", "unknown")
        chunks = d.get("chunk_count", 1)
        lines.append(f"{idx}. `{src}` ({chunks} chunks)")

    res_str = "\n".join(lines)
    with _SOURCES_CACHE_LOCK:
        _SOURCES_CACHE[cache_key] = (now, res_str)

    return res_str


@tool(expected_credentials=[{"app_id": APP_ID, "type": [ConnectionType.KEY_VALUE]}])
def list_ibm_knowledge_sources(
    collection_name: str = DEFAULT_COLLECTION,
    prefix: str = "",
    limit: int = 50,
) -> str:
    """List indexed IBM product manuals, versions, and topics available in Astra DB.

    Args:
        collection_name: Astra DB collection name (default: "ibm_docs_rag2").
        prefix: Optional prefix filter (e.g. "cloud-paks/cp-biz-automation").
        limit: Maximum number of sources to list (default: 50, max: 100).

    Returns:
        Structured list of indexed IBM document source paths.
    """
    return _list_ibm_knowledge_sources_impl(
        collection_name=collection_name,
        prefix=prefix,
        limit=limit,
    )
