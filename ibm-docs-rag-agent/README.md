# IBM Product Documentation RAG & Visual Architecture Agent (wxo + MCP Server)

An enterprise-grade Retrieval-Augmented Generation (RAG) agent and **Model Context Protocol (MCP)** server. Grounded in IBM product documentation vectorized and stored in **DataStax Astra DB** (e.g. `ibm_docs_rag2`), the agent answers technical, architecture, troubleshooting, and configuration questions with a strict **confidence threshold (similarity score ≥ 0.75)**, plain-English synthesis, structured formatting, and **visual Mermaid diagrams**.

---

## 🚀 Key Features

- **Strict Confidence Grounding (≥ 0.75)**: Eliminates hallucinations by discarding chunks below 0.75 similarity. If no content meets the threshold, the agent explicitly informs the user, reports the highest observed score, and suggests query refinements.
- **Multi-Query Expansion & Re-Ranking**: Generates technical domain synonyms (e.g., abbreviations like `CP4BA`, `BAW`, `ODM`, `SSO`, `LDAP`), queries Astra DB in parallel, deduplicates chunks, and re-ranks results.
- **Product & Version Scoping**: Isolates answers to specific product releases (e.g. `24.0.1` vs `26.0.0`).
- **Cross-Version Documentation Comparison**: Retrieves comparative evidence across two releases to summarize upgrades, deprecations, or security changes.
- **Visual Mermaid Diagram Generation**: Produces clean, validated Mermaid code blocks (`flowchart`, `sequenceDiagram`, `stateDiagram-v2`, `architecture`) to illustrate component interactions and procedures.
- **Full Document Deep-Dive**: Fetches and reassembles sequential chunks to provide full context when needed.
- **Dynamic Collection Support**: Configurable collection name (defaults to `ibm_docs_rag2`).

---

## 🛠️ Tools Inventory

| Tool Name | Purpose | Key Parameters |
| :--- | :--- | :--- |
| `search_ibm_knowledge` | Primary semantic vector search with threshold filtering. | `query`, `collection_name="ibm_docs_rag2"`, `top_k=5`, `min_confidence=0.75` |
| `search_with_query_expansion` | Multi-query expansion across technical synonyms & parallel re-ranking. | `query`, `collection_name="ibm_docs_rag2"`, `top_k=5`, `min_confidence=0.75` |
| `search_by_product_version` | Scopes semantic search by product family and release version. | `query`, `product_or_topic`, `version`, `min_confidence=0.75` |
| `compare_product_versions` | Side-by-side comparative retrieval across two product versions. | `topic`, `version_a`, `version_b`, `min_confidence=0.75` |
| `generate_mermaid_diagram` | Generates syntax-validated Mermaid architecture/flow diagrams. | `title`, `diagram_type="flowchart"`, `direction="TD"`, `nodes`, `edges` |
| `get_ibm_doc_content` | Reassembles complete document text from sequential chunks. | `source_name`, `collection_name="ibm_docs_rag2"`, `max_chunks=15` |
| `list_ibm_knowledge_sources` | Catalogs indexed IBM product manuals and topics in Astra DB. | `collection_name="ibm_docs_rag2"`, `prefix=""`, `limit=50` |

---

## 📋 Response Structure

Every response follows a consistent, enterprise-ready format:

1. 📌 **Direct Answer / Executive Summary**: 2-3 clear sentences answering the question directly.
2. ⚙️ **Key Details / Step-by-Step Guidance**: Bullet points, configuration parameters, or comparative tables.
3. 📐 **Visual Architecture / Workflow Diagram** (when applicable): Validated Mermaid diagram block.
4. 📚 **Source References & Confidence Scores**: Exact document paths and similarity scores (e.g. `0.890`, `0.780`).
5. 💡 **Related Topics / Next Steps**: Suggested follow-up areas.

---

## ⚙️ watsonx Orchestrate Deployment & Setup

### 1. Connection Configuration
In the watsonx Orchestrate UI, navigate to **Connections** and create a **Key-Value** connection named `astra` with the following keys:
- `ASTRA_DB_API_ENDPOINT`: `https://<db-id>-<region>.apps.astra.datastax.com`
- `ASTRA_DB_APPLICATION_TOKEN`: `AstraCS:...`

*(Alternatively, for local testing, set these as environment variables.)*

### 2. Import Agent via watsonx Orchestrate ADK / IBM Bob
1. Open the project in IBM Bob or VS Code with the watsonx Orchestrate ADK extension.
2. Open `ibm-docs-rag-agent/agent.yaml`.
3. Click **Import / Deploy Agent** to watsonx Orchestrate.
4. Verify that the tools in `rag_tools.py` are bound and the `astra` connection is assigned.

---

## 🔌 Model Context Protocol (MCP) Server

The agent includes a production-ready FastMCP server ([`server.py`](ibm-docs-rag-agent/server.py)) that exposes all RAG capabilities, resources, and prompt templates to MCP clients (such as IBM Bob, Claude Desktop, Cursor, and watsonx).

### MCP Server Features
- **7 Tools**: Full access to semantic search, query expansion, version comparison, Mermaid diagrams, doc reconstruction, and source catalogs.
- **MCP Resources**:
  - `ibm-docs://catalog`: Real-time index catalog of all indexed IBM documentation sources and chunk counts.
  - `ibm-docs://doc/{source_path}`: Direct access to full reassembled documentation text.
- **MCP Prompts**:
  - `ask_ibm_doc`: Structured prompt for answering IBM product questions grounded in Astra DB with confidence citations.
  - `compare_ibm_versions`: Comparative prompt for analyzing technical differences between product releases.
  - `troubleshoot_ibm_issue`: Diagnostic prompt for symptom, error code, and remediation troubleshooting.

### Running the MCP Server
```bash
# Using uv:
uv run --directory ibm-docs-rag-agent --with "mcp<2" --with astrapy --with pydantic --with httpx python server.py

# Or in a dedicated virtualenv:
cd ibm-docs-rag-agent
pip install -r requirements.txt
python server.py
```

### Registering in `mcp.json`

#### Option A: Remote Streamable HTTP Server (OpenShift Deployed)
```json
{
  "mcpServers": {
    "ibm-docs-rag": {
      "type": "streamable-http",
      "url": "https://ibm-docs-rag-mcp-astra-ingest.apps.itz-5ju5e0.infra01-lb.lon04.techzone.ibm.com/mcp"
    }
  }
}
```

#### Option B: Local Stdio Process
```json
{
  "mcpServers": {
    "ibm-docs-rag-local": {
      "command": "python",
      "args": ["/absolute/path/to/ibm-docs-rag-agent/server.py"],
      "env": {
        "ASTRA_DB_API_ENDPOINT": "${env:ASTRA_DB_API_ENDPOINT}",
        "ASTRA_DB_APPLICATION_TOKEN": "${env:ASTRA_DB_APPLICATION_TOKEN}"
      }
    }
  }
}
```

---

## ☁️ OpenShift Deployment & Lifecycle

The MCP server is packaged with container definitions and OpenShift manifests:
- **Dockerfile**: Non-root Red Hat UBI9 Python 3.12 container running Uvicorn ASGI server.
- **ImageStream & BuildConfig**: Automated binary Docker builds.
- **Deployment & Service**: Pod replicas with health check probes on `/health` (port 8080).
- **Route**: HTTPS Edge-terminated Route redirecting insecure traffic.

### Build and Deploy Commands:
```bash
# Apply OpenShift manifests
oc apply -f ibm-docs-rag-agent/openshift/

# Start binary build from source
oc start-build ibm-docs-rag-mcp --from-dir=ibm-docs-rag-agent --follow -n astra-ingest

# Monitor rollout
oc rollout status deployment/ibm-docs-rag-mcp -n astra-ingest
```

---

## 🧪 Running the Unit Test Suite

Run the full test suite (29 unit tests covering RAG tools, FastMCP tool registration, resources, and prompts):

```bash
uv run --directory ibm-docs-rag-agent --with "mcp<2" --with astrapy --with pydantic --with httpx --with pytest --with pytest-asyncio pytest tests -v
```

---

## 📝 Example Queries

### 1. Security Configuration
> **User**: *"How do I configure Keycloak SSO authentication in CP4BA 26.0.0?"*
>
> **Agent Action**: Calls `search_ibm_knowledge` or `search_by_product_version`.
>
> **Output**: Direct summary, LDAP/Keycloak setup steps, Mermaid authentication sequence diagram, citations with similarity scores (e.g. `0.890`).

### 2. Version Comparison
> **User**: *"What are the deployment troubleshooting differences between CP4BA 24.0.1 and 26.0.0?"*
>
> **Agent Action**: Calls `compare_product_versions(topic="deployment troubleshooting", version_a="24.0.1", version_b="26.0.0")`.
>
> **Output**: Comparative table highlighting changes in operator pod logging, CR statuses, and MQ connectivity.

### 3. Query Below Confidence Threshold (< 0.75)
> **User**: *"How to configure custom third-party database plugin XYZ?"*
>
> **Agent Output**:
> ```markdown
> ⚠️ **Confidence Threshold Notice**: No documentation met the minimum confidence score of **0.75** for query: *'custom third-party database plugin XYZ'*.
> - **Highest Similarity Score Observed**: `0.612` (below required threshold of `0.75`).
> - **Target Collection**: `ibm_docs_rag2`
>
> **Recommended Refinements**:
> 1. Use standard IBM Db2 or PostgreSQL database keywords.
> 2. Search available topics using `list_ibm_knowledge_sources`.
> ```
