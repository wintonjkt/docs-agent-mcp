# IBM Documentation RAG & Ingestion MCP Suite

An enterprise-ready documentation knowledge system and Model Context Protocol (MCP) server suite for IBM product documentation, powered by DataStax Astra DB vector search and IBM Cloud Object Storage.

---

## 📂 Repository Structure

- [`ibm-docs-rag-agent/`](ibm-docs-rag-agent/): Enterprise IBM Documentation RAG Agent and **FastMCP Server** with semantic vector search ($\ge 0.75$ confidence threshold), domain query expansion, version comparison, and Mermaid diagram generation. Deployable over Stdio, SSE/HTTP, or Red Hat OpenShift.
- [`astra-ingest-app/`](astra-ingest-app/): High-throughput ingestion pipeline and FastAPI web application for parsing, chunking, and indexing Markdown documentation from IBM Cloud Object Storage (COS) into Astra DB vector collections.

---

## 🚀 Quick Start

### 1. IBM Documentation RAG MCP Server
```bash
# Run locally over Stdio
cd ibm-docs-rag-agent
uv run --with "mcp<2" --with astrapy --with pydantic --with httpx python server.py

# Or deploy to Red Hat OpenShift
oc apply -f ibm-docs-rag-agent/openshift/
oc start-build ibm-docs-rag-mcp --from-dir=ibm-docs-rag-agent --follow -n astra-ingest
```

### 2. Astra Ingest Pipeline Web App
```bash
# Run locally
cd astra-ingest-app
uv run --with fastapi --with uvicorn --with astrapy --with ibm-cos-sdk python -m app.main

# Or deploy to Red Hat OpenShift
oc apply -f astra-ingest-app/openshift/
oc start-build astra-ingest-app --from-dir=astra-ingest-app --follow -n astra-ingest
```

---

## 🧪 Testing

```bash
# Run all RAG agent & MCP server unit tests (29 tests)
uv run --directory ibm-docs-rag-agent --with "mcp<2" --with astrapy --with pydantic --with httpx --with pytest --with pytest-asyncio pytest tests -v

# Run all Astra Ingest pipeline tests (43 tests)
uv run --directory astra-ingest-app --with pytest --with pytest-asyncio pytest tests -v
```
