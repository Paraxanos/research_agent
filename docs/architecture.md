# Research Agent Architecture

This workspace is now the clean build target for the full research agent.

Detailed walkthrough:

- `docs/architecture_walkthrough.md`

## Why We Pivoted

The original single-file React artifact spec is good for a class demo, but it is not the best foundation for a serious research assistant. We are keeping the UI shell as a fast preview, but the production path is:

- React frontend
- FastAPI backend
- LangGraph orchestration
- LangChain model integration
- Pinecone vector database
- Gemini embeddings
- Grok generation

This gives us better separation of concerns, safer key handling, stronger retrieval, and room for evaluation and observability.

## Planned System Shape

### Frontend

- Existing `research_agent.jsx` shell stays as the visual starting point.
- It now calls backend endpoints for upload, retrieval-backed chat, and style profile state.
- Reviewer and Comparator mode controls will stay in the UI.

### Backend

- `backend/src/research_agent/api.py`
  - FastAPI application and HTTP routes
- `backend/src/research_agent/runtime.py`
  - Runtime entrypoint that invokes the compiled LangGraph
- `backend/src/research_agent/graph/`
  - State definition and graph builder
- `backend/src/research_agent/config.py`
  - Typed app settings
- `backend/src/research_agent/schemas.py`
  - Request and response models

### Retrieval and Ingestion

Implemented now:

- PDF upload endpoint
- PDF parsing with `pypdf`
- Chunking with LangChain text splitters
- Pinecone indexing with Gemini embeddings
- Mode-specific retrieval and prompt assembly
- Persistent style profile storage for writer mode

## Retrieval Note

We are using Pinecone's managed approximate nearest-neighbor retrieval path. I did not find an official user-facing HNSW configuration option in Pinecone's current serverless index docs, so the implementation does not fake an HNSW setting that the provider does not expose.

## Build Milestones

1. Backend foundation: config, schemas, FastAPI, LangGraph skeleton
2. PDF ingestion pipeline and Pinecone vector storage
3. Global Brain end-to-end chat
4. Local Brain grounded retrieval with citations
5. Reviewer and Comparator workflows
6. Writer mode with style memory
7. Session memory, evaluation, and polish
