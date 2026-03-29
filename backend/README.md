# Research Agent Backend

This is the backend for the research agent.

## Current milestone

- FastAPI app with upload, delete, chat, health, style-profile, and retrieval-preview endpoints
- LangGraph runtime with retrieval-backed chat
- Hybrid retrieval (dense Pinecone + sparse lexical) with reranking
- Local hashing embeddings by default (Gemini optional)
- Groq generation for chat and style memory
- React frontend served directly by the backend

## Provider split

- Pinecone: vector database
- Local hash / Gemini: embeddings
- Groq: generation

## Important note on HNSW

Pinecone's current public create-index docs expose managed index options like serverless configuration, dimension, and metric. They do not expose a user-facing HNSW setting in the serverless flow. This backend therefore uses Pinecone's managed ANN retrieval path rather than pretending there is a configurable HNSW toggle.

## Environment

Copy `backend/.env.example` to `backend/.env` and set:

- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `GROQ_API_KEY`
- `GENERATION_PROVIDER` (`auto` | `groq` | `gemini`; default `auto`)
- `GENERATION_FALLBACK_ORDER` (default `gemini,groq`)
- `GENERATION_PROVIDER_COOLDOWN_SECONDS` (default `600`)
- `EMBEDDING_PROVIDER` (`local` | `auto` | `gemini`; default `local`)

`local`: fully offline hashing embeddings (no Gemini quota).
`auto`: try Gemini first, then fallback to local on quota/rate errors.

LangSmith tracing (optional):
- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=research-agent`
- `LANGSMITH_ENDPOINT=https://api.smith.langchain.com` (optional)
- `LANGCHAIN_TRACING_V2=true` (optional compatibility flag)

## Run

After dependencies are installed:

```powershell
uvicorn research_agent.api:app --reload --app-dir src
```

From the workspace root, this direct launch path also works:

```powershell
$env:PYTHONPATH='backend\src'
.\.venv\Scripts\python.exe -m uvicorn research_agent.api:app --port 8010
```
