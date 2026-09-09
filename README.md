# RAG Semantic Search

A self-hosted retrieval-augmented generation (RAG) stack for querying your own knowledge base. Content is scraped from the web via Firecrawl, cleaned, chunked, embedded with Ollama, stored in Qdrant, reranked with a cross-encoder, and summarized by an LLM with per-source attribution. Collections can hold any domain — legal, medical, financial, travel, or anything else you load.

## Stack

| Service    | Port  | Purpose                                          |
|------------|-------|--------------------------------------------------|
| qdrant     | 6353  | Vector DB (REST), 6354 gRPC                      |
| backend    | 8765  | FastAPI — scrape, chunk, embed, ingest, query    |
| reranker   | —     | bge-reranker-v2-m3 cross-encoder (internal)      |
| admin-ui   | 8770  | Collections, scraping, document management       |
| query-ui   | 8771  | Semantic search + curated LLM answers            |

External services: **Ollama** (embeddings + LLM, via `app-network`) and **Firecrawl** (self-hosted scraping API).

## Pipeline

1. **Scrape** — Admin UI posts a URL to Firecrawl; boilerplate is stripped via configurable include/exclude CSS selectors (`SCRAPE_INCLUDE_TAGS` / `SCRAPE_EXCLUDE_TAGS`), plus a markdown cleaner that removes images, ads, link-only lines, and duplicated blocks.
2. **Chunk** — cleaned markdown is split on paragraph boundaries (~1500 chars, 200 overlap); section headings are prepended to each chunk so every chunk is self-describing. All chunks of a page share a `doc_id`.
3. **Embed & store** — chunks are embedded (`bge-m3`, 1024d) and upserted to Qdrant with metadata (`source`, `title`, `line`, `topic`, …).
4. **Query** — vector search → optional cross-encoder rerank over a larger candidate pool → optional LLM curated answer with natural per-source attribution.

## Usage

```bash
docker compose up -d --build
```

- **Admin UI**: http://localhost:8770 — create collections (a **Private** toggle prefixes the name with `private-`, hiding it from the Query UI), ingest text, scrape URLs (with a "Preview Tags" selector picker), manage documents (edit re-chunks, refetch re-scrapes the source URL, delete removes all chunks of a doc), and tune **Query Settings**: top-k, candidate-k rerank pool, min similarity score, metadata filters, LLM answer, rerank, hnsw_ef slider, and exact search. Settings persist in `./data/backend/settings.json`.
- **Query UI**: http://localhost:8771 — pick a collection and ask questions in plain English. Curated answers include a numbered **Sources** list mapping `[n]` citations to their links. **Pro Mode** reveals the scored chunks behind each answer and a "Documents in this knowledge base" list. Mobile viewports get a desktop-required splash.
- **API docs**: http://localhost:8765/docs

### Screenshots

![Query UI](docs/query-ui.png)

![Admin UI](docs/admin-ui.png)

## Configuration

Copy `.env.example` to `.env` and adjust:

- `OLLAMA_URL`, `EMBED_MODEL`, `EMBED_DIM` — embedding model config
- `LLM_MODEL` — Ollama model for curated answers (default `llama3.2:latest`)
- `RERANKER_URL`, `RERANK_MODEL` — cross-encoder reranker; empty `RERANKER_URL` falls back to LLM scoring
- `FIRECRAWL_URL`, `FIRECRAWL_API_KEY` — self-hosted Firecrawl endpoint
- `SCRAPE_INCLUDE_TAGS`, `SCRAPE_EXCLUDE_TAGS` — default CSS selectors applied to every scrape (per-request overrides in the Admin UI)
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY` — Qdrant connection
- `*_PORT` — host port mappings

`.env` is gitignored; commit `.env.example` only.

> `EMBED_DIM` must match the model's output size (nomic-embed-text=768, bge-m3=1024). Existing collections keep the dimension they were created with.

## Persistence

Qdrant data is bind-mounted to `./data/qdrant`; reranker model cache to `./data/reranker`; query settings to `./data/backend`. All survive container removal.

## Networking

Ollama and Firecrawl are reached via the external `app-network` docker network. If your Ollama container isn't on it, either connect it (`docker network connect app-network ollama`) or set `OLLAMA_URL=http://host.docker.internal:11434` and add `extra_hosts: ["host.docker.internal:host-gateway"]` to the backend service.

## Security & Threat Model

This demo is intentionally run without the controls that would be mandatory in production. That is a teaching choice, not an oversight.

The FastAPI backend exposes unauthenticated `POST/PUT/DELETE` routes for creating collections, ingesting text, scraping URLs, and editing points. CORS is set to `*` so the static Query UI can call the API directly from the browser. That means any client that can reach the backend port can write to the knowledge base, scrape arbitrary URLs, and consume Ollama / embedding / reranker resources.

**Why that is acceptable here:**

- The demo is self-hosted, uses only publicly available input data, and is not exposed to real users or real PII.
- This discussion highlights a real-world challenge experienced by companies: a RAG knowledge base with unauthenticated write access can be poisoned by anyone who can reach the API.
- The author's production system is separate and hardened.

**What a production deployment would require:**

Authentication and role separation, scoped scraping with a domain allowlist, rate limiting, egress filtering, audit logging, content review before ingestion, and placement of the admin/write paths behind an identity-aware proxy such as Cloudflare Access.
