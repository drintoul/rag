import asyncio
import json
import os
import re
import uuid
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")
RERANKER_URL = os.getenv("RERANKER_URL", "").rstrip("/")
FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://firecrawl-api:3002").rstrip("/")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY") or None

# Boilerplate selectors excluded from every scrape (overlays, nav, consent, etc.)
DEFAULT_EXCLUDE_TAGS = [
    t.strip()
    for t in os.getenv(
        "SCRAPE_EXCLUDE_TAGS",
        "nav,header,footer,aside,script,style,noscript,iframe,form,"
        "[role=banner],[role=navigation],[role=dialog],[aria-modal=true],"
        ".modal,.popup,.overlay,.cookie-banner,.cookie-consent,.consent-banner,"
        ".country-banner,.country-selector,.cmp-country-banner,#country-banner-id,"
        ".nav,.navbar,.menu,.sidebar,.footer,.header,.advertisement,.ad,.social-share"
    ).split(",")
    if t.strip()
]

# Common main-content containers tried on every scrape (fallback: full page)
DEFAULT_INCLUDE_TAGS = [
    t.strip()
    for t in os.getenv(
        "SCRAPE_INCLUDE_TAGS",
        "main,article,[role=main],.main-wrapper,.main-content,#main-content,"
        "#main,.content,#content,.post-content,.article-body,.entry-content,"
        ".cmp-content-block,.page-content,.site-content"
    ).split(",")
    if t.strip()
]

app = FastAPI(title="Cruise RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=30)


# ---------- Models ----------

class CreateCollection(BaseModel):
    name: str = Field(..., min_length=1)
    distance: str = "Cosine"


class Document(BaseModel):
    text: str
    metadata: dict[str, Any] = {}


class IngestRequest(BaseModel):
    documents: list[Document]


class UpdatePointRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = {}


class PreviewRequest(BaseModel):
    url: str


class ScrapeRequest(BaseModel):
    url: str
    metadata: dict[str, Any] = {}
    max_chars: int = 60000
    chunk_size: int = 1500
    chunk_overlap: int = 200
    only_main_content: bool = True
    exclude_tags: list[str] = []
    include_tags: list[str] = []


class QueryRequest(BaseModel):
    collection: Optional[str] = None
    query: str
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    filter_key: Optional[str] = None
    filter_value: Optional[str] = None
    generate: Optional[bool] = None
    llm_model: Optional[str] = None
    rerank: Optional[bool] = None
    rerank_model: Optional[str] = None
    candidate_k: Optional[int] = None
    exact: Optional[bool] = None
    hnsw_ef: Optional[int] = None


SETTINGS_FILE = os.getenv("SETTINGS_FILE", "/data/settings.json")
DEFAULT_SETTINGS = {
    "collection": "canada-constitution",
    "top_k": 5,
    "score_threshold": None,
    "filter_key": None,
    "filter_value": None,
    "generate": True,
    "rerank": True,
    "candidate_k": 20,
    "hnsw_ef": None,
    "exact": False,
}


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(s: dict):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f)


# ---------- Helpers ----------

async def embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Ollama embed failed: {resp.text}")
        return resp.json()["embeddings"]


async def generate_answer(query: str, hits: list[dict], model: Optional[str] = None) -> str:
    blocks = []
    for i, h in enumerate(hits):
        p = h.get("payload") or {}
        text = p.get("text", "")
        if not text:
            continue
        src = p.get("title") or p.get("source") or ""
        line = p.get("line", "")
        label = " — ".join(x for x in [line, src] if x)
        blocks.append(f"[{i+1}] {f'({label})' if label else ''}\n{text}")
    context_block = "\n\n".join(blocks)
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "Each context block is numbered and may have a source label in parentheses. "
        "When blocks come from different sources or entities (e.g. different cruise lines), "
        "clearly attribute each part of your answer to its source — never blend policies or facts "
        "across sources into a single undifferentiated answer. "
        "Cite the specific source for each fact using its source label from the context. "
        "Begin with the source name, e.g. 'According to the British North America Act, 1867, …' or 'As stated in [source label], …'. "
        "Do NOT say 'according to the context', 'the context states', or other generic phrases. "
        "do NOT append bracketed citations like '(Source: [1] …)' — the UI lists sources separately. "
        "Include ALL relevant details from the context (e.g. every location, date, or option mentioned). "
        "If the context does not contain the answer, say so. Be concise but complete.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model or LLM_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0}},
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Ollama generate failed: {resp.text}")
        return resp.json()["response"].strip()


async def rerank_score(query: str, text: str, model: str, client: httpx.AsyncClient) -> float:
    prompt = (
        "Rate how relevant the document is to the query on a scale of 0 to 10.\n"
        "Rules:\n"
        "- First identify the exact entity/term the query asks about.\n"
        "- Score 9-10 ONLY if the document explicitly mentions that exact term and answers the query.\n"
        "- Score 0-3 if the document only contains similar-sounding names without the exact term.\n"
        "- Score 4-8 if the document is on-topic but only partially answers.\n"
        "Reply with ONLY a single number.\n\n"
        f"Query: {query}\nDocument: {text[:2000]}\nScore:"
    )
    try:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 8}},
        )
        if resp.status_code != 200:
            return -1.0
        m = re.search(r"\d+(\.\d+)?", resp.json().get("response", ""))
        return float(m.group()) if m else -1.0
    except Exception:
        return -1.0


async def rerank_tei(query: str, hits: list[dict]) -> list[dict]:
    texts = [(h["payload"] or {}).get("text", "") for h in hits]
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "texts": texts},
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Reranker failed: {resp.text}")
        scored = resp.json()  # [{"index": i, "score": s}]
    for item in scored:
        hits[item["index"]]["rerank_score"] = item["score"]
    hits.sort(key=lambda h: h.get("rerank_score", float("-inf")), reverse=True)
    return hits


async def rerank(query: str, hits: list[dict], model: Optional[str] = None) -> tuple[list[dict], str]:
    if RERANKER_URL:
        try:
            return await rerank_tei(query, hits), "cross-encoder"
        except HTTPException:
            raise
        except Exception:
            pass  # fall back to LLM scoring
    model = model or LLM_MODEL
    async with httpx.AsyncClient(timeout=120) as client:
        scores = await asyncio.gather(*[
            rerank_score(query, (h["payload"] or {}).get("text", ""), model, client)
            for h in hits
        ])
    for h, s in zip(hits, scores):
        h["rerank_score"] = s
    # docs that failed scoring keep their vector rank, below scored ones
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits, "llm"


def clean_markdown(md: str) -> str:
    """Strip scrape noise: images, template stubs, link-only lines, excess blanks."""
    # multi-line link blocks: [\ \n ### Title \ \n Label](url)
    md = re.sub(r"\[\\\s*\n(?:[^\n]*\n)*?[^\n]*?\]\([^)]*\)", "", md)
    out = []
    for line in md.split("\n"):
        s = line.strip()
        if not s:
            out.append("")
            continue
        # template stubs like {{ notfound }}
        if re.fullmatch(r"\{\{.*?\}\}", s):
            continue
        # boilerplate lines
        if s in {"Advertisement", "×", "More from our network"} or s.startswith("Credit:"):
            continue
        # image-credit tails and related-links lines
        if re.match(r"^/\s", s) or "via Getty Images" in s or s.startswith("Related:"):
            continue
        # stray link-block remnants
        if s in {"[\\", "\\"} or s.endswith("\\"):
            continue
        # orphaned link tails: "Label](url)" with no opening bracket
        if re.match(r"^[^\[\n]*\]\([^)]*\)\s*$", s):
            continue
        # strip inline images anywhere: ![...](...) and [![...](...)](...)
        s = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", s)
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s).strip()
        if not s:
            continue
        # lines that are only a link: [text](url)
        if re.fullmatch(r"\[[^\]]*\]\([^)]*\)", s):
            continue
        out.append(s)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    # drop repeated paragraphs (e.g. title blocks rendered twice in the page)
    seen: set[str] = set()
    final = []
    for para in cleaned.split("\n\n"):
        key = re.sub(r"[\s#*=\-_]+", " ", para.split("\n")[0]).strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        final.append(para)
    return "\n\n".join(final)


HEADING_RE = re.compile(r"^(#{1,6}\s+\S|.+\n[=-]{3,}\s*$)")


def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    """Split on paragraph boundaries; prepend the current section heading to each
    chunk so every chunk is self-describing. Overlap carries the tail of the
    previous chunk into the next (cut at a word boundary)."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    heading = ""

    def emit(body: str):
        body = body.strip()
        if body:
            chunks.append(f"{heading}\n\n{body}" if heading else body)

    def tail(s: str) -> str:
        if overlap <= 0 or len(s) <= overlap:
            return ""
        t = s[-overlap:]
        sp = t.find(" ")
        return t[sp + 1:] if sp >= 0 else t

    for p in paras:
        if HEADING_RE.match(p):
            emit(cur)
            cur = ""
            heading = p.split("\n")[0].lstrip("#").strip()
            continue
        # oversized paragraph: hard-split into slices
        while len(p) > size:
            emit(cur)
            cur = ""
            emit(p[:size])
            p = p[size - overlap:]
        if cur and len(cur) + len(p) + 2 > size:
            emit(cur)
            cur = tail(cur)
        cur = f"{cur}\n\n{p}" if cur else p
    emit(cur)
    return chunks or ([text[:size]] if text.strip() else [])


async def fetch_markdown(url: str, include_tags: list[str], exclude_tags: list[str],
                         only_main: bool = True) -> tuple[str, dict]:
    """Scrape a URL via Firecrawl; returns (cleaned markdown, page metadata)."""
    headers = {"Content-Type": "application/json"}
    if FIRECRAWL_API_KEY:
        headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"

    async def do_scrape(use_includes: bool) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{FIRECRAWL_URL}/v1/scrape",
                json={
                    "url": url,
                    "formats": ["markdown"],
                    "onlyMainContent": only_main,
                    "excludeTags": list(dict.fromkeys(DEFAULT_EXCLUDE_TAGS + exclude_tags)),
                    **({"includeTags": include_tags} if use_includes else {}),
                },
                headers=headers,
            )
        if resp.status_code != 200:
            raise HTTPException(502, f"Firecrawl scrape failed: {resp.text}")
        data = resp.json()
        if not data.get("success"):
            raise HTTPException(502, f"Firecrawl error: {data}")
        return data["data"]

    page = await do_scrape(use_includes=bool(include_tags))
    md = page.get("markdown", "")
    if not md.strip() and include_tags:
        page = await do_scrape(use_includes=False)
        md = page.get("markdown", "")
    if not md.strip():
        raise HTTPException(422, "Firecrawl returned empty content")
    return clean_markdown(md), page.get("metadata", {})


def collection_or_404(name: str):
    if not qdrant.collection_exists(name):
        raise HTTPException(404, f"Collection '{name}' not found")


# ---------- Routes ----------

@app.get("/api/health")
async def health():
    try:
        collections = qdrant.get_collections().collections
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
        collections = []
    ollama_ok = False
    firecrawl_ok = False
    reranker_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            ollama_ok = (await client.get(f"{OLLAMA_URL}/api/tags")).status_code == 200
            firecrawl_ok = (await client.get(f"{FIRECRAWL_URL}/")).status_code == 200
            if RERANKER_URL:
                reranker_ok = (await client.get(f"{RERANKER_URL}/health")).status_code == 200
    except Exception:
        pass
    return {
        "qdrant": qdrant_ok,
        "ollama": ollama_ok,
        "firecrawl": firecrawl_ok,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "reranker": reranker_ok,
        "collections": len(collections),
    }


@app.get("/api/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            data = (await client.get(f"{OLLAMA_URL}/api/tags")).json()
            return {"models": [m["name"] for m in data.get("models", [])]}
    except Exception as e:
        raise HTTPException(502, f"Ollama unreachable: {e}")


@app.get("/api/collections")
async def list_collections():
    result = []
    for c in qdrant.get_collections().collections:
        info = qdrant.get_collection(c.name)
        result.append({
            "name": c.name,
            "points": info.points_count,
            "vectors": info.indexed_vectors_count,
            "status": info.status.value if info.status else None,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance.value,
        })
    return {"collections": result}


@app.post("/api/collections")
async def create_collection(req: CreateCollection):
    if qdrant.collection_exists(req.name):
        raise HTTPException(409, f"Collection '{req.name}' already exists")
    dist = {"cosine": Distance.COSINE, "euclid": Distance.EUCLID, "dot": Distance.DOT}.get(
        req.distance.lower(), Distance.COSINE
    )
    qdrant.create_collection(
        collection_name=req.name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=dist),
    )
    return {"created": req.name, "vector_size": EMBED_DIM, "distance": dist.value}


@app.delete("/api/collections/{name}")
async def delete_collection(name: str):
    collection_or_404(name)
    qdrant.delete_collection(name)
    return {"deleted": name}


@app.get("/api/collections/{name}/points")
async def list_points(name: str, limit: int = 50, offset: Optional[str] = None):
    collection_or_404(name)
    points, next_offset = qdrant.scroll(
        collection_name=name,
        limit=min(limit, 200),
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    return {
        "points": [{"id": str(p.id), "payload": p.payload} for p in points],
        "next_offset": str(next_offset) if next_offset else None,
    }


@app.post("/api/collections/{name}/ingest")
async def ingest(name: str, req: IngestRequest):
    collection_or_404(name)
    if not req.documents:
        raise HTTPException(400, "No documents provided")
    texts = [d.text for d in req.documents]
    vectors = await embed(texts)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=v,
            payload={"text": d.text, **d.metadata},
        )
        for d, v in zip(req.documents, vectors)
    ]
    qdrant.upsert(collection_name=name, points=points)
    return {"ingested": len(points), "collection": name}


@app.put("/api/collections/{name}/points/{point_id}")
async def update_point(name: str, point_id: str, req: UpdatePointRequest):
    collection_or_404(name)
    [vector] = await embed([req.text])
    qdrant.upsert(
        collection_name=name,
        points=[PointStruct(id=point_id, vector=vector,
                            payload={"text": req.text, **req.metadata})],
    )
    return {"updated": point_id}


@app.delete("/api/collections/{name}/points/{point_id}")
async def delete_point(name: str, point_id: str):
    collection_or_404(name)
    qdrant.delete(collection_name=name, points_selector=[point_id])
    return {"deleted": point_id}


@app.post("/api/scrape/preview")
async def scrape_preview(req: PreviewRequest):
    """Fetch a page's raw HTML and return candidate CSS selectors for include/exclude."""
    headers = {"Content-Type": "application/json"}
    if FIRECRAWL_API_KEY:
        headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{FIRECRAWL_URL}/v1/scrape",
            json={"url": req.url, "formats": ["rawHtml"]},
            headers=headers,
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"Firecrawl preview failed: {resp.text}")
    data = resp.json()
    if not data.get("success"):
        raise HTTPException(502, f"Firecrawl error: {data}")
    html = data["data"].get("rawHtml", "")
    if not html:
        raise HTTPException(422, "Firecrawl returned no HTML")

    from collections import Counter

    ids = Counter(re.findall(r'id="([A-Za-z][\w:-]*)"', html))
    classes = Counter(
        c
        for attr in re.findall(r'class="([^"]+)"', html)
        for c in attr.split()
        if re.fullmatch(r"[A-Za-z][\w-]*", c)
    )
    tags = Counter(re.findall(r"<([a-z][a-z0-9]*)\s", html))
    skip_tags = {"script", "style", "link", "meta", "path", "svg", "br", "img",
                 "input", "span", "b", "i", "u", "a", "li", "ul", "ol"}
    return {
        "ids": [{"selector": f"#{i}", "count": n} for i, n in ids.most_common(40)],
        "classes": [{"selector": f".{c}", "count": n} for c, n in classes.most_common(60)],
        "tags": [{"selector": t, "count": n} for t, n in tags.most_common(30) if t not in skip_tags],
    }


async def ingest_chunks(name: str, url: str, title: str, req: ScrapeRequest,
                        doc_id: Optional[str] = None) -> dict:
    """Scrape, clean, chunk, embed and upsert. Returns summary dict."""
    md, fc_meta = await fetch_markdown(
        url, req.include_tags or DEFAULT_INCLUDE_TAGS, req.exclude_tags,
        req.only_main_content,
    )
    text = md[: req.max_chars]
    chunks = chunk_text(text, req.chunk_size, req.chunk_overlap)
    doc_id = doc_id or str(uuid.uuid4())
    title = title or fc_meta.get("title", "")
    vectors = await embed(chunks)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=v,
            payload={
                "text": c,
                "source": url,
                "title": title,
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "include_tags": req.include_tags,
                "exclude_tags": req.exclude_tags,
                **({"doc_text": text} if i == 0 else {}),
                **req.metadata,
            },
        )
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    qdrant.upsert(collection_name=name, points=points)
    return {"ingested": len(points), "collection": name, "url": url,
            "title": title, "chars": len(text), "doc_id": doc_id}


@app.post("/api/collections/{name}/scrape")
async def scrape_and_ingest(name: str, req: ScrapeRequest):
    collection_or_404(name)
    return await ingest_chunks(name, req.url, "", req)


def doc_points(name: str, doc_id: str) -> list:
    pts, offset = [], None
    while True:
        res, offset = qdrant.scroll(
            collection_name=name,
            scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            limit=100, offset=offset, with_payload=True, with_vectors=False,
        )
        pts.extend(res)
        if offset is None:
            return pts


@app.get("/api/collections/{name}/docs/{doc_id}")
async def get_doc(name: str, doc_id: str):
    collection_or_404(name)
    pts = doc_points(name, doc_id)
    if not pts:
        raise HTTPException(404, "Document not found")
    pts.sort(key=lambda p: (p.payload or {}).get("chunk_index", 0))
    first = pts[0].payload or {}
    text = first.get("doc_text")
    if text is None:
        # legacy docs without doc_text: join chunks, stripping the prepended
        # heading and overlap tail that duplicate content
        parts = []
        prev_heading = None
        for p in pts:
            t = (p.payload or {}).get("text", "")
            lines = t.split("\n\n", 1)
            if prev_heading and len(lines) == 2 and lines[0].strip() == prev_heading:
                t = lines[1]
            prev_heading = lines[0].strip() if len(lines) == 2 else None
            parts.append(t)
        text = "\n\n".join(parts)
    meta = {
        k: v for k, v in first.items()
        if k not in ("text", "doc_text", "doc_id", "chunk_index", "chunk_count",
                     "include_tags", "exclude_tags")
    }
    return {"doc_id": doc_id, "text": text, "metadata": meta,
            "include_tags": first.get("include_tags") or [],
            "exclude_tags": first.get("exclude_tags") or [],
            "chunks": len(pts)}


class UpdateDocRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = {}


@app.put("/api/collections/{name}/docs/{doc_id}")
async def update_doc(name: str, doc_id: str, req: UpdateDocRequest):
    collection_or_404(name)
    pts = doc_points(name, doc_id)
    if not pts:
        raise HTTPException(404, "Document not found")
    first = pts[0].payload or {}
    chunks = chunk_text(req.text)
    vectors = await embed(chunks)
    qdrant.delete(
        collection_name=name,
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )
    points = [
        PointStruct(
            id=str(uuid.uuid4()), vector=v,
            payload={
                "text": c, "doc_id": doc_id, "chunk_index": i,
                "chunk_count": len(chunks),
                "source": first.get("source"), "title": first.get("title", ""),
                "include_tags": first.get("include_tags") or [],
                "exclude_tags": first.get("exclude_tags") or [],
                **({"doc_text": req.text} if i == 0 else {}),
                **req.metadata,
            },
        )
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    qdrant.upsert(collection_name=name, points=points)
    return {"updated": doc_id, "chunks": len(chunks)}


@app.delete("/api/collections/{name}/docs/{doc_id}")
async def delete_doc(name: str, doc_id: str):
    collection_or_404(name)
    qdrant.delete(
        collection_name=name,
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )
    return {"deleted": doc_id}


@app.post("/api/collections/{name}/points/{point_id}/refetch")
async def refetch_point(name: str, point_id: str):
    """Re-scrape a point's source URL with its stored tags and update it in place."""
    collection_or_404(name)
    found = qdrant.retrieve(collection_name=name, ids=[point_id], with_payload=True)
    if not found:
        raise HTTPException(404, "Point not found")
    payload = found[0].payload or {}
    url = payload.get("source")
    if not url:
        raise HTTPException(422, "Point has no source URL to refetch")
    meta = {
        k: v for k, v in payload.items()
        if k not in ("text", "source", "title", "include_tags", "exclude_tags",
                     "doc_id", "chunk_index", "chunk_count", "doc_text")
    }
    req = ScrapeRequest(
        url=url,
        metadata=meta,
        include_tags=payload.get("include_tags") or [],
        exclude_tags=payload.get("exclude_tags") or [],
    )
    doc_id = payload.get("doc_id")
    # delete all sibling chunks of this doc (or just this point if unchunked)
    if doc_id:
        qdrant.delete(
            collection_name=name,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )
    else:
        qdrant.delete(collection_name=name, points_selector=[point_id])
    result = await ingest_chunks(name, url, payload.get("title", ""), req, doc_id=doc_id)
    return {"refetched": doc_id or point_id, "url": url, **result}


@app.get("/api/collections/{name}/docs")
async def list_docs(name: str):
    """List distinct documents (grouped by doc_id) in a collection."""
    collection_or_404(name)
    docs: dict[str, dict] = {}
    offset = None
    while True:
        res, offset = qdrant.scroll(
            collection_name=name, limit=200, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in res:
            pl = p.payload or {}
            key = pl.get("doc_id") or str(p.id)
            if key not in docs:
                docs[key] = {
                    "doc_id": pl.get("doc_id"),
                    "title": pl.get("title") or pl.get("source") or "(untitled)",
                    "source": pl.get("source"),
                    "chunks": 0,
                }
            docs[key]["chunks"] += 1
        if offset is None:
            break
    return {"docs": sorted(docs.values(), key=lambda d: d["title"])}


@app.get("/api/settings")
async def get_settings():
    return load_settings()


@app.put("/api/settings")
async def put_settings(req: dict):
    s = load_settings()
    s.update({k: v for k, v in req.items() if k in DEFAULT_SETTINGS})
    save_settings(s)
    return s


@app.post("/api/query")
async def query(req: QueryRequest):
    s = load_settings()
    collection = req.collection or s["collection"]
    if not collection:
        raise HTTPException(400, "No collection configured — set one in Admin UI query settings")
    top_k = req.top_k if req.top_k is not None else s["top_k"]
    threshold = req.score_threshold if req.score_threshold is not None else s["score_threshold"]
    filter_key = req.filter_key or s["filter_key"]
    filter_value = req.filter_value if req.filter_value is not None else s["filter_value"]
    generate = req.generate if req.generate is not None else s["generate"]
    rerank_on = req.rerank if req.rerank is not None else s["rerank"]
    candidate_k = req.candidate_k if req.candidate_k is not None else s["candidate_k"]

    collection_or_404(collection)
    [vector] = await embed([req.query])
    qfilter = None
    if filter_key and filter_value is not None:
        qfilter = Filter(
            must=[FieldCondition(key=filter_key, match=MatchValue(value=filter_value))]
        )
    exact = req.exact if req.exact is not None else s.get("exact", False)
    hnsw_ef = req.hnsw_ef if req.hnsw_ef is not None else s.get("hnsw_ef")
    fetch_k = max(candidate_k, top_k) if rerank_on else top_k
    search_params = None
    if exact or hnsw_ef:
        search_params = SearchParams(exact=bool(exact), hnsw_ef=hnsw_ef)
    results = qdrant.query_points(
        collection_name=collection,
        query=vector,
        limit=fetch_k,
        query_filter=qfilter,
        score_threshold=threshold,
        search_params=search_params,
        with_payload=True,
    )
    hits = [
        {"id": str(p.id), "score": p.score,
         "payload": {k: v for k, v in (p.payload or {}).items() if k != "doc_text"}}
        for p in results.points
    ]
    reranker_used = None
    if rerank_on and hits:
        hits, reranker_used = await rerank(req.query, hits, req.rerank_model)
        hits = hits[:top_k]
    answer = None
    if generate and hits:
        if any((h["payload"] or {}).get("text") for h in hits):
            answer = await generate_answer(req.query, hits, req.llm_model)
    return {
        "collection": collection,
        "query": req.query,
        "answer": answer,
        "reranker": reranker_used,
        "llm_model": req.llm_model or LLM_MODEL if generate else None,
        "results": hits,
    }
