import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

MODEL_ID = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

app = FastAPI(title="Reranker")
model = CrossEncoder(MODEL_ID, max_length=512)


class RerankRequest(BaseModel):
    query: str
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = [[req.query, t] for t in req.texts]
    scores = model.predict(pairs)
    # TEI-compatible response shape: [{"index": i, "score": s}]
    return [
        {"index": i, "score": float(s)}
        for i, s in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ]
