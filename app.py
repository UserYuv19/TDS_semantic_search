import time
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# App
# -------------------------
app = FastAPI(title="Semantic Search with Re-ranking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

# -------------------------
# Load documents
# -------------------------
with open("docs.json", "r", encoding="utf-8") as f:
    DOCS = json.load(f)

TOTAL_DOCS = len(DOCS)

# -------------------------
# Embeddings (cached)
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english")
doc_texts = [d["content"] for d in DOCS]
DOC_EMBEDDINGS = vectorizer.fit_transform(doc_texts)

# -------------------------
# Request model
# -------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

# -------------------------
# Re-ranking function
# -------------------------
def rerank(query: str, docs: list[dict]) -> list[tuple[dict, float]]:
    q_vec = vectorizer.transform([query])
    d_vecs = vectorizer.transform([d["content"] for d in docs])
    sims = cosine_similarity(q_vec, d_vecs)[0]

    reranked = []
    for doc, score in zip(docs, sims):
        bonus = sum(
            1 for w in query.lower().split()
            if w in doc["content"].lower()
        ) * 0.05

        final_score = float(score + bonus)
        reranked.append((doc, final_score))

    return reranked

# -------------------------
# Search endpoint
# -------------------------
@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    # Safety: empty query
    if not req.query.strip():
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": TOTAL_DOCS
            }
        }

    # 1️⃣ Embed query
    q_vec = vectorizer.transform([req.query])

    # 2️⃣ Vector search
    sims = cosine_similarity(q_vec, DOC_EMBEDDINGS)[0]
    top_idx = np.argsort(sims)[::-1][: max(1, req.k)]

    candidates = []
    for i in top_idx:
        raw_score = float(sims[i])
        if np.isnan(raw_score):
            raw_score = 0.0

        candidates.append({
            "id": DOCS[i]["id"],
            "content": DOCS[i]["content"],
            "metadata": DOCS[i].get("metadata", {}),
            "score": raw_score
        })

    # 3️⃣ Re-ranking
    reranked = False
    if req.rerank and candidates:
        reranked = True
        pairs = rerank(req.query, candidates)
        candidates = [
            {**doc, "score": float(score)}
            for doc, score in pairs
        ]

    # 4️⃣ SAFE normalization (guaranteed valid)
    if candidates:
        scores = [float(d["score"]) for d in candidates]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # Force slight descending values
            for idx, d in enumerate(candidates):
                d["score"] = round(1.0 - (idx * 0.01), 3)
        else:
            for d in candidates:
                normalized = (d["score"] - min_score) / (max_score - min_score)
                normalized = max(0.0, min(1.0, normalized))
                d["score"] = round(normalized, 3)

    # 5️⃣ Sort + trim
    candidates.sort(key=lambda x: x["score"], reverse=True)
    results = candidates[: req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": TOTAL_DOCS
        }
    }
