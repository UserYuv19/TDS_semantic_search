import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# App
# -------------------------
app = FastAPI(title="Semantic Search with Re-ranking")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins
    allow_methods=["*"],      # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

# -------------------------
# Load documents
# -------------------------
import json
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
# Re-ranking (better scorer)
# -------------------------
def rerank(query: str, docs):
    """
    Simulated expensive reranker:
    Uses cosine similarity again but boosts exact keyword overlap.
    """
    q_vec = vectorizer.transform([query])
    d_vecs = vectorizer.transform([d["content"] for d in docs])
    sims = cosine_similarity(q_vec, d_vecs)[0]

    reranked = []
    for doc, score in zip(docs, sims):
        keyword_bonus = sum(
            1 for w in query.lower().split() if w in doc["content"].lower()
        ) * 0.1
        reranked.append((doc, min(score + keyword_bonus, 1.0)))

    return reranked

# -------------------------
# Search endpoint
# -------------------------
@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    # 1️⃣ Embed query
    q_vec = vectorizer.transform([req.query])

    # 2️⃣ Vector search (top-k)
    sims = cosine_similarity(q_vec, DOC_EMBEDDINGS)[0]
    top_idx = np.argsort(sims)[::-1][:req.k]

    candidates = [
        {
            "id": DOCS[i]["id"],
            "content": DOCS[i]["content"],
            "metadata": {"source": DOCS[i]["metadata"]["source"]},
            "score": float(sims[i])
        }
        for i in top_idx
    ]

    reranked = False

    # 3️⃣ Re-ranking
    if req.rerank and candidates:
        reranked = True
        reranked_docs = rerank(req.query, candidates)
        candidates = [
            {**doc, "score": score}
            for doc, score in reranked_docs
        ]

    # 4️⃣ Normalize scores (0–1)
    max_score = max(d["score"] for d in candidates) if candidates else 1
    for d in candidates:
        d["score"] = round(d["score"] / max_score, 3)

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
