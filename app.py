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
# Re-ranking function (MUST EXIST)
# -------------------------
def rerank(query: str, docs: list[dict]) -> list[tuple[dict, float]]:
    q_vec = vectorizer.transform([query])
    d_vecs = vectorizer.transform([d["content"] for d in docs])
    sims = cosine_similarity(q_vec, d_vecs)[0]

    reranked_pairs = []
    for doc, score in zip(docs, sims):
        bonus = sum(
            1 for w in query.lower().split()
            if w in doc["content"].lower()
        ) * 0.1

        final_score = min(float(score + bonus), 1.0)
        reranked_pairs.append((doc, final_score))

    return reranked_pairs

# -------------------------
# Search endpoint
# -------------------------
@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    # Embed query
    q_vec = vectorizer.transform([req.query])

    # Vector search
    sims = cosine_similarity(q_vec, DOC_EMBEDDINGS)[0]
    top_idx = np.argsort(sims)[::-1][:req.k]

    candidates = []
    for i in top_idx:
        candidates.append({
            "id": DOCS[i]["id"],
            "content": DOCS[i]["content"],
            "metadata": DOCS[i].get("metadata", {}),
            "score": float(sims[i])
        })

    reranked = False

    # Re-rank
    if req.rerank and candidates:
        reranked = True
        reranked_pairs = rerank(req.query, candidates)
        candidates = [
            {**doc, "score": float(score)}
            for doc, score in reranked_pairs
        ]

    # 🔒 FORCE SAFE NORMALIZATION
    scores = [c["score"] for c in candidates if isinstance(c["score"], (int, float))]
    max_score = max(scores) if scores else 1.0

    for c in candidates:
        if max_score > 0:
            c["score"] = round(float(c["score"]) / max_score, 3)
        else:
            c["score"] = 0.0

        # absolute safety clamp
        c["score"] = max(0.0, min(1.0, c["score"]))

    # Sort + trim
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
