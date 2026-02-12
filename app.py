import time
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

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
# Embeddings
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
# Re-ranking
# -------------------------
def rerank_docs(query: str, docs: list):
    q_vec = vectorizer.transform([query])
    d_vecs = vectorizer.transform([d["content"] for d in docs])
    sims = cosine_similarity(q_vec, d_vecs)[0]

    reranked = []
    for doc, sim in zip(docs, sims):
        bonus = sum(
            1 for w in query.lower().split()
            if w in doc["content"].lower()
        ) * 0.1

        score = float(sim + bonus)
        reranked.append({**doc, "score": score})

    return reranked

# -------------------------
# Search endpoint
# -------------------------
@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    q_vec = vectorizer.transform([req.query])
    sims = cosine_similarity(q_vec, DOC_EMBEDDINGS)[0]

    top_idx = np.argsort(sims)[::-1][:req.k]

    # ✅ ALWAYS initialize score
    candidates = []
    for i in top_idx:
        candidates.append({
            "id": DOCS[i]["id"],
            "content": DOCS[i]["content"],
            "metadata": {"source": DOCS[i]["metadata"]["source"]},
            "score": float(sims[i]) if not np.isnan(sims[i]) else 0.0
        })

    reranked_flag = False

    if req.rerank and candidates:
        reranked_flag = True
        candidates = rerank_docs(req.query, candidates)

    # Normalize SAFELY
    scores = [d["score"] for d in candidates if isinstance(d["score"], (int, float))]
    max_score = max(scores) if scores else 0.0

    for d in candidates:
        if max_score > 0:
            d["score"] = round(float(d["score"]) / max_score, 3)
        else:
            d["score"] = 0.0

        # ✅ FINAL CLAMP (absolute safety)
        if d["score"] < 0:
            d["score"] = 0.0
        if d["score"] > 1:
            d["score"] = 1.0

    # Sort + trim
    candidates.sort(key=lambda x: x["score"], reverse=True)
    results = candidates[:req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": TOTAL_DOCS
        }
    }
