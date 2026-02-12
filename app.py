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
# Re-ranking (expensive scorer)
# -------------------------
def rerank_docs(query: str, docs: list):
    q_vec = vectorizer.transform([query])
    d_vecs = vectorizer.transform([d["content"] for d in docs])
    sims = cosine_similarity(q_vec, d_vecs)[0]

    reranked = []
    for doc, score in zip(docs, sims):
        keyword_bonus = sum(
            1 for w in query.lower().split()
            if w in doc["content"].lower()
        ) * 0.1

        final_score = min(float(score) + keyword_bonus, 1.0)
        reranked.append({**doc, "score": final_score})

    return reranked

# -------------------------
# Search endpoint
# -------------------------
@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    # 1️⃣ Embed query
    q_vec = vectorizer.transform([req.query])

    # 2️⃣ Vector search
    sims = cosine_similarity(q_vec, DOC_EMBEDDINGS)[0]
    top_idx = np.argsort(sims)[::-1][:req.k]

    candidates = []
    for i in top_idx:
        candidates.append({
            "id": DOCS[i]["id"],
            "content": DOCS[i]["content"],
            "metadata": {"source": DOCS[i]["metadata"]["source"]},
            "score": float(sims[i])
        })

    reranked_flag = False

    # 3️⃣ Re-ranking
    if req.rerank and candidates:
        reranked_flag = True
        candidates = rerank_docs(req.query, candidates)

    # 4️⃣ Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 5️⃣ Trim results
    results = candidates[:req.rerankK]

    # 6️⃣ Normalize scores (0–1 safely)
    if results:
        max_score = max(d["score"] for d in results)
        for d in results:
            if max_score > 0:
                d["score"] = round(d["score"] / max_score, 3)
            else:
                d["score"] = 0.0

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": TOTAL_DOCS
        }
    }
