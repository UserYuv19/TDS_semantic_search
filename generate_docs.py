import json

docs = []
for i in range(92):
    docs.append({
        "id": i,
        "content": f"API documentation example about authentication topic {i}.",
        "metadata": {"source": f"doc_{i}.md"}
    })

with open("docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2)
