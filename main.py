import os
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

app = FastAPI()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("intfloat/multilingual-e5-base")

class Query(BaseModel):
    text: str
    top_k: int = 3

@app.post("/search")
def search(q: Query):
    vector = model.encode(q.text).tolist()
    hits = qdrant.search(
        collection_name="docs",
        query_vector=vector,
        limit=q.top_k
    )
    return {"results": [hit.payload for hit in hits]}
