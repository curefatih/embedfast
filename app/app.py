import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

from fastembed import TextEmbedding

from app.request import DocumentRequest

text_embeds = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.getenv("MAIN_TEXT_EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    text_embeds["main"] = TextEmbedding(model_name=model_name)
    yield
    text_embeds.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/api/v1/text-embed")
async def read_root(document_request: DocumentRequest):
    embedding_model: TextEmbedding = text_embeds["main"]
    embeddings_list = list(embedding_model.embed(document_request.documents))
    return list(map(lambda embed: embed.tolist(), embeddings_list))


@app.get("/health")
async def read_item():
    return "OK"
