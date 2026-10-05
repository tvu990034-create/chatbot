"""FastAPI server for the local chatbot with built-in web UI."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from local_chatbot.config import ChatbotConfig
from local_chatbot.graph import LocalChatGraph

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Local AI Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = ChatbotConfig()
chatbot = LocalChatGraph(cfg)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ClearRequest(BaseModel):
    session_id: str = "default"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": chatbot.engine.is_loaded,
        "model_path": cfg.model_path,
        "knowledge_base": cfg.knowledge_base_path,
        "langgraph": chatbot._graph is not None,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    result = chatbot.chat(req.message, session_id=req.session_id)

    async def stream():
        words = result.response.split()
        for word in words:
            yield f"data: {json.dumps({'type': 'token', 'value': word + ' '})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'source': result.source, 'sources': result.sources, 'latency_ms': result.latency_ms})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/chat/sync")
def chat_sync(req: ChatRequest):
    result = chatbot.chat(req.message, session_id=req.session_id)
    return {
        "response": result.response,
        "source": result.source,
        "sources": result.sources,
        "latency_ms": result.latency_ms,
    }


@app.post("/session/clear")
def clear_session(req: ClearRequest):
    chatbot.clear_session(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


def main():
    import uvicorn

    print("\n" + "=" * 60)
    print("  LOCAL AI CHATBOT")
    print("=" * 60)
    print(f"  Model:    {cfg.model_path}")
    print(f"  Loaded:   {chatbot.engine.is_loaded} (will load on first request)")
    print(f"  RAG:      {cfg.knowledge_base_path}")
    print(f"  LangGraph: {chatbot._graph is not None}")
    print(f"  URL:      http://localhost:{cfg.port}")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
