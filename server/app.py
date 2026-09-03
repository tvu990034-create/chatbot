"""
server/app.py
~~~~~~~~~~~~~
FastAPI backend — streamlined for maximum speed.
No performance equations, no overhead.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting local-chatbot API server …")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Local AI chatbot - streamlined for maximum speed",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32_000)
    history: list[Message] = Field(default_factory=list)
    model: str | None = Field(None, description="Override the default LLM model")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=32_000)
    use_rag: bool = Field(True)
    use_agent: bool = Field(True, description="Route through LangGraph agent")
    use_cache: bool = Field(True, description="Simple cache lookup")
    use_router: bool = Field(True, description="Model router")
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    id: str
    message: str
    model: str
    elapsed_ms: int
    sources: list[str] = Field(default_factory=list)
    cache_hit: bool = False


class RAGIngestRequest(BaseModel):
    rebuild: bool = Field(False)


class RAGQueryRequest(BaseModel):
    question: str
    provider: str = Field("auto")


# ---------------------------------------------------------------------------
# Info / health
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
async def root():
    return {
        "name":    settings.app_name,
        "version": settings.app_version,
        "model":   settings.default_model,
        "backend": settings.local_backend.value,
        "rag":     settings.rag_provider.value,
    }


@app.get("/health", tags=["info"])
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/v1/health", tags=["info"])
async def health_v1():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/models", tags=["llm"])
async def list_models():
    # Return default models
    return {
        "models": [
            "ollama/phi3:mini",
            "ollama/gemma2:2b",
            "ollama/llama3.2",
            "ollama/tinyllama",
        ],
        "default": settings.default_model
    }


@app.get("/api/v1/models", tags=["llm"])
async def list_models_v1():
    # Return default models for React UI
    return [
        "ollama/phi3:mini",
        "ollama/gemma2:2b",
        "ollama/llama3.2",
        "ollama/tinyllama",
    ]


@app.get("/backend/status", tags=["backend"])
async def backend_status():
    from backends.model_server import get_backend_status
    return get_backend_status()


# ---------------------------------------------------------------------------
# Metrics  (Eq1–Eq9 dashboard)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Chat (non-streaming)
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_endpoint(req: ChatRequest):
    request_id  = str(uuid.uuid4())[:8]
    t0          = time.perf_counter()
    history     = [{"role": m.role, "content": m.content} for m in req.history]
    sources: list[str] = []
    cache_hit   = False

    effective_max_tokens = req.max_tokens

    try:
        if req.use_agent:
            from agents.langgraph_agent import achat
            reply = await achat(req.message, history=history)
        elif req.use_rag:
            from rag.llama_index_rag import get_rag
            result  = await get_rag().aquery(req.message)
            reply   = result["answer"]
            sources = result.get("sources", [])
        else:
            from gateway.litellm_gateway import achat
            msgs = history + [{"role": "user", "content": req.message}]
            reply = await achat(
                msgs,
                model=req.model,
                temperature=req.temperature,
                max_tokens=effective_max_tokens,
                system_prompt=req.system_prompt,
                use_cache=req.use_cache,
                use_router=req.use_router,
            )
            # Check if cache was hit
            from gateway.simple_cache import get_cache
            cache = get_cache()
            cache_hit = False  # Simple cache doesn't track hits

    except Exception as exc:
        logger.exception("Chat error [%s]: %s", request_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    model_used = req.model or settings.default_model

    return ChatResponse(
        id=request_id,
        message=reply,
        model=model_used,
        elapsed_ms=elapsed_ms,
        sources=sources,
        cache_hit=cache_hit,
    )


class ChatV1Request(BaseModel):
    messages: list[dict]
    model: str = "ollama/phi3:mini"
    provider: str = "ollama"
    use_cache: bool = True
    performance_mode: str = "balanced"
    images: list[str] = []


@app.post("/api/v1/chat", tags=["chat"])
async def chat_v1_endpoint(req: ChatV1Request):
    """
    React UI compatible chat endpoint.
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    
    # Convert messages format
    history = [{"role": m.get("role"), "content": m.get("content")} for m in req.messages]
    
    # Apply performance mode settings
    temperature = 0.7
    max_tokens = 512
    
    if req.performance_mode == "speed":
        temperature = 0.5
        max_tokens = 256
    elif req.performance_mode == "quality":
        temperature = 0.9
        max_tokens = 1024
    
    # Check cache first before calling API - use both simple and semantic cache
    query = history[-1]['content'] if history else ""
    cached_reply = None
    if req.use_cache and query:
        # Try semantic cache first (more intelligent matching)
        try:
            from gateway.semantic_cache import SemanticCache
            semantic_cache = SemanticCache()
            cached_reply = semantic_cache.get(query)
            if cached_reply:
                logger.info(f"Semantic cache hit for query: {query[:40]}...")
        except Exception as e:
            logger.warning(f"Semantic cache failed: {e}")
        
        # Fallback to simple cache
        if not cached_reply:
            from gateway.simple_cache import get_cache
            cache = get_cache()
            cached_reply = cache.get(query)
            logger.info(f"Cache check for '{query[:30]}...': {'HIT' if cached_reply else 'MISS'}")
    
    if cached_reply:
        reply = cached_reply
        cache_hit = True
        elapsed = time.perf_counter() - t0
        logger.info(f"Returning cached response in {elapsed:.3f}s")
        return {
            "response": reply,
            "model": req.model,
            "duration": elapsed,
            "optimizations_applied": 1,
            "cache_hit": True,
        }
    
    cache_hit = False
    
    # Check cache first for instant responses
    if req.use_cache and query:
        from gateway.simple_cache import get_cache
        cache = get_cache()
        cached = cache.get(query)
        if cached:
            reply = cached
            cache_hit = True
            logger.info(f"Cache hit for query: {query[:40]}...")
    
    # If not cached, call the gateway
    if not cache_hit:
        try:
            from gateway.litellm_gateway import chat
            import asyncio
            
            # For ollama, don't pass api_base, let litellm handle it
            # Run synchronous chat in thread pool with balanced timeout
            loop = asyncio.get_event_loop()
            reply = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    chat,
                    history,
                    req.model,
                    temperature,
                    max_tokens,
                    None,  # api_base
                    req.use_cache,
                ),
                timeout=30.0  # 30 second timeout for quality
            )
        except Exception as exc:
            logger.exception("Chat v1 error [%s]: %s", request_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    elapsed = time.perf_counter() - t0
    
    return {
        "response": reply,
        "model": req.model,
        "duration": elapsed,
        "cache_hit": cache_hit,
        "optimizations_applied": 1,  # Response caching
    }


# ---------------------------------------------------------------------------
# Chat (streaming SSE)
# ---------------------------------------------------------------------------

@app.post("/chat/stream", tags=["chat"])
async def chat_stream_endpoint(req: ChatRequest):
    """
    Server-Sent Events stream.
    Each event: {"delta": "...", "done": false}
    Final:       {"delta": "",    "done": true, "metrics": {...}}
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def event_generator() -> AsyncGenerator[str, None]:
        import json
        t0 = time.perf_counter()
        try:
            from gateway.litellm_gateway import achat_stream
            msgs = history + [{"role": "user", "content": req.message}]
            async for chunk in achat_stream(
                msgs,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                system_prompt=req.system_prompt,
                use_router=req.use_router,
            ):
                yield f"data: {json.dumps({'delta': chunk, 'done': False})}\n\n"
        except Exception as exc:
            logger.exception("Stream error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"
            return

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        metrics_snap = {
            "elapsed_ms":   elapsed_ms,
        }
        yield f"data: {json.dumps({'delta': '', 'done': True, 'metrics': metrics_snap})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

@app.post("/rag/ingest", tags=["rag"])
async def rag_ingest(req: RAGIngestRequest, files: list[UploadFile] = File(default=[])):
    import shutil
    from config import RAGProvider

    saved: list[str] = []
    if files:
        for f in files:
            dest = settings.rag_docs_dir / (f.filename or "upload.txt")
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(str(dest))

    result: dict[str, Any] = {"saved_files": saved}

    if settings.rag_provider in (RAGProvider.LLAMA_INDEX, RAGProvider.BOTH):
        try:
            from rag.llama_index_rag import get_rag
            rag = get_rag()
            if saved:
                result["llama_index_nodes_added"] = rag.add_documents(saved)
            else:
                rag.build_index(force_rebuild=req.rebuild)
                result["llama_index"] = "reindexed"
        except Exception as exc:
            result["llama_index_error"] = str(exc)

    if settings.rag_provider in (RAGProvider.HAYSTACK, RAGProvider.BOTH):
        try:
            from rag.haystack_pipeline import get_haystack_rag
            result["haystack_docs_stored"] = get_haystack_rag().ingest(saved or None)
        except Exception as exc:
            result["haystack_error"] = str(exc)

    return result


@app.post("/rag/query", tags=["rag"])
async def rag_query(req: RAGQueryRequest):
    from config import RAGProvider
    out: dict[str, Any] = {}

    use_li = req.provider in ("llama_index", "auto") and \
        settings.rag_provider in (RAGProvider.LLAMA_INDEX, RAGProvider.BOTH)
    use_hs = req.provider in ("haystack", "auto") and \
        settings.rag_provider in (RAGProvider.HAYSTACK, RAGProvider.BOTH)

    if use_li:
        try:
            from rag.llama_index_rag import get_rag
            out["llama_index"] = await get_rag().aquery(req.question)
        except Exception as exc:
            out["llama_index_error"] = str(exc)

    if use_hs:
        try:
            from rag.haystack_pipeline import get_haystack_rag
            out["haystack"] = get_haystack_rag().query(req.question)
        except Exception as exc:
            out["haystack_error"] = str(exc)

    if not out:
        raise HTTPException(status_code=400, detail="No RAG provider active.")
    return out
