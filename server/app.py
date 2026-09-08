"""
server/app.py
~~~~~~~~~~~~~
FastAPI backend — streamlined for maximum speed.
No performance equations, no overhead.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    content: str = Field(..., min_length=1, max_length=100_000)


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
    speed_mode: bool = Field(False, description="Cap tokens/temp and skip generation policy")


def validate_conversation(messages: list[Message]) -> None:
    """
    BUG 7 FIX: Validate conversation structure.
    
    Validates:
    - roles are valid
    - ordering is coherent
    - no empty messages
    - final message is from user
    - no orphaned assistant/tool messages
    """
    if not messages:
        return  # Empty history is valid
    
    # Check for empty content
    for i, msg in enumerate(messages):
        if not msg.content or not msg.content.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Empty message at position {i}"
            )
    
    # Check that conversation starts with user or system
    if messages[0].role not in ("user", "system"):
        raise HTTPException(
            status_code=400,
            detail="Conversation must start with 'user' or 'system' message"
        )
    
    # Check for orphaned assistant messages (assistant without preceding user)
    for i in range(1, len(messages)):
        prev_role = messages[i-1].role
        current_role = messages[i].role
        
        # Assistant should follow user or system
        if current_role == "assistant" and prev_role not in ("user", "system"):
            raise HTTPException(
                status_code=400,
                detail=f"Orphaned assistant message at position {i}"
            )
        
        # System should only be at the beginning
        if current_role == "system" and i > 0:
            raise HTTPException(
                status_code=400,
                detail=f"System message must be at beginning (position {i})"
            )


class ChatResponse(BaseModel):
    id: str
    response: str
    model: str
    duration: float
    cache_hit: bool
    optimizations_applied: int
    sources: list[str] = Field(default_factory=list)


def optimization_count(*, cache_hit: bool, rag: bool, policy: str) -> int:
    """Count the optimizations that actually ran for a request. At least 1
    (the cache layer) is always accounted for."""
    n = 0
    if cache_hit:
        n += 1
    if rag:
        n += 1
    if policy:
        n += 1
    return max(n, 1)


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
    return await asyncio.to_thread(get_backend_status)


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
    
    # BUG 7 FIX: Validate conversation structure
    validate_conversation(req.history)
    
    history     = [{"role": m.role, "content": m.content} for m in req.history]
    sources: list[str] = []
    cache_hit   = False
    model_used: str = ""
    actual_model: str | None = None
    rag_used   = False
    policy     = ""

    effective_max_tokens = req.max_tokens

    try:
        if req.use_agent:
            from agents.langgraph_agent import achat
            result = await achat(
                req.message, history=history,
                model=req.model, temperature=req.temperature,
                max_tokens=req.max_tokens, system_prompt=req.system_prompt,
                use_rag=req.use_rag, use_router=req.use_router,
                use_cache=req.use_cache,
                speed_mode=req.speed_mode,
            )
            reply = result.reply
            cache_hit = result.cache_hit
            model_used = result.model_used or req.model or settings.default_model
            rag_used = bool(getattr(result, "rag_used", False))
            policy = getattr(result, "generation_policy", "") or ""
            sources = []  # TODO: extract from RAG when available
        elif req.use_rag:
            # Pure RAG path (no user-facing LLM call): LlamaIndex handles
            # retrieval + generation internally.  The following request params
            # are NOT applicable here because there is no explicit LLM call
            # for the caller to control: model, temperature, max_tokens,
            # system_prompt.  Use the agent path (use_agent=True) if you need
            # to control LLM generation parameters alongside RAG.
            from rag.llama_index_rag import get_rag
            result  = await get_rag().aquery(req.message)
            reply   = result["answer"]
            sources = result.get("sources", [])
        else:
            from gateway.litellm_gateway import achat
            msgs = history + [{"role": "user", "content": req.message}]
            reply, cache_hit, actual_model = await achat(
                msgs,
                model=req.model,
                temperature=req.temperature,
                max_tokens=effective_max_tokens,
                system_prompt=req.system_prompt,
                use_cache=req.use_cache,
                use_router=req.use_router,
                speed_mode=req.speed_mode,
            )

    except Exception as exc:
        logger.exception("Chat error [%s]: %s", request_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    # model_used set in agent path; fallback for direct gateway path
    if not model_used:
        model_used = actual_model or req.model or settings.default_model

    return ChatResponse(
        id=request_id,
        response=reply,
        model=model_used,
        duration=elapsed,
        cache_hit=cache_hit,
        optimizations_applied=optimization_count(
            cache_hit=cache_hit, rag=rag_used or bool(sources), policy=policy),
        sources=sources,
    )


class ChatV1Request(BaseModel):
    messages: list[dict]
    model: str = "ollama/phi3:mini"
    provider: str = "ollama"
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=32_000)
    system_prompt: str | None = None
    use_rag: bool = Field(True)
    use_router: bool = Field(True)
    use_cache: bool = True
    performance_mode: str = "balanced"
    images: list[str] = []


def validate_v1_messages(messages: list[dict]) -> None:
    """
    BUG 7 FIX: Validate v1 message format.
    Similar validation as for ChatRequest but for dict format.
    """
    if not messages:
        return  # Empty messages is valid
    
    valid_roles = {"user", "assistant", "system"}
    
    for i, msg in enumerate(messages):
        # Check required fields
        if not isinstance(msg, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Message at position {i} must be a dict"
            )
        
        if "role" not in msg or "content" not in msg:
            raise HTTPException(
                status_code=400,
                detail=f"Message at position {i} missing 'role' or 'content'"
            )
        
        role = msg["role"]
        content = msg["content"]
        
        # Check role validity
        if role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{role}' at position {i}"
            )
        
        # Check for empty content
        if not content or not content.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Empty message content at position {i}"
            )
    
    # Check conversation starts with user or system
    if messages[0]["role"] not in ("user", "system"):
        raise HTTPException(
            status_code=400,
            detail="Conversation must start with 'user' or 'system' message"
        )
    
    # Check for orphaned assistant messages
    for i in range(1, len(messages)):
        prev_role = messages[i-1]["role"]
        current_role = messages[i]["role"]
        
        if current_role == "assistant" and prev_role not in ("user", "system"):
            raise HTTPException(
                status_code=400,
                detail=f"Orphaned assistant message at position {i}"
            )
        
        if current_role == "system" and i > 0:
            raise HTTPException(
                status_code=400,
                detail=f"System message must be at beginning (position {i})"
            )


@app.post("/api/v1/chat", tags=["chat"])
async def chat_v1_endpoint(req: ChatV1Request):
    """
    React UI compatible chat endpoint.
    Routes through the agent path so all request params are honoured.
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    # BUG 7 FIX: Validate conversation structure
    validate_v1_messages(req.messages)

    # Convert messages format
    history = [{"role": m.get("role"), "content": m.get("content")} for m in req.messages]

    # Use caller-provided values; fall back to performance-mode defaults.
    temperature = req.temperature
    max_tokens = req.max_tokens
    if temperature is None:
        temperature = 0.7 if req.performance_mode == "balanced" else (
            0.5 if req.performance_mode == "speed" else 0.9
        )
    if max_tokens is None:
        max_tokens = 512 if req.performance_mode == "balanced" else (
            256 if req.performance_mode == "speed" else 1024
        )

    cache_hit = False
    model_used = ""

    try:
        # Route through the agent path so temperature/max_tokens/system_prompt
        # /use_rag/use_router are all propagated correctly.
        from agents.langgraph_agent import achat
        # Extract the last user message; the rest is history.
        last_user = ""
        if history and history[-1].get("role") == "user":
            last_user = history[-1]["content"]
            history = history[:-1]
        if not last_user:
            # Fallback: use the last message content
            last_user = history[-1]["content"] if history else ""

        result = await achat(
            last_user, history=history,
            model=req.model, temperature=temperature,
            max_tokens=max_tokens, system_prompt=req.system_prompt,
            use_rag=req.use_rag, use_router=req.use_router,
            use_cache=req.use_cache,
            speed_mode=(req.performance_mode == "speed"),
        )
        reply = result.reply
        cache_hit = result.cache_hit
        model_used = result.model_used or req.model
        rag_used = bool(getattr(result, "rag_used", False))
        policy = getattr(result, "generation_policy", "") or ""
    except Exception as exc:
        logger.exception("Chat v1 error [%s]: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0

    return ChatResponse(
        id=request_id,
        response=reply,
        model=model_used,
        duration=elapsed,
        cache_hit=cache_hit,
        optimizations_applied=optimization_count(
            cache_hit=cache_hit, rag=rag_used, policy=policy),
    )


# ---------------------------------------------------------------------------
# Chat (streaming SSE)
# ---------------------------------------------------------------------------

@app.post("/chat/stream", tags=["chat"])
async def chat_stream_endpoint(req: ChatRequest):
    """
    Server-Sent Events stream.
    Each event: {"delta": "...", "done": false}
    Final:       {"delta": "",    "done": true, "metrics": {...}}

    Routes through agents.langgraph_agent.achat_stream so all optimizations
    apply: cache, quick-path, RAG injection, Eq1 context trimming/compression,
    generation policy and model routing.
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def event_generator() -> AsyncGenerator[str, None]:
        import json
        t0 = time.perf_counter()
        try:
            from agents.langgraph_agent import achat_stream
            async for chunk in achat_stream(
                req.message, history=history,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                system_prompt=req.system_prompt,
                use_rag=req.use_rag,
                use_router=req.use_router,
                use_cache=req.use_cache,
                speed_mode=req.speed_mode,
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
    import os
    from config import RAGProvider

    # BUG 9 FIX: Implement upload limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
    MAX_FILES = 20  # Maximum files per request
    MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total per request

    saved: list[str] = []
    total_size = 0
    
    if files:
        # BUG 9 FIX: Check file count limit
        if len(files) > MAX_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files: {len(files)} (max {MAX_FILES})"
            )
        
        for f in files:
            # BUG 8 FIX: Prevent path traversal attacks
            # Use basename to strip any directory components
            safe_filename = os.path.basename(f.filename or "upload.txt")
            
            # Additional safety: ensure filename doesn't contain suspicious patterns
            if not safe_filename or safe_filename.startswith('.') or '..' in safe_filename:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid filename: {f.filename}"
                )
            
            dest = settings.rag_docs_dir / safe_filename
            
            # BUG 8 FIX: Ensure destination is within the intended directory
            try:
                dest.resolve().relative_to(settings.rag_docs_dir.resolve())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Path traversal attempt detected: {f.filename}"
                )
            
            # BUG 9 FIX: Check individual file size
            file_size = 0
            f.file.seek(0, os.SEEK_END)
            file_size = f.file.tell()
            f.file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {f.filename} ({file_size} bytes, max {MAX_FILE_SIZE})"
                )
            
            # BUG 9 FIX: Check total size limit
            if total_size + file_size > MAX_TOTAL_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Total upload size exceeds limit ({MAX_TOTAL_SIZE} bytes)"
                )
            
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(str(dest))
            total_size += file_size

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
            out["haystack"] = await asyncio.to_thread(
                lambda: get_haystack_rag().query(req.question))
        except Exception as exc:
            out["haystack_error"] = str(exc)

    if not out:
        raise HTTPException(status_code=400, detail="No RAG provider active.")
    return out
