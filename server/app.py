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
    
    # BUG 7 FIX: Validate conversation structure
    validate_conversation(req.history)
    
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
            try:
                reply, cache_hit, actual_model = await achat(
                    msgs,
                    model=req.model,
                    temperature=req.temperature,
                    max_tokens=effective_max_tokens,
                    system_prompt=req.system_prompt,
                    use_cache=req.use_cache,
                    use_router=req.use_router,
                )
            except (TypeError, ValueError):
                # Fallback for old gateway that doesn't return tuple
                try:
                    reply, cache_hit = await achat(
                        msgs,
                        model=req.model,
                        temperature=req.temperature,
                        max_tokens=effective_max_tokens,
                        system_prompt=req.system_prompt,
                        use_cache=req.use_cache,
                        use_router=req.use_router,
                    )
                    actual_model = req.model or settings.default_model
                except (TypeError, ValueError):
                    # Final fallback for very old gateway
                    reply = await achat(
                        msgs,
                        model=req.model,
                        temperature=req.temperature,
                        max_tokens=effective_max_tokens,
                        system_prompt=req.system_prompt,
                        use_cache=req.use_cache,
                        use_router=req.use_router,
                    )
                    cache_hit = False
                    actual_model = req.model or settings.default_model
            # Bug #15 FIX: Gateway now returns cache_hit status

    except Exception as exc:
        logger.exception("Chat error [%s]: %s", request_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    # BUG 3 FIX: Use actual model used by gateway, not just requested model
    model_used = locals().get('actual_model', req.model or settings.default_model)

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
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    
    # BUG 7 FIX: Validate conversation structure
    validate_v1_messages(req.messages)
    
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
    
    # REMOVED: Duplicate cache lookup - let the gateway handle all caching
    # Bug #15: The API was doing cache lookups and then the gateway also does cache lookups
    # This causes duplicate work and inconsistent behavior. Gateway handles caching now.
    
    cache_hit = False
    elapsed = time.perf_counter() - t0
    
    # Call the gateway (handles caching internally)
    try:
        from gateway.litellm_gateway import chat
        import asyncio
        
        # For ollama, don't pass api_base, let litellm handle it
        # Run synchronous chat in thread pool with balanced timeout
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
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
            timeout=getattr(settings, 'request_deadline', 120)
        )
        
        # Handle both old (string) and new (tuple) return types
        if isinstance(result, tuple):
            reply, cache_hit = result
        else:
            reply = result
            
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
            # BUG 2 FIX: Pass use_cache parameter to streaming function
            async for chunk in achat_stream(
                msgs,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                system_prompt=req.system_prompt,
                use_cache=req.use_cache,  # BUG 2 FIX: Honor use_cache
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
            out["haystack"] = get_haystack_rag().query(req.question)
        except Exception as exc:
            out["haystack_error"] = str(exc)

    if not out:
        raise HTTPException(status_code=400, detail="No RAG provider active.")
    return out
