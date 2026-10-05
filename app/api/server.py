from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cache import PreWarmedCache
from config import AdvancedConfig
from identity import ConversationState, deterministic_chat_identity_key
from retrieval import (
    PromptBuilder,
    RetrievalConfig,
    RetrievalPipeline,
    MinimalPromptBuilder,
    ZeroTokenResponder,
    FAQDatabase,
)
from cloud_prompt import CloudPromptBuilder
from phrase_normalizer import PhraseNormalizer

# ------------------------------------------------------------------
#  Cloud‑only inference engines (no llama.cpp)
# ------------------------------------------------------------------
class CloudOpenAiEngine:
    """Deterministic cloud engine backed by OpenAI / compatible API."""
    def __init__(self, cfg: dict):
        import openai
        self.model = cfg.get("model", "gpt-4.1-nano")
        self.client = openai.OpenAI(
            api_key=cfg.get("api_key", os.environ.get("OPENAI_API_KEY")),
            base_url=cfg.get("base_url", None),
        )
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 256)

    def generate(self, state: ConversationState, user_input: str, seed: int,
                 prompt_override: Optional[str] = None):
        if prompt_override:
            prompt = prompt_override
        else:
            history = "\n".join(f"{t.role}: {t.content}" for t in state.history)
            prompt = f"system: You are a fast assistant.\n{history}\nuser: {user_input}\nassistant:"

        if prompt.startswith("[CACHE_SYSTEM]"):
            prompt = prompt[len("[CACHE_SYSTEM]"):]
            parts = prompt.split("\n", 1)
            system_text = parts[0]
            user_text = parts[1] if len(parts) > 1 else ""
            messages = [
                {"role": "system", "content": system_text, "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": user_text},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )
        reply = response.choices[0].message.content.strip()

        new_state = ConversationState(
            history=list(state.history),
            token_ids=list(state.token_ids),
            kv_cache=None,
        )
        new_state.append("user", user_input)
        new_state.append("assistant", reply)
        metrics = {
            "prompt_tokens": 0.0,
            "generated_tokens": float(len(reply.split())),
            "speculative_enabled": 0.0,
            "kv_pruned_tokens": 0.0,
            "tome_merged_tokens": 0.0,
        }
        return reply, new_state, metrics


class CloudSimulatedEngine:
    """Fake engine for testing without any API key – just a sleep."""
    def __init__(self, latency_ms: float = 50):
        self.latency_ms = latency_ms

    def generate(self, state: ConversationState, user_input: str, seed: int,
                 prompt_override: Optional[str] = None):
        import time
        time.sleep(self.latency_ms / 1000.0)
        reply = f"Simulated cloud answer to: {user_input}"
        new_state = ConversationState(
            history=list(state.history),
            token_ids=list(state.token_ids),
            kv_cache=None,
        )
        new_state.append("user", user_input)
        new_state.append("assistant", reply)
        metrics = {
            "prompt_tokens": 0.0,
            "generated_tokens": float(len(reply.split())),
            "speculative_enabled": 0.0,
            "kv_pruned_tokens": 0.0,
            "tome_merged_tokens": 0.0,
        }
        return reply, new_state, metrics


# ------------------------------------------------------------------
#  App configuration
# ------------------------------------------------------------------
@dataclass(slots=True)
class AppConfig:
    system_prompt: str = "You are a fast, deterministic assistant."
    default_seed: int = 42
    documents: List[str] = field(default_factory=lambda: [
        "TinyLlama is an open language model suitable for local inference.",
        "KV-cache stores attention keys and values from prior tokens to avoid recomputation.",
        "Speculative decoding uses a draft model then verifies with a target model.",
        "BM25 is a lexical ranking function for sparse retrieval.",
        "FAISS HNSW provides fast approximate nearest-neighbor search.",
    ])
    doc_ids: List[str] = field(default_factory=lambda: [f"doc-{i}" for i in range(5)])
    graph_edges: List[tuple[int, int]] = field(default_factory=lambda: [(0, 1), (1, 2), (2, 3), (0, 4)])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    seed: int = 42


# ------------------------------------------------------------------
#  NEW: Markov Predictor wrapper
# ------------------------------------------------------------------
try:
    from predictor import FastPredictor
except ImportError:
    FastPredictor = None

# ------------------------------------------------------------------
#  NEW: Neural Bloom Filter wrapper
# ------------------------------------------------------------------
try:
    from nbf import UnifiedNeuralBloomFilter
except ImportError:
    UnifiedNeuralBloomFilter = None

# ------------------------------------------------------------------
#  NEW: Conversation Memory
# ------------------------------------------------------------------
try:
    from conversation_memory import ConversationMemory, MemoryConfig, Compressor
except ImportError:
    ConversationMemory = None


class ChatApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        adv = AdvancedConfig.from_env()

        # Cloud engine
        backend = os.getenv("INFERENCE_BACKEND", "simulatedcloud")
        if backend == "simulatedcloud":
            self.inference = CloudSimulatedEngine()
        else:
            openai_cfg = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL", None),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", str(adv.dynamic_tokens_max_cap))),
            }
            self.inference = CloudOpenAiEngine(openai_cfg)

        # Retrieval pipeline
        self.retrieval = RetrievalPipeline(
            documents=cfg.documents,
            doc_ids=cfg.doc_ids,
            cfg=RetrievalConfig(
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                use_pagerank_boost=adv.use_pagerank_boost,
                pagerank_boost_gamma=adv.pagerank_boost_gamma,
                use_pagerank_prune=adv.use_pagerank_prune,
                pagerank_prune_beta=adv.pagerank_prune_beta,
                pagerank_prune_keep_k=adv.pagerank_prune_keep_k,
                pagerank_prune_score_floor=adv.pagerank_prune_score_floor,
                use_dynamic_index_switch=adv.use_dynamic_index_switch,
                short_query_max_words=adv.short_query_max_words,
                use_essential_keywords=adv.use_essential_keywords,
                essential_keywords_alpha=adv.essential_keywords_alpha,
                bm25_top_k=100,
                dense_top_k=200,
                rerank_top_k=5,
                chunk_limit=adv.chunk_limit,
                score_threshold=adv.score_threshold,
            ),
            graph_edges=cfg.graph_edges,
        )

        # Prompt builder
        use_cloud_prompt = os.getenv("USE_CLOUD_PROMPT_CACHE", "true").lower() == "true"
        conciseness = os.getenv("CONCISENESS_MODE", "false").lower() == "true"
        if use_cloud_prompt:
            self.prompt_builder = CloudPromptBuilder(
                system_prompt=cfg.system_prompt,
                use_cache_control=True,
                conciseness_short_queries=conciseness,
            )
        else:
            self.prompt_builder = MinimalPromptBuilder()

        # Cache
        normalizer = PhraseNormalizer(map_path="phrase_map.json")
        self.cache = PreWarmedCache(normalizer=normalizer)

        # Session storage
        self.sessions: Dict[str, ConversationState] = {}
        # NEW: per‑session memory objects
        self.session_memories: Dict[str, ConversationMemory] = {}

        # Zero‑token / FAQ
        self.zero_token = ZeroTokenResponder() if adv.zero_token_enabled else None
        self.faq = FAQDatabase.from_json(adv.faq_db_path) if adv.faq_enabled and os.path.exists(adv.faq_db_path) else FAQDatabase()

        # Feature flags
        self.adv = adv

        # Additional FRGL heuristic
        self.use_frgl_heuristic = os.getenv("USE_FRGL_HEURISTIC", "false").lower() == "true"
        self.frgl_score_threshold = float(os.getenv("FRGL_SCORE_THRESHOLD", "0.9"))
        self.frgl_max_chunk_tokens = int(os.getenv("FRGL_MAX_CHUNK_TOKENS", "150"))

        # ────── NEW MODULES ──────
        # Neural Bloom Filter
        self.nbf: Optional[UnifiedNeuralBloomFilter] = None
        if adv.use_nbf and UnifiedNeuralBloomFilter is not None:
            self.nbf = UnifiedNeuralBloomFilter(
                weights_path=adv.nbf_weights_path,
                table_path=adv.nbf_table_path,
                m_bits=adv.nbf_m_bits,
                f_bits=adv.nbf_f_bits,
                tau=adv.nbf_tau,
            )

        # Markov Predictor
        self.predictor: Optional[FastPredictor] = None
        if adv.use_markov_predictor and FastPredictor is not None:
            self.predictor = FastPredictor(
                top_lists_path=adv.markov_toplist_path,
                state_index_path=adv.markov_state_index_path,
                unigram_row_path=adv.markov_unigram_path,
                time_decay_path=adv.markov_time_decay_path,
                canonical_strings=[],  # will be loaded inside FastPredictor
            )

        # Conversation Memory compressor
        self.memory_compressor: Optional[Compressor] = None
        if adv.use_conversation_memory:
            self.memory_compressor = Compressor(
                model=adv.memory_compressor_model,
                max_tokens=adv.memory_m2_tokens,
            )

        # Prefetch cache for predicted queries
        self.prefetch_cache: Dict[str, List[Tuple[str, str, float]]] = {}

    def _get_state(self, session_id: str) -> ConversationState:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationState()
        return self.sessions[session_id]

    def _get_memory(self, session_id: str, user_query: str) -> ConversationMemory:
        """Get or create a ConversationMemory for this session."""
        if session_id not in self.session_memories:
            cfg = MemoryConfig(
                W=self.adv.memory_w,
                M1_tokens=self.adv.memory_m1_tokens,
                M2_tokens=self.adv.memory_m2_tokens,
                cache_provider="openai",
            )
            self.session_memories[session_id] = ConversationMemory(
                config=cfg, compressor=self.memory_compressor
            )
        return self.session_memories[session_id]

    async def _predict_and_prefetch(self, session_id: str, last_assistant_response: str):
        """Fire‑and‑forget: predict next queries and pre‑fetch retrieval."""
        if not self.predictor:
            return
        try:
            # Build canonical history from the last 3 user queries (stored in session_state)
            state = self._get_state(session_id)
            # simplest: extract last 3 user turns
            user_turns = [t.content for t in state.history if t.role == "user"][-3:]
            # we need canonical IDs – for now we can use a hash or simple string as placeholder
            # In a real system, you'd keep a mapping of query -> canonical_id.
            # For now, we skip prediction if no canonical IDs exist.
            # We'll implement a basic fallback that uses the predictor with dummy IDs.
            # (The full implementation requires a canonical ID lookup.)
            pass
        except Exception:
            pass

    async def stream_chat(self, req: ChatRequest):
        state = self._get_state(req.session_id)

        # 1. Cache check
        cached = self.cache.get(req.message)
        if cached is not None:
            yield f"data: {json.dumps({'type': 'token', 'value': cached})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # 2. Zero‑token / FAQ
        if self.zero_token:
            canned = self.zero_token.respond(req.message)
            if canned:
                yield f"data: {json.dumps({'type': 'token', 'value': canned})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
        faq_answer = self.faq.get(req.message)
        if faq_answer:
            yield f"data: {json.dumps({'type': 'token', 'value': faq_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # ────── NEW: Neural Bloom Filter interceptor ──────
        if self.nbf is not None:
            # We need an embedding of the query. We can reuse the dense retriever's embedder.
            # For simplicity, get it from the retrieval pipeline.
            emb = self.retrieval.dense.embedder.encode([req.message], normalize_embeddings=True)[0]
            nbf_res = self.nbf.query(emb)
            if nbf_res is not None:
                answer, conf = nbf_res
                yield f"data: {json.dumps({'type': 'token', 'value': answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'source': 'nbf', 'confidence': conf})}\n\n"
                return

        # 3. Retrieval (with score gating)
        retrieval_start = time.perf_counter()
        retrieved = self.retrieval.retrieve(req.message)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # FRGL heuristic
        if (self.use_frgl_heuristic and retrieved and
            retrieved[0][2] >= self.frgl_score_threshold and
            len(retrieved[0][1].split()) <= self.frgl_max_chunk_tokens):
            yield f"data: {json.dumps({'type': 'token', 'value': retrieved[0][1]})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'source': 'frgl_heuristic'})}\n\n"
            return

        # Score gating
        if self.adv.use_score_gating and (not retrieved or retrieved[0][2] < self.adv.score_threshold):
            fallback = "I'm sorry, I don't have information about that."
            yield f"data: {json.dumps({'type': 'token', 'value': fallback})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # 4. Build prompt
        retrieved_chunks = [r[1] for r in retrieved[:self.adv.chunk_limit]] if retrieved else []

        # ────── NEW: use Conversation Memory if enabled ──────
        if self.adv.use_conversation_memory and self.memory_compressor is not None:
            memory = self._get_memory(req.session_id, req.message)
            prompt, cache_params = memory.get_prompt(req.message)
            # (cache_params can be used if the engine supports structured caching)
        else:
            prompt = self.prompt_builder.build(
                history=[],
                user_input=req.message,
                retrieved_chunks=retrieved_chunks,
            )
            if self.adv.one_liner_mode:
                prompt = "Answer in one sentence.\n" + prompt

        # 5. Generate response
        response, new_state, metrics = self.inference.generate(
            state=state,
            user_input=req.message,
            seed=req.seed,
            prompt_override=prompt,
        )
        self.sessions[req.session_id] = new_state

        # ────── NEW: Update memory after generation ──────
        if self.adv.use_conversation_memory and self.memory_compressor is not None:
            memory = self._get_memory(req.session_id, req.message)
            memory.add_turn(req.message, response)

        self.cache.put(req.message, response)

        # 6. Stream tokens (removed artificial delay for speed)
        for token in response.split():
            yield f"data: {json.dumps({'type': 'token', 'value': token + ' '})}\n\n"
        done = {
            "type": "done",
            "metrics": metrics,
            "retrieval_latency_ms": retrieval_latency_ms,
        }
        yield f"data: {json.dumps(done)}\n\n"

        # ────── NEW: Launch prediction task ──────
        if self.adv.use_markov_predictor:
            asyncio.create_task(self._predict_and_prefetch(req.session_id, response))

        self.sessions[req.session_id] = new_state
        self.cache.put(req.message, response)


# ---- FastAPI app ----
app = FastAPI(title="Cloud‑Native Speed Demon Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_app = ChatApp(AppConfig())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(chat_app.stream_chat(req), media_type="text/event-stream")

@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_json()
        req = ChatRequest(**data)
        async for sse_event in chat_app.stream_chat(req):
            payload = sse_event.replace("data: ", "").strip()
            if payload:
                await ws.send_text(payload)