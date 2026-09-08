"""
rag/llama_index_rag.py
~~~~~~~~~~~~~~~~~~~~~~
RAG pipeline built on LlamaIndex (llama-index-core).

Responsibilities
----------------
* Ingest documents from the configured docs directory.
* Build / load a persistent vector index backed by ChromaDB.
* Expose query() and aquery() for retrieval-augmented generation.
* Expose add_documents() to ingest new files at runtime.

This uses the installed llama-index-core package, not the source
folder in AI-Chatbot/llama_index directly.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

# Optional advanced optimizations
_koopman_mixer = None
try:
    from gateway.advanced_optimizations import get_koopman_mixer
    _koopman_available = True
except ImportError as exc:
    logger.warning(
        "Koopman mixer not available (ImportError: %s). "
        "Advanced optimizations disabled.", exc
    )
    _koopman_available = False
except Exception as exc:
    logger.error(
        "Unexpected error loading Koopman mixer: %s. "
        "Advanced optimizations disabled.", exc
    )
    _koopman_available = False

# ---------------------------------------------------------------------------
# Lazy imports – these are heavy; only load when the module is first used.
# ---------------------------------------------------------------------------

def _get_dependencies():
    """Import LlamaIndex + ChromaDB components (deferred to avoid slow startup)."""
    try:
        import chromadb
        from llama_index.core import (
            Settings as LISettings,
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
        )
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.litellm import LiteLLM
        from llama_index.vector_stores.chroma import ChromaVectorStore
        return (
            chromadb, LISettings, SimpleDirectoryReader,
            StorageContext, VectorStoreIndex,
            SentenceSplitter, HuggingFaceEmbedding,
            LiteLLM, ChromaVectorStore,
        )
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex or ChromaDB not installed. "
            "Run: pip install llama-index-core llama-index-embeddings-huggingface "
            "llama-index-vector-stores-chroma llama-index-llms-litellm chromadb"
        ) from exc


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class LlamaIndexRAG:
    """
    LlamaIndex-powered RAG pipeline.

    Parameters
    ----------
    docs_dir:         Directory to ingest documents from.
    persist_dir:      Where to store the ChromaDB collection.
    collection_name:  ChromaDB collection name.
    top_k:            Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        docs_dir: Path | None = None,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.docs_dir         = docs_dir or settings.rag_docs_dir
        self.persist_dir      = persist_dir or settings.chroma_persist_dir
        self.collection_name  = collection_name or settings.rag_collection_name
        self.top_k            = top_k or settings.rag_top_k
        self._index: Any      = None
        self._query_engine: Any = None
        # Serializes mutation of the shared query-engine llm kwargs so
        # concurrent aquery()/query() calls cannot overwrite each other's budget.
        self._engine_kwargs_lock = threading.Lock()
        self._engine_kwargs_async_lock: Any = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _configure_llama_settings(self) -> None:
        """Wire LiteLLM and HuggingFace embeddings into LlamaIndex globals."""
        (
            _, LISettings, _, _, _,
            _, HuggingFaceEmbedding,
            LiteLLM, _,
        ) = _get_dependencies()

        from config import get_litellm_model

        LISettings.llm = LiteLLM(
            model=get_litellm_model(),
            max_tokens=settings.litellm_max_tokens,
            temperature=settings.litellm_temperature,
        )
        LISettings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        LISettings.chunk_size    = settings.rag_chunk_size
        LISettings.chunk_overlap = settings.rag_chunk_overlap
        logger.info("LlamaIndex configured with model=%s embed=%s",
                    get_litellm_model(), settings.embedding_model)

    def _get_chroma_vector_store(self):
        """Create or open a persistent ChromaDB collection."""
        (
            chromadb, _, _, _, _,
            _, _, _, ChromaVectorStore,
        ) = _get_dependencies()

        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = client.get_or_create_collection(self.collection_name)
        return ChromaVectorStore(chroma_collection=collection)

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build (or load) the vector index.
        If the ChromaDB collection already has documents and force_rebuild
        is False, just load the existing index.
        """
        (
            _, _, SimpleDirectoryReader,
            StorageContext, VectorStoreIndex,
            SentenceSplitter, _, _, _,
        ) = _get_dependencies()

        self._configure_llama_settings()
        vector_store  = self._get_chroma_vector_store()
        storage_ctx   = StorageContext.from_defaults(vector_store=vector_store)

        # Check if collection already has data
        chroma_collection = vector_store.client.get_collection(self.collection_name) \
            if hasattr(vector_store, "client") else None
        has_data = (
            chroma_collection is not None
            and chroma_collection.count() > 0
        )

        if has_data and not force_rebuild:
            logger.info("Loading existing LlamaIndex index from ChromaDB …")
            self._index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_ctx
            )
        else:
            logger.info("Building LlamaIndex index from %s …", self.docs_dir)
            docs_path = Path(self.docs_dir)
            
            # Check if docs directory exists and has files
            if not docs_path.exists():
                logger.warning("docs_dir %s does not exist – index will be empty.", self.docs_dir)
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store, storage_context=storage_ctx
                )
            elif not any(docs_path.iterdir()):
                logger.warning("docs_dir %s is empty – index will be empty.", self.docs_dir)
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store, storage_context=storage_ctx
                )
            else:
                documents = SimpleDirectoryReader(str(self.docs_dir), recursive=True).load_data()
                splitter  = SentenceSplitter(
                    chunk_size=settings.rag_chunk_size,
                    chunk_overlap=settings.rag_chunk_overlap,
                )
                self._index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_ctx,
                    transformations=[splitter],
                    show_progress=True,
                )
                logger.info("Indexed %d document(s).", len(documents))

        self._query_engine = self._index.as_query_engine(similarity_top_k=self.top_k)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _eq8_gate(self, source_nodes: list) -> tuple[float, int | None]:
        """
        Eq8 – Confidence-Gated Token Budget.

        Derives a single confidence score from the top retrieved node's
        similarity score, then maps it to a max_tokens budget via:

            M(c) = M_min + (M_max - M_min) * (1 - e^(-θ·c))

        Returns (confidence, max_tokens_budget).
        max_tokens_budget is None when confidence < gate_threshold
        (hard gate → skip LLM call entirely).
        """
        from gateway.perf_math import eq8_confidence_to_max_tokens

        if not source_nodes:
            # No retrieved context → treat as zero confidence
            confidence = 0.0
        else:
            # Average the top-k node scores, clamp to [0, 1]
            scores = [
                float(getattr(n, "score", 0.0) or 0.0)
                for n in source_nodes
            ]
            raw_avg = sum(scores) / len(scores)
            # Similarity scores from ChromaDB are often cosine in [-1,1];
            # normalise to [0,1] so Eq8 thresholds are meaningful.
            confidence = max(0.0, min(1.0, (raw_avg + 1.0) / 2.0))

        budget = eq8_confidence_to_max_tokens(
            score=confidence,
            m_min=getattr(settings, "eq8_m_min", 20),
            m_max=getattr(settings, "eq8_m_max", 256),
            theta=getattr(settings, "eq8_theta", 5.0),
            gate_threshold=getattr(settings, "eq8_gate_threshold", 0.45),
        )
        logger.debug("Eq8 confidence=%.3f → max_tokens=%s", confidence, budget)
        return confidence, budget

    def retrieve(self, question: str) -> dict[str, Any]:
        """
        Retrieval-only: fetch top-k chunks WITHOUT running the LLM.

        Used by the agent's RAG prefetch so a single request does not spawn an
        extra full LLM generation just to feed context to the main agent call.

        Returns dict with keys: answer (""), chunks (list[str]),
                                sources (list[str]), retrieval_confidence (float)
        """
        if self._query_engine is None:
            self.build_index()

        retriever = self._index.as_retriever(similarity_top_k=self.top_k)
        source_nodes = retriever.retrieve(question)
        confidence, budget = self._eq8_gate(source_nodes)

        return {
            "answer": "",
            "chunks": [n.get_content() for n in source_nodes],
            "sources": [
                n.metadata.get("file_name", "unknown") for n in source_nodes
            ],
            "retrieval_confidence": round(confidence, 4),
        }

    def query(self, question: str) -> dict[str, Any]:
        """
        Retrieve context and generate an answer synchronously.

        Eq8 is applied after retrieval: the top-node similarity score is
        mapped to a dynamic token budget.  If confidence is below the hard
        gate the LLM call is skipped and a low-confidence notice is returned.

        Koopman-inspired mixing can be enabled to blend multiple RAG results.

        Returns
        -------
        dict with keys: answer (str), sources (list[str]),
                        retrieval_confidence (float), eq8_max_tokens (int|None)
        """
        if self._query_engine is None:
            self.build_index()

        # --- retrieval only (no synthesis yet) ---
        retriever = self._index.as_retriever(similarity_top_k=self.top_k)
        source_nodes = retriever.retrieve(question)

        confidence, budget = self._eq8_gate(source_nodes)
        
        # NOTE: Koopman-inspired mixing could be applied here for multiple RAG providers
        # if _koopman_available and settings.koopman_mixing_enabled:
        #     mixer = get_koopman_mixer()
        #     # Would mix results from multiple RAG sources

        if budget is None:
            # Eq8 hard gate: confidence too low to justify a full-generation
            # budget.  If we have any retrieved nodes, still answer with the
            # minimum budget instead of refusing outright (the LLM can decline
            # or answer briefly).  Only refuse when retrieval returned nothing.
            if not source_nodes:
                logger.info(
                    "Eq8 hard gate: no context retrieved (confidence=%.3f) – skipping LLM.",
                    confidence,
                )
                return {
                    "answer": (
                        "I could not find sufficiently relevant context to answer "
                        "confidently. Please try rephrasing your question or uploading "
                        "more relevant documents."
                    ),
                    "sources": [],
                    "retrieval_confidence": round(confidence, 4),
                    "eq8_max_tokens": None,
                }
            logger.info(
                "Eq8 low confidence (%.3f) – answering with minimum budget.", confidence)
            budget = int(getattr(settings, "eq8_m_min", 20))

        # Temporarily override max_tokens for this call.  Guarded by a lock and
        # restored in a finally block so exceptions/concurrency can never leave
        # a foreign budget in the shared engine kwargs.
        with self._engine_kwargs_lock:
            llm_kwargs = getattr(self._query_engine, "_llm_kwargs", None)
            original_max = llm_kwargs.get("max_tokens", None) if llm_kwargs else None
            had_key = bool(llm_kwargs and "max_tokens" in llm_kwargs)
            try:
                if llm_kwargs is not None:
                    llm_kwargs["max_tokens"] = budget
                response = self._query_engine.query(question)
            except Exception:
                raise
            finally:
                if llm_kwargs is not None:
                    if had_key:
                        llm_kwargs["max_tokens"] = original_max
                    else:
                        llm_kwargs.pop("max_tokens", None)

        sources = [
            node.metadata.get("file_name", "unknown")
            for node in getattr(response, "source_nodes", source_nodes)
        ]
        return {
            "answer": str(response),
            "sources": sources,
            "retrieval_confidence": round(confidence, 4),
            "eq8_max_tokens": budget,
        }

    async def aquery(self, question: str) -> dict[str, Any]:
        """
        Async variant of query() with Eq8 confidence-gated token budget.
        """
        if self._query_engine is None:
            self.build_index()

        import asyncio

        # Async retrieval
        retriever = self._index.as_retriever(similarity_top_k=self.top_k)
        source_nodes = await asyncio.get_running_loop().run_in_executor(
            None, lambda: retriever.retrieve(question)
        )

        confidence, budget = self._eq8_gate(source_nodes)

        if budget is None:
            logger.info("Eq8 low confidence async (%.3f) – answering with minimum budget.", confidence)
            source_nodes = source_nodes or []
            if not source_nodes:
                return {
                    "answer": (
                        "I could not find sufficiently relevant context to answer "
                        "confidently. Please try rephrasing your question or uploading "
                        "more relevant documents."
                    ),
                    "sources": [],
                    "retrieval_confidence": round(confidence, 4),
                    "eq8_max_tokens": None,
                }
            budget = int(getattr(settings, "eq8_m_min", 20))

        if self._engine_kwargs_async_lock is None:
            self._engine_kwargs_async_lock = asyncio.Lock()
        lock = self._engine_kwargs_async_lock
        async with lock:
            llm_kwargs = getattr(self._query_engine, "_llm_kwargs", None)
            original_max = llm_kwargs.get("max_tokens", None) if llm_kwargs else None
            had_key = bool(llm_kwargs and "max_tokens" in llm_kwargs)
            try:
                if llm_kwargs is not None:
                    llm_kwargs["max_tokens"] = budget
                response = await self._query_engine.aquery(question)
            except Exception:
                raise
            finally:
                if llm_kwargs is not None:
                    if had_key:
                        llm_kwargs["max_tokens"] = original_max
                    else:
                        llm_kwargs.pop("max_tokens", None)
        sources = [
            node.metadata.get("file_name", "unknown")
            for node in getattr(response, "source_nodes", source_nodes)
        ]
        return {
            "answer": str(response),
            "sources": sources,
            "retrieval_confidence": round(confidence, 4),
            "eq8_max_tokens": budget,
        }
        sources = [
            node.metadata.get("file_name", "unknown")
            for node in getattr(response, "source_nodes", source_nodes)
        ]
        return {
            "answer": str(response),
            "sources": sources,
            "retrieval_confidence": round(confidence, 4),
            "eq8_max_tokens": budget,
        }

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_documents(self, file_paths: list[str | Path]) -> int:
        """
        Ingest additional files into the existing index.

        Returns the number of new nodes added.
        """
        (
            _, _, SimpleDirectoryReader,
            _, _,
            SentenceSplitter, _, _, _,
        ) = _get_dependencies()

        if self._index is None:
            self.build_index()

        str_paths = [str(p) for p in file_paths]
        documents = SimpleDirectoryReader(input_files=str_paths).load_data()
        splitter  = SentenceSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(documents)
        self._index.insert_nodes(nodes)
        self._query_engine = self._index.as_query_engine(similarity_top_k=self.top_k)
        logger.info("Added %d node(s) from %d file(s).", len(nodes), len(file_paths))
        return len(nodes)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_rag_instance: LlamaIndexRAG | None = None


def get_rag() -> LlamaIndexRAG:
    """Return (and lazily initialise) the module-level RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LlamaIndexRAG()
        _rag_instance.build_index()
    return _rag_instance
