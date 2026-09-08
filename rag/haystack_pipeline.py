"""
rag/haystack_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~
RAG pipeline built on Haystack 3.x (haystack-ai).

Provides:
* HaystackRAG.build_pipeline()  – assembles index + retrieval pipelines
* HaystackRAG.ingest()          – embed and store documents
* HaystackRAG.query()           – retrieve + generate
* get_haystack_rag()            – module-level singleton

All heavy imports are deferred to avoid slowing down startup when
Haystack is not the active RAG provider.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _get_haystack():
    """Import Haystack components (deferred)."""
    try:
        from haystack import Document, Pipeline
        from haystack.components.builders import PromptBuilder
        from haystack.components.converters import (
            MarkdownToDocument,
            PyPDFToDocument,
            TextFileToDocument,
        )
        from haystack.components.embedders import (
            SentenceTransformersDocumentEmbedder,
            SentenceTransformersTextEmbedder,
        )
        from haystack.components.generators import OpenAIGenerator
        from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
        from haystack.components.writers import DocumentWriter
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        return (
            Document, Pipeline, PromptBuilder,
            MarkdownToDocument, PyPDFToDocument, TextFileToDocument,
            SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder,
            OpenAIGenerator,
            DocumentCleaner, DocumentSplitter,
            InMemoryEmbeddingRetriever, DocumentWriter,
            InMemoryDocumentStore,
        )
    except ImportError as exc:
        raise ImportError(
            "Haystack not installed. Run: pip install haystack-ai sentence-transformers"
        ) from exc


# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE = """
You are a helpful AI assistant. Use the following retrieved documents to answer
the question. If the documents do not contain relevant information, say so
and answer from your general knowledge.

Context:
{% for doc in documents %}
--- [{{ loop.index }}] {{ doc.meta.get('file_path', 'unknown') }} ---
{{ doc.content }}
{% endfor %}

Question: {{ question }}

Answer:
""".strip()


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class HaystackRAG:
    """
    Haystack 3.x RAG pipeline using in-memory document store + sentence-transformers.

    For production use, swap InMemoryDocumentStore for ChromaDocumentStore or
    any of Haystack's 9 supported vector backends.
    """

    def __init__(
        self,
        docs_dir: Path | None = None,
        top_k: int | None = None,
        prompt_template: str = _DEFAULT_TEMPLATE,
    ) -> None:
        self.docs_dir        = docs_dir or settings.rag_docs_dir
        self.top_k           = top_k or settings.rag_top_k
        self.prompt_template = prompt_template
        self._document_store: Any = None
        self._index_pipeline: Any = None
        self._query_pipeline: Any = None
        self._is_built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_pipeline(self) -> None:
        """Assemble both the indexing pipeline and the query pipeline."""
        (
            _, Pipeline, PromptBuilder,
            MarkdownToDocument, PyPDFToDocument, TextFileToDocument,
            SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder,
            OpenAIGenerator,
            DocumentCleaner, DocumentSplitter,
            InMemoryEmbeddingRetriever, DocumentWriter,
            InMemoryDocumentStore,
        ) = _get_haystack()

        # Shared document store
        self._document_store = InMemoryDocumentStore()

        # ---- Indexing pipeline ----------------------------------------
        idx = Pipeline()
        idx.add_component("cleaner",   DocumentCleaner())
        idx.add_component("splitter",  DocumentSplitter(
            split_by="word",
            split_length=settings.rag_chunk_size,
            split_overlap=settings.rag_chunk_overlap,
        ))
        idx.add_component("embedder",  SentenceTransformersDocumentEmbedder(
            model=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        ))
        idx.add_component("writer",    DocumentWriter(document_store=self._document_store))
        idx.connect("cleaner",  "splitter")
        idx.connect("splitter", "embedder")
        idx.connect("embedder", "writer")
        self._index_pipeline = idx

        # ---- Query pipeline -------------------------------------------
        qry = Pipeline()
        qry.add_component("embedder",  SentenceTransformersTextEmbedder(
            model=settings.embedding_model,
            device=settings.embedding_device,
        ))
        qry.add_component("retriever", InMemoryEmbeddingRetriever(
            document_store=self._document_store,
            top_k=self.top_k,
        ))
        qry.add_component("prompt",    PromptBuilder(template=self.prompt_template))
        qry.add_component("llm",       OpenAIGenerator(
            api_base_url=settings.litellm_api_base or "http://localhost:4000",
            model=settings.default_model.split("/")[-1],
            api_key="dummy-key-litellm-handles-auth",
            generation_kwargs={
                "max_tokens": settings.litellm_max_tokens,
                "temperature": settings.litellm_temperature,
            },
        ))
        qry.connect("embedder.embedding", "retriever.query_embedding")
        qry.connect("retriever.documents", "prompt.documents")
        qry.connect("prompt",              "llm")
        self._query_pipeline = qry

        self._is_built = True
        logger.info("Haystack pipelines built (doc_store=%s)", type(self._document_store).__name__)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, file_paths: list[str | Path] | None = None) -> int:
        """
        Ingest files from file_paths (or the configured docs_dir if None).

        Returns the number of documents stored.
        """
        if not self._is_built:
            self.build_pipeline()

        (
            Document, _, _, _, _, _, _, _, _, _, _, _, _, _,
        ) = _get_haystack()[:14]

        paths: list[Path] = []
        if file_paths:
            paths = [Path(p) for p in file_paths]
        else:
            docs_path = Path(self.docs_dir)
            if docs_path.exists():
                paths = list(docs_path.rglob("*"))
                paths = [p for p in paths if p.is_file()]

        if not paths:
            logger.warning("No files to ingest from %s", self.docs_dir)
            return 0

        # Build plain Document objects from text files
        documents: list[Any] = []
        for p in paths:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                documents.append(Document(content=content, meta={"file_path": str(p)}))
            except Exception as exc:
                logger.warning("Skipping %s: %s", p, exc)

        if not documents:
            return 0

        self._index_pipeline.run({"cleaner": {"documents": documents}})
        count = self._document_store.count_documents()
        logger.info("Haystack document store now has %d document(s).", count)
        return count

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> dict[str, Any]:
        """
        Retrieval-only: embed the query and fetch top-k documents WITHOUT
        calling the LLM generator (OpenAIGenerator).

        Used by the agent's RAG prefetch so a single request does not spawn an
        extra full LLM generation just to feed context to the main agent call.

        Returns dict with keys: answer (""), chunks (list[str]), sources (list[str])
        """
        if not self._is_built:
            self.build_pipeline()
            self.ingest()

        embedder  = self._query_pipeline.get_component("embedder")
        retriever = self._query_pipeline.get_component("retriever")

        emb_out   = embedder.run({"text": question})
        query_emb = emb_out["embedding"]
        docs      = retriever.run({"query_embedding": query_emb})["documents"]

        return {
            "answer": "",
            "chunks": [d.content for d in docs],
            "sources": [d.meta.get("file_path", "unknown") for d in docs],
        }

    def query(self, question: str) -> dict[str, Any]:
        """
        Retrieve relevant context and generate an answer.

        Returns dict with keys: answer (str), sources (list[str])
        """
        if not self._is_built:
            self.build_pipeline()
            self.ingest()

        result = self._query_pipeline.run({
            "embedder": {"text": question},
            "prompt":   {"question": question},
        })

        replies   = result.get("llm", {}).get("replies", [""])
        answer    = replies[0] if replies else ""
        docs      = result.get("retriever", {}).get("documents", [])
        sources   = [d.meta.get("file_path", "unknown") for d in docs]
        return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_haystack_instance: HaystackRAG | None = None


def get_haystack_rag() -> HaystackRAG:
    """Return (and lazily initialise) the module-level Haystack RAG instance."""
    global _haystack_instance
    if _haystack_instance is None:
        _haystack_instance = HaystackRAG()
        _haystack_instance.build_pipeline()
        _haystack_instance.ingest()
    return _haystack_instance
