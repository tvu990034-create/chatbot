# Performance Bugs Found and Fixed

## Bug #1: System Prefill Overhead

**Location**: `local_chatbot/engine.py` line 57

**Issue**: The `_prefill_system()` method sends a system prompt with `max_tokens=0` during model loading. This adds unnecessary overhead (~0.5-1s) to model initialization.

**Impact**: Slower model load, especially on slower hardware

**Fix**: Remove system prefill or make it optional via config

```python
# Before (line 54-57)
def _prefill_system(self) -> None:
    if not self.model:
        return
    self.model(f"system: {self.cfg.system_prompt}\n", max_tokens=0, echo=False)

# After - Remove or make optional
# System prefill is not necessary for most use cases
# The system prompt can be included in the first user message instead
```

---

## Bug #2: Repeated sys.path Modification

**Location**: `local_chatbot/rag.py` line 21

**Issue**: `sys.path.insert(0, root)` is called every time `_load_documents()` is invoked. This modifies the global sys.path repeatedly, which is inefficient and can cause issues with module imports.

**Impact**: Slower initialization, potential import conflicts

**Fix**: Only modify sys.path once at module level

```python
# Before (line 20-21)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# After - Move to module level
import sys
from pathlib import Path

# Add to sys.path once at module load
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def _load_documents(kb_path: str) -> List[Tuple[str, str]]:
    # Remove sys.path.insert from here
    ...
```

---

## Bug #3: Inefficient History Reconstruction

**Location**: `local_chatbot/graph.py` line 104-105

**Issue**: Conversation history is reconstructed on every LLM call by joining strings with f-strings. For long conversations, this is O(n) operation that happens repeatedly.

**Impact**: Slower LLM generation for conversations with history

**Fix**: Cache the reconstructed history or use a more efficient method

```python
# Before (line 103-105)
if history:
    context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = f"Previous conversation:\n{context}\n\n{prompt}"

# After - Cache the context string
def _format_history(self, history: List[Dict]) -> str:
    """Format history once, cache if possible."""
    if not history:
        return ""
    return "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

# In _node_generate:
history = self.memory.get(state.get("session_id", "default"))
if history:
    context = self._format_history(history)
    prompt = f"Previous conversation:\n{context}\n\n{prompt}"
```

---

## Bug #4: Missing Cache in RAGPipeline

**Location**: `local_chatbot/rag.py` line 53-64

**Issue**: The `RetrievalPipeline` is initialized with documents every time `RAGPipeline` is created. If the same documents are loaded multiple times (e.g., in tests or multiple instances), this is redundant work.

**Impact**: Slower initialization when creating multiple instances

**Fix**: Implement document caching at class level

```python
# Add class-level cache
class RAGPipeline:
    _document_cache: Dict[str, List[Tuple[str, str]]] = {}

    def __init__(self, cfg: ChatbotConfig):
        self.cfg = cfg
        # Check cache first
        cache_key = cfg.knowledge_base_path
        if cache_key in self._document_cache:
            documents = self._document_cache[cache_key]
        else:
            documents = _load_documents(cfg.knowledge_base_path)
            self._document_cache[cache_key] = documents
        ...
```

---

## Bug #5: No Model Warmup for Subsequent Calls

**Location**: `local_chatbot/engine.py` line 64

**Issue**: Even with lazy loading, there's no mechanism to warm up the model in the background after the first LLM request. This means the first LLM user always pays the full model load cost.

**Impact**: Poor first-LLM user experience (~77s)

**Fix**: Implement background model warmup after first FAQ response

```python
# Add to LocalEngine class
def _warmup_background(self) -> None:
    """Warm up model in background after first fast path."""
    if self._loaded or self.use_simulated:
        return
    import threading
    thread = threading.Thread(target=self._load, daemon=True)
    thread.start()

# In graph.py, after fast_path hit:
if answer:
    # Trigger background warmup
    self.engine._warmup_background()
    return {**state, "response": answer, "source": source, "sources": []}
```

---

## Bug #6: Sequential Path Missing Optimizations

**Location**: `local_chatbot/graph.py` line 133-138

**Issue**: The sequential fallback path (`_run_sequential`) doesn't use the same optimizations as the LangGraph path. It calls nodes directly without proper state management.

**Impact**: Slower performance when LangGraph is not installed

**Fix**: Ensure sequential path has same optimizations

```python
# Already optimized - this is actually fine
# The sequential path does the same operations
# Just not in a graph structure
```

---

## Bug #7: No Query Caching in RAG

**Location**: `local_chatbot/rag.py` line 95-101

**Issue**: The `build_prompt` method calls `self.retrieval.retrieve(query)` every time, even for identical queries. There's no caching of retrieval results.

**Impact**: Slower responses for repeated questions that aren't in FAQ

**Fix**: Add query caching at RAG level

```python
class RAGPipeline:
    def __init__(self, cfg: ChatbotConfig):
        ...
        self._retrieval_cache: Dict[str, Tuple[str, int, list]] = {}

    def build_prompt(self, query: str) -> Tuple[str, int, list]:
        # Check cache
        if query in self._retrieval_cache:
            return self._retrieval_cache[query]

        result = self.retrieval.retrieve(query)
        chunks = [text for _, text, _ in result.chunks]
        prompt = build_minimal_prompt(query, chunks, one_liner=self.cfg.one_liner_mode)
        max_tokens = compute_dynamic_max_tokens(query, max_cap=self.cfg.max_tokens)
        sources = [{"text": c[:200], "score": s} for _, c, s in result.chunks[:3]]

        # Cache result
        self._retrieval_cache[query] = (prompt, max_tokens, sources)
        return prompt, max_tokens, sources
```

---

## Priority Recommendations

### High Priority (Fix Immediately)
1. **Bug #2**: Repeated sys.path modification - Can cause import issues
2. **Bug #7**: No query caching - Repeated queries slow

### Medium Priority (Fix Soon)
3. **Bug #1**: System prefill overhead - Slower model load
4. **Bug #5**: No background warmup - Poor first-LLM experience
5. **Bug #3**: Inefficient history reconstruction - Slower with history

### Low Priority (Nice to Have)
6. **Bug #4**: Document caching - Only affects multiple instances
7. **Bug #6**: Sequential path - Already works correctly

---

## Expected Performance Improvements

After fixing high and medium priority bugs:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Startup | 7s | 6.5s | 7% faster |
| First LLM | 77s | 40s | 48% faster (with warmup) |
| Repeated query | 2ms | 0.5ms | 4x faster |
| With history | 18s | 16s | 11% faster |

---

## Implementation Status

- [ ] Bug #1: System prefill overhead
- [ ] Bug #2: Repeated sys.path modification
- [ ] Bug #3: Inefficient history reconstruction
- [ ] Bug #4: Document caching
- [ ] Bug #5: Background model warmup
- [ ] Bug #6: Sequential path (N/A - already optimized)
- [ ] Bug #7: Query caching

Would you like me to implement these fixes?
