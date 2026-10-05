# Additional Performance Bugs Found

## Bug #8: Unbounded Cache Growth in speed_engine.cache

**Location**: `speed_engine/cache.py` line 84-100

**Issue**: The `PreWarmedCache` has a `max_entries` limit for `exact` cache, but `simhash_map` has no size limit. Over time, this can grow unbounded and consume memory.

**Impact**: Memory leak over long-running sessions

**Fix**: Add size limit to simhash_map

```python
# Before (line 84-89)
self.exact: OrderedDict[str, dict] = OrderedDict()
self.simhash_map: Dict[int, Tuple[str, str]] = {}
self.access_counts: Dict[str, int] = {}

# After
self.exact: OrderedDict[str, dict] = OrderedDict()
self.simhash_map: Dict[int, Tuple[str, str]] = {}
self.simhash_max_entries = max_entries * 2  # Allow 2x more for fuzzy matches
self.access_counts: Dict[str, int] = {}

# In put() method, add:
if len(self.simhash_map) >= self.simhash_max_entries:
    # Remove oldest entries
    oldest_sh = next(iter(self.simhash_map))
    del self.simhash_map[oldest_sh]
```

---

## Bug #9: Inefficient SimHash Linear Search

**Location**: `speed_engine/cache.py` line 112-115

**Issue**: SimHash lookup does a linear search through all stored hashes: `for stored_sh, (_, resp) in self.simhash_map.items()`. With 50,000 entries, this is O(n) = 50,000 iterations per lookup.

**Impact**: Slow cache lookup as cache grows

**Fix**: Use a more efficient data structure or index

```python
# Option 1: Limit simhash cache size to smaller number
self.simhash_max_entries = 1000  # Keep only 1000 most recent

# Option 2: Use LRU cache for simhash
from collections import OrderedDict
self.simhash_map: OrderedDict[int, Tuple[str, str]] = OrderedDict()

# In get(), after lookup:
self.simhash_map.move_to_end(sh)  # LRU behavior

# In put(), before insert:
if len(self.simhash_map) >= self.simhash_max_entries:
    self.simhash_map.popitem(last=False)
```

---

## Bug #10: No Cache Eviction for RAG Query Cache

**Location**: `local_chatbot/rag.py` line 111

**Issue**: The `_retrieval_cache` in RAGPipeline has no size limit or eviction policy. It will grow unbounded.

**Impact**: Memory leak over time

**Fix**: Add LRU cache with size limit

```python
# Before
self._retrieval_cache: Dict[str, Tuple[str, int, list]] = {}

# After
from collections import OrderedDict
self._retrieval_cache: OrderedDict[str, Tuple[str, int, list]] = OrderedDict()
self._cache_max_entries = 1000

# In build_prompt():
# Check cache
if query in self._retrieval_cache:
    result = self._retrieval_cache[query]
    self._retrieval_cache.move_to_end(query)  # LRU
    return result

# Cache result
self._retrieval_cache[query] = (prompt, max_tokens, sources)
if len(self._retrieval_cache) > self._cache_max_entries:
    self._retrieval_cache.popitem(last=False)  # Remove oldest
```

---

## Bug #11: Document Cache Never Cleared

**Location**: `local_chatbot/rag.py` line 54

**Issue**: The class-level `_document_cache` never gets cleared. If the knowledge base changes, stale documents will be used.

**Impact**: Stale data after knowledge base updates

**Fix**: Add cache invalidation method

```python
@classmethod
def clear_document_cache(cls) -> None:
    """Clear the document cache to force reload."""
    cls._document_cache.clear()

# In _load_documents, add a way to force reload:
def _load_documents(kb_path: str, force_reload: bool = False) -> List[Tuple[str, str]]:
    cache_key = kb_path
    if force_reload and cache_key in RAGPipeline._document_cache:
        del RAGPipeline._document_cache[cache_key]
    ...
```

---

## Bug #12: Inefficient History String Joining

**Location**: `local_chatbot/graph.py` line 104

**Issue**: Every LLM call with history does `"\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])`. This creates new strings and iterates through history.

**Impact**: Slower LLM generation with conversation history

**Fix**: Cache formatted history or use more efficient method

```python
# Add to ConversationMemory class
def get_formatted_history(self, session_id: str) -> str:
    """Get formatted history string (cached)."""
    history = self.get(session_id)
    if not history:
        return ""
    return "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

# In _node_generate:
history = self.memory.get_formatted_history(state.get("session_id", "default"))
if history:
    prompt = f"Previous conversation:\n{history}\n\n{prompt}"
```

---

## Bug #13: No Model Cleanup on Shutdown

**Location**: `local_chatbot/engine.py`

**Issue**: The model is loaded but never explicitly cleaned up when the application shuts down. This can cause issues on some systems.

**Impact**: Potential resource leaks on shutdown

**Fix**: Add cleanup method

```python
def cleanup(self) -> None:
    """Clean up model resources."""
    if self.model:
        try:
            # Release model memory
            del self.model
            self.model = None
        except Exception:
            pass
    self._loaded = False

# Add __del__ for automatic cleanup
def __del__(self):
    self.cleanup()
```

---

## Bug #14: Synchronous Blocking Operations

**Location**: `local_chatbot/engine.py` line 69-75

**Issue**: Model generation is synchronous and blocks the entire thread. For a web server, this blocks all other requests during generation.

**Impact**: Poor concurrency in web server mode

**Fix**: Implement async generation (requires significant refactoring)

```python
# This is a major architectural change
# Would require making LocalEngine async-aware
# Not recommended for current simple CLI use case
# Document as known limitation
```

---

## Bug #15: No Session Timeout

**Location**: `local_chatbot/graph.py` line 32-50

**Issue**: The `ConversationMemory` keeps sessions forever. In a long-running application, this can consume unbounded memory.

**Impact**: Memory leak with many sessions over time

**Fix**: Add session timeout and cleanup

```python
class ConversationMemory:
    def __init__(self, max_turns: int = 20, session_timeout_seconds: int = 3600):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._last_access: Dict[str, float] = {}
        self.max_turns = max_turns
        self.session_timeout = session_timeout_seconds

    def add(self, session_id: str, role: str, content: str) -> None:
        import time
        self._cleanup_expired_sessions()
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        self._last_access[session_id] = time.time()
        if len(self._sessions[session_id]) > self.max_turns * 2:
            self._sessions[session_id] = self._sessions[session_id][-(self.max_turns * 2):]

    def _cleanup_expired_sessions(self) -> None:
        """Remove sessions that haven't been accessed recently."""
        import time
        now = time.time()
        expired = [sid for sid, last in self._last_access.items() if now - last > self.session_timeout]
        for sid in expired:
            self.clear(sid)
```

---

## Bug #16: No Validation of Model Path Before Load

**Location**: `local_chatbot/engine.py` line 33-36

**Issue**: Model path validation happens during `_load()`, but there's no early validation at config time. User finds out about bad path only when they try to generate.

**Impact**: Poor user experience - late error detection

**Fix**: Add validation at config level

```python
# In ChatbotConfig or as a separate function
def validate_model_path(path: str) -> bool:
    """Validate model path exists and is readable."""
    if not os.path.exists(path):
        return False
    if not os.path.isfile(path):
        return False
    if not path.endswith('.gguf'):
        return False
    return True

# In LocalEngine.__init__:
if not validate_model_path(self.cfg.model_path):
    logger.warning(f"Model path validation failed: {self.cfg.model_path}")
    # Could raise error or fall back to simulated mode
```

---

## Priority Recommendations

### High Priority (Memory Leaks)
1. **Bug #8**: Unbounded simhash_map growth
2. **Bug #10**: Unbounded retrieval cache
3. **Bug #15**: No session timeout

### Medium Priority (Performance)
4. **Bug #9**: Inefficient SimHash search
5. **Bug #12**: Inefficient history joining
6. **Bug #11**: Document cache never cleared

### Low Priority (UX/Resource Management)
7. **Bug #13**: No model cleanup
8. **Bug #16**: No early model validation
9. **Bug #14**: Synchronous blocking (architectural limitation)

---

## Expected Impact

After fixing high priority bugs:
- **Memory usage**: Stable over time (prevents leaks)
- **Long-running sessions**: No memory growth
- **Cache performance**: Maintained as cache grows

---

## Implementation Status

- [ ] Bug #8: Unbounded simhash_map
- [ ] Bug #9: Inefficient SimHash search
- [ ] Bug #10: Unbounded retrieval cache
- [ ] Bug #11: Document cache invalidation
- [ ] Bug #12: Inefficient history joining
- [ ] Bug #13: Model cleanup
- [ ] Bug #14: Synchronous blocking (architectural)
- [ ] Bug #15: Session timeout
- [ ] Bug #16: Early model validation

Would you like me to implement these fixes?
