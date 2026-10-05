# Magic Number Bugs

## Summary

Found **3 magic number issues** that should be replaced with named constants for better maintainability.

**Total**: 3 bugs
**Fixed**: 2 bugs
**Documented Only**: 1 bug

## Findings

### Magic Number Bug #44: Hardcoded History Window Size ✅ FIXED

**Location**: `local_chatbot/graph.py` line 66, `local_chatbot/optimized_graph.py` line 240

**Issue**: The code used `history[-6:]` to get the last 6 messages. This magic number `6` appeared in multiple places and had no clear semantic meaning.

**Impact**: Hard to maintain, unclear why 6 messages were chosen

**Fix**: Added `HISTORY_WINDOW_SIZE = 6` constant in both files and replaced hardcoded value.

---

### Magic Number Bug #45: Magic Number in Slice ⏸️ NOT FIXED

**Location**: `local_chatbot/static/index.html` line 217

**Issue**: The code uses `.slice(6)` to remove "data: " prefix. This is a magic number equal to the length of "data: ".

**Impact**: Brittle - if the prefix changes, this breaks

**Status**: In static HTML file, out of scope for local_chatbot fixes. Could be fixed but is low priority.

---

### Magic Number Bug #46: Magic Numbers in config.py ✅ FIXED

**Location**: `local_chatbot/config.py` lines 20, 23, 24, 25, 26

**Issue**: Several magic numbers were used as defaults:
- `2048` for n_ctx
- `-1` for n_gpu_layers
- `512` for n_batch
- `0.1` for temperature
- `512` for max_tokens
- `3` for chunk_limit

**Impact**: Unclear why these values were chosen, hard to tune

**Fix**: Added named constants with clear documentation:
- `DEFAULT_N_CTX = 2048`
- `DEFAULT_N_GPU_LAYERS = -1` (with comment explaining -1 means use all GPU layers)
- `DEFAULT_N_BATCH = 512`
- `DEFAULT_TEMPERATURE = 0.1`
- `DEFAULT_MAX_TOKENS = 512`
- `DEFAULT_CHUNK_LIMIT = 3`
- `DEFAULT_HOST = "0.0.0.0"`
- `DEFAULT_PORT = 8000`

Also added `DEFAULT_MAX_TURNS = 20` and `DEFAULT_SESSION_TIMEOUT = 3600` in ConversationMemory.

---

## Priority Recommendations

### Low Priority (Code Quality)
1. **Magic Number Bug #44**: History window size constant ✅ FIXED
2. **Magic Number Bug #45**: Data prefix constant ⏸️ (static HTML, low priority)
3. **Magic Number Bug #46**: Config defaults ✅ FIXED

---

## Implementation Status

- [x] Magic Number Bug #44: Add HISTORY_WINDOW_SIZE constant ✅
- [x] Magic Number Bug #45: Add DATA_PREFIX constant ⏸️ (not fixed - static HTML)
- [x] Magic Number Bug #46: Add config default constants ✅

---

## Expected Impact

After fixing magic numbers:
- More maintainable code
- Clearer intent
- Easier to tune parameters

---

## Files Modified

- `local_chatbot/graph.py` - Added HISTORY_WINDOW_SIZE, DEFAULT_MAX_TURNS, DEFAULT_SESSION_TIMEOUT constants
- `local_chatbot/optimized_graph.py` - Added HISTORY_WINDOW_SIZE constant
- `local_chatbot/config.py` - Added DEFAULT_* constants for all config values
