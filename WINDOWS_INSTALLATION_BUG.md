# Windows Installation Bug Found

## Bug Description

When running `pip install -r requirements.txt` on Windows, the installation fails with:

```
error: linker `link.exe` not found
error: could not compile `windows_x86_64_msvc` (build script) due to 1 previous error
Failed to build pydantic-core llama-cpp-python
```

## Root Cause

- `pydantic-core` requires compilation with Rust/C++
- `llama-cpp-python` requires compilation with CMake/C++
- Both require Microsoft C++ Build Tools on Windows
- The README did not mention this prerequisite

## Fix Applied

Updated README.md to add:

1. **Windows Prerequisites section**:
   - Link to Microsoft C++ Build Tools download
   - Instructions to select "Desktop development with C++"
   - Alternative command to use pre-built wheels only: `pip install --only-binary :all: -r requirements.txt`

2. **Changed `python` to `py`**:
   - `py` is more reliable on Windows for Python launcher
   - Added note: "If `py` doesn't work, try `python` instead"

## Impact

**Critical**: This bug blocks all Windows users from installing dependencies and using the chatbot.

## Next Steps

For users without C++ Build Tools:
1. Option A: Install Microsoft C++ Build Tools (recommended)
2. Option B: Use pre-built wheels only: `pip install --only-binary :all: -r requirements.txt`
3. Option C: Use a pre-compiled llama-cpp-python wheel from the [llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases)

## Verification

Commands tested:
- ✅ `py -m venv .venv` - Works
- ✅ `.venv\Scripts\activate` - Works
- ❌ `pip install -r requirements.txt` - Fails without C++ Build Tools
- ✅ `New-Item -ItemType Directory -Path models -ErrorAction SilentlyContinue` - Already exists (expected)
- ✅ `Copy-Item .env.example .env` - Works

## Commit

**Commit**: `b42c4cc` - Fix README - add Windows C++ Build Tools prerequisite

**Pushed**: ✅ Yes
