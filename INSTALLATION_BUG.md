# Installation Bug Report

## Summary

Found **installation bug** where scikit-learn cannot be installed on Windows without Microsoft C++ Build Tools. This package is not needed for the local chatbot.

---

## Installation Bug #55: Unnecessary scikit-learn Dependency

**Location**: `requirements.txt` line 13

**Issue**: `scikit-learn==1.3.2` is listed in requirements.txt but:
- Not used by `local_chatbot/` module (verified with grep)
- Only used by legacy cloud app (`app/retrieval.py`, `retrieval.py`)
- Requires Microsoft C++ Build Tools to compile on Windows
- Blocks installation for Windows users without build tools

**Impact**: Users cannot install dependencies on Windows without installing Microsoft C++ Build Tools first

**Current State**:
- Local chatbot does NOT import or use scikit-learn
- Cloud app (legacy) uses scikit-learn but is not the primary application
- Installation fails with: "Microsoft Visual C++ 14.0 or greater is required"

**Fix**: Remove scikit-learn from requirements.txt (local chatbot doesn't need it)

---

## Other Potentially Unused Dependencies

The following dependencies are likely only for the legacy cloud app and should be reviewed:

- `python-jose[cryptography]` - JWT for cloud app authentication
- `passlib[bcrypt]` - Password hashing for cloud app
- `sqlalchemy` - Database ORM for cloud app
- `aiosqlite` - Async SQLite for cloud app
- `openai` - OpenAI client for cloud app
- `kaggle` - Kaggle API (likely unused)

These should be removed or moved to a separate `requirements-cloud.txt` if needed for the legacy app.

---

## Recommendation

Create two requirement files:
1. `requirements.txt` - For local chatbot (core dependencies only)
2. `requirements-cloud.txt` - For legacy cloud app (all dependencies including scikit-learn)

This will:
- Make local chatbot installation easier
- Reduce dependencies for local users
- Keep cloud app dependencies available if needed
- Follow best practices for multi-application repositories
