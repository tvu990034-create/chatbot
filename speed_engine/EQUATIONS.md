# Speed Engine — Equation Index

Implementations follow **`engine/EQUATION_INVENTORY.md`** (full extraction from PDF/TXT specs).

| Module | Equations |
|--------|-----------|
| `prefilter.py` | EQ-ZERO-TOKEN, EQ-FAQ, EQ-BLOOM-FP/OPT-K/DOUBLE-HASH, EQ-BLOOM-PRECISION, EQ-NORMALIZE |
| `cache.py` | EQ-SIMHASH, EQ-PREWARM-CASCADE, EQ-PREWARM-HIT, EQ-SESSION-TTFT, EQ-POPULARITY-EVICT |
| `retrieval.py` | EQ-BM25, EQ-QUERY-TRUNC, EQ-DYNAMIC-INDEX, EQ-DENSE-COSINE, EQ-CROSS-ENCODER, EQ-SCORE-GATE |
| `pagerank.py` | EQ-PAGERANK-STATIONARY, EQ-PPR-PUSH, EQ-PAGERANK-PRUNE |
| `prompt.py` | EQ-MINIMAL-PROMPT, EQ-DYNAMIC-TOKENS, EQ-ONE-LINER, EQ-COST-DELTA, EQ-SINGLE-CHUNK |
| `generation.py` | EQ-BOREDOM, EQ-ROUTER, EQ-SPEC-SINGLE/DUAL, simulated cloud (EQ-TTFB-CACHE) |
| `pipeline.py` | Ordered wiring: prefilter → cache → retrieval → gate → prompt → generate → cache put |

Run server: `uvicorn server_speed:app --port 8001`  
Verify: `python benchmarks/speed_verify.py`
