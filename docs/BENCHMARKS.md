# Measured benchmarks — optimized vs raw baseline

All runs on a CPU-only box (Ollama `127.0.0.1:11434`, qwen3:4b unless
noted). Wall-clock times swing with backend load; ratios are
directional. Per-question texts for the head-to-head rows below.

## User experience vs baseline (same questions)

| Question | Optimized | Raw baseline |
|---|---|---|
| Hello there! (instant) | 0 ms | 22.6–29.1 s |
| What is 12*8? (instant) | 0–1 ms | 44.1–47.3 s |
| 12*8 (direct) | 6.6 s — "The answer is 96" | 47.3 s — "The product of 12 and 8 is calculated as follows:…" |
| And 13*13? (agent + history) | 0.1 s — "169" | 33.0 s — "13 multiplied by 13 equals **169**. Here's the quick calculation…" |
| Hello there! (stream, first bytes) | 5.9 s | 27.3 s — greeting |
| Capital of France? | 12 s — "Paris." | 159.4 s — full Paris essay (431 chars) |
| Transformer in one sentence | 42.5–48.5 s | 51.7–83.0 s |
| Reverse-string function | 83.5 s | 116.2 s |

## Cross-run comparison (5 curated questions, `bench --n 5`)

Fair protocol: dual warmup (cold model load never measured), alternating
run order per row (no warm-model bias for either path), think-off parity
on thinking models, per-row error isolation.

| Question | Balanced (calm) | Balanced (loaded) | Speed |
|---|---|---|---|
| Hello | 0.00x | 0.00x | 0.00x |
| 12*8 | 0.00x | err (raw timeout) | 0.00x |
| Transformer | 0.58x | err | 0.82x |
| Reverse fn | err (raw timeout) | err | 0.72x |
| Capital | 0.30x | 0.11x | 0.94x |

`0.00x` rows are routing (no model call), not generation speed.

## Throughput honesty (length-normalized)

`bench` prints ms/char alongside ratios: ms alone cannot tell a faster
answer from a shorter one. Measured: 192–287 ms/char optimized vs
151–355 ms/char raw — **no systematic throughput edge either way**.
Wall-clock wins come from skipped work (routing, cache) and brevity,
not faster token generation. CPU tok/s is physics; the honest levers
are calling the model less and picking the smaller model sooner.

## Accuracy (GSM8K, n=5 — anecdote scale)

| Path | Answered | Correct | Accuracy |
|---|---|---|---|
| Optimized | 5 | 4 | 0.800 |
| Raw | 4 | 4 | 1.000 (1 timeout) |

A 3-question probe on reasoning budgets: short box 0/3, full budget
3/3, raw 2/3. The 100-item quiet-machine eval is still open — that is
the run that settles the accuracy story.

## Environment responsiveness (user-perceived)

CLI instant answers ~1 s · factual chat 3–15 s · REPL two questions
~2 s · API boot ~2 s · Gradio UI ~22 s · agent explainers minutes
(CPU-bound) · server stream/greeting and multiturn follow-ups now
instant via gateway-first ordering and follow-up-math normalization.
