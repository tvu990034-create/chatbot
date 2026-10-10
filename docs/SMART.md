# Smarter methods — every way this stack buys answer quality

Honest status first: trivia wins are brevity, not brilliance. The
mechanisms below target *correctness*; proof status is marked per
item. The run that settles accuracy at scale (100-item quiet-machine
GSM8K eval) is still open.

## 1. Verification pass (proven on drafts)

Short speed drafts below the calibration bar get re-solved by a
bounded verify call charged to the same deadline (per-item worst case
cannot grow). Calibration scaler + deferral rule decide who gets
checked. Balanced (eval) path deliberately does NOT second-guess:
replacing a correct terse verdict with prose flips graded accuracy.

## 2. Full think budgets for reasoning (proven 3/3 vs 0/3 probe)

qwen3 needs ~2000+ hidden tokens on GSM8K: a short box returned
truncated reasoning and scored 0/3, the full budget scored 3/3 (raw
baseline 2/3, capped at 128 tokens). Reasoning queries are exempt
from every small cap in the system (`budget_math` returns no cap).

## 3. Multi-sample consensus (tested, opt-in)

`routing_math.majority_consensus` over CISC samples; LoadBalancer
with dwell + EWMA across endpoints. Consensus paths exist and are
tested; single-sample balanced remains the default eval path.

## 4. Retrieval fusion (live)

Agent RAG prefetches LlamaIndex + Haystack in parallel and fuses
rankings with RRF; Eq8 confidence gates synthesis budgets (low
confidence → skip the LLM call, say so instead of hallucinating).

## 5. Graceful degradation (live, measured)

Optimized → plain retry → raw generator → neutral "couldn't finish"
message. Degraded texts are never cached (a transient failure must
not become a permanent wrong answer); degenerate RAG output falls
through to the direct path with stale sources cleared.

## 6. Online temperature (live)

OGD temperature learns from verify verdicts; Thompson-sampling
adapter over `bandit_math` available.

## 7. What "smarter" is NOT

Shorter answers (trivia brevity), skipped work (routing/cache), or
faster tokens (no tok/s edge on this CPU). Those are speed, honestly
labeled as speed in `docs/BENCHMARKS.md`.
