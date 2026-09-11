"""
optimizer_cli.py – make the local AI faster and smarter right from your terminal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Turbo is a CLI front-end that routes every prompt through the optimizer stack
(UniversalEnhancedGateway) so chats are faster (RESPONSE cache, semantic cache,
speed-mode budgets, complexity routing, prefix matching) and smarter (planning,
self-consistency, tool use, controlled decoding) without you changing anything.

It is also wired into the main CLI as the `turbo` command group:

  python main.py turbo chat "create a logo"        one-shot via the optimizer
  python main.py turbo chat "2 + 2" --baseline      compare vs. the raw path
  python main.py turbo run                          interactive REPL
  python main.py turbo check                        verify every optimization module
  python main.py turbo stats                        live optimizer telemetry
  python main.py turbo bench                        optimized vs. baseline timing
  python main.py turbo eval                         HF benchmark: optimized vs. raw
  python main.py turbo --help

It also runs standalone:

  python optimizer_cli.py chat "hello"
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
DEFAULT_MODE = "speed"
EVAL_MODE = "speed"
MAX_TURNS = 12
_CACHE_DIR = Path(__file__).resolve().parent / ".hf_cache"

logging.getLogger().setLevel(logging.WARNING)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    name="turbo",
    help="Make the local AI (Ollama) faster and smarter via the optimizer stack.",
    add_completion=False,
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]TURBO[/bold cyan] - faster + smarter local AI via the optimizer stack",
        border_style="cyan",
    ))


def _quiet_library_logging() -> None:
    for name in ("gateway", "LiteLLM", "urllib3", "httpx", "httpcore",
                 "huggingface_hub", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _resolve_model(model: Optional[str]) -> str:
    if model:
        return model
    from config import settings
    return getattr(settings, "local_model_name", None) or settings.default_model


def _trim_context(messages: List[Dict[str, str]],
                  max_turns: int = MAX_TURNS) -> bool:
    """Drop the oldest non-system turns once history exceeds max_turns.

    Returns True if messages were trimmed (so callers can surface one notice).
    """
    system_msgs = 0
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs += 1
        else:
            break
    history = messages[system_msgs:]
    budget = max(2, max_turns * 2)
    if len(history) > budget:
        del messages[system_msgs:system_msgs + (len(history) - budget)]
        return True
    return False


def make_optimized_gateway(model: Optional[str] = None, mode: str = DEFAULT_MODE):
    from gateway.universal_enhanced_gateway import get_universal_gateway
    _quiet_library_logging()
    return get_universal_gateway(model_name=_resolve_model(model),
                                 enable_all_optimizations=True,
                                 performance_mode=mode)


def raw_chat(messages: List[Dict[str, str]], model: Optional[str] = None,
             max_tokens: Optional[int] = None) -> Tuple[str, bool, str]:
    from gateway.litellm_gateway import chat
    from gateway.universal_enhanced_gateway import THINKING_MODEL_MARKERS
    model_name = _resolve_model(model)
    kwargs = dict(messages=messages, model=model_name, max_tokens=max_tokens,
                  use_cache=False)
    # Thinking models (qwen3) burn their whole generation budget on hidden
    # chain-of-thought and return empty content; the baseline must run under
    # the same think-off setting as the optimized path to be comparable.
    if any(m in model_name.lower() for m in THINKING_MODEL_MARKERS):
        kwargs["reasoning_effort"] = "none"
    return chat(**kwargs)


def _gateway_stats(gw) -> Dict[str, Any]:
    if hasattr(gw, "get_optimization_stats"):
        try:
            return gw.get_optimization_stats()
        except Exception:  # noqa: BLE001
            pass
    return {
        "cache_hits": int(getattr(gw, "cache_hits", 0) or 0),
        "cache_misses": int(getattr(gw, "cache_misses", 0) or 0),
        "performance_metrics": dict(getattr(gw, "performance_metrics", {}) or {}),
    }


def _with_system(system_prompt: Optional[str],
                 messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not system_prompt:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": system_prompt}] + list(messages)


# ---------------------------------------------------------------------------
# Core logic (injectable so tests run fully offline)
# ---------------------------------------------------------------------------


def run_chat(query: str, model: Optional[str] = None, mode: str = DEFAULT_MODE,
             gateway=None, baseline: bool = False,
             messages: Optional[List[Dict[str, str]]] = None,
             system_prompt: Optional[str] = None,
             show: bool = False) -> Dict[str, Any]:
    """Send one query through the optimizer; optionally time the raw path too."""
    msgs = _with_system(system_prompt, messages or [{"role": "user", "content": query}])
    gw = gateway if gateway is not None else make_optimized_gateway(model, mode)
    t0 = time.perf_counter()
    reply = gw.chat(msgs)
    optimized_ms = (time.perf_counter() - t0) * 1000.0
    rec: Dict[str, Any] = {
        "query": query,
        "reply": reply,
        "optimized_ms": optimized_ms,
        "cache_hits": int(getattr(gw, "cache_hits", 0) or 0),
        "cache_misses": int(getattr(gw, "cache_misses", 0) or 0),
    }
    if baseline:
        t0 = time.perf_counter()
        try:
            response, hit, used_model = raw_chat(msgs, model=_resolve_model(model))
            baseline_ms = (time.perf_counter() - t0) * 1000.0
            rec["baseline_ms"] = baseline_ms
            rec["baseline_hit"] = hit
        except Exception as exc:  # noqa: BLE001
            rec["baseline_error"] = f"{type(exc).__name__}: {exc}"
    if show:
        console.print(f"[bold cyan]Assistant:[/bold cyan] {reply}")
        if rec.get("baseline_error"):
            console.print(f"[yellow]Baseline unavailable: {rec['baseline_error']}[/yellow]")
        elif rec.get("baseline_ms") is not None:
            ratio = rec["optimized_ms"] / max(rec["baseline_ms"], 0.001)
            console.print(Panel.fit(
                f"[green]optimized[/green] {rec['optimized_ms']:.0f} ms   "
                f"[yellow]baseline[/yellow] {rec['baseline_ms']:.0f} ms   "
                f"[bold]{ratio:.2f}x[/bold] vs raw",
                border_style="dim",
            ))
    return rec


CURATED_QUESTIONS = [
    "Hello there!",
    "What is 12 * 8?",
    "Explain what a transformer model does in one sentence.",
    "Write a Python function that reverses a string.",
    "What is the capital of France?",
]


def run_bench(n: int = 5, mode: str = DEFAULT_MODE, model: Optional[str] = None,
              use_baseline: bool = True, questions: Optional[str] = None,
              gateway=None, raw=None, show: bool = True,
              raw_max_tokens: int = 128, warmup: bool = True) -> Dict[str, Any]:
    """Time optimized (cold + warm) vs. raw latency per question.

    Individual questions never abort the run: failures are recorded on the row
    (cold_error / warm_error / baseline_error) so one timeout can't kill the
    whole benchmark. A warmup pass (when enabled) moves the one-time model
    load out of the measured rows.
    """
    qs = ([q.strip() for q in questions.split("|") if q.strip()]
          if questions else list(CURATED_QUESTIONS))
    qs = qs[:max(1, n)] or CURATED_QUESTIONS[:max(1, n)]
    gw = gateway if gateway is not None else make_optimized_gateway(model, mode)
    raw_fn = raw if raw is not None else raw_chat
    model_name = _resolve_model(model)
    if warmup:
        probe = [{"role": "user", "content": "warmup"}]
        try:
            gw.chat(probe)
        except Exception:  # noqa: BLE001
            pass
        if use_baseline:
            try:
                raw_fn(probe, model=model_name, max_tokens=raw_max_tokens)
            except Exception:  # noqa: BLE001
                pass
    rows: List[Dict[str, Any]] = []
    for question in qs:
        msgs = [{"role": "user", "content": question}]
        row: Dict[str, Any] = {"question": question}
        try:
            t0 = time.perf_counter()
            gw.chat(msgs)
            row["cold_ms"] = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            row["cold_error"] = f"{type(exc).__name__}: {exc}"
        try:
            t0 = time.perf_counter()
            gw.chat(msgs)
            row["warm_ms"] = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            row["warm_error"] = f"{type(exc).__name__}: {exc}"
        if use_baseline:
            try:
                t0 = time.perf_counter()
                raw_fn(msgs, model=model_name, max_tokens=raw_max_tokens)
                row["baseline_ms"] = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:  # noqa: BLE001
                row["baseline_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    if show:
        table = Table(title="Optimized vs. baseline (local AI)",
                      show_header=True, border_style="dim")
        table.add_column("Question", style="white")
        table.add_column("Cold (ms)", justify="right", style="cyan")
        table.add_column("Warm (ms)", justify="right", style="green")
        if use_baseline:
            table.add_column("Raw (ms)", justify="right", style="yellow")
            table.add_column("vs raw", justify="right", style="bold")
        for row in rows:
            cold = f"{row['cold_ms']:.0f}" if "cold_ms" in row else "[red]err[/red]"
            warm = f"{row['warm_ms']:.0f}" if "warm_ms" in row else "[red]err[/red]"
            if use_baseline:
                if "baseline_ms" in row:
                    ratio = row["cold_ms"] / max(row["baseline_ms"], 0.001)
                    base = f"{row['baseline_ms']:.0f}" if "baseline_ms" in row else "[red]err[/red]"
                    table.add_row(row["question"], cold, warm, base,
                                  (f"{ratio:.2f}x" if "cold_ms" in row else "[red]-[/red]"))
                else:
                    table.add_row(row["question"], cold, warm, "[red]err[/red]", "[red]-[/red]")
            else:
                table.add_row(row["question"], cold, warm)
        console.print(table)
        console.print("[dim]cold/raw: the ratio of optimized-cold to raw latency "
                      "(lower = optimized faster). warm 0 ms = served from cache.[/dim]")
    return {"mode": mode, "model": model_name, "rows": rows}


_SMOKE_PROBES: List[Dict[str, Any]] = [
    {"module": "gateway.equations.cache_math", "name": "cos_sim",
     "call": lambda m: m.cos_sim([1.0, 0.0], [1.0, 0.0]) > 0.9999},
    {"module": "gateway.equations.cache_math", "name": "temporal_decay",
     "call": lambda m: 0.0 < m.temporal_decay(now=10, created_at=0, half_life=5) <= 1.0},
    {"module": "gateway.equations.cache_math", "name": "kl_divergence",
     "call": lambda m: m.kl_divergence([0.5, 0.5], [0.5, 0.5]) < 1e-9},
    {"module": "gateway.equations.cache_math", "name": "SemanticCacheMath.rank",
     "call": lambda m: len(m.SemanticCacheMath(k=2).rank(
         [1.0, 0.0], {"a": [1.0, 0.0]}, now=10.0)) == 1},
    {"module": "gateway.equations.cache_math", "name": "evict_scores",
     "call": lambda m: isinstance(m.evict_scores(
         [("k", {"freq": 1, "last_access": 0, "token_cost": 1, "demand": 0.5})],
         now=10.0), list)},
    {"module": "gateway.equations.routing_math", "name": "majority_consensus",
     "call": lambda m: m.majority_consensus(["a", "a", "b"])[0] == "a"},
    {"module": "gateway.equations.routing_math", "name": "confidence_token_budget",
     "call": lambda m: float(m.confidence_token_budget(0.9)) >= 20},
    {"module": "gateway.equations.routing_math", "name": "LoadBalancer",
     "call": lambda m: m.LoadBalancer().observe("a", 3) is None
                       and m.LoadBalancer().recommend() is None},
    {"module": "gateway.equations.routing_math", "name": "difficulty_score",
     "call": lambda m: 0.0 <= m.difficulty_score("What is quantum entanglement?") <= 1.0},
    {"module": "gateway.equations.routing_math", "name": "should_route_to_strong",
     "call": lambda m: m.should_route_to_strong(0.9) is True},
    {"module": "gateway.equations.bandit_math", "name": "ThompsonSampler.sample",
     "call": lambda m: m.ThompsonSampler(n_arms=3).sample() in (0, 1, 2)},
    {"module": "gateway.equations.bandit_math", "name": "UCB1.select",
     "call": lambda m: m.UCB1(n_arms=3).counts() == [0, 0, 0]},
    {"module": "gateway.equations.bandit_math", "name": "ucb",
     "call": lambda m: len(m.ucb([1, 1], [1.0, 0.5])) == 2},
    {"module": "gateway.equations.bandit_math", "name": "epsilon_greedy_select",
     "call": lambda m: m.epsilon_greedy_select(0.1, [1, 1], [1.0, 0.5]) in (0, 1)},
    {"module": "gateway.equations.memory_math", "name": "recency_weight",
     "call": lambda m: 0.0 <= m.recency_weight(position=0, total=2, gamma=0.9) <= 1.0},
    {"module": "gateway.equations.memory_math", "name": "compress_to_token_budget",
     "call": lambda m: isinstance(m.compress_to_token_budget("hello world foo bar", 2), str)},
    {"module": "gateway.equations.memory_math", "name": "ContextCompactor",
     "call": lambda m: hasattr(m, "ContextCompactor")},
    {"module": "gateway.equations.retrieval_math", "name": "hybrid_relevance",
     "call": lambda m: 0.0 <= m.hybrid_relevance(0.5, 0.7) <= 1.0},
    {"module": "gateway.equations.retrieval_math", "name": "topk_stable",
     "call": lambda m: len(m.topk_stable([3.0, 1.0, 2.0], k=2)) == 2},
    {"module": "gateway.equations.retrieval_math", "name": "qf_similarity",
     "call": lambda m: 0.0 <= m.qf_similarity("What is gravity?", "Explain gravity.") <= 1.0},
    {"module": "gateway.equations.calibration_math", "name": "softmax",
     "call": lambda m: abs(sum(m.softmax([0.2, 0.8])) - 1.0) < 1e-6},
    {"module": "gateway.equations.calibration_math", "name": "calibrated_confidence",
     "call": lambda m: 0.0 <= m.calibrated_confidence(0.9) <= 1.0},
    {"module": "gateway.equations.calibration_math", "name": "temperature_scale",
     "call": lambda m: len(m.temperature_scale([0.2, 0.8])) == 2},
    {"module": "gateway.equations.calibration_math", "name": "brier_score",
     "call": lambda m: m.brier_score([[1.0, 0.0], [0.0, 1.0]], [0, 1]) >= 0.0},
    {"module": "gateway.equations.planning_math", "name": "uct_score",
     "call": lambda m: m.uct_score(0.5, 10, 2) >= 0.0},
    {"module": "gateway.equations.planning_math", "name": "agreement_fraction",
     "call": lambda m: m.agreement_fraction(["a", "a"]) == 1.0},
    {"module": "gateway.equations.planning_math", "name": "depth_limit_knowledge",
     "call": lambda m: 0 <= m.depth_limit_knowledge(0.8) <= 8},
    {"module": "gateway.equations.not_implementable", "name": "list_not_implementable",
     "call": lambda m: isinstance(m.list_not_implementable(), list)},
    {"module": "gateway.perf_math", "name": "eq18_query_complexity",
     "call": lambda m: 0.0 <= m.eq18_query_complexity("What is a transformer?") <= 1.0},
    {"module": "gateway.perf_math", "name": "eq19_should_use_bon",
     "call": lambda m: isinstance(m.eq19_should_use_bon(0.9, 0.005), bool)},
]


def run_check(verbose: bool = True) -> Dict[str, Any]:
    """Verify every optimization module loads and its key routine runs (offline)."""
    import importlib
    _quiet_library_logging()
    results: List[Dict[str, Any]] = []
    for probe in _SMOKE_PROBES:
        ok = False
        detail = ""
        try:
            module = importlib.import_module(probe["module"])
            if not probe["call"](module):
                detail = "returned unexpected value"
            else:
                ok = True
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
        results.append({"module": probe["module"], "name": probe["name"],
                        "ok": ok, "detail": detail})

    wiring: Dict[str, Any] = {}
    try:
        from gateway.equation_wiring import install_semantic_cache, install_ewma_router
        from gateway.opt_core import get_router_state
        from gateway.simple_cache import get_cache

        cache = get_cache()
        router = get_router_state()
        sem_res = install_semantic_cache(cache)
        ewma_res = install_ewma_router(router)
        wiring["semantic_cache"] = bool(sem_res) or bool(getattr(cache, "_semantic_installed", False))
        wiring["ewma_router"] = bool(ewma_res) or bool(getattr(router, "_ewma_installed", False))
        wiring["koopman_rag"] = "skipped (RAG provider not loaded)"
    except Exception as exc:  # noqa: BLE001
        wiring = {"error": f"{type(exc).__name__}: {exc}"}

    failed = [r for r in results if not r["ok"]]
    wiring_error = bool(wiring.get("error"))
    wiring_enabled = wiring.get("semantic_cache") and wiring.get("ewma_router")
    success = (not failed) and (not wiring_error) and bool(wiring_enabled)

    if verbose:
        table = Table(title="Optimization modules", border_style="dim")
        table.add_column("Module", style="cyan", no_wrap=True)
        table.add_column("Check", style="white")
        table.add_column("Result", style="bold")
        for r in results:
            label = "[green]OK[/green]" if r["ok"] else "[red]FAIL[/red]"
            detail = (f" [dim]- {r['detail']}[/dim]" if r["detail"] else "")
            table.add_row(r["module"], r["name"], label + detail)
        console.print(table)
        wiring_table = Table(title="Equation wiring", border_style="dim")
        wiring_table.add_column("Component", style="cyan")
        wiring_table.add_column("Installed", style="bold")
        for key, value in wiring.items():
            if value is True:
                label = "[green]yes[/green]"
            elif value is False:
                label = "[red]no[/red]"
            else:
                label = str(value)
            wiring_table.add_row(key, label)
        console.print(wiring_table)
        if success:
            console.print(Panel.fit(
                f"[green]All {len(results)} optimization module checks passed "
                f"and equation wiring is installed.[/green]",
                border_style="green",
            ))
        else:
            console.print(Panel.fit(
                f"[red]{len(failed)} check(s) failed or wiring missing.[/red]",
                border_style="red",
            ))
    return {"checks": results, "wiring": wiring, "success": success}


# ---------------------------------------------------------------------------
# HF-dataset evaluation (optimized vs. raw)
# ---------------------------------------------------------------------------

MATH_PROMPT = (
    "Solve the following math word problem. "
    "Answer with just the final number.\n\nQuestion: {question}\n\nAnswer:"
)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_QUESTION_KEYS = ("question", "input", "prompt", "problem_statement", "problem",
                  "instruction", "PROMPT", "base_description", "description",
                  "text")


def _norm_token(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize_row(record: Dict[str, Any], label_names=None) -> Dict[str, Any]:
    """Map heterogeneous dataset rows to {question, answer | answer_index+choices, context}."""
    row: Dict[str, Any] = {}
    question = ""
    for key in _QUESTION_KEYS:
        value = record.get(key)
        if value:
            question = str(value)
            break
    row["question"] = question
    row["context"] = str(record.get("passage") or record.get("context") or "")
    answer = record.get("answer")
    label = record.get("label")
    choices = record.get("choices")
    if isinstance(answer, str) and answer:
        row["answer"] = answer
    elif isinstance(answer, int) and isinstance(choices, list):
        row["answer_index"] = answer
        row["choices"] = choices
    elif isinstance(answer, (list, tuple)) and answer:
        row["answer"] = " | ".join(str(a) for a in answer if a)
    elif label is not None:
        if isinstance(choices, list) and not isinstance(label, (dict, str)):
            row["answer_index"] = int(label)
            row["choices"] = choices
        elif label_names:
            row["answer"] = str(label_names[int(label)])
        else:
            row["answer"] = str(label)
    elif answer is not None:
        row["answer"] = str(answer)
    return row


def load_dataset(dataset: str, config: str, split: str,
                 n: int, seed: int = 1) -> List[Dict[str, Any]]:
    """Load a HuggingFace dataset (cached on the F: drive) and shuffle a slice."""
    import os
    os.environ.setdefault("HF_HOME", str(_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_DIR / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(_CACHE_DIR / "hub"))
    import datasets
    try:
        ds = datasets.load_dataset(dataset, config, split=split,
                                   trust_remote_code=True)
    except Exception as first_err:  # noqa: BLE001
        try:
            ds = datasets.load_dataset(dataset, config, split=split,
                                       trust_remote_code=False)
        except Exception:
            raise RuntimeError(
                f"Cannot load '{dataset}' (config='{config}', split='{split}'). "
                f"Dataset may require an obsolete loading script.\n"
                f"Original error: {first_err}") from first_err
    label_names = None
    features = getattr(ds, "features", None)
    if features and "label" in features:
        names = getattr(features["label"], "names", None)
        if names:
            label_names = list(names)
    size = len(ds)
    indices = random.Random(seed).sample(range(size), min(max(1, n), size))
    return [_normalize_row(dict(ds[i]), label_names) for i in indices]


def _extract_gold(answer_text: str) -> Optional[str]:
    if not answer_text:
        return None
    marker = re.search(r"####\s*(.+)", answer_text)
    haystack = marker.group(1) if marker else answer_text
    found = _NUM_RE.findall(haystack)
    if found:
        return found[-1].replace(",", "")
    return None


def _answer_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    if not pred or not gold:
        return False
    pred_num = _extract_number(pred)
    gold_num = _extract_number(gold) or gold.replace(",", "")
    if pred_num is None:
        return False
    try:
        return float(pred_num) == float(gold_num)
    except (TypeError, ValueError):
        return pred_num.rstrip(".") == gold_num.rstrip(".")


def _extract_number(text: str) -> Optional[str]:
    if not text:
        return None
    found = _NUM_RE.findall(text)
    if not found:
        return None
    return found[-1].replace(",", "")


def _guess_extract_mode(item: Dict[str, Any]) -> str:
    if "answer_index" in item and isinstance(item.get("choices"), list):
        return "letter"
    text = str(item.get("answer") or "")
    if not text.strip():
        return "latency"
    if "####" in text:
        return "numeric"
    if text.split() and text.strip().lower().split()[0] in ("yes", "no", "true", "false"):
        return "word"
    return "numeric"


def _gold_value(item: Dict[str, Any], mode: str) -> Optional[str]:
    if mode == "letter":
        try:
            return chr(65 + int(item["answer_index"]))
        except (KeyError, TypeError, ValueError):
            return None
    if mode == "exact":
        return str(item.get("answer", ""))
    answer = str(item.get("answer", ""))
    if mode == "word":
        first = answer.strip().lower().split()
        return first[0] if first else None
    return _extract_gold(answer)


def _eval_prompt(question: str, item: Dict[str, Any], mode: str) -> str:
    if mode == "letter":
        options = item.get("choices") or []
        opts = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(options))
        return (f"Answer the following multiple-choice question with only the "
                f"correct option letter (A, B, C, D, ...).\n\n"
                f"Question: {question}\n\n{opts}\n\nAnswer:")
    if mode == "word":
        context = item.get("context") or ""
        return (f"{context}\n\nQuestion: {question}\n\n"
                f"Answer with exactly one word: yes or no.")
    if mode == "exact":
        return (f"Answer the following question with a short, concise answer. "
                f"Give only the answer, with no explanation.\n\n"
                f"Question: {question}\n\nAnswer:")
    if mode == "latency":
        return (f"Complete the following task as best you can. "
                f"Read the task carefully, work through it, and provide your "
                f"final result at the end.\n\nTask: {question}\n\nResult:")
    return MATH_PROMPT.format(question=question)


def _pred_value(pred: str, mode: str) -> Optional[str]:
    if not pred:
        return None
    if mode == "word":
        hit = re.search(r"\b(yes|no|true|false)\b", pred, re.IGNORECASE)
        return hit.group(1).lower() if hit else None
    if mode == "letter":
        hit = re.search(r"\b[A-E]\b", pred)
        return hit.group(0) if hit else None
    if mode == "exact":
        return _norm_token(pred)
    return _extract_number(pred)


def _is_correct(pred: Optional[str], gold: Optional[str], mode: str) -> bool:
    if not pred or not gold:
        return False
    if mode == "numeric":
        return _answer_equal(pred, gold)
    if mode == "exact":
        p = _norm_token(pred)
        for alt in str(gold).split("|"):
            a = _norm_token(alt)
            if a and (p == a or a in p):
                return True
        return False
    return pred.strip().lower() == gold.strip().lower()


def run_eval(dataset: str = "openai/gsm8k", config: str = "main",
             split: str = "test", n: int = 40, seed: int = 1,
             mode: str = EVAL_MODE, model: Optional[str] = None,
             use_baseline: bool = True, out: Optional[str] = None,
             extract: str = "auto", baseline_max_tokens: int = 160,
             gateway=None, raw=None, samples: Optional[List[Dict[str, Any]]] = None,
             show: bool = True) -> Dict[str, Any]:
    """Evaluate a HF dataset through optimized vs. raw paths and compare.

    Never aborts on a bad sample: per-sample errors are recorded on the row.
    `extract` controls answer parsing: auto (guessed per dataset), numeric,
    letter (A-E multiple choice), word (yes/no/true/false), exact (short,
    free-form answers via token-insensitive matching) or latency (no gold
    answer; only response text and timing are recorded).
    """
    if samples is None:
        samples = load_dataset(dataset, config, split, n, seed)
    samples = samples[:max(1, n)]
    from config import settings
    settings.generation_timeout = max(
        int(getattr(settings, "generation_timeout", 15)), 240)
    gw = gateway if gateway is not None else make_optimized_gateway(model, mode)
    raw_fn = raw if raw is not None else raw_chat
    model_name = _resolve_model(model)
    max_tokens = baseline_max_tokens
    scored = extract != "latency"

    rows: List[Dict[str, Any]] = []
    for item in samples:
        question = str(item.get("question", ""))
        run_mode = _guess_extract_mode(item) if extract == "auto" else extract
        if run_mode == "latency":
            scored = False
        gold = _gold_value(item, run_mode) if scored else None
        content = _eval_prompt(question, item, run_mode)
        msgs = [{"role": "user", "content": content}]
        row: Dict[str, Any] = {"question": question, "gold": gold,
                               "extract_mode": run_mode}
        t0 = time.perf_counter()
        try:
            pred = gw.chat(msgs)
            row["optimized_ms"] = (time.perf_counter() - t0) * 1000.0
            row["optimized_pred"] = pred
            if scored:
                row["optimized_extracted"] = _pred_value(pred, run_mode)
                row["optimized_correct"] = _is_correct(
                    row["optimized_extracted"], gold, run_mode)
        except Exception as exc:  # noqa: BLE001
            row["optimized_error"] = f"{type(exc).__name__}: {exc}"
        if use_baseline:
            t0 = time.perf_counter()
            try:
                resp, _hit, _used = raw_fn(msgs, model=model_name, max_tokens=max_tokens)
                row["baseline_ms"] = (time.perf_counter() - t0) * 1000.0
                row["baseline_pred"] = resp
                if scored:
                    row["baseline_correct"] = _is_correct(
                        _pred_value(resp, run_mode), gold, run_mode)
            except Exception as exc:  # noqa: BLE001
                row["baseline_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    def _stats(key_ms: str, key_correct: str) -> Dict[str, Any]:
        done = [r for r in rows if key_ms in r]
        ms = [r[key_ms] for r in done]
        cache_hits = sum(1 for r in done if r[key_ms] < 50)
        if scored:
            correct = sum(1 for r in done if r.get(key_correct))
            accuracy = (correct / len(done)) if done else 0.0
        else:
            correct = None
            accuracy = None
        return {"answered": len(done), "correct": correct,
                "accuracy": accuracy,
                "avg_ms": (sum(ms) / len(ms)) if ms else 0.0,
                "total_s": round(sum(ms) / 1000.0, 2) if ms else 0.0,
                "cache_like_fast": cache_hits}

    optimized = _stats("optimized_ms", "optimized_correct")
    baseline = _stats("baseline_ms", "baseline_correct") if use_baseline else None

    summary = {"dataset": dataset, "config": config, "split": split,
               "mode": mode, "model": model_name, "n": len(rows),
               "scored": scored, "optimized": optimized,
               "baseline": baseline, "rows": rows}
    if out:
        try:
            summary["_rows"] = rows
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(
                json.dumps({k: v for k, v in summary.items()
                            if k != "_rows"}, indent=2), encoding="utf-8")
            summary.pop("_rows", None)
        except Exception as exc:  # noqa: BLE001
            summary["save_error"] = f"{type(exc).__name__}: {exc}"

    if show:
        table = Table(title=f"{dataset} ({split}) - optimized vs. raw",
                      border_style="dim")
        table.add_column("Question", style="white", no_wrap=False)
        table.add_column("Optimized", justify="right", style="green")
        table.add_column("Raw", justify="right", style="yellow")
        for row in rows:
            if scored:
                opt = ("[green]OK[/green]" if row.get("optimized_correct")
                       else ("[red]err[/red]" if "optimized_error" in row else "[red]X[/red]"))
                raw_col = ("[green]OK[/green]" if row.get("baseline_correct")
                           else ("[red]err[/red]" if "baseline_error" in row else "[red]X[/red]"))
            else:
                opt = ("[red]err[/red]" if "optimized_error" in row else "[green]done[/green]")
                raw_col = ("[red]err[/red]" if "baseline_error" in row else "[green]done[/green]")
            table.add_row(row["question"][:60], opt, raw_col)
        console.print(table)

        def _acc(v):
            return "n/a" if v is None else f"{v:.3f}"

        summary_table = Table(title="Evaluation summary", border_style="dim")
        summary_table.add_column("Path", style="cyan")
        for key in ("answered", "correct", "accuracy", "avg_ms", "total_s"):
            summary_table.add_column(key, justify="right")
        summary_table.add_row(
            "optimized",
            str(optimized["answered"]), _acc(optimized["correct"]),
            _acc(optimized["accuracy"]), f"{optimized['avg_ms']:.0f}",
            f"{optimized['total_s']:.1f}")
        if baseline:
            summary_table.add_row(
                "raw",
                str(baseline["answered"]), _acc(baseline["correct"]),
                _acc(baseline["accuracy"]), f"{baseline['avg_ms']:.0f}",
                f"{baseline['total_s']:.1f}")
        console.print(summary_table)
        if not scored:
            console.print("[dim]latency-only mode: no ground-truth answers in "
                          "this dataset; accuracy is n/a by design.[/dim]")
        elif baseline and (optimized["accuracy"] is not None
                           and baseline["accuracy"] is not None):
            console.print(
                f"[dim]accuracy delta: "
                f"{optimized['accuracy'] - baseline['accuracy']:+.3f}   "
                f"latency: {optimized['avg_ms']:.0f}ms vs {baseline['avg_ms']:.0f}ms raw[/dim]")
        if out:
            console.print(f"[dim]report saved to {out}[/dim]")
    return summary


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send through the optimizer"),
    mode: str = typer.Option(DEFAULT_MODE, "--mode", help="speed, balanced or quality"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    baseline: bool = typer.Option(False, "--baseline",
                                  help="Also run the raw path and compare latency"),
    system_prompt: Optional[str] = typer.Option(None, "--system"),
) -> None:
    """Send a single message through the optimizer stack."""
    _banner()
    run_chat(message, model=model, mode=mode, baseline=baseline,
             system_prompt=system_prompt, show=True)


@app.command()
def run(
    mode: str = typer.Option(DEFAULT_MODE, "--mode", help="speed, balanced or quality"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    system_prompt: Optional[str] = typer.Option(None, "--system"),
) -> None:
    """Interactive REPL. In-chat commands: /quit /reset /stats /help."""
    _banner()
    gw = make_optimized_gateway(model, mode)
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    trimmed_notice = False
    console.print("[dim]Type your message. Commands: /quit /reset /stats /help[/dim]")
    while True:
        try:
            text = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            raise typer.Exit(0)
        if not text:
            continue
        if text in ("/quit", "/exit"):
            raise typer.Exit(0)
        if text == "/reset":
            messages = ([{"role": "system", "content": system_prompt}]
                        if system_prompt else [])
            trimmed_notice = False
            console.print("[dim]Conversation reset.[/dim]")
            continue
        if text == "/stats":
            console.print_json(data=_gateway_stats(gw))
            continue
        if text == "/help":
            console.print("[dim]Commands: /quit /reset /stats /help[/dim]")
            continue
        messages.append({"role": "user", "content": text})
        t0 = time.perf_counter()
        try:
            reply = gw.chat(messages)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{exc}[/red]")
            messages.pop()
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        console.print(f"[bold cyan]Assistant:[/bold cyan] {reply}")
        console.print(f"[dim]{elapsed_ms:.0f} ms[/dim]")
        messages.append({"role": "assistant", "content": reply})
        if _trim_context(messages, MAX_TURNS) and not trimmed_notice:
            trimmed_notice = True
            console.print(f"[dim]Context trimmed to the last {MAX_TURNS} turns "
                          "to keep long sessions fast.[/dim]")


@app.command()
def check(verbose: bool = typer.Option(True, "--verbose/--quiet")) -> None:
    """Verify every optimization module loads, runs, and is wired in (offline)."""
    _banner()
    summary = run_check(verbose=verbose)
    if not summary["success"]:
        raise typer.Exit(1)


@app.command()
def stats(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show live optimizer telemetry from the universal gateway."""
    data = _gateway_stats(make_optimized_gateway())
    if json_output:
        console.print_json(data=data)
        return
    table = Table(title="Optimizer telemetry", border_style="dim")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            table.add_row(key, json.dumps(value, default=str)[:200])
        else:
            table.add_row(key, str(value))
    console.print(table)


@app.command()
def bench(
    n: int = typer.Option(5, "--n", help="Number of questions to use"),
    mode: str = typer.Option(DEFAULT_MODE, "--mode", help="speed, balanced or quality"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    baseline: bool = typer.Option(True, "--baseline/--no-baseline",
                                  help="Also time the raw path"),
    questions: Optional[str] = typer.Option(None, "--questions",
                                            help="Pipe-delimited custom questions"),
    raw_max_tokens: int = typer.Option(128, "--raw-max-tokens",
                                       help="Token cap for the raw path (avoids timeouts)"),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup",
                                help="Warm the model first so rows measure true per-query cost"),
) -> None:
    """Time the optimized path (cold + warm) vs. the raw path per question."""
    _banner()
    run_bench(n=n, mode=mode, model=model, use_baseline=baseline,
              questions=questions, show=True, raw_max_tokens=raw_max_tokens,
              warmup=warmup)


@app.command()
def eval(
    dataset: str = typer.Option("openai/gsm8k", "--dataset", "-d",
                                help="HuggingFace benchmark dataset id"),
    config: str = typer.Option("main", "--config", help="Dataset config/version"),
    split: str = typer.Option("test", "--split", help="Which split to evaluate"),
    n: int = typer.Option(40, "--n", help="Number of questions to evaluate"),
    seed: int = typer.Option(1, "--seed"),
    mode: str = typer.Option(EVAL_MODE, "--mode", help="speed, balanced or quality"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    baseline: bool = typer.Option(True, "--baseline/--no-baseline",
                                  help="Also run each question through the raw path"),
    extract: str = typer.Option("auto", "--extract",
                                help="Answer parsing: auto, numeric, letter, "
                                     "word, exact or latency"),
    baseline_max_tokens: int = typer.Option(160, "--baseline-max-tokens",
                                            help="Token cap for the raw path"),
    out: Optional[str] = typer.Option(None, "--out",
                                      help="JSON report path (default: benchmark_results/eval_report.json)"),
) -> None:
    """Evaluate a HF benchmark through optimized vs. raw paths and compare.

    Dataset shapes auto-detected: '####'-style math answers (numeric),
    choices+answer-index MC (letter), yes/no label datasets (word), rows
    without ground truth (latency). Pass --extract exact for short free-form
    answers (e.g. SimpleQA, HLE).
    """
    _banner()
    if out is None:
        out = str(Path(__file__).resolve().parent / "benchmark_results"
                  / f"eval_{dataset.replace('/', '_')}_{split}.json")
    run_eval(dataset=dataset, config=config, split=split, n=n, seed=seed,
             mode=mode, model=model, use_baseline=baseline, out=out,
             extract=extract, baseline_max_tokens=baseline_max_tokens, show=True)


if __name__ == "__main__":
    app()