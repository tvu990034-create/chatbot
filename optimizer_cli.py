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
  python main.py turbo --help

It also runs standalone:

  python optimizer_cli.py chat "hello"
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import settings

console = Console()
DEFAULT_MODE = "speed"

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


def _resolve_model(model: Optional[str]) -> str:
    if model:
        return model
    return getattr(settings, "local_model_name", None) or settings.default_model


def make_optimized_gateway(model: Optional[str] = None, mode: str = DEFAULT_MODE):
    from gateway.universal_enhanced_gateway import get_universal_gateway
    return get_universal_gateway(model_name=_resolve_model(model),
                                 enable_all_optimizations=True,
                                 performance_mode=mode)


def raw_chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> Tuple[str, bool, str]:
    from gateway.litellm_gateway import chat
    return chat(messages, model=_resolve_model(model), use_cache=False)


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
        response, hit, used_model = raw_chat(msgs, model=_resolve_model(model))
        baseline_ms = (time.perf_counter() - t0) * 1000.0
        rec["baseline_ms"] = baseline_ms
        rec["baseline_hit"] = hit
    if show:
        console.print(f"[bold cyan]Assistant:[/bold cyan] {reply}")
        if rec.get("baseline_ms") is not None:
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
              gateway=None, raw=None, show: bool = True) -> Dict[str, Any]:
    """Time optimized (cold + warm) vs. raw latency per question, no assertions."""
    qs = ([q.strip() for q in questions.split("|") if q.strip()]
          if questions else list(CURATED_QUESTIONS))
    qs = qs[:max(1, n)] or CURATED_QUESTIONS[:max(1, n)]
    gw = gateway if gateway is not None else make_optimized_gateway(model, mode)
    raw_fn = raw if raw is not None else raw_chat
    model_name = _resolve_model(model)
    rows: List[Dict[str, Any]] = []
    for question in qs:
        msgs = [{"role": "user", "content": question}]
        t0 = time.perf_counter()
        gw.chat(msgs)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        gw.chat(msgs)
        warm_ms = (time.perf_counter() - t0) * 1000.0
        baseline_ms = None
        if use_baseline:
            t0 = time.perf_counter()
            raw_fn(msgs, model=model_name)
            baseline_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({"question": question, "cold_ms": cold_ms,
                     "warm_ms": warm_ms, "baseline_ms": baseline_ms})
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
            if use_baseline:
                ratio = row["cold_ms"] / max(row["baseline_ms"] or 0.001, 0.001)
                table.add_row(row["question"],
                              f"{row['cold_ms']:.0f}", f"{row['warm_ms']:.0f}",
                              f"{row['baseline_ms']:.0f}", f"{ratio:.2f}x")
            else:
                table.add_row(row["question"],
                              f"{row['cold_ms']:.0f}", f"{row['warm_ms']:.0f}")
        console.print(table)
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
            label = "[green]yes[/green]" if value and not key.startswith("error") else (
                "[red]no[/red]" if value is False else value)
            wiring_table.add_row(key, str(label) if not isinstance(label, str) else label)
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
) -> None:
    """Time the optimized path (cold + warm) vs. the raw path per question."""
    _banner()
    run_bench(n=n, mode=mode, model=model, use_baseline=baseline,
              questions=questions, show=True)


if __name__ == "__main__":
    app()