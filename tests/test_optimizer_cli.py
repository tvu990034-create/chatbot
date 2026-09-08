"""
Tests for the turbo CLI (optimizer_cli.py).

All behavior tests run fully offline: the optimizer smoke checks are pure
Python, and any model calls are replaced with fake gateways.
"""
from typer.testing import CliRunner

import optimizer_cli

runner = CliRunner()


class FakeGateway:
    def __init__(self, reply="pong"):
        self.reply = reply
        self.cache_hits = 3
        self.cache_misses = 1

    def chat(self, messages):
        return self.reply


def test_turbo_command_group_registered():
    result = runner.invoke(optimizer_cli.app, ["--help"])
    assert result.exit_code == 0
    for cmd_name in ("chat", "run", "check", "stats", "bench"):
        assert cmd_name in result.stdout


def test_turbo_registered_on_main_app():
    import main

    result = runner.invoke(main.app, ["turbo", "--help"])
    assert result.exit_code == 0
    assert "turbo" in result.stdout


def test_turbo_help_runs_offline():
    result = runner.invoke(optimizer_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "turbo" in result.stdout


def test_check_passes_offline():
    summary = optimizer_cli.run_check(verbose=False)
    assert summary["success"] is True
    assert summary["checks"]


def test_check_reports_every_module():
    summary = optimizer_cli.run_check(verbose=False)
    modules = {c["module"] for c in summary["checks"]}
    for expected in (
        "gateway.equations.cache_math",
        "gateway.equations.routing_math",
        "gateway.equations.bandit_math",
        "gateway.equations.memory_math",
        "gateway.equations.retrieval_math",
        "gateway.equations.calibration_math",
        "gateway.equations.planning_math",
        "gateway.equations.not_implementable",
        "gateway.perf_math",
    ):
        assert expected in modules


def test_check_typer_command_offline():
    result = runner.invoke(optimizer_cli.app, ["check", "--quiet"])
    assert result.exit_code == 0


def test_chat_with_fake_gateway(monkeypatch):
    monkeypatch.setattr(optimizer_cli, "make_optimized_gateway",
                        lambda model=None, mode=optimizer_cli.DEFAULT_MODE: FakeGateway())
    result = optimizer_cli.run_chat("ping", gateway=None)
    assert result["reply"] == "pong"
    assert result["cache_hits"] == 3
    assert result["optimized_ms"] >= 0
    assert result.get("baseline_ms") is None


def test_chat_with_baseline_uses_raw_path(monkeypatch):
    monkeypatch.setattr(optimizer_cli, "make_optimized_gateway",
                        lambda model=None, mode=optimizer_cli.DEFAULT_MODE: FakeGateway())
    calls = []

    def fake_raw(messages, model=None):
        calls.append(messages)
        return ("base reply", False, "fake")

    monkeypatch.setattr(optimizer_cli, "raw_chat", fake_raw)
    result = optimizer_cli.run_chat("hi", gateway=None, baseline=True, model="fake:1")
    assert result["baseline_ms"] is not None
    assert calls
    assert calls[0][0]["role"] == "user"


def test_chat_typer_command_offline(monkeypatch):
    monkeypatch.setattr(optimizer_cli, "make_optimized_gateway",
                        lambda model=None, mode=optimizer_cli.DEFAULT_MODE: FakeGateway())
    result = runner.invoke(optimizer_cli.app, ["chat", "hello"])
    assert result.exit_code == 0
    assert "pong" in result.stdout


def test_run_bench_hermetic():
    def fake_raw(messages, model=None):
        return ("base", False, model)

    summary = optimizer_cli.run_bench(n=2, gateway=FakeGateway(), raw=fake_raw,
                                      use_baseline=True, show=False)
    assert len(summary["rows"]) == 2
    for row in summary["rows"]:
        assert row["cold_ms"] >= 0
        assert row["warm_ms"] >= 0
        assert row["baseline_ms"] is not None


def test_stats_offline(monkeypatch):
    monkeypatch.setattr(optimizer_cli, "make_optimized_gateway", lambda *a, **k: FakeGateway())
    result = runner.invoke(optimizer_cli.app, ["stats", "--json"])
    assert result.exit_code == 0