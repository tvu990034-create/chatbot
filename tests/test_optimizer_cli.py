"""
Tests for the turbo CLI (optimizer_cli.py).

All behavior tests run fully offline: the optimizer smoke checks are pure
Python, and any model calls are replaced with fake gateways.
"""
from typer.testing import CliRunner
import pytest

import optimizer_cli

runner = CliRunner()


class FakeGateway:
    def __init__(self, reply="pong"):
        self.reply = reply
        self.cache_hits = 3
        self.cache_misses = 1

    def chat(self, messages, **kwargs):
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


def test_chat_baseline_error_keeps_optimized_reply(monkeypatch):
    monkeypatch.setattr(optimizer_cli, "make_optimized_gateway",
                        lambda model=None, mode=optimizer_cli.DEFAULT_MODE: FakeGateway())

    def broken_raw(messages, model=None, max_tokens=None):
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(optimizer_cli, "raw_chat", broken_raw)
    result = optimizer_cli.run_chat("hi", gateway=None, baseline=True, show=False)
    assert result["reply"] == "pong"
    assert "baseline_error" in result
    assert result.get("baseline_ms") is None


def test_chat_typer_command_offline(monkeypatch):
    monkeypatch.setattr(optimizer_cli, "make_optimized_gateway",
                        lambda model=None, mode=optimizer_cli.DEFAULT_MODE: FakeGateway())
    result = runner.invoke(optimizer_cli.app, ["chat", "hello"])
    assert result.exit_code == 0
    assert "pong" in result.stdout


def test_run_bench_hermetic():
    def fake_raw(messages, model=None, max_tokens=None):
        return ("base", False, model)

    summary = optimizer_cli.run_bench(n=2, gateway=FakeGateway(), raw=fake_raw,
                                      use_baseline=True, show=False)
    assert len(summary["rows"]) == 2
    for row in summary["rows"]:
        assert row["cold_ms"] >= 0
        assert row["warm_ms"] >= 0
        assert row["baseline_ms"] is not None


def test_run_bench_raw_error_does_not_abort():
    def broken_raw(messages, model=None, max_tokens=None):
        raise TimeoutError("simulated timeout")

    summary = optimizer_cli.run_bench(n=3, gateway=FakeGateway(), raw=broken_raw,
                                      use_baseline=True, show=False)
    assert len(summary["rows"]) == 3
    for row in summary["rows"]:
        assert "cold_ms" in row
        assert "baseline_error" in row
        assert "baseline_ms" not in row


def test_stats_offline(monkeypatch):
    monkeypatch.setattr(optimizer_cli, "make_optimized_gateway", lambda *a, **k: FakeGateway())
    result = runner.invoke(optimizer_cli.app, ["stats", "--json"])
    assert result.exit_code == 0


def _history(turns):
    msgs = []
    for i in range(turns):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def test_trim_context_keeps_window():
    msgs = [{"role": "system", "content": "sys"}] + _history(30)
    assert optimizer_cli._trim_context(msgs, max_turns=5) is True
    assert msgs[0] == {"role": "system", "content": "sys"}
    turns = sum(1 for m in msgs if m["role"] == "user")
    assert turns == 5


def test_trim_context_within_budget_noop():
    msgs = _history(2)
    assert optimizer_cli._trim_context(msgs, max_turns=5) is False
    assert len(msgs) == 4


def test_trim_context_no_system_prompt():
    msgs = _history(30)
    assert optimizer_cli._trim_context(msgs, max_turns=4) is True
    assert len(msgs) == 8


def test_extract_gold_from_gsm8k_answer():
    gold = optimizer_cli._extract_gold("John has 3 apples and gets 2 more. #### 5 apples")
    assert gold == "5"


def test_extract_number_from_prediction():
    assert optimizer_cli._extract_number("The answer is 42.") == "42"
    assert optimizer_cli._extract_number("no numbers here") is None


def test_extract_number_prefers_main_clause_over_parentheticals():
    assert optimizer_cli._extract_number(
        "The hospital makes a profit of $10,000 per day by calculating total "
        "revenue ($40,000) from patients minus total costs ($30,000) for "
        "doctor time.") == "10000"
    assert optimizer_cli._extract_number("#### 10000") == "10000"
    assert optimizer_cli._extract_number(
        "resulting in $1,040 x $1.20 = $1,248.") == "1248"


def test_answer_equal_parses_floats():
    assert optimizer_cli._answer_equal("105", "105.0") is True
    assert optimizer_cli._answer_equal("7 apples", "7") is True
    assert optimizer_cli._answer_equal("8", "9") is False


def test_run_eval_restores_global_generation_timeout():
    """run_eval raises settings.generation_timeout to 240 during the loop but
    must restore the original value afterwards, so the leak of a 240s floor
    into every later model call in the same process is fixed."""
    from config import settings
    original = settings.generation_timeout
    try:
        settings.generation_timeout = 37
        samples = [{"question": "what is 2+2?", "answer": "4 #### 4"}]

        class G:
            cache_hits = 0
            cache_misses = 1

            def chat(self, messages, **kwargs):
                return "4"

        summary = optimizer_cli.run_eval(
            n=1, gateway=G(), raw=lambda m, model=None, max_tokens=None: ("4", False, None),
            samples=samples, show=False, use_baseline=True)
        assert summary["optimized"]["answered"] == 1
        assert settings.generation_timeout == 37, f"leaked {settings.generation_timeout}"
    finally:
        settings.generation_timeout = original


def test_run_eval_hermetic():
    class ContentGateway:
        cache_hits = 0
        cache_misses = 1

        def chat(self, messages, **kwargs):
            query = messages[-1]["content"]
            return "4" if "+" in query else "9"

    samples = [
        {"question": "What is 2 + 2?", "answer": "The answer is 4. #### 4"},
        {"question": "What is 3 * 3?", "answer": "The answer is 9. #### 9"},
    ]
    calls = []

    def fake_raw(messages, model=None, max_tokens=None):
        calls.append(messages)
        return ("The answer is 4.", False, model)

    summary = optimizer_cli.run_eval(
        n=2, gateway=ContentGateway(), raw=fake_raw,
        samples=samples, show=False, use_baseline=True)
    optimized = summary["optimized"]
    baseline = summary["baseline"]
    assert optimized["answered"] == 2
    assert optimized["correct"] == 2
    assert optimized["accuracy"] == 1.0
    assert baseline["accuracy"] == 0.5
    assert len(summary["rows"]) == 2
    # 2 samples + 1 warmup probe (run_eval warms the model so the first row
    # does not include the cold model-load time)
    assert len(calls) == 3
    assert calls[-1][-1]["role"] == "user"


def test_run_eval_never_aborts_on_bad_sample():
    class ExplodingGateway:
        def chat(self, messages, **kwargs):
            raise RuntimeError("boom")

    def broken_raw(messages, model=None, max_tokens=None):
        raise TimeoutError("oom")

    samples = [{"question": "q1", "answer": "5 #### 5"}]
    summary = optimizer_cli.run_eval(
        n=1, gateway=ExplodingGateway(), raw=broken_raw, samples=samples,
        show=False, use_baseline=True)
    assert "optimized_error" in summary["rows"][0]
    assert "baseline_error" in summary["rows"][0]
    assert summary["optimized"]["answered"] == 0
    assert summary["baseline"]["answered"] == 0


def test_eval_command_registered():
    result = runner.invoke(optimizer_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.stdout


def test_normalize_and_extract_modes():
    mmlu = {"question": "q", "answer_index": 2, "choices": ["a", "b", "c", "d"]}
    boolq = {"question": "Q", "context": "P", "answer": "yes"}
    gsm = {"question": "q", "answer": "4 #### 4"}
    assert optimizer_cli._guess_extract_mode(mmlu) == "letter"
    assert optimizer_cli._guess_extract_mode(boolq) == "word"
    assert optimizer_cli._guess_extract_mode(gsm) == "numeric"
    assert optimizer_cli._gold_value(mmlu, "letter") == "C"
    assert optimizer_cli._gold_value(boolq, "word") == "yes"
    assert optimizer_cli._pred_value("The correct answer is B.", "letter") == "B"
    assert optimizer_cli._pred_value("No, I disagree.", "word") == "no"
    assert optimizer_cli._pred_value("The answer is 7.", "numeric") == "7"
    assert optimizer_cli._is_correct("B", "B", "letter")
    assert optimizer_cli._is_correct("yes", "YES", "word")
    assert optimizer_cli._is_correct("7", "7.0", "numeric")


def test_exact_and_latency_modes():
    assert optimizer_cli._guess_extract_mode({"question": "no answer row"}) == "latency"
    assert optimizer_cli._gold_value({"answer": "Michio Sugeno"}, "exact") == "Michio Sugeno"
    assert optimizer_cli._pred_value("The winner was Michio Sugeno.", "exact") == "the winner was michio sugeno"
    assert optimizer_cli._is_correct("Michio Sugeno.", "michio sugeno", "exact")
    assert optimizer_cli._is_correct("He is most known as Annick Bricaud", "Annick Bricaud", "exact")
    assert optimizer_cli._is_correct("April 1998", "April 1998", "exact")
    assert not optimizer_cli._is_correct("May 2000", "April 1998", "exact")
    assert optimizer_cli._is_correct("Vladimir", "Vladimir | Volodymyr", "exact")
    assert optimizer_cli._is_correct("Volodymyr", "Vladimir | Volodymyr", "exact")
    assert "Task:" in optimizer_cli._eval_prompt("do it", {}, "latency")
    assert "no explanation" in optimizer_cli._eval_prompt("q", {}, "exact")
    assert optimizer_cli._pred_value("", "exact") is None


def test_run_eval_latency_mode():
    samples = [{"question": "Do the thing."}, {"question": "Other task."}]

    class Gateway:
        cache_hits = 0
        cache_misses = 1

        def chat(self, messages, **kwargs):
            return f"Done with {messages[-1]['content'][:20]}"

    def fake_raw(messages, model=None, max_tokens=None):
        return ("Raw result.", False, model)

    summary = optimizer_cli.run_eval(n=2, gateway=Gateway(), raw=fake_raw,
                                     samples=samples, show=False,
                                     use_baseline=True)
    # extract="auto": run-level scored stays True (the mode supports grading),
    # but per-row row_scored is False because every sample is latency — no row
    # carries a correctness verdict and accuracy is None.
    assert summary["scored"] is True
    assert summary["optimized"]["accuracy"] is None
    assert summary["baseline"]["accuracy"] is None
    row = summary["rows"][0]
    assert "optimized_pred" in row
    assert "optimized_correct" not in row
    assert "baseline_correct" not in row


def test_normalize_row_question_fallbacks():
    for key, value in (("problem", "AIME math?"),
                       ("problem_statement", "Fix the bug."),
                       ("PROMPT", "Use the tools."),
                       ("base_description", "Implement a sampler.")):
        item = optimizer_cli._normalize_row({key: value})
        assert item["question"] == value
    joined = optimizer_cli._normalize_row({"answer": ["Vladimir", "Volodymyr"]})
    assert joined["answer"] == "Vladimir | Volodymyr"


def test_eval_prompt_variants():
    letter = optimizer_cli._eval_prompt("Q", {"choices": ["a", "b"]}, "letter")
    assert "A) a" in letter and "B) b" in letter
    word = optimizer_cli._eval_prompt("Q", {"context": "P", "question": "Q"}, "word")
    assert "P" in word and "yes or no" in word
    num = optimizer_cli._eval_prompt("Q", {}, "numeric")
    assert "Question: Q" in num


def test_run_eval_word_mode():
    samples = [{"question": "Q1", "context": "P", "answer": "yes"},
               {"question": "Q2", "context": "P", "answer": "no"}]

    class YesGateway:
        cache_hits = 0
        cache_misses = 1

        def chat(self, messages, **kwargs):
            return "Yes."

    def fake_raw(messages, model=None, max_tokens=None):
        return ("Not sure.", False, None)

    summary = optimizer_cli.run_eval(n=2, gateway=YesGateway(), raw=fake_raw,
                                     samples=samples, show=False, use_baseline=True)
    assert summary["optimized"]["answered"] == 2
    assert summary["optimized"]["correct"] == 1
    assert summary["baseline"]["answered"] == 2
    assert summary["baseline"]["correct"] == 0


def test_load_dataset_normalizes_label_names(monkeypatch):
    import datasets as ds_mod

    class FakeNames:
        names = ["no", "yes"]

    class FakeFeatures(dict):
        pass

    feats = FakeFeatures(label=FakeNames())

    class FakeDS:
        features = feats

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {"question": f"q{i}", "label": i % 2, "passage": f"ctx{i}"}

    monkeypatch.setattr(ds_mod, "load_dataset", lambda *a, **k: FakeDS())
    items = optimizer_cli.load_dataset("bogus/x", "y", "test", n=2, seed=1)
    assert len(items) == 2
    for item in items:
        assert item["answer"] in ("no", "yes")
        assert item["context"].startswith("ctx")


def test_load_dataset_normalizes_choices(monkeypatch):
    import datasets as ds_mod

    class FakeDS:
        features = None

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return {"question": f"q{i}", "choices": list("abcd"), "answer": i % 4}

    monkeypatch.setattr(ds_mod, "load_dataset", lambda *a, **k: FakeDS())
    items = optimizer_cli.load_dataset("x/y", "z", "test", n=3, seed=2)
    assert items[0]["choices"] == list("abcd")
    assert items[0]["answer_index"] in (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# EqSet-U / EqSet-V (SPEED_MATH_WISHES.md): unit-tolerance + chain verifier
# ---------------------------------------------------------------------------


def test_eqset_u_trailing_unit_lines_stay_bare_numbers():
    # EqSet-U: a single whitelisted unit on the trailing number line must not
    # defeat the bare-number verdict.
    assert optimizer_cli._extract_number("40 chickens.") == "40"
    assert optimizer_cli._extract_number("the answer is 40 dollars") == "40"
    assert optimizer_cli._extract_number("#### 60 cups") == "60"
    assert optimizer_cli._extract_number("she gave 60 cups.") == "60"
    assert optimizer_cli._extract_number("Total $1,248.") == "1248"
    # a unit must NOT be required: plain lines still short-circuit
    assert optimizer_cli._extract_number("#### 10000") == "10000"


def test_eqset_u_disjoint_from_parenthetical_logic():
    # Parenthetical priorities are unchanged by the unit whitelist.
    assert optimizer_cli._extract_number(
        "profit of $10,000 per day ($40,000 revenue, $30,000 costs)") == "10000"


def test_eqset_v_verifies_valid_chain():
    ok, end = optimizer_cli._verify_arithmetic_chain(
        "He gave 20 * 3 = 60 cups total.")
    assert ok is True
    assert end == 60.0


def test_eqset_v_flags_wrong_identity():
    ok, end = optimizer_cli._verify_arithmetic_chain("20 * 3 = 600")
    assert ok is False
    assert end == 60.0


def test_eqset_v_handles_no_triple():
    ok, end = optimizer_cli._verify_arithmetic_chain("no arithmetic here")
    assert ok is True
    assert end is None


def test_eqset_v_verifies_multiple_triples_with_x_notation():
    ok, end = optimizer_cli._verify_arithmetic_chain(
        "8 x 2 = 16; 16 + 4 = 20")
    assert ok is True
    assert end == 20.0


def test_run_bench_cold_error_preserves_baseline_ratio():
    """Bugfix: run_bench computed `row["cold_ms"] / baseline_ms` even when the
    cold call errored (no cold_ms key), raising KeyError and killing the whole
    bench report.  A failed cold + warm + OK baseline must render, not crash."""

    class FlakyGateway:
        cache_hits = 0
        cache_misses = 1

        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("cold timed out")
            return "pong"

    def fake_raw(messages, model=None, max_tokens=None):
        return ("base", False, model)

    summary = optimizer_cli.run_bench(n=1, gateway=FlakyGateway(), raw=fake_raw,
                                      use_baseline=True, show=False,
                                      warmup=False)
    row = summary["rows"][0]
    assert "cold_error" in row
    assert "cold_ms" not in row
    assert "baseline_ms" in row


def test_run_eval_latency_row_does_not_flip_scored_for_rest():
    """Bugfix: run_eval mutated the shared `scored` flag when a latency-mode
    row appeared, silently stripping accuracy reporting from every LATER row.
    A scored row following a latency row must still be graded."""

    class OneShotCorrectGateway:
        cache_hits = 0
        cache_misses = 1

        def chat(self, messages, **kwargs):
            return "yes"

    def sample(q, answer=None):
        item = {"question": q}
        if answer is not None:
            item["answer"] = answer
        return item

    samples = [
        sample("just time me.", None),          # latency mode (no gold)
        sample("is 2 prime?", "yes"),          # word mode — must stay scored
        sample("is 3 prime?", "yes"),
    ]

    def fake_raw(messages, model=None, max_tokens=None):
        return ("no", False, model)

    summary = optimizer_cli.run_eval(n=3, gateway=OneShotCorrectGateway(),
                                     raw=fake_raw, samples=samples, show=False,
                                     use_baseline=True)
    # Run-level scored flag comes from extract != "latency" (auto here).
    assert summary["scored"] is True
    latency, scored1, scored2 = summary["rows"]
    assert latency["extract_mode"] == "latency"
    assert "optimized_correct" not in latency
    assert scored1["optimized_correct"] is True
    assert scored2["optimized_correct"] is True
    # Accuracy denominator counts scorable rows, not diluted by latency row.
    assert summary["optimized"]["answered"] == 3
    assert summary["optimized"]["correct"] == 2
    assert summary["optimized"]["accuracy"] == 1.0


def test_make_gateway_rejects_bad_mode():
    """An unknown mode silently behaved as quality; fail loudly instead."""
    with pytest.raises(ValueError, match="unknown performance mode"):
        optimizer_cli.make_optimized_gateway("phi3:mini", "turbo")


def test_bench_warmup_bypasses_cache():
    """The warmup generation must not store a junk 'warmup' answer that a
    later real query could hit."""

    class RecordingGateway(FakeGateway):
        def __init__(self):
            super().__init__("pong")
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return self.reply

    gw = RecordingGateway()
    optimizer_cli.run_bench(n=1, gateway=gw, raw=lambda *a, **k: ("r", False, "m"),
                            show=False, warmup=True)
    first_msgs, first_kwargs = gw.calls[0]
    assert first_msgs == [{"role": "user", "content": "warmup"}]
    assert first_kwargs.get("use_cache") is False


def test_run_eval_restores_timeout_when_gateway_fails(monkeypatch):
    """The eval-timeout boost must not leak if gateway creation raises."""
    from config import settings
    monkeypatch.setattr(settings, "generation_timeout", 15)

    def boom(*a, **k):
        raise RuntimeError("no backend")

    monkeypatch.setattr(optimizer_cli, "make_optimized_gateway", boom)
    samples = [{"question": "What is 2+2?", "answer": "4"}]
    with pytest.raises(RuntimeError, match="no backend"):
        optimizer_cli.run_eval(samples=samples, n=1, show=False)
    assert settings.generation_timeout == 15


def test_main_chat_agent_forwards_model_and_no_rag():
    """main.py chat --use-agent must forward --model/--no-rag (were dropped)."""
    from unittest.mock import patch
    import main
    with patch("agents.langgraph_agent.chat",
               return_value="canned") as mock_chat:
        result = CliRunner().invoke(
            main.app, ["chat", "--use-agent", "--no-rag",
                       "--model", "mymodel", "hi"])
    assert result.exit_code == 0
    mock_chat.assert_called_once()
    _args, kwargs = mock_chat.call_args
    assert kwargs.get("model") == "mymodel"
    assert kwargs.get("use_rag") is False


def test_main_chat_direct_uses_gateway():
    """Direct path still routes through the balanced gateway."""
    from unittest.mock import MagicMock, patch
    import main
    gw = MagicMock()
    gw.chat.return_value = "direct hi"
    with patch("gateway.universal_enhanced_gateway.get_universal_gateway",
               return_value=gw):
        result = CliRunner().invoke(main.app, ["chat", "hi"])
    assert result.exit_code == 0
    assert "direct hi" in result.stdout


def test_main_ingest_none_provider_warns(monkeypatch):
    """Provider 'none' must say so instead of silently ingesting nothing."""
    import main
    from config import settings, RAGProvider
    monkeypatch.setattr(settings, "rag_provider", RAGProvider.NONE)
    result = CliRunner().invoke(main.app, ["ingest", "--path", "."])
    assert result.exit_code == 0
    assert "nothing ingested" in result.stdout