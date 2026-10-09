"""Tests for gateway.equations.budget_math (ported legacy EQ-DYNAMIC-TOKENS)."""


class TestDynamicMaxTokens:
    def test_empty_returns_base(self):
        from gateway.equations.budget_math import dynamic_max_tokens
        assert dynamic_max_tokens("") == 20
        assert dynamic_max_tokens("   ") == 20
        assert dynamic_max_tokens(None) == 20

    def test_short_question_gets_small_budget(self):
        from gateway.equations.budget_math import dynamic_max_tokens
        # 3 words -> 20 + 2*3 = 26
        assert dynamic_max_tokens("What is 12*8?") == 26

    def test_scales_with_length_inside_caps(self):
        from gateway.equations.budget_math import dynamic_max_tokens
        short = dynamic_max_tokens("Hi there friend")
        long_q = dynamic_max_tokens(" ".join(["word"] * 40))
        assert 10 <= short < long_q <= 256

    def test_explainer_trigger_gets_long_cap(self):
        from gateway.equations.budget_math import dynamic_max_tokens
        assert dynamic_max_tokens(
            "Explain what a transformer model does") == 150
        assert dynamic_max_tokens(
            "compare cats versus dogs in detail") == 150

    def test_long_cap_respects_max_cap(self):
        from gateway.equations.budget_math import dynamic_max_tokens
        assert dynamic_max_tokens("Explain this", max_cap=100) == 100

    def test_floor_holds(self):
        from gateway.equations.budget_math import dynamic_max_tokens
        assert dynamic_max_tokens("Hi", base=0, floor=10) == 10

    def test_reasoning_gets_no_cap(self):
        # Thinking models burn ~10x visible budget in hidden CoT: any
        # small cap truncates into garbage, so these return None and the
        # caller must use the configured default.  A short budget here is
        # a shorter wrong answer, not speed.
        from gateway.equations.budget_math import dynamic_max_tokens
        assert dynamic_max_tokens("Prove that sqrt(2) is irrational") is None
        assert dynamic_max_tokens("Why is the sky blue?") is None
        assert dynamic_max_tokens("Derive the result") is None
