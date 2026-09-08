"""
tests/test_equations_memory.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for gateway/equations/memory_math.py — memory / context-window /
attention-heuristic / token-budget equations.
"""

import math
import pytest

from gateway.equations.memory_math import (
    recency_weight,
    priority_score,
    message_priority,
    MemoryWindow,
    consolidation_score,
    exponential_rehearsal,
    should_summarize,
    truncate_decay_stop,
    is_structured_text,
    compress_to_token_budget,
    ContextCompactor,
    attention_importance,
    token_importance,
    prune_filler_tokens,
    desired_output_length,
    response_length_policy,
    should_keep_turn,
    SlidingWindow,
    drop_oldest,
)


def _tokens(messages):
    from gateway.opt_core import estimate_tokens
    return sum(estimate_tokens(str(m.get("content", ""))) + 4 for m in messages)


def test_recency_weight_decreases_with_age_backward_in_time():
    total = 6
    # Newest position has weight 1.0; moving back decays by gamma each step.
    newest = recency_weight(total - 1, total)
    oldest = recency_weight(0, total)
    assert newest == pytest.approx(1.0)
    assert oldest < newest
    # Monotone non-decreasing as position advances toward the newest slot.
    weights = [recency_weight(p, total) for p in range(total)]
    assert weights == sorted(weights)
    # Step width matches gamma.
    assert recency_weight(3, total, gamma=0.9) == pytest.approx(0.9 * recency_weight(4, total, gamma=0.9))


def test_priority_score_bounded_and_saliency_weighted():
    assert 0.0 <= priority_score(1.0, 1.0, 1.0) <= 1.0
    assert priority_score(0.0, 1.0, 1.0) == 0.0  # no recency -> no priority
    high = priority_score(1.0, 0.9, 0.9)
    low = priority_score(1.0, 0.1, 0.1)
    assert high > low


def test_message_priority_saliency_and_mentions():
    base = {"role": "user", "content": "hello how are you"}
    salient = {"role": "user", "content": "IMPORTANT task: remember the deadline"}
    assert message_priority(salient, now=0.0, age=0.0, mention_count=0) > \
        message_priority(base, now=0.0, age=0.0, mention_count=0)
    # Mentions amplify; age decays.
    mentioned = message_priority(base, now=0.0, age=0.0, mention_count=9)
    plain = message_priority(base, now=0.0, age=0.0, mention_count=0)
    assert mentioned > plain
    fresh = message_priority(base, now=0.0, age=10.0, mention_count=0)
    stale = message_priority(base, now=0.0, age=10000.0, mention_count=0)
    assert fresh > stale
    assert 0.0 <= plain <= 1.0


def test_memory_window_trim_keeps_newest_and_fits_budget():
    w = MemoryWindow()
    for i in range(25):
        w.add({"role": "user", "content": f"message number {i}"},
              {"tokens": 10, "age": float(i) * 60.0})
    budget = 90  # holds newest (10) plus 8 more at 10 tokens each = 90
    dropped = w.trim_to_budget(budget)
    stats = w.stats()
    assert stats["tokens_used"] <= budget
    assert stats["newest_content"] == "message number 24"
    assert stats["count"] == 9
    assert len(dropped) == 25 - 9
    # Newest survived even though it is not the highest-scoring message.
    assert all(m.get("content") != "message number 24" for m in dropped)


def test_memory_window_sorted_by_value_and_reset():
    w = MemoryWindow()
    for i in range(10):
        w.add({"role": "user", "content": f"message {i}"},
              {"tokens": 5, "age": float(10 - i) * 3600.0,
               "mention_count": 0 if i % 2 else 5})
    top = w.sorted_by_value(3)
    assert len(top) == 3
    assert w.sorted_by_value(0) == []
    w.reset()
    assert w.stats()["count"] == 0
    assert w.stats()["tokens_used"] == 0


def test_consolidation_grows_with_age_and_revisits():
    young = consolidation_score(0.9, age=0.0, revisit_count=0)
    old = consolidation_score(0.9, age=30 * 86400.0, revisit_count=0)
    old_visited = consolidation_score(0.9, age=30 * 86400.0, revisit_count=9)
    assert 0.0 <= young <= old <= old_visited <= 1.0
    assert exponential_rehearsal(0.9, 0) == pytest.approx(0.9)
    assert exponential_rehearsal(0.9, 5) > exponential_rehearsal(0.9, 0)
    assert exponential_rehearsal(0.9, 10) <= 1.0


def test_should_summarize_near_budget():
    assert should_summarize(50, 100) is False
    assert should_summarize(70, 100) is True   # exactly the 0.7 fraction
    assert should_summarize(90, 100) is True
    assert should_summarize(1, 0) is True      # degenerate budget -> summarize
    assert should_summarize(0, 100) is False


def test_truncate_decay_stop_fits_and_marginal_stop():
    seq = [(5.0, 20), (4.0, 30), (3.0, 25)]
    remaining, dropped = truncate_decay_stop(seq, 100, decay=0.8)
    assert dropped == [] and len(remaining) == 3  # already fits
    # Decay low enough that dropping the second item is not worth it.
    seq2 = [(10.0, 100), (10.0, 60)]
    remaining2, dropped2 = truncate_decay_stop(seq2, 50, decay=0.02, min_margin=0.05)
    assert len(dropped2) == 1  # second drop marginal < 5% -> stopped early
    assert sum(t for _, t in remaining2) == 60  # still over budget, deliberately kept
    assert truncate_decay_stop([], 50, decay=0.8) == ([], [])


def test_compress_to_token_budget_and_compactor_preserve_structure_and_current():
    cc = ContextCompactor()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    for i in range(30):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role,
                     "content": f"filler leading to the eventual request #{i} with padding words"})
    msgs.append({"role": "user", "content": "current final request now"})

    original_tokens = _tokens(msgs)
    budget = 140
    compact, dropped, hint = cc.compact(msgs, budget, keep_ratio=0.5)

    assert compact[0]["role"] == "system"
    assert compact[-1]["role"] == "user"
    assert compact[-1]["content"] == "current final request now"
    assert _tokens(compact) <= budget
    assert _tokens(compact) < original_tokens
    assert dropped > 0
    assert "compressed" in hint
    # Caller's messages are untouched.
    assert len(msgs) == 32 and msgs[-1]["content"] == "current final request now"


def test_compress_exists_and_respects_budget():
    long_text = ("This is a fairly long stretch of ordinary prose. " * 40).strip()
    out = compress_to_token_budget(long_text, 40)
    from gateway.opt_core import estimate_tokens
    assert estimate_tokens(out) <= 40
    assert compress_to_token_budget("", 40) == ""
    assert compress_to_token_budget("hi there", 100) == "hi there"


def test_prune_filler_leaves_code_and_json_intact():
    json_str = '{"handler": "complete", "max_tokens": 1024, "messages": [{"role": "user", "content": "the and or the"}]}'
    snippet = "def compute(x, y):\n    return x * y + 1\n\nprint(compute(2, 3))"
    assert is_structured_text(json_str) is True
    assert is_structured_text(snippet) is True
    assert prune_filler_tokens(json_str, ["the", "and", "or"]) == json_str
    assert prune_filler_tokens(snippet, ["def", "return", "print"]) == snippet
    # Plain prose gets pruned.
    pruned = prune_filler_tokens("the quick and the dead", ["the", "and"])
    assert pruned == "quick dead"
    assert "the" not in pruned


def test_attention_importance_and_token_importance_bounded():
    assert 0.0 <= attention_importance(0, 0, 0.0) <= 1.0
    assert attention_importance(500, 0, 1.0) > attention_importance(0, 0, 0.0)
    assert 0.0 <= token_importance(2.0, 0, sep=1.0) <= 1.0
    assert token_importance(10.0, 0, sep=1.0) > token_importance(0.0, 0, sep=1.0)


def test_desired_output_length_and_response_policy():
    assert 64 <= desired_output_length(10) <= 4096
    assert desired_output_length(10, is_coding=True) >= desired_output_length(10)
    assert desired_output_length(10, is_math=True) >= desired_output_length(10)
    assert response_length_policy(999, min_tokens=10, max_tokens=20) == 20
    assert response_length_policy(1, min_tokens=10, max_tokens=20) == 10
    assert response_length_policy(15, min_tokens=20, max_tokens=10) == 15  # swapped bounds


def test_should_keep_turn_high_cosine_true():
    theme = [1.0, 0.0, 0.0, 0.0]
    on_topic = [1.0, 0.0, 0.0, 0.0]
    off_topic = [0.0, 0.0, 0.0, 1.0]
    assert should_keep_turn(on_topic, theme, threshold=0.7) is True
    assert should_keep_turn(off_topic, theme, threshold=0.7) is False
    # Empty embeddings compare at 0.0 -> dropped for a positive threshold.
    assert should_keep_turn([], theme, threshold=0.7) is False


def test_sliding_window_and_drop_oldest():
    sw = SlidingWindow()
    for i in range(10):
        sw.add(i)
    assert sw.window(3) == [7, 8, 9]
    assert sw.window(0) == []
    assert sw.window(100) == list(range(10))
    assert sw.stats()["added_total"] == 10
    capped = SlidingWindow(maxlen=4)
    for i in range(10):
        capped.add(i)
    assert len(capped.window(10)) == 4
    assert drop_oldest(list(range(10)), 3) == [7, 8, 9]
    assert drop_oldest([], 3) == []
    assert drop_oldest([1, 2, 3], 0) == []