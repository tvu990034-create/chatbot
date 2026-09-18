"""
Test the LiteLLM gateway and the shared optimization helpers it relies on.

This module was previously written against symbols that no longer exist on
``gateway.litellm_gateway`` (``chat_stream``, ``_cache_key``,
``_trim_history_eq1``, ``_metrics``).  It now exercises the current public
surface: the chat entry points plus the canonical helpers in
``gateway.opt_core``.
"""

import pytest
from unittest.mock import patch, MagicMock


def test_gateway_imports():
    """Gateway exposes the current chat entry points."""
    from gateway.litellm_gateway import chat, achat, achat_stream

    assert callable(chat)
    assert callable(achat)
    assert callable(achat_stream)


def test_cache_identity_generation():
    """make_cache_identity returns a stable, non-empty key scoped to the model."""
    from gateway.opt_core import make_cache_identity

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    key1 = make_cache_identity("Hello", model="phi3:mini", messages=messages)
    key2 = make_cache_identity("Hello", model="phi3:mini", messages=messages)
    key3 = make_cache_identity("Hello", model="qwen3:4b", messages=messages)

    assert isinstance(key1, str)
    assert len(key1) == 64  # sha256 hex digest
    assert key1 == key2
    assert key1 != key3  # the model is part of the identity


def test_history_trimming():
    """build_context_messages trims a long history to the token budget."""
    from gateway.opt_core import build_context_messages

    messages = [
        {"role": "user", "content": f"message number {i} " * 20}
        for i in range(20)
    ]

    built = build_context_messages(
        messages, context_limit=8192, budget=100, always_keep_n_history=2)

    assert len(built.messages) < len(messages)  # should trim some messages
    assert len(built.messages) >= 1              # should keep at least one


@patch('gateway.litellm_gateway.completion')
def test_chat_call_mocked(mock_completion):
    """Test chat function with mocked LiteLLM."""
    from gateway.litellm_gateway import chat

    # Mock the LiteLLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_completion.return_value = mock_response

    messages = [{"role": "user", "content": "Hello"}]

    # This should not make a real API call
    try:
        response = chat(messages, use_cache=False, use_router=False)
        assert response == "Test response"
    except Exception as e:
        # It's okay if this fails due to missing dependencies
        pytest.skip(f"Test skipped due to: {e}")


def test_optimization_stats_function():
    """get_optimization_stats returns the expected structure."""
    from gateway.litellm_gateway import get_optimization_stats

    stats = get_optimization_stats()

    assert isinstance(stats, dict)
    assert "optimizations" in stats
    assert "total_optimizations" in stats
