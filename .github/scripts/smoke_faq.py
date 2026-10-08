"""CI smoke test: FAQ fast path answers instantly with source 'faq'.

Extracted from .github/workflows/deploy.yml (inline ``python -c`` blocks
cannot be indented inside a YAML literal block, which broke parsing).
Run from the repo root:  python .github/scripts/smoke_faq.py
"""
import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from local_chatbot.config import ChatbotConfig  # noqa: E402
from local_chatbot.graph import LocalChatGraph  # noqa: E402

cfg = ChatbotConfig()
chatbot = LocalChatGraph(cfg)

start = time.time()
result = chatbot.chat("What is the capital of France?")
latency = time.time() - start

print("Response: %s" % result.response)
print("Source: %s" % result.source)
print("Latency: %.2fs" % latency)
assert result.source == "faq", \
    "Expected FAQ source, got %s" % result.source
assert latency < 1.0, "FAQ response too slow: %.2fs" % latency
print("Fast path test PASSED")
