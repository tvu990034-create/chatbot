"""CI smoke test: construct the chatbot graph within budget.

Extracted from .github/workflows/deploy.yml (inline ``python -c`` blocks
cannot be indented inside a YAML literal block, which broke parsing).
Run from the repo root:  python .github/scripts/smoke_init.py
"""
import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from local_chatbot.config import ChatbotConfig  # noqa: E402
from local_chatbot.graph import LocalChatGraph  # noqa: E402

start = time.time()
cfg = ChatbotConfig()
chatbot = LocalChatGraph(cfg)
init_time = time.time() - start

print("Init time: %.2fs" % init_time)
assert init_time < 5.0, "Too slow: %.2fs" % init_time
print("CLI init test PASSED")
