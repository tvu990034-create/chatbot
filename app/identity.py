"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from functools import lru_cache

_sha256_cache: Dict[bytes, str] = {}


def _sha256_bytes(data: bytes) -> str:
    if data in _sha256_cache:
        return _sha256_cache[data]
    result = hashlib.sha256(data).hexdigest()
    # Limit cache size to prevent memory bloat
    if len(_sha256_cache) < 10000:
        _sha256_cache[data] = result
    return result


def compute_model_hash(weights_file_path: str) -> str:
    """
    Equation 2:
        H = SHA256(weights_file)
    """
    hasher = hashlib.sha256()
    try:
        with open(weights_file_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return "model_file_not_found"
    except Exception as e:
        return f"hash_error: {str(e)[:20]}"


@dataclass(slots=True)
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class ConversationState:
    """Manages conversation state with enhanced context tracking."""
    history: List[ConversationTurn] = field(default_factory=list)
    max_history: int = 10
    context_summary: str = ""
    token_ids: List[int] = field(default_factory=list)
    kv_cache: Optional[Any] = None
    _canonical_json_cache: Optional[str] = None

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn with automatic history management."""
        turn = ConversationTurn(role=role, content=content)
        self.history.append(turn)
        
        # Trim history if it exceeds max length
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Disable context summary for performance (can be enabled in production)
        # if len(self.history) % 3 == 0:
        #     self._update_context_summary()
        
        self._canonical_json_cache = None  # Invalidate cache

    def _update_context_summary(self) -> None:
        """Generate a summary of the conversation context."""
        if not self.history:
            self.context_summary = ""
            return
        
        recent_turns = self.history[-3:]
        summary_parts = []
        for turn in recent_turns:
            # Handle both ConversationTurn objects and dictionaries
            if isinstance(turn, dict):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
            else:
                role = turn.role
                content = turn.content
            summary_parts.append(f"{role}: {content[:50]}...")
        
        self.context_summary = " | ".join(summary_parts)

    def get_context_window(self, window_size: int = 5) -> List[Dict]:
        """Get the most recent conversation turns as a list of dictionaries."""
        recent_turns = self.history[-window_size:]
        result = []
        for turn in recent_turns:
            # Handle both ConversationTurn objects and dictionaries
            if isinstance(turn, dict):
                result.append(turn)
            else:
                result.append({"role": turn.role, "content": turn.content})
        return result

    def clear(self) -> None:
        """Clear the conversation history."""
        self.history = []
        self.context_summary = ""

    def canonical_json(self) -> str:
        if self._canonical_json_cache is None:
            self._canonical_json_cache = json.dumps(
                {
                    "history": [dataclasses.asdict(t) for t in self.history],
                    "token_ids": self.token_ids,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        return self._canonical_json_cache

    def state_fingerprint(self) -> str:
        return _sha256_bytes(self.canonical_json().encode("utf-8"))


def autoregressive_joint_probability(
    conditionals: Sequence[float],
) -> float:
    """
    Equation 4:
        P(y1..yT|...) = product_t P(yt | y<t,...)
    """
    p = 1.0
    for c in conditionals:
        p *= float(c)
    return p


def deterministic_chat_identity_key(
    model_hash: str,
    state: ConversationState,
    user_input: str,
    seed: int,
) -> str:
    """
    Equation 1 identity key material:
        Chat(H,C,x,seed) is deterministic for fixed inputs.
    """
    payload = json.dumps(
        {
            "H": model_hash,
            "C": state.canonical_json(),
            "x": user_input,
            "seed": int(seed),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_bytes(payload.encode("utf-8"))

