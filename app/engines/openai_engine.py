"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
OpenAI Cloud Engine for Cloud AI Chatbot
"""
import os
from typing import Tuple, Dict
import openai

from ..identity import ConversationState


class CloudOpenAiEngine:
    """OpenAI-based inference engine for cloud deployment"""
    
    def __init__(self, cfg: dict, http_client=None):
        self.model = cfg.get('model', 'gpt-4.1-nano')
        self.client = openai.OpenAI(
            api_key=cfg.get('api_key', os.environ.get('OPENAI_API_KEY')),
            base_url=cfg.get('base_url', None),
            http_client=http_client,
        )
        self.temperature = cfg.get('temperature', 0.0)
        self.max_tokens = cfg.get('max_tokens', 256)

    def generate(self, state: ConversationState, user_input: str, seed: int, prompt_override: str = None, enable_reasoning: bool = False) -> Tuple[str, ConversationState, Dict]:
        if prompt_override:
            prompt = prompt_override
        else:
            history_text = '\n'.join(f"{t.role}: {t.content}" for t in state.history)
            prompt = f"system: You are a fast assistant.\n{history_text}\nuser: {user_input}\nassistant:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=seed,
            )
            try:
                reply = response.choices[0].message.content.strip()
            except (IndexError, AttributeError) as e:
                reply = f"Error parsing response: {str(e)[:100]}"
        except Exception as e:
            reply = f"Error generating response: {str(e)[:100]}"
        
        new_state = ConversationState(
            history=list(state.history),
            token_ids=list(state.token_ids),
            kv_cache=None,
        )
        new_state.append('user', user_input)
        new_state.append('assistant', reply)
        
        return reply, new_state, {
            'prompt_tokens': 0,
            'generated_tokens': len(reply.split()),
            'speculative_enabled': 0,
            'kv_pruned_tokens': 0,
            'tome_merged_tokens': 0,
        }
