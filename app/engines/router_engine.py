"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
DynamicRouterEngine - Route queries to different models based on complexity
Simple queries go to small model (faster/cheaper), complex queries to main model
"""
import os
from typing import Tuple, Dict
import openai
import logging

from ..identity import ConversationState

logger = logging.getLogger(__name__)


class DynamicRouterEngine:
    """Dynamic routing engine that classifies queries and routes to appropriate model"""
    
    def __init__(self, cfg: dict):
        self.simple_model = cfg.get('simple_model', os.environ.get('FASTCLOUD_SIMPLE_MODEL', 'gpt-3.5-turbo'))
        self.default_model = cfg.get('default_model', os.environ.get('FASTCLOUD_DEFAULT_MODEL', 'gpt-4.1-nano'))
        self.simple_max_tokens = cfg.get('simple_max_tokens', int(os.environ.get('FASTCLOUD_SIMPLE_MAX_TOKENS', '64')))
        self.default_max_tokens = cfg.get('max_tokens', 256)
        
        # Create OpenAI clients for both models
        api_key = cfg.get('api_key', os.environ.get('OPENAI_API_KEY'))
        base_url = cfg.get('base_url', None)
        
        self.simple_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.default_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        self.temperature = cfg.get('temperature', 0.0)
        
        # Keywords that indicate complexity
        self.complex_keywords = ['explain', 'how to', 'why', 'describe', 'analyze', 'compare', 'detail', 'elaborate']
        
        # Metrics tracking
        self.simple_count = 0
        self.complex_count = 0
        
        logger.info(f"DynamicRouterEngine initialized: simple={self.simple_model}, default={self.default_model}")
    
    def _classify_query(self, query: str) -> bool:
        """
        Classify query as simple (True) or complex (False)
        Returns True for simple, False for complex
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Check for complex keywords
        has_complex_keyword = any(keyword in query_lower for keyword in self.complex_keywords)
        
        # Simple if: word count ≤ 8 AND no complex keywords
        is_simple = word_count <= 8 and not has_complex_keyword
        
        return is_simple
    
    def generate(self, state: ConversationState, user_input: str, seed: int, prompt_override: str = None, enable_reasoning: bool = False) -> Tuple[str, ConversationState, Dict]:
        """Generate response with dynamic model routing"""
        # Build prompt
        if prompt_override:
            prompt = prompt_override
        else:
            history_text = '\n'.join(f"{t.role}: {t.content}" for t in state.history)
            prompt = f"system: You are a fast assistant.\n{history_text}\nuser: {user_input}\nassistant:"
        
        # Classify query
        is_simple = self._classify_query(user_input)
        
        if is_simple:
            self.simple_count += 1
            client = self.simple_client
            model = self.simple_model
            max_tokens = self.simple_max_tokens
            logger.info(f"DynamicRouter: routing to simple model ({model})")
        else:
            self.complex_count += 1
            client = self.default_client
            model = self.default_model
            max_tokens = self.default_max_tokens
            logger.info(f"DynamicRouter: routing to default model ({model})")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating with {model}: {e}")
            reply = f"Error generating response: {str(e)[:100]}"
        
        # Update conversation state
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
            'router_classification': 'simple' if is_simple else 'complex',
            'router_model_used': model,
        }
