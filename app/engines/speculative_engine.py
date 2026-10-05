"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
SpeculativeCloudEngine - Speculative decoding with draft and verifier models
Uses a cheap draft model to propose tokens and a large verifier to validate them
"""
import os
from typing import Tuple, Dict
import openai
import logging

from ..identity import ConversationState

logger = logging.getLogger(__name__)


class SpeculativeCloudEngine:
    """Speculative decoding engine with draft and verifier models"""
    
    def __init__(self, cfg: dict, http_client=None):
        self.draft_model = cfg.get('draft_model', os.environ.get('SPECULATIVE_DRAFT_MODEL', 'gpt-3.5-turbo'))
        self.verifier_model = cfg.get('verifier_model', os.environ.get('SPECULATIVE_VERIFIER_MODEL', 'gpt-4o'))
        
        # Create clients for both models
        api_key = cfg.get('api_key', os.environ.get('OPENAI_API_KEY'))
        base_url = cfg.get('base_url', None)
        
        self.draft_client = openai.OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self.verifier_client = openai.OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        
        self.temperature = cfg.get('temperature', 0.0)
        self.max_tokens = cfg.get('max_tokens', 256)
        self.draft_max_tokens = cfg.get('draft_max_tokens', 5)
        
        # Metrics tracking
        self.draft_tokens_generated = 0
        self.verifier_tokens_generated = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0
        
        logger.info(f"SpeculativeCloudEngine initialized: draft={self.draft_model}, verifier={self.verifier_model}")
    
    def _generate_draft(self, prompt: str, seed: int) -> str:
        """Generate draft tokens from the cheap draft model"""
        try:
            response = self.draft_client.chat.completions.create(
                model=self.draft_model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                max_tokens=self.draft_max_tokens,
                seed=seed,
                stream=False,
            )
            draft = response.choices[0].message.content.strip()
            self.draft_tokens_generated += len(draft.split())
            return draft
        except Exception as e:
            logger.error(f"Draft generation error: {e}")
            return ""
    
    def _verify_draft(self, prompt: str, draft: str, seed: int) -> Tuple[str, int]:
        """
        Verify draft tokens using the large verifier model
        Returns (accepted_text, num_accepted_tokens)
        """
        if not draft:
            return "", 0
        
        try:
            # Send original prompt + draft to verifier with logprobs
            full_prompt = f"{prompt} {draft}"
            response = self.verifier_client.chat.completions.create(
                model=self.verifier_model,
                messages=[{'role': 'user', 'content': full_prompt}],
                temperature=self.temperature,
                max_tokens=len(draft.split()),
                seed=seed,
                logprobs=True,
                top_logprobs=5,
                stream=False,
            )
            
            verifier_output = response.choices[0].message.content.strip()
            self.verifier_tokens_generated += len(verifier_output.split())
            
            # Simple acceptance: if verifier output starts with draft tokens, accept them
            # This is a simplified approach - in production, you'd compare token-by-token
            draft_words = draft.split()
            verifier_words = verifier_output.split()
            
            accepted_count = 0
            for i, (dw, vw) in enumerate(zip(draft_words, verifier_words)):
                if dw.lower() == vw.lower():
                    accepted_count += 1
                else:
                    break
            
            accepted_text = " ".join(draft_words[:accepted_count])
            self.accepted_tokens += accepted_count
            self.rejected_tokens += len(draft_words) - accepted_count
            
            return accepted_text, accepted_count
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return "", 0
    
    def generate(self, state: ConversationState, user_input: str, seed: int, prompt_override: str = None, enable_reasoning: bool = False) -> Tuple[str, ConversationState, Dict]:
        """Generate response using speculative decoding"""
        # Build prompt
        if prompt_override:
            prompt = prompt_override
        else:
            history_text = '\n'.join(f"{t.role}: {t.content}" for t in state.history)
            prompt = f"system: You are a fast assistant.\n{history_text}\nuser: {user_input}\nassistant:"
        
        # Reset metrics for this generation
        self.draft_tokens_generated = 0
        self.verifier_tokens_generated = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0
        
        # Speculative decoding loop
        full_response = ""
        current_prompt = prompt
        tokens_generated = 0
        
        while tokens_generated < self.max_tokens:
            # Generate draft tokens
            draft = self._generate_draft(current_prompt, seed)
            if not draft:
                break
            
            # Verify draft tokens
            accepted_text, accepted_count = self._verify_draft(current_prompt, draft, seed)
            
            if accepted_count > 0:
                # Append accepted tokens
                full_response += (" " if full_response else "") + accepted_text
                tokens_generated += accepted_count
                current_prompt = f"{current_prompt} {accepted_text}"
            else:
                # If no tokens accepted, break to avoid infinite loop
                # In production, you might want to fall back to verifier-only generation
                break
            
            # If we didn't accept all draft tokens, the remaining need to be regenerated
            if accepted_count < len(draft.split()):
                # For simplicity, we break here. In production, continue with remaining tokens
                break
        
        # If we didn't generate enough tokens, fall back to direct verifier generation
        if tokens_generated < 16:  # Minimum reasonable response
            logger.info("Falling back to direct verifier generation")
            try:
                response = self.verifier_client.chat.completions.create(
                    model=self.verifier_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=seed,
                )
                full_response = response.choices[0].message.content.strip()
                self.verifier_tokens_generated += len(full_response.split())
            except Exception as e:
                logger.error(f"Fallback generation error: {e}")
                full_response = "Error generating response"
        
        # Update conversation state
        new_state = ConversationState(
            history=list(state.history),
            token_ids=list(state.token_ids),
            kv_cache=None,
        )
        new_state.append('user', user_input)
        new_state.append('assistant', full_response)
        
        # Calculate metrics
        total_generated = len(full_response.split())
        
        logger.info(
            f"SpeculativeCloudEngine: draft_tokens={self.draft_tokens_generated}, "
            f"verifier_tokens={self.verifier_tokens_generated}, "
            f"accepted={self.accepted_tokens}, rejected={self.rejected_tokens}, "
            f"total_output={total_generated}"
        )
        
        return full_response, new_state, {
            'prompt_tokens': 0,
            'generated_tokens': total_generated,
            'speculative_enabled': 1,
            'kv_pruned_tokens': 0,
            'tome_merged_tokens': 0,
            'draft_tokens_generated': self.draft_tokens_generated,
            'verifier_tokens_generated': self.verifier_tokens_generated,
            'accepted_tokens': self.accepted_tokens,
            'rejected_tokens': self.rejected_tokens,
        }
