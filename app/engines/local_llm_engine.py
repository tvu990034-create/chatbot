"""
Local LLM Inference Engine with Reasoning Capabilities
Uses llama-cpp-python for fully local inference with chain-of-thought reasoning
"""
import os
from typing import Tuple, Dict, Optional
from ..identity import ConversationState, ConversationTurn


class LocalLLMEngine:
    """Local LLM inference engine with chain-of-thought reasoning"""
    
    def __init__(self, model_path: str = None, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """
        Initialize local LLM engine
        
        Args:
            model_path: Path to GGUF model file (e.g., models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        """
        self.model_path = model_path or self._find_model()
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        self._load_model()
        
        # Reasoning prompt templates
        self.reasoning_system_prompt = """You are a helpful AI assistant that thinks through problems step by step.
When answering questions, you should:
1. Break down the problem into smaller parts
2. Consider different approaches
3. Reason through the solution
4. Provide a clear, well-reasoned answer

Think step by step and show your reasoning process."""
        
        self.standard_system_prompt = """You are a helpful AI assistant with expertise in machine learning, natural language processing, and technical systems.
Provide clear, accurate, and detailed answers."""
    
    def _find_model(self) -> str:
        """Find available GGUF model in models directory"""
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".gguf"):
                    return os.path.join(models_dir, file)
        # Fallback to common path
        return os.path.join(os.path.dirname(__file__), "..", "..", "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    def _load_model(self):
        """Load the LLM model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            print(f"Loaded local LLM from {self.model_path}")
        except ImportError:
            print("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            print("Falling back to simulated responses.")
            self.model = None
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    
    def generate(
        self, 
        state: ConversationState, 
        user_input: str, 
        seed: int = None, 
        prompt_override: str = None,
        enable_reasoning: bool = True
    ) -> Tuple[str, ConversationState, Dict]:
        """
        Generate response with optional chain-of-thought reasoning
        
        Args:
            state: Conversation state
            user_input: User's message
            seed: Random seed for reproducibility
            prompt_override: Override the default prompt
            enable_reasoning: Enable chain-of-thought reasoning
        
        Returns:
            Tuple of (response, new_state, metrics)
        """
        if self.model is None:
            # Fallback to simple responses if model not loaded
            return self._fallback_response(state, user_input, seed)
        
        # Build prompt
        if prompt_override:
            prompt = prompt_override
        else:
            system_prompt = self.reasoning_system_prompt if enable_reasoning else self.standard_system_prompt
            prompt = self._build_prompt(state, user_input, system_prompt)
        
        # Add reasoning instruction if enabled
        if enable_reasoning and "step by step" not in prompt.lower():
            prompt += "\n\nPlease think step by step to answer this question."
        
        try:
            # Generate response
            response = self.model(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "user:", "Assistant:", "assistant:"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            # Extract reasoning and final answer if reasoning was enabled
            if enable_reasoning:
                generated_text = self._extract_reasoning(generated_text)
            
            # Update conversation state
            new_state = state
            new_state.history.append(ConversationTurn(role="user", content=user_input))
            new_state.history.append(ConversationTurn(role="assistant", content=generated_text))
            
            metrics = {
                "tokens_generated": response['usage']['total_tokens'] if 'usage' in response else len(generated_text.split()),
                "inference_time_ms": response.get('timings', {}).get('prompt_ms', 0) + response.get('timings', {}).get('prediction_ms', 0),
                "cache_hit": False,
                "reasoning_enabled": enable_reasoning,
                "model_path": self.model_path
            }
            
            return generated_text, new_state, metrics
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return self._fallback_response(state, user_input, seed)
    
    def _build_prompt(self, state: ConversationState, user_input: str, system_prompt: str) -> str:
        """Build conversation prompt from history"""
        prompt = f"System: {system_prompt}\n\n"
        
        # Add conversation history (last 10 turns to save context)
        for turn in state.history[-10:]:
            prompt += f"{turn.role.capitalize()}: {turn.content}\n"
        
        prompt += f"User: {user_input}\nAssistant:"
        return prompt
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract the final answer from reasoning text"""
        # If the text contains reasoning markers, try to extract just the answer
        lines = text.split('\n')
        answer_lines = []
        in_reasoning = False
        
        for line in lines:
            line_lower = line.lower()
            # Look for transition from reasoning to answer
            if any(marker in line_lower for marker in ['therefore', 'thus', 'so', 'answer:', 'conclusion:', 'in summary']):
                in_reasoning = False
                answer_lines.append(line)
            elif in_reasoning:
                continue  # Skip reasoning lines
            else:
                answer_lines.append(line)
            
            # Detect reasoning markers
            if any(marker in line_lower for marker in ['let me think', 'first', 'second', 'step', 'consider']):
                in_reasoning = True
        
        if answer_lines:
            return '\n'.join(answer_lines).strip()
        return text
    
    def _fallback_response(self, state: ConversationState, user_input: str, seed: int = None) -> Tuple[str, ConversationState, Dict]:
        """Fallback response when model is not available"""
        import random
        
        if seed is not None:
            random.seed(seed + hash(user_input))
        else:
            random.seed(hash(user_input))
        
        # Simple keyword-based responses
        responses = [
            "I understand your question. Let me think about this step by step.",
            "That's an interesting question. Here's my reasoning process:",
            "Let me break this down into parts to give you a thorough answer.",
            "I need to consider several factors to answer this properly."
        ]
        
        response = random.choice(responses)
        
        # Try to use retrieved context if available in state
        if hasattr(state, 'retrieved_context') and state.retrieved_context:
            response += f"\n\nBased on the information: {state.retrieved_context[:200]}..."
        
        new_state = state
        new_state.history.append(ConversationTurn(role="user", content=user_input))
        new_state.history.append(ConversationTurn(role="assistant", content=response))
        
        metrics = {
            "tokens_generated": len(response.split()),
            "inference_time_ms": 0.0,
            "cache_hit": False,
            "reasoning_enabled": False,
            "fallback": True
        }
        
        return response, new_state, metrics
    
    def is_available(self) -> bool:
        """Check if the model is available"""
        return self.model is not None
