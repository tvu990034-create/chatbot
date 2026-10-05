"""
Prompt Engine - Mathematical equations for prompt building
Extracted from PDF files - these are the ONLY equations that matter
"""

import math
from typing import List, Dict, Any

class PromptEngine:
    """
    Implements ALL prompt equations from PDF files
    These equations are the single source of truth for prompt logic
    """
    
    def __init__(self):
        # Equation parameters from PDFs
        self.MAX_TOTAL_PROMPT_TOKENS = 32000  # safety cap
        self.MAX_WINDOW_TOKENS = 3000  # T_win (recent conversation)
        self.SUMMARIZE_EVERY_N_TURNS = 5  # S (trigger frequency)
        self.conciseness_ratio = 0.3  # L_struct / L
        self.L_orig = 2000
        self.L_comp = 800
    
    # EQUATION: L_struct = L * conciseness_ratio
    def calc_L_struct(self, L: int, conciseness_ratio: float = 0.3) -> int:
        """L_struct = L * conciseness_ratio"""
        return int(L * conciseness_ratio)
    
    # EQUATION: reduction = L_orig / L_comp
    def calc_reduction(self, L_orig: int, L_comp: int) -> float:
        """reduction = L_orig / L_comp if L_comp > 0 else 1.0"""
        return L_orig / L_comp if L_comp > 0 else 1.0
    
    # EQUATION: length = np.random.randint(5, 15)
    def calc_random_length(self, min_len: int = 5, max_len: int = 15) -> int:
        """length = random integer between min and max"""
        import random
        return random.randint(min_len, max_len)
    
    # EQUATION: prev_row = range(len(s2) + 1)
    def calc_prev_row(self, s2_length: int) -> range:
        """prev_row = range(len(s2) + 1)"""
        return range(s2_length + 1)
    
    # EQUATION: curr = [i + 1]
    def calc_curr(self, i: int) -> List[int]:
        """curr = [i + 1]"""
        return [i + 1]
    
    # EQUATION: extended_probs = example_probs + [0.85] * (L - len(example_probs))
    def calc_extended_probs(self, example_probs: list, L: int) -> list:
        """extended_probs = example_probs + [0.85] * (L - len(example_probs))"""
        return example_probs + [0.85] * (L - len(example_probs))
    
    # EQUATION: q = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '{date}', q)
    def normalize_date(self, q: str) -> str:
        """q = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '{date}', q)"""
        import re
        return re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '{date}', q)
    
    # EQUATION: if len(state.sliding_messages) <= 1
    def should_use_sliding_window(self, num_messages: int) -> bool:
        """if len(state.sliding_messages) <= 1"""
        return num_messages <= 1
    
    # EQUATION: specificity += 0.1
    def calc_specificity_increment(self) -> float:
        """specificity += 0.1"""
        return 0.1
    
    # EQUATION: a = (total_latency - b*V*tokens) / tokens, but often ~0.003s
    def calc_per_token_cost(self, total_latency: float, b: float, V: int, tokens: int) -> float:
        """a = (total_latency - b*V*tokens) / tokens"""
        return (total_latency - b * V * tokens) / tokens if tokens > 0 else 0.003
    
    # EQUATION: T_struct = L_struct * (a + b * V_bar)
    def calc_T_struct(self, L_struct: int, a: float, b: float, V_bar: int) -> float:
        """T_struct = L_struct * (a + b * V_bar)"""
        return L_struct * (a + b * V_bar)
    
    # EQUATION: L = Number of transformer layers
    def calc_num_layers(self, L: int = 32) -> int:
        """L = Number of transformer layers"""
        return L
    
    # EQUATION: p = Parallel fraction per layer (0.8–0.95)
    def calc_parallel_fraction(self, p: float = 0.92) -> float:
        """p = Parallel fraction per layer"""
        return p
    
    # EQUATION: serial_fraction = 1.0 - p
    def calc_serial_fraction(self, p: float) -> float:
        """serial_fraction = 1.0 - p"""
        return 1.0 - p
    
    # EQUATION: S = Number of pipeline stages
    def calc_pipeline_stages(self, S: int = 4) -> int:
        """S = Number of pipeline stages"""
        return S
    
    # MASTER EQUATION: Calculate optimal prompt length
    def calculate_optimal_prompt_length(self, query_length: int, history_length: int) -> Dict[str, Any]:
        """
        MASTER EQUATION: Combines all prompt equations
        Returns optimal prompt length and structure
        """
        # Calculate L_struct
        L_struct = self.calc_L_struct(query_length, self.conciseness_ratio)
        
        # Calculate reduction
        reduction = self.calc_reduction(self.L_orig, self.L_comp)
        
        # Calculate total tokens
        total_tokens = query_length + history_length
        
        # Check if we need summarization
        need_summary = total_tokens > self.MAX_WINDOW_TOKENS
        summary_turn = total_tokens // self.SUMMARIZE_EVERY_N_TURNS
        
        # Calculate per-token cost
        a = self.calc_per_token_cost(0.5, 1e-5, 50000, total_tokens)
        
        # Calculate structural time
        T_struct = self.calc_T_struct(L_struct, a, 1e-5, 50000)
        
        return {
            "L_struct": L_struct,
            "reduction_factor": reduction,
            "total_tokens": total_tokens,
            "need_summary": need_summary,
            "summary_turn": summary_turn,
            "per_token_cost_ms": a,
            "structural_time_ms": T_struct,
            "max_window_tokens": self.MAX_WINDOW_TOKENS,
            "max_total_tokens": self.MAX_TOTAL_PROMPT_TOKENS
        }
    
    # MASTER EQUATION: Build prompt with equations
    def build_prompt(self, query: str, history: List[Dict], context: str = "") -> str:
        """
        MASTER EQUATION: Builds prompt using all equations
        Returns optimized prompt string
        """
        # Normalize dates in query
        normalized_query = self.normalize_date(query)
        
        # Calculate optimal lengths
        prompt_metrics = self.calculate_optimal_prompt_length(len(query), len(str(history)))
        
        # Build prompt components
        system_prompt = "You are a helpful assistant."
        
        # Add context if available
        if context:
            context_part = f"\nContext: {context}"
        else:
            context_part = ""
        
        # Add history if within window
        if prompt_metrics["total_tokens"] <= self.MAX_WINDOW_TOKENS:
            history_part = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        else:
            # Summarize history
            history_part = f"[History summarized - {prompt_metrics['summary_turn']} turns]"
        
        # Build final prompt
        prompt = f"{system_prompt}{context_part}\n{history_part}\nuser: {normalized_query}\nassistant:"
        
        return prompt


# Global prompt engine instance
prompt_engine = PromptEngine()
