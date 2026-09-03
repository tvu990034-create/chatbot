"""
reasoning/react.py
~~~~~~~~~~~~~~~~~~
ReAct: Reasoning + Acting framework for language agents.

Based on: "ReAct: Synergizing Reasoning and Acting in Language Models"
https://arxiv.org/abs/2210.03629
https://github.com/ysymyth/ReAct

ReAct combines reasoning traces and task-specific actions in an interleaved manner,
enabling agents to reason about task execution and dynamically adjust their approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import deque

from config import settings

logger = logging.getLogger(__name__)


class ThoughtType(str, Enum):
    """Types of thoughts in ReAct reasoning."""
    OBSERVATION = "observation"  # What the agent observes
    THOUGHT = "thought"  # Internal reasoning
    ACTION = "action"  # Action to take
    RESULT = "result"  # Result of action


@dataclass
class Step:
    """A single step in ReAct reasoning."""
    step_type: ThoughtType
    content: str
    observation: str = ""
    action_result: str = ""
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActConfig:
    """Configuration for ReAct framework."""
    max_steps: int = 10
    max_iterations: int = 5
    temperature: float = 0.7
    allow_self_correction: bool = True
    stop_on_success: bool = True
    verbose_thoughts: bool = True


class ReActAgent:
    """
    ReAct agent with reasoning + acting capabilities.
    
    The agent operates in an interleaved manner:
    1. Reason about current state (Thought)
    2. Decide on action (Action)
    3. Execute action and observe result (Observation)
    4. Repeat until task completion
    """
    
    def __init__(self, config: ReActConfig | None = None):
        self.config = config or ReActConfig()
        self.llm_caller = self._get_llm_caller()
        self.tool_registry: dict[str, Callable] = {}
        self._tool_cache: dict[str, Any] = {}  # Cache for deterministic tool results
        self._register_default_tools()
    
    def _get_llm_caller(self) -> Callable:
        """Get the LLM caller function."""
        try:
            from gateway.litellm_gateway import chat
            return chat
        except ImportError:
            logger.warning("LiteLLM gateway not available, using fallback")
            return self._fallback_llm
    
    def _fallback_llm(self, messages: list[dict], **kwargs) -> str:
        """Fallback LLM caller when gateway is unavailable."""
        logger.error("LLM gateway unavailable, returning mock response")
        return "Mock response - LLM gateway not configured"
    
    def _get_tool_cache_key(self, action_name: str, action_args: dict) -> str:
        """Generate cache key for tool result."""
        import hashlib
        import json
        cache_string = f"{action_name}:{json.dumps(action_args, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_deterministic_tool(self, action_name: str) -> bool:
        """Check if a tool produces deterministic results."""
        deterministic_tools = {
            "calculate",  # Mathematical calculations are deterministic
            "lookup",     # Knowledge base lookups are deterministic
            "analyze",    # Analysis of same data should be deterministic
            "compare",    # Comparison of same items is deterministic
        }
        return action_name in deterministic_tools
    
    def _register_default_tools(self) -> None:
        """Register default tools for the agent."""
        # Information gathering tools
        self.register_tool("search", self._search_tool, "Search for information online")
        self.register_tool("calculate", self._calculate_tool, "Perform mathematical calculations")
        self.register_tool("lookup", self._lookup_tool, "Look up information in knowledge base")
        
        # Code execution tools
        self.register_tool("python_execute", self._python_execute_tool, "Execute Python code")
        self.register_tool("file_read", self._file_read_tool, "Read file contents")
        
        # Analysis tools
        self.register_tool("analyze", self._analyze_tool, "Analyze data or information")
        self.register_tool("compare", self._compare_tool, "Compare multiple items")
    
    def register_tool(self, name: str, func: Callable, description: str) -> None:
        """Register a tool for the agent to use."""
        self.tool_registry[name] = {
            "func": func,
            "description": description
        }
        logger.debug(f"Registered tool: {name}")
    
    def solve(self, problem: str, context: str = "") -> tuple[str, list[Step]]:
        """
        Solve a problem using ReAct reasoning.
        
        Args:
            problem: The problem to solve
            context: Additional context for the problem
            
        Returns:
            Tuple of (solution, reasoning_steps)
        """
        logger.info(f"Starting ReAct reasoning for problem: {problem[:100]}...")
        
        steps: list[Step] = []
        current_context = context
        observation = ""
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"ReAct iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Build reasoning chain
            reasoning_steps = []
            
            for step_num in range(self.config.max_steps):
                # Generate thought
                thought = self._generate_thought(problem, current_context, observation, steps)
                thought_step = Step(
                    step_type=ThoughtType.THOUGHT,
                    content=thought,
                    timestamp=self._get_timestamp()
                )
                steps.append(thought_step)
                reasoning_steps.append(f"Thought: {thought}")
                
                # Decide on action
                action_decision = self._decide_action(problem, current_context, observation, steps)
                
                if "finish" in action_decision.lower() or "done" in action_decision.lower():
                    # Task completion
                    final_step = Step(
                        step_type=ThoughtType.RESULT,
                        content=self._extract_solution(action_decision),
                        timestamp=self._get_timestamp()
                    )
                    steps.append(final_step)
                    logger.info("ReAct: Task completed successfully")
                    return final_step.content, steps
                
                # Execute action
                action_name, action_args = self._parse_action(action_decision)
                
                if action_name in self.tool_registry:
                    try:
                        # Check cache for deterministic tools
                        cache_key = self._get_tool_cache_key(action_name, action_args)
                        if cache_key in self._tool_cache:
                            result = self._tool_cache[cache_key]
                            logger.debug(f"Tool cache hit for {action_name}")
                        else:
                            result = self.tool_registry[action_name]["func"](**action_args)
                            # Cache deterministic tools
                            if self._is_deterministic_tool(action_name):
                                self._tool_cache[cache_key] = result
                                logger.debug(f"Cached result for {action_name}")
                        
                        action_result = f"Action: {action_name}({action_args})\nResult: {result}"
                    except Exception as exc:
                        action_result = f"Action: {action_name}({action_args})\nError: {exc}"
                        logger.warning(f"Tool execution failed: {exc}")
                else:
                    action_result = f"Unknown tool: {action_name}. Available tools: {list(self.tool_registry.keys())}"
                
                action_step = Step(
                    step_type=ThoughtType.ACTION,
                    content=action_decision,
                    action_result=action_result,
                    timestamp=self._get_timestamp()
                )
                steps.append(action_step)
                reasoning_steps.append(f"Action: {action_decision}")
                reasoning_steps.append(f"Observation: {action_result}")
                
                # Update observation for next iteration
                observation = action_result
                
                # Update context with new information
                current_context = f"{context}\n\nPrevious actions:\n" + "\n".join(reasoning_steps[-6:])
                
                # Check if we should continue
                if self._should_stop(observation, steps):
                    solution = self._generate_final_solution(problem, steps)
                    final_step = Step(
                        step_type=ThoughtType.RESULT,
                        content=solution,
                        timestamp=self._get_timestamp()
                    )
                    steps.append(final_step)
                    return solution, steps
            
            # Self-correction if enabled
            if self.config.allow_self_correction and iteration < self.config.max_iterations - 1:
                correction = self._generate_correction(problem, steps)
                if correction and "continue" not in correction.lower():
                    # Apply correction
                    logger.info(f"Applying self-correction: {correction[:100]}...")
                    current_context = f"{context}\n\nCorrection needed: {correction}"
        
        # Generate final solution if we've exhausted iterations
        solution = self._generate_final_solution(problem, steps)
        final_step = Step(
            step_type=ThoughtType.RESULT,
            content=solution,
            timestamp=self._get_timestamp()
        )
        steps.append(final_step)
        
        return solution, steps
    
    def _generate_thought(self, problem: str, context: str, observation: str, previous_steps: list[Step]) -> str:
        """Generate a thought about the current situation."""
        recent_steps = previous_steps[-3:] if previous_steps else []
        steps_summary = "\n".join([f"{s.step_type.value}: {s.content[:100]}" for s in recent_steps])
        
        prompt = f"""You are solving a problem using reasoning and acting. Think about the current situation and decide your next step.

Problem: {problem}

Context: {context}

Recent steps:
{steps_summary}

Current observation: {observation}

What is your current thought about the situation? What should you consider next?"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.temperature,
                max_tokens=200
            )
            return response.strip()
        except Exception as exc:
            logger.warning(f"Failed to generate thought: {exc}")
            return "I need to analyze the current situation and plan my next step."
    
    def _decide_action(self, problem: str, context: str, observation: str, previous_steps: list[Step]) -> str:
        """Decide on the next action to take."""
        tools_list = "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tool_registry.items()
        ])
        
        prompt = f"""Based on your current thought, decide your next action.

Problem: {problem}

Available tools:
{tools_list}

Current observation: {observation}

Decide your next action. Format as: "Action: tool_name(arg1=value1, arg2=value2)" or "Action: finish(solution)" if you have solved the problem."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.temperature,
                max_tokens=150
            )
            return response.strip()
        except Exception as exc:
            logger.warning(f"Failed to decide action: {exc}")
            return "Action: finish(Unable to determine action)"
    
    def _parse_action(self, action_decision: str) -> tuple[str, dict[str, Any]]:
        """Parse action decision into tool name and arguments."""
        # Simple parsing - looks for "Action: tool_name(args)"
        try:
            if "finish" in action_decision.lower():
                return "finish", {}
            
            # Extract tool name and arguments
            if ":" in action_decision:
                action_part = action_decision.split(":", 1)[1].strip()
            else:
                action_part = action_decision.strip()
            
            # Parse tool name
            tool_name = action_part.split("(")[0].strip()
            
            # Parse arguments
            args = {}
            if "(" in action_part and ")" in action_part:
                args_str = action_part.split("(", 1)[1].split(")")[0].strip()
                if args_str:
                    # Simple argument parsing
                    for arg in args_str.split(","):
                        if "=" in arg:
                            key, value = arg.split("=", 1)
                            args[key.strip()] = value.strip().strip('"\'')
            
            return tool_name, args
            
        except Exception as exc:
            logger.warning(f"Failed to parse action: {exc}")
            return "finish", {}
    
    def _should_stop(self, observation: str, steps: list[Step]) -> bool:
        """Determine if we should stop reasoning."""
        # Stop if we have a clear solution
        stop_indicators = ["solution:", "answer:", "result:", "conclusion:"]
        observation_lower = observation.lower()
        return any(indicator in observation_lower for indicator in stop_indicators)
    
    def _generate_final_solution(self, problem: str, steps: list[Step]) -> str:
        """Generate the final solution based on reasoning steps."""
        steps_summary = "\n".join([
            f"{s.step_type.value}: {s.content if s.step_type != ThoughtType.ACTION else s.action_result}"
            for s in steps
        ])
        
        prompt = f"""Based on the following reasoning steps, provide a final solution to the problem.

Problem: {problem}

Reasoning steps:
{steps_summary}

Provide a clear, concise final solution."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=0.3,
                max_tokens=500
            )
            return response.strip()
        except Exception as exc:
            logger.warning(f"Failed to generate final solution: {exc}")
            return "Unable to generate final solution due to error."
    
    def _generate_correction(self, problem: str, steps: list[Step]) -> str:
        """Generate a self-correction if stuck."""
        steps_summary = "\n".join([
            f"{s.step_type.value}: {s.content[:100]}"
            for s in steps[-5:]
        ])
        
        prompt = f"""Review the following reasoning steps and suggest a correction if the approach seems stuck or suboptimal.

Problem: {problem}

Recent steps:
{steps_summary}

If the current approach is working well, respond with "continue". If not, suggest a specific correction or alternative approach."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=0.7,
                max_tokens=200
            )
            return response.strip()
        except Exception as exc:
            logger.warning(f"Failed to generate correction: {exc}")
            return "continue"
    
    def _extract_solution(self, action_decision: str) -> str:
        """Extract solution from finish action."""
        if "finish" in action_decision.lower():
            # Extract solution from finish action
            if "(" in action_decision and ")" in action_decision:
                solution = action_decision.split("(", 1)[1].split(")")[0].strip()
                return solution.strip('"\'')
        return action_decision
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    # Default tool implementations
    def _search_tool(self, query: str) -> str:
        """Mock search tool."""
        return f"Search results for '{query}': This is a mock search result. In production, this would call a real search API."
    
    def _calculate_tool(self, expression: str) -> str:
        """Calculate mathematical expression."""
        try:
            # Safe evaluation of mathematical expressions
            import ast
            import operator as op
            
            operators = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv,
                ast.Pow: op.pow,
                ast.USub: op.neg,
            }
            
            def eval_node(node):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.Num):  # Python < 3.8 compatibility
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return operators[type(node.op)](eval_node(node.operand))
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")
            
            tree = ast.parse(expression, mode='eval')
            result = eval_node(tree.body)
            return str(result)
        except Exception as exc:
            return f"Calculation error: {exc}"
    
    def _lookup_tool(self, query: str) -> str:
        """Mock lookup tool."""
        return f"Lookup results for '{query}': This is a mock lookup result. In production, this would query a knowledge base."
    
    def _python_execute_tool(self, code: str) -> str:
        """Execute Python code safely."""
        try:
            # Very restricted execution for safety
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                # Only allow safe operations
                safe_globals = {
                    "__builtins__": {
                        "print": print,
                        "len": len,
                        "range": range,
                        "sum": sum,
                        "max": max,
                        "min": min,
                        "abs": abs,
                        "round": round,
                    }
                }
                exec(code, safe_globals, {})
            
            return output.getvalue() or "Code executed successfully"
        except Exception as exc:
            return f"Execution error: {exc}"
    
    def _file_read_tool(self, path: str) -> str:
        """Read file contents."""
        try:
            from pathlib import Path
            file_path = Path(path)
            if file_path.exists():
                return file_path.read_text()[:1000]  # Limit to 1000 chars
            else:
                return f"File not found: {path}"
        except Exception as exc:
            return f"File read error: {exc}"
    
    def _analyze_tool(self, data: str) -> str:
        """Analyze data or information."""
        return f"Analysis of '{data[:100]}...': This is a mock analysis. In production, this would perform actual analysis."
    
    def _compare_tool(self, items: str) -> str:
        """Compare multiple items."""
        return f"Comparison of '{items[:100]}...': This is a mock comparison. In production, this would perform actual comparison."


def get_react_agent(config: ReActConfig | None = None) -> ReActAgent:
    """Get a configured ReAct agent."""
    return ReActAgent(config or ReActConfig())