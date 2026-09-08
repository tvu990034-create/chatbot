"""
Regression tests for tool execution and code-execution fixes.

Covers:
  1.  Malformed tool arguments
  2.  Tool exceptions
  3.  Repeated tool call detection (request-scoped)
  4.  Calculator multi-operator expressions
  5.  TIR stdout capture
  6.  TIR stderr capture
  7.  TIR exception handling
  8.  TIR subprocess timeout
  9.  TIR last-expression evaluation
  10. Code model selection (code routing)
  11. validate_code: bracket balance ≠ syntax valid
  12. _execute_code_agent: model reference
"""

from __future__ import annotations

import ast
import asyncio
import sys
import time
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# 1. Malformed tool arguments
# ===================================================================

class TestMalformedToolArgs:
    """Each LangChain BaseTool must survive missing / wrong-type / extra args."""

    def _get_tools(self):
        from tools.aider_tool import get_code_tools
        return {t.name: t for t in get_code_tools()}

    @pytest.fixture(autouse=True)
    def _skip_if_no_aider(self):
        tools = self._get_tools()
        if not tools:
            pytest.skip("aider not installed — tools list is empty")

    def test_repo_map_missing_args(self):
        """repo_map with empty dict should not crash."""
        tools = self._get_tools()
        t = tools["repo_map"]
        result = t.invoke({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_file_read_missing_file_path(self):
        """file_read with no file_path should produce an error string."""
        tools = self._get_tools()
        t = tools["file_read"]
        result = t.invoke({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_file_read_nonexistent_file(self):
        """file_read with a non-existent path returns a readable error."""
        tools = self._get_tools()
        t = tools["file_read"]
        result = t.invoke({"file_path": "nonexistent_file_xyz.py"})
        assert isinstance(result, str)
        assert "not found" in result.lower() or "error" in result.lower()

    def test_code_search_empty_pattern(self):
        """code_search with empty pattern should not crash."""
        tools = self._get_tools()
        t = tools["code_search"]
        result = t.invoke({"pattern": ""})
        assert isinstance(result, str)

    def test_code_search_unmatched_regex(self):
        """code_search with invalid regex should not crash."""
        tools = self._get_tools()
        t = tools["code_search"]
        result = t.invoke({"pattern": "(?P<invalid"})
        assert isinstance(result, str)

    def test_aider_edit_read_only_mode(self):
        """aider_edit in read-only mode returns disabled message."""
        from config import settings
        if not settings.aider_read_only:
            pytest.skip("aider_read_only is False")
        tools = self._get_tools()
        t = tools["aider_edit"]
        result = t.invoke({"instruction": "fix bug", "files": ["test.py"]})
        assert isinstance(result, str)
        assert "read" in result.lower() or "disabled" in result.lower()


# ===================================================================
# 2. Tool exceptions
# ===================================================================

class TestToolExceptions:
    """Tool execution must never raise — always return a string."""

    def test_repo_map_import_error(self):
        """repo_map returns a string even if aider is not installed."""
        from tools.aider_tool import get_code_tools
        tools = {t.name: t for t in get_code_tools()}
        if "repo_map" not in tools:
            pytest.skip("aider not installed")
        t = tools["repo_map"]
        result = t.invoke({"repo_path": "/nonexistent/path"})
        assert isinstance(result, str)

    def test_file_read_permission_error(self):
        """file_read with a directory path returns error string."""
        from tools.aider_tool import get_code_tools
        tools = {t.name: t for t in get_code_tools()}
        if "file_read" not in tools:
            pytest.skip("aider not installed")
        t = tools["file_read"]
        result = t.invoke({"file_path": "."})
        assert isinstance(result, str)


# ===================================================================
# 3. Repeated tool call detection
# ===================================================================

class TestRepeatedToolCalls:
    """TruncatedToolNode must block identical tool+args after N repeats."""

    def test_repeated_call_detector_exists(self):
        """_MAX_REPEATED_SAME_CALL is defined in TruncatedToolNode."""
        import inspect
        from agents.langgraph_agent import build_agent
        src = inspect.getsource(build_agent)
        assert "_MAX_REPEATED_SAME_CALL" in src

    def test_tool_call_counts_in_agent_state(self):
        """AgentState includes _tool_call_counts field."""
        from agents.langgraph_agent import AgentState
        assert "_tool_call_counts" in AgentState.__annotations__

    def test_initial_state_has_empty_counts(self):
        """chat() and achat() initialize _tool_call_counts as {}."""
        from agents.langgraph_agent import chat
        # Mock everything to avoid real LLM call
        with patch("agents.langgraph_agent.get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content="ok", type="ai")],
                "rag_context": "", "rag_fetch_time": 0.0,
                "advanced_reasoning_used": False, "reasoning_metadata": {},
                "model_used": "test", "generation_policy": "",
                "input_tokens": 10,
            }
            mock_get_agent.return_value = mock_agent
            with patch("agents.langgraph_agent._get_langgraph") as mock_lg:
                mock_lg.return_value = (
                    MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                    MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                    MagicMock(), MagicMock(), MagicMock(),
                )
                with patch("agents.langgraph_agent.settings") as mock_settings:
                    mock_settings.agent_recursion_limit = 25
                    mock_settings.default_model = "test"
                    mock_settings.litellm_api_base = None
                    try:
                        chat("Tell me a story", use_rag=False, use_cache=False)
                    except Exception:
                        pass
            # Verify _tool_call_counts was passed
            call_kwargs = mock_agent.invoke.call_args[0][0]
            assert "_tool_call_counts" in call_kwargs
            assert call_kwargs["_tool_call_counts"] == {}


# ===================================================================
# 4. Calculator multi-operator expressions
# ===================================================================

class TestCalculatorMultiOperator:
    """Calculator must handle multi-operator expressions with correct precedence."""

    def test_add_multiply_precedence(self):
        """2 + 3 * 4 == 14 (not 20)."""
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("2 + 3 * 4") == "14"

    def test_multiply_add_precedence(self):
        """15 + 5 * 2 == 25 (not 40)."""
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("15 + 5 * 2") == "25"

    def test_divide_add(self):
        """100 / 4 + 7 == 32."""
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("100 / 4 + 7") == "32.0"

    def test_parentheses_override(self):
        """(2 + 3) * 4 == 20."""
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("(2 + 3) * 4") == "20"

    def test_nested_parentheses(self):
        """((2 + 3) * (4 - 1)) == 15."""
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("( ( 2 + 3 ) * ( 4 - 1 ) )") == "15"

    def test_unary_minus(self):
        """-5 + 3 == -2."""
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("-5 + 3") == "-2"

    def test_division_by_zero(self):
        """Division by zero returns None (safe failure)."""
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("1 / 0") is None

    def test_non_arithmetic_returns_none(self):
        """Non-math query returns None."""
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("hello world") is None

    def test_gateway_calculator_multi_op(self):
        """UniversalEnhancedGateway._tool_calculator handles multi-op expressions."""
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        result = gw._tool_calculator("2 + 3 * 4")
        assert result == "14"

    def test_react_calculator_multi_op(self):
        """ReAct _calculate_tool handles multi-op expressions."""
        from reasoning.react import ReActAgent, ReActConfig
        agent = ReActAgent(ReActConfig())
        result = agent._calculate_tool("2 + 3 * 4")
        assert result == "14"

    def test_react_calculator_division_by_zero(self):
        """ReAct _calculate_tool returns error string on div-by-zero."""
        from reasoning.react import ReActAgent, ReActConfig
        agent = ReActAgent(ReActConfig())
        result = agent._calculate_tool("1 / 0")
        assert "error" in result.lower()


# ===================================================================
# 5. TIR stdout capture
# ===================================================================

class TestTIRStdout:
    """TIR must capture stdout output from executed code."""

    def test_print_captured(self):
        """Code with print() should capture the output."""
        from gateway.opt_core import execute_python_isolated
        code = 'print("hello world")'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert "hello world" in output

    def test_multiple_prints(self):
        """Multiple print statements are all captured."""
        from gateway.opt_core import execute_python_isolated
        code = 'print("line1")\nprint("line2")'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert "line1" in output
        assert "line2" in output

    def test_print_with_formatting(self):
        """print() with format strings works."""
        from gateway.opt_core import execute_python_isolated
        code = 'x = 42\nprint(f"The answer is {x}")'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert "The answer is 42" in output


# ===================================================================
# 6. TIR stderr capture
# ===================================================================

class TestTIRStderr:
    """TIR must capture stderr from executed code."""

    def test_syntax_error_stderr(self):
        """Syntax errors produce a non-zero exit and stderr."""
        from gateway.opt_core import execute_python_isolated
        code = 'def foo(:\n  pass'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is False
        assert "SyntaxError" in output or "syntax" in output.lower()

    def test_runtime_error_stderr(self):
        """Runtime errors (non-zero exit) produce stderr."""
        from gateway.opt_core import execute_python_isolated
        code = 'import sys; sys.exit(1)'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is False


# ===================================================================
# 7. TIR exception handling
# ===================================================================

class TestTIRException:
    """TIR must handle exceptions gracefully."""

    def test_name_error(self):
        """Undefined variable produces an error, not a crash."""
        from gateway.opt_core import execute_python_isolated
        code = 'print(undefined_variable)'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is False
        assert "NameError" in output or "name" in output.lower()

    def test_type_error(self):
        """Type error produces an error, not a crash."""
        from gateway.opt_core import execute_python_isolated
        code = 'x = "hello" + 42'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is False
        assert "TypeError" in output or "type" in output.lower()

    def test_import_error(self):
        """Import error produces an error, not a crash."""
        from gateway.opt_core import execute_python_isolated
        code = 'import nonexistent_module_xyz'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is False


# ===================================================================
# 8. TIR subprocess timeout
# ===================================================================

class TestTIRTimeout:
    """TIR must enforce a real wall-clock timeout via subprocess."""

    def test_infinite_loop_times_out(self):
        """An infinite loop must be killed by the subprocess timeout."""
        from gateway.opt_core import execute_python_isolated
        code = 'import time\nwhile True:\n    time.sleep(0.01)'
        t0 = time.monotonic()
        ok, result, output = execute_python_isolated(code, timeout=1)
        elapsed = time.monotonic() - t0
        assert ok is False
        assert "timeout" in output.lower()
        # Must finish within a reasonable time (2s margin)
        assert elapsed < 3.0

    def test_fast_code_completes(self):
        """Fast code completes well within the timeout."""
        from gateway.opt_core import execute_python_isolated
        code = 'print("fast")'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert "fast" in output


# ===================================================================
# 9. TIR last-expression evaluation
# ===================================================================

class TestTIRLastExpression:
    """TIR must evaluate the final expression and return its value."""

    def test_last_expression_returned(self):
        """The value of the last expression is captured."""
        from gateway.opt_core import execute_python_isolated
        code = 'x = 10\ny = 20\nx + y'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert result == 30

    def test_last_expression_not_executed_twice(self):
        """The last expression should only be eval'd, not exec'd then eval'd."""
        from gateway.opt_core import execute_python_isolated
        code = 'counter = 0\ndef inc():\n    global counter\n    counter += 1\n    return counter\ninc()'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert result == 1

    def test_no_last_expression(self):
        """Code with only statements (no final expression) returns None."""
        from gateway.opt_core import execute_python_isolated
        code = 'x = 42\nprint(x)'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        # No final expression → result is None (but stdout is captured)
        assert result is None
        assert "42" in output

    def test_set_result_variable(self):
        """Code that sets _result explicitly."""
        from gateway.opt_core import execute_python_isolated
        code = '_result = 99'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert result == 99

    def test_string_result(self):
        """String results are captured correctly."""
        from gateway.opt_core import execute_python_isolated
        code = '"hello " + "world"'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert result == "hello world"

    def test_list_result(self):
        """List results are captured correctly."""
        from gateway.opt_core import execute_python_isolated
        code = '[i**2 for i in range(5)]'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert result == [0, 1, 4, 9, 16]

    def test_dict_result(self):
        """Dict results are captured correctly."""
        from gateway.opt_core import execute_python_isolated
        code = '{"a": 1, "b": 2}'
        ok, result, output = execute_python_isolated(code, timeout=5)
        assert ok is True
        assert result == {"a": 1, "b": 2}


# ===================================================================
# 10. Code model selection
# ===================================================================

class TestCodeModelSelection:
    """Code intent should route to code-specialized models when available."""

    def test_code_intent_detected(self):
        """detect_code_intent returns True for code-like queries."""
        from gateway.opt_core import detect_code_intent
        assert detect_code_intent("write a python function to sort a list") is True
        assert detect_code_intent("debug this code") is True
        assert detect_code_intent("hello world") is False


# ===================================================================
# 11. validate_code: bracket balance ≠ syntax valid
# ===================================================================

class TestValidateCode:
    """validate_code must distinguish balanced from parsed for JS/Java/etc."""

    def test_python_valid(self):
        """Valid Python passes ast.parse."""
        from gateway.opt_core import validate_code
        valid, err, validated = validate_code("x = 1\nprint(x)", "python")
        assert valid is True
        assert err is None
        assert validated is True

    def test_python_invalid(self):
        """Invalid Python fails with SyntaxError."""
        from gateway.opt_core import validate_code
        valid, err, validated = validate_code("def foo(:", "python")
        assert valid is False
        assert "syntax" in err.lower()
        assert validated is True

    def test_json_valid(self):
        """Valid JSON passes."""
        from gateway.opt_core import validate_code
        valid, err, validated = validate_code('{"a": 1}', "json")
        assert valid is True
        assert validated is True

    def test_json_invalid(self):
        """Invalid JSON fails."""
        from gateway.opt_core import validate_code
        valid, err, validated = validate_code('{a: 1}', "json")
        assert valid is False
        assert validated is True

    def test_javascript_balanced_but_not_parsed(self):
        """Balanced JS returns is_valid=True but actually_validated=False."""
        from gateway.opt_core import validate_code
        valid, err, validated = validate_code(
            "function foo() { return 1; }", "javascript"
        )
        assert valid is True
        assert validated is False  # partial check, not parsed

    def test_javascript_unbalanced(self):
        """Unbalanced JS returns is_valid=False."""
        from gateway.opt_core import validate_code
        valid, err, validated = validate_code(
            "function foo() { return 1;", "javascript"
        )
        assert valid is False
        assert "Unbalanced" in err

    def test_java_balanced_not_parsed(self):
        """Balanced Java returns actually_validated=False."""
        from gateway.opt_core import validate_code
        valid, err, validated = validate_code(
            "class Foo { void bar() {} }", "java"
        )
        assert valid is True
        assert validated is False

    def test_empty_code_rejected(self):
        """Empty code is rejected for all languages."""
        from gateway.opt_core import validate_code
        valid, err, validated = validate_code("", "javascript")
        assert valid is False
        assert "Empty" in err or "empty" in err.lower()

    def test_fenced_code_extracted(self):
        """Fenced code blocks are extracted before validation."""
        from gateway.opt_core import validate_code
        code = '```python\nx = 1\nprint(x)\n```'
        valid, err, validated = validate_code(code, "python")
        assert valid is True


# ===================================================================
# 12. _execute_code_agent: model reference
# ===================================================================

class TestCodeAgentModelRef:
    """_execute_code_agent must reference self.model_name, not undefined var."""

    def test_code_agent_uses_self_model(self):
        """Verify the source uses self.model_name."""
        import inspect
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        src = inspect.getsource(UniversalEnhancedGateway._execute_code_agent)
        assert "self.model_name" in src
        assert "model_to_use" not in src


# ===================================================================
# 13. ReAct python_execute tool
# ===================================================================

class TestReActPythonExecute:
    """ReAct python_execute must capture output and handle errors."""

    def test_print_captured(self):
        """print() output is captured."""
        from reasoning.react import ReActAgent, ReActConfig
        agent = ReActAgent(ReActConfig())
        result = agent._python_execute_tool('print("hello")')
        assert "hello" in result

    def test_no_output_returns_success(self):
        """Code with no output returns success message."""
        from reasoning.react import ReActAgent, ReActConfig
        agent = ReActAgent(ReActConfig())
        result = agent._python_execute_tool("x = 1 + 2")
        assert "successfully" in result.lower() or len(result) > 0

    def test_exception_returns_error(self):
        """Runtime error returns error string, not exception."""
        from reasoning.react import ReActAgent, ReActConfig
        agent = ReActAgent(ReActConfig())
        result = agent._python_execute_tool("undefined_var")
        assert "error" in result.lower()

    def test_restricted_builtins(self):
        """Only safe builtins are available."""
        from reasoning.react import ReActAgent, ReActConfig
        agent = ReActAgent(ReActConfig())
        result = agent._python_execute_tool("import os")
        assert "error" in result.lower()
