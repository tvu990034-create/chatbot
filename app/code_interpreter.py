"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

import subprocess
import logging
import asyncio
import os
import sys
import tempfile
from typing import Optional, Dict, Any
from dataclasses import dataclass
import shlex

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str
    execution_time_ms: float


class CodeInterpreter:
    """
    Sandboxed Python code interpreter.
    Runs code in a subprocess with timeout for safety.
    """
    
    def __init__(self, timeout_seconds: int = 5, enabled: bool = False):
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
    
    def execute(self, code: str, session_id: str = "default") -> ExecutionResult:
        """
        Execute Python code in a sandboxed subprocess.
        
        Args:
            code: Python code to execute
            session_id: Session identifier for tracking
            
        Returns:
            ExecutionResult with output, error, and timing
        """
        if not self.enabled:
            return ExecutionResult(
                success=False,
                output="",
                error="Code interpreter is disabled",
                execution_time_ms=0
            )
        
        start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Run the code in a subprocess with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env={
                        **os.environ,
                        'PYTHONPATH': os.pathsep.join(sys.path)
                    }
                )
                
                execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000 if asyncio.get_event_loop().is_running() else 0
                
                return ExecutionResult(
                    success=result.returncode == 0,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time_ms=execution_time_ms
                )
            
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout_seconds} seconds",
                    execution_time_ms=self.timeout_seconds * 1000
                )
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
        
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=0
            )
    
    async def execute_async(self, code: str, session_id: str = "default") -> ExecutionResult:
        """
        Execute Python code asynchronously.
        
        Args:
            code: Python code to execute
            session_id: Session identifier for tracking
            
        Returns:
            ExecutionResult with output, error, and timing
        """
        if not self.enabled:
            return ExecutionResult(
                success=False,
                output="",
                error="Code interpreter is disabled",
                execution_time_ms=0
            )
        
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Run the code in a subprocess with timeout
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={
                        **os.environ,
                        'PYTHONPATH': os.pathsep.join(sys.path)
                    }
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout_seconds
                    )
                    
                    execution_time_ms = (loop.time() - start_time) * 1000
                    
                    return ExecutionResult(
                        success=process.returncode == 0,
                        output=stdout.decode('utf-8'),
                        error=stderr.decode('utf-8'),
                        execution_time_ms=execution_time_ms
                    )
                
                except asyncio.TimeoutError:
                    # Kill the process if it times out
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass
                    
                    return ExecutionResult(
                        success=False,
                        output="",
                        error=f"Execution timed out after {self.timeout_seconds} seconds",
                        execution_time_ms=self.timeout_seconds * 1000
                    )
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
        
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=0
            )
    
    def detect_code_request(self, message: str) -> bool:
        """
        Detect if a message is requesting code execution.
        
        Args:
            message: User message
            
        Returns:
            True if code execution is requested
        """
        code_keywords = [
            "run this code",
            "execute this",
            "run python",
            "execute python",
            "```python",
            "```",
            "code:",
            "calculate:",
            "compute:",
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in code_keywords)
