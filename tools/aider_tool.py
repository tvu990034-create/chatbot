"""
tools/aider_tool.py
~~~~~~~~~~~~~~~~~~~
Code-aware tool that wraps key Aider capabilities for use inside the
LangGraph agent.

What this provides
------------------
* RepoMapTool   – generate a compact map of a codebase (using Aider's
                  tree-sitter-based repomap) so the agent can reason
                  about code structure without loading entire files.
* FileReadTool  – safely read a specific file from the repo.
* CodeSearchTool – grep-like symbol / text search across the repo.
* AiderEditTool – (optional, off by default) apply LLM-generated edits
                  to files via Aider's edit strategies.

All tools are implemented as LangChain-compatible BaseTool subclasses
so they can be passed directly to a LangGraph ToolNode.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _get_langchain_tools():
    try:
        from langchain_core.tools import BaseTool
        return BaseTool
    except ImportError as exc:
        raise ImportError("langchain-core not installed: pip install langchain-core") from exc


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class RepoMapInput(BaseModel):
    repo_path: str = Field(
        default="",
        description="Absolute path to the git repository root. "
                    "Leave empty to use the configured default.",
    )
    max_context_window: int = Field(
        8000,
        description="Approximate token budget for the repo map.",
    )


class FileReadInput(BaseModel):
    file_path: str = Field(description="Path to the file, relative to the repo root.")
    start_line: int = Field(1,  description="First line to read (1-indexed).")
    end_line:   int = Field(200, description="Last line to read (inclusive).")


class CodeSearchInput(BaseModel):
    pattern: str = Field(description="Text or regex pattern to search for.")
    repo_path: str = Field(default="", description="Repo root (empty = default).")
    file_glob: str = Field(default="**/*.py", description="Glob pattern for files to search.")


class AiderEditInput(BaseModel):
    instruction: str = Field(
        description="Natural-language edit instruction for Aider."
    )
    files: list[str] = Field(
        description="List of file paths (relative to repo root) to edit."
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class RepoMapTool(_get_langchain_tools()):  # type: ignore[misc]
    """
    Generate a concise structural map of a codebase using Aider's repomap.

    Returns a text summary showing classes, functions, and their relationships,
    which helps the LLM reason about large codebases without reading every file.
    """

    name: str = "repo_map"
    description: str = (
        "Generate a structural map of a code repository. "
        "Use this to understand the codebase layout before reading or editing files."
    )
    args_schema: Type[BaseModel] = RepoMapInput

    def _run(self, repo_path: str = "", max_context_window: int = 8000) -> str:
        repo = Path(repo_path or settings.aider_repo_path)
        if not repo.exists():
            return f"Repository path does not exist: {repo}"
        
        # Try full Aider integration first
        try:
            from aider.repomap import RepoMap
            from aider.io import InputOutput
        except ImportError:
            # Fallback to simplified implementation without Aider
            logger.info("Aider not installed, using simplified repo map")
            return self._simple_repo_map(repo, max_context_window)
        
        try:
            io = InputOutput(yes=True)
            rm = RepoMap(
                map_tokens=max_context_window,
                root=str(repo),
                main_model=None,
                io=io,
                repo_content_prefix="",
            )
            # Gather all Python files as the "other" files
            all_files = [str(p) for p in repo.rglob("*.py") if ".git" not in p.parts]
            repo_map = rm.get_repo_map(chat_files=[], other_files=all_files)
            return repo_map or "Repository map is empty (no recognised source files found)."
        except Exception as exc:
            logger.exception("RepoMapTool failed: %s", exc)
            # Fallback to simplified implementation
            return self._simple_repo_map(repo, max_context_window)

    def _simple_repo_map(self, repo: Path, max_context_window: int) -> str:
        """Generate a simplified repo map without full Aider dependency."""
        try:
            import re
            structure = []
            total_chars = 0
            
            # Get common file patterns
            for pattern in ["*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.java", "*.go", "*.rs"]:
                for file_path in repo.rglob(pattern):
                    if ".git" in file_path.parts or "__pycache__" in file_path.parts:
                        continue
                    
                    rel_path = file_path.relative_to(repo)
                    structure.append(f"📄 {rel_path}")
                    
                    # Try to extract classes and functions for Python files
                    if file_path.suffix == ".py":
                        try:
                            content = file_path.read_text(encoding="utf-8", errors="ignore")
                            classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
                            functions = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
                            
                            if classes:
                                structure.append(f"  ├── Classes: {', '.join(classes)}")
                            if functions:
                                structure.append(f"  ├── Functions: {', '.join(functions[:5])}")
                        except Exception:
                            pass
                    
                    total_chars += len(str(struct[-1]))
                    if total_chars > max_context_window * 4:  # Rough token estimate
                        structure.append("  ... (truncated)")
                        break
            
            if not structure:
                return "No recognised source files found in repository."
            
            return "# Repository Structure\n" + "\n".join(structure)
        except Exception as exc:
            logger.exception("Simple repo map failed: %s", exc)
            return f"Failed to generate repo map: {exc}"

    async def _arun(self, **kwargs: Any) -> str:
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(None, self._run, **kwargs)


class FileReadTool(_get_langchain_tools()):  # type: ignore[misc]
    """Read a specific file (or line range) from the repository."""

    name: str = "file_read"
    description: str = (
        "Read the contents of a file in the repository. "
        "Specify start_line / end_line to limit output for large files."
    )
    args_schema: Type[BaseModel] = FileReadInput

    def _run(self, file_path: str, start_line: int = 1, end_line: int = 200) -> str:
        repo = Path(settings.aider_repo_path)
        full_path = repo / file_path if not Path(file_path).is_absolute() else Path(file_path)

        if not full_path.exists():
            return f"File not found: {full_path}"
        if not full_path.is_file():
            return f"Not a file: {full_path}"

        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            s = max(0, start_line - 1)
            e = min(total, end_line)
            selected = lines[s:e]
            header = f"# {file_path}  (lines {s+1}–{e} of {total})\n"
            return header + "\n".join(
                f"{s+i+1:4d} | {line}" for i, line in enumerate(selected)
            )
        except Exception as exc:
            return f"Error reading {file_path}: {exc}"

    async def _arun(self, **kwargs: Any) -> str:
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(None, self._run, **kwargs)


class CodeSearchTool(_get_langchain_tools()):  # type: ignore[misc]
    """Search for a text pattern across source files in the repository."""

    name: str = "code_search"
    description: str = (
        "Search for a text or regex pattern in the repository source files. "
        "Returns matching lines with file paths and line numbers."
    )
    args_schema: Type[BaseModel] = CodeSearchInput

    def _run(
        self, pattern: str, repo_path: str = "", file_glob: str = "**/*.py"
    ) -> str:
        repo = Path(repo_path or settings.aider_repo_path)
        if not repo.exists():
            return f"Repository path does not exist: {repo}"

        matches: list[str] = []
        for path in repo.rglob(file_glob.lstrip("**/") or "*.py"):
            if ".git" in path.parts:
                continue
            try:
                for i, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    import re
                    if re.search(pattern, line):
                        rel = path.relative_to(repo)
                        matches.append(f"{rel}:{i}: {line.rstrip()}")
            except Exception:
                continue

        if not matches:
            return f"No matches found for pattern: {pattern!r}"
        preview = matches[:50]
        suffix = f"\n… and {len(matches)-50} more match(es)." if len(matches) > 50 else ""
        return "\n".join(preview) + suffix

    async def _arun(self, **kwargs: Any) -> str:
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(None, self._run, **kwargs)


class AiderEditTool(_get_langchain_tools()):  # type: ignore[misc]
    """
    Apply LLM-driven code edits using Aider (CLI subprocess).

    DISABLED by default (settings.aider_read_only == True).
    Set AIDER_READ_ONLY=false in .env to enable writes.
    """

    name: str = "aider_edit"
    description: str = (
        "Apply code changes to one or more files using Aider. "
        "Provide a natural-language instruction and the list of files to edit. "
        "WARNING: This writes to disk. Only available when AIDER_READ_ONLY=false."
    )
    args_schema: Type[BaseModel] = AiderEditInput

    def _run(self, instruction: str, files: list[str]) -> str:
        if settings.aider_read_only:
            return (
                "AiderEditTool is disabled (AIDER_READ_ONLY=true). "
                "Set AIDER_READ_ONLY=false in .env to enable file edits."
            )

        # Check if aider is installed
        try:
            import importlib.util
            if importlib.util.find_spec("aider") is None:
                return "Aider not installed. Install code tools with: pip install -r requirements-code-tools.txt"
        except Exception:
            return "Aider not installed. Install code tools with: pip install -r requirements-code-tools.txt"

        from config import get_litellm_model

        repo = Path(settings.aider_repo_path)
        cmd = [
            sys.executable, "-m", "aider",
            "--model",    get_litellm_model(),
            "--message",  instruction,
            "--yes",
            "--no-pretty",
        ]
        if not settings.aider_auto_commit:
            cmd.append("--no-auto-commits")
        cmd += [str(repo / f) for f in files]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(repo),
                timeout=300,
            )
            output = result.stdout + result.stderr
            return output[:4000] if len(output) > 4000 else output
        except subprocess.TimeoutExpired:
            return "Aider edit timed out after 5 minutes."
        except Exception as exc:
            return f"Aider edit failed: {exc}"

    async def _arun(self, **kwargs: Any) -> str:
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(None, self._run, **kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_code_tools() -> list:
    """Return the list of code-aware tools to give to the agent."""
    try:
        from aider.repomap import RepoMap
        from aider.io import InputOutput
        # Aider is available, include all tools
        tools = [
            RepoMapTool(),
            FileReadTool(),
            CodeSearchTool(),
        ]
        if not settings.aider_read_only:
            tools.append(AiderEditTool())
        return tools
    except ImportError:
        # Aider is not installed, return basic tools only
        logger.warning("Aider not installed, code tools disabled. Install with: pip install aider-chat")
        return []
