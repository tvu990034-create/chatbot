"""
Repository hygiene: properties CI (Linux) enforces but Windows forgives.

- No UTF-8 BOM in any tracked .py file (broke an AST-based check silently;
  editors must not reintroduce it).
- No case-mismatched imports: Windows resolves `import Foo` for foo.py,
  Linux raises ImportError. Only TOP-LEVEL imports are checked (function
  level lazy imports are resolved by Python itself at call time).
- No bare print() debugging in library code (gateway/agents/server/rag):
  per-request stdout spam + slow console IO. CLIs (main/optimizer_cli)
  may print to the terminal.
"""
from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIRST_PARTY = {
    "gateway", "agents", "rag", "server", "tests",
    "config", "config_manager", "optimizer_cli", "main",
    "tools", "ui", "backends", "factuality", "merging",
    "reasoning", "finetuning", "prompt_compression",
}

# Libraries must never print directly (use logging); CLI entry points may.
NO_PRINT_DIRS = ("gateway", "agents", "server", "rag")


def _py_files():
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in (".venv", "__pycache__", ".git", ".hf_cache",
                                "node_modules", ".pytest_cache")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def test_no_bom_in_python_files():
    bad = []
    for path in _py_files():
        with open(path, "rb") as fh:
            if fh.read(3) == b"\xef\xbb\xbf":
                bad.append(os.path.relpath(path, ROOT))
    assert not bad, "BOM found in: %s" % bad


def _resolve(parts, relpath):
    cur = ROOT
    for p in parts:
        try:
            entries = os.listdir(cur)
        except OSError:
            return
        if p in entries:
            cur = os.path.join(cur, p)
            continue
        if p + ".py" in entries:
            return
        raise AssertionError(
            "%s: import %s has no case-exact match under %s"
            % (relpath, ".".join(parts), os.path.relpath(cur, ROOT)))


def test_top_level_imports_match_case():
    for path in _py_files():
        with open(path, encoding="utf-8-sig") as fh:
            tree = ast.parse(fh.read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    if parts[0] in FIRST_PARTY:
                        _resolve(parts, os.path.relpath(path, ROOT))
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:
                    continue
                parts = node.module.split(".")
                if parts[0] in FIRST_PARTY:
                    _resolve(parts, os.path.relpath(path, ROOT))


def _is_main_guard(node):
    """True for `if __name__ == "__main__":` (demo blocks may print)."""
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (
        isinstance(t, ast.Compare)
        and isinstance(t.left, ast.Name)
        and t.left.id == "__name__"
    )


def _walk(node, in_main, hits, relpath):
    if _is_main_guard(node):
        in_main = True
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print" and not in_main):
        hits.append(relpath)
        return
    for child in ast.iter_child_nodes(node):
        _walk(child, in_main, hits, relpath)


def test_no_bare_print_in_libraries():
    bad = []
    for path in _py_files():
        rel = os.path.relpath(path, ROOT)
        if rel.split(os.sep)[0] not in NO_PRINT_DIRS:
            continue
        with open(path, encoding="utf-8-sig") as fh:
            tree = ast.parse(fh.read())
        _walk(tree, False, bad, rel)
    assert not bad, "bare print() in library code: %s" % sorted(set(bad))
