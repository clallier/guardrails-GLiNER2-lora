"""
Static Code Smells & Style Guide Enforcement Test Suite.

This module statically checks function and method complexities.
It runs as a custom AST checker inside pytest.

In addition to this custom checker, the project supports industry-standard static analysis tools:
1. Ruff (Linter & Formatter): Checks compliance with PEP-8 guidelines.
   Command: `uv run ruff check src/`
2. Pylint (Refactoring Smells): Detects duplicate code and structural flaws.
   Command: `uv run pylint src/`
3. Radon (Complexity Metric): Measures cognitive and cyclomatic complexity.
   Command: `uv run radon cc src/ -a`
4. Bandit (Security Audits): Scans codebases for potential security vulnerabilities.
   Command: `uv run bandit -r src/`
"""

import ast
import glob
import os
import shutil
import subprocess

import pytest


def get_function_body_line_count(node: ast.AST, file_lines: list[str]) -> int:
    """
    Calculates the exact number of logical code lines inside a function or method body.
    Excludes signature, decorators, blank lines, standalone comments, and docstrings.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
        return 0

    # Determine the boundaries of the active body nodes, using explicit type guards
    linenos: list[int] = []
    for child in node.body:
        val = getattr(child, "lineno", None)
        if isinstance(val, int):
            linenos.append(val)

    if not linenos:
        return 0

    start_line = min(linenos)

    end_linenos: list[int] = []
    for child in node.body:
        val = getattr(child, "end_lineno", None)
        if isinstance(val, int):
            end_linenos.append(val)

    end_line = max(end_linenos) if end_linenos else max(linenos)

    body_lines = file_lines[start_line - 1 : end_line]

    # Count only active lines of logic (skipping blanks and comments)
    logical_lines = 0
    for line in body_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            logical_lines += 1

    # Exclude docstring from logical code lines if present as the first statement
    if node.body:
        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
            if isinstance(first_stmt.value.value, str):
                doc_start = first_stmt.lineno
                doc_end = getattr(first_stmt, "end_lineno", None) or doc_start
                if doc_start is not None and doc_end is not None:
                    doc_lines_count = sum(
                        1
                        for line in file_lines[doc_start - 1 : doc_end]
                        if line.strip() and not line.strip().startswith("#")
                    )
                    logical_lines = max(0, logical_lines - doc_lines_count)

    return logical_lines


def test_enforce_function_length_limits():
    """
    Static analysis unit test that scans the entire workspace and asserts
    that every custom function or method contains no more than 20 lines of logic.
    """
    max_allowed_lines = 20
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
    py_files = glob.glob(os.path.join(src_dir, "**/*.py"), recursive=True)

    violations = []

    for file_path in py_files:
        # Exclude legacy/vendor code if any
        if "venv" in file_path or ".venv" in file_path:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            file_lines = content.splitlines()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                logical_count = get_function_body_line_count(node, file_lines)

                # Check for class context to identify method name
                func_name = node.name
                parent = getattr(node, "parent", None)
                if parent and isinstance(parent, ast.ClassDef):
                    func_name = f"{parent.name}.{node.name}"

                if logical_count > max_allowed_lines:
                    rel_path = os.path.relpath(file_path, os.path.dirname(src_dir))
                    violations.append(
                        f"[{rel_path}:{node.lineno}] '{func_name}' has {logical_count} logical "
                        f"lines (Max allowed is {max_allowed_lines})"
                    )

    if violations:
        error_msg = "\n".join(violations)
        pytest.fail(
            f"Style Guide Violation: The following functions/methods "
            f"exceed the 20-line logic limit:\n{error_msg}"
        )


def test_enforce_class_layout_ordering():
    """
    Static analysis unit test that asserts class structures adhere to layout rules:
    1. Internal Constants (class-level variables/assignments) first.
    2. Public API (methods without a leading single underscore,
        plus double-underscored magic methods) second.
    3. Private Internal Helpers (methods prefixed with a single underscore) third.
    """
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
    py_files = glob.glob(os.path.join(src_dir, "**/*.py"), recursive=True)
    violations = []

    for file_path in py_files:
        if "venv" in file_path or ".venv" in file_path:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # Class-level elements order check
            # We track the maximum index / state of encountered nodes.
            # States:
            # 0: Constants / Class-level variables
            # 1: Public API Methods (no leading single underscore, or starts with '__')
            # 2: Private Internal Helpers
            # (leading single underscore, e.g. starts with '_' but not '__')
            current_state = 0

            for child in node.body:
                # 1. Check for Constants (assignments at the class body level)
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    if current_state > 0:
                        rel_path = os.path.relpath(file_path, os.path.dirname(src_dir))
                        violations.append(
                            f"[{rel_path}:{child.lineno}] Class '{node.name}' has class variable "
                            "after method definitions"
                        )
                # 2. Check for Methods
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = child.name
                    # Determine category:
                    # Double underscore methods like __init__ or __call__ are considered
                    # Public API / setup
                    if method_name.startswith("_") and not method_name.startswith("__"):
                        # Private helper (State 2)
                        current_state = max(current_state, 2)
                    else:
                        # Public API (State 1)
                        if current_state > 1:
                            rel_path = os.path.relpath(file_path, os.path.dirname(src_dir))
                            violations.append(
                                f"[{rel_path}:{child.lineno}] Class '{node.name}' has public "
                                f"method '{method_name}' defined after private helper method(s)"
                            )
                        current_state = max(current_state, 1)

    if violations:
        error_msg = "\n".join(violations)
        pytest.fail(f"Style Guide Violation: Class layout ordering is incorrect:\n{error_msg}")


def test_ruff_linting():
    """Runs Ruff check on the src directory and asserts no errors."""
    if not shutil.which("ruff"):
        pytest.fail("Ruff is not installed. Please run: uv sync --all-extras")

    result = subprocess.run(["ruff", "check", "src"], capture_output=True, text=True)  # noqa: S607
    if result.returncode != 0:
        pytest.fail(f"Ruff Lint Failures:\n{result.stdout}\n{result.stderr}")


def test_bandit_security_scan():
    """Runs Bandit security scan on the src directory and asserts no vulnerabilities."""
    if not shutil.which("bandit"):
        pytest.fail("Bandit is not installed. Please run: uv sync --all-extras")

    result = subprocess.run(["bandit", "-r", "src", "-q"], capture_output=True, text=True)  # noqa: S607
    if result.returncode != 0:
        pytest.fail(f"Bandit Security Scan Violations Found:\n{result.stdout}\n{result.stderr}")


def test_radon_complexity_scan():
    """Runs Radon complexity analysis on the src directory
    and checks for nesting/complexity issues."""
    if not shutil.which("radon"):
        pytest.fail("Radon is not installed. Please run: uv sync --all-extras")

    result = subprocess.run(["radon", "cc", "src", "-s"], capture_output=True, text=True)  # noqa: S607
    if result.returncode != 0:
        pytest.fail(f"Radon complexity check failed:\n{result.stdout}\n{result.stderr}")


def test_pylint_linting():
    """Runs Pylint on the src directory and asserts no architectural smells."""
    if not shutil.which("pylint"):
        pytest.fail("Pylint is not installed. Please run: uv sync --all-extras")

    result = subprocess.run(["pylint", "src", "--errors-only"], capture_output=True, text=True)  # noqa: S607
    if result.returncode != 0:
        pytest.fail(f"Pylint Lint Errors Found:\n{result.stdout}\n{result.stderr}")
