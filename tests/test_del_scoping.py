"""Guard: `del name` in a function must refer to something that function can delete.

Python treats `del x` as a *binding* operation. Inside a function, it marks `x`
local — so deleting a name the function never binds, and never declared
`nonlocal` or `global`, raises UnboundLocalError. Always. Not sometimes.

This is valid syntax, so `py_compile` accepts it and the CI compile job passes.
It is invisible to AST-shape tests. And in this repo it lands in code no local
test can execute, because torch is not installed — so the failure only appears
at the end of a real GPU training run, after all the work is done.

That is exactly what happened: refactoring `train_sync`'s teardown into a nested
`_free_memory()` moved

    del model, optimizer, scheduler, dataset, dataloader

from the function that bound those names into one that did not. Every training
run would have finished, then raised UnboundLocalError on the way out, and been
reported as a failed job with no model exported. The fix is `nonlocal`, which is
also what makes the delete do its actual job: dropping the enclosing binding is
the only way `gc.collect()` can reclaim the GPU tensors.

The check is repo-wide and not specific to that function, because the mistake is
a property of Python scoping rather than of any one refactor.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_files() -> list[Path]:
    """Every first-party module: services, benchmarks, and the tests themselves."""
    skip = {".venv", ".git", "__pycache__", "node_modules", "cache", "models", "output"}
    found: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        found.append(path)
    return sorted(found)


def _bound_names(func: ast.AST) -> set[str]:
    """Names the function itself binds, so `del` on them is legitimate.

    Only this function's own scope — a nested function's bindings belong to it,
    not to its parent, so they are not descended into.
    """
    bound: set[str] = set()

    args = getattr(func, "args", None)
    if args is not None:
        for group in (args.posonlyargs, args.args, args.kwonlyargs):
            bound.update(a.arg for a in group)
        for solo in (args.vararg, args.kwarg):
            if solo is not None:
                bound.add(solo.arg)

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            # A nested def/class binds its own name here, but its body is a
            # separate scope and is checked on its own pass.
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(child.name)
                continue
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                bound.update(child.names)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                bound.add(child.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            visit(child)

    visit(func)
    return bound


def _offenders(tree: ast.AST) -> list[tuple[int, str, str]]:
    """(line, function, name) for each undeletable `del` target."""
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound = _bound_names(node)
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Delete):
                continue
            # Only bare names are scope-sensitive; `del d[k]` and `del o.attr`
            # are attribute/subscript deletes and are always fine.
            for target in sub.targets:
                if isinstance(target, ast.Name) and target.id not in bound:
                    found.append((sub.lineno, node.name, target.id))
    return found


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_del_targets_are_deletable_in_their_scope(path: Path):
    offenders = _offenders(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} deletes names its own scope never binds:\n  "
        + "\n  ".join(f"line {line}: {func}() -> del {name}" for line, func, name in offenders)
        + "\nPython makes a `del` target local to the function, so this raises "
          "UnboundLocalError at runtime. Declare `nonlocal`/`global` if the "
          "intent is to drop the outer binding, which is also the only form "
          "that lets gc reclaim the object."
    )


def test_the_scan_actually_covers_the_services():
    """A silently empty parametrisation would make this file decorative."""
    scanned = {p.parts[len(REPO_ROOT.parts)] for p in _python_files()}
    for expected in ("piper-training-service", "stt-service", "frontend-service"):
        assert expected in scanned, f"{expected} was not scanned"


def test_the_check_catches_the_shape_that_slipped_through():
    """Pinned against a synthetic case, so the detector cannot rot into a no-op."""
    broken = ast.parse(
        "def outer():\n"
        "    model = 1\n"
        "    def free():\n"
        "        del model\n"
        "    free()\n"
    )
    assert _offenders(broken) == [(4, "free", "model")]

    fixed = ast.parse(
        "def outer():\n"
        "    model = 1\n"
        "    def free():\n"
        "        nonlocal model\n"
        "        del model\n"
        "    free()\n"
    )
    assert _offenders(fixed) == []


def test_deleting_a_name_the_function_binds_itself_is_allowed():
    ok = ast.parse("def f():\n    tmp = 1\n    del tmp\n")
    assert _offenders(ok) == []


def test_subscript_and_attribute_deletes_are_not_flagged():
    ok = ast.parse("def f(d, o):\n    del d['k']\n    del o.attr\n")
    assert _offenders(ok) == []
