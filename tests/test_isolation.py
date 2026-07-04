"""Guardrail: the fitting core must import only NumPy/SciPy/stdlib.

Two sibling projects reuse ``psyvis_ml.fitting`` verbatim, so it must not import anything
else from ``psyvis_ml`` and must not reach outside its own subpackage. This is enforced
statically (AST) so a stray ``from ..datasets import X`` fails the build, not just runtime.
"""

import ast
import importlib.util
import pathlib
import sys

import pytest

_FITTING_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "psyvis_ml" / "fitting"

# Everything the fitting core is allowed to import at the top level.
_ALLOWED_ROOTS = {"numpy", "scipy", "math", "warnings", "dataclasses", "typing", "__future__"}


def _fitting_modules():
    return sorted(_FITTING_DIR.glob("*.py"))


def test_fitting_dir_exists_and_has_modules():
    mods = _fitting_modules()
    assert mods, "no fitting modules found"
    names = {p.name for p in mods}
    assert {"sigmoids.py", "likelihood.py", "fit.py", "bootstrap.py"} <= names


@pytest.mark.parametrize("path", _fitting_modules(), ids=lambda p: p.name)
def test_no_forbidden_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root in _ALLOWED_ROOTS, (
                    f"{path.name}: forbidden absolute import {alias.name!r}"
                )
                assert not alias.name.startswith("psyvis_ml"), (
                    f"{path.name}: fitting core must not import psyvis_ml"
                )
        elif isinstance(node, ast.ImportFrom):
            # Relative imports within the fitting subpackage are level == 1 (from .mod).
            # level >= 2 (from ..pkg) escapes the subpackage and is forbidden.
            assert node.level < 2, (
                f"{path.name}: relative import escapes the fitting subpackage "
                f"(level={node.level}, module={node.module!r})"
            )
            if node.level == 0:  # absolute import
                root = (node.module or "").split(".")[0]
                assert root in _ALLOWED_ROOTS, (
                    f"{path.name}: forbidden absolute import from {node.module!r}"
                )
                assert not (node.module or "").startswith("psyvis_ml"), (
                    f"{path.name}: fitting core must not import psyvis_ml"
                )


def test_fitting_imports_without_the_rest_of_the_package():
    # Load psyvis_ml.fitting directly from its path, with the parent package's other
    # subpackages absent from the loaded set, to prove there is no hidden coupling.
    for mod in list(sys.modules):
        if mod.startswith("psyvis_ml"):
            del sys.modules[mod]
    spec = importlib.util.spec_from_file_location(
        "psyvis_ml_fitting_isolated",
        _FITTING_DIR / "__init__.py",
        submodule_search_locations=[str(_FITTING_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # would raise if it needed the rest of the package
    assert hasattr(module, "fit_psychometric")
