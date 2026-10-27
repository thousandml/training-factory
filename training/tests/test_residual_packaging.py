"""Regression test: residual helper must ship with both source and customer bundles."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]

def test_residual_helper_exists_and_is_in_customer_bundle():
    assert (ROOT / "residual_names.py").is_file()
    module = ast.parse((ROOT / "make_customer_bundle.py").read_text())
    public = next(node.value for node in module.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC" for t in node.targets))
    assert "residual_names.py" in ast.literal_eval(public)

def test_imports_are_present():
    assert "from residual_names import resolve_residual" in (ROOT / "load_model.py").read_text()
    assert "from residual_names import resolve_residual" in (ROOT / "inspect_checkpoint.py").read_text()
