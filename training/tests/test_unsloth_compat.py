import sys
import types
import pytest
from unsloth_compat import configure_checkpointing, optimizer_name


def test_native_does_not_import_unsloth():
    configure_checkpointing("native")
    assert optimizer_name("adamw_torch") == "adamw_torch"


def test_smart_checkpointing_calls_installed_patch(monkeypatch):
    called = []
    package = types.ModuleType("unsloth_zoo")
    package.__path__ = []
    module = types.ModuleType("unsloth_zoo.gradient_checkpointing")
    module.patch_unsloth_smart_gradient_checkpointing = lambda: called.append(True)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", package)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.gradient_checkpointing", module)
    configure_checkpointing("unsloth")
    assert called == [True]


def test_missing_patch_fails_closed(monkeypatch):
    package = types.ModuleType("unsloth_zoo")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "unsloth_zoo", package)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.gradient_checkpointing", types.ModuleType("unsloth_zoo.gradient_checkpointing"))
    with pytest.raises(RuntimeError, match="lacks"):
        configure_checkpointing("unsloth")
