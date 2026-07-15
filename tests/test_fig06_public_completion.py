"""Focused tests for Figure 6 public-export completion status."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_fig06_module():
    spec = importlib.util.spec_from_file_location(
        "fig06_d5_block_selection_for_status_tests",
        PROJECT_ROOT / "scripts" / "fig06_d5_block_selection.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _result(status: str) -> dict[str, object]:
    return {"status": status, "errors": [], "warnings": []}


def test_publication_completion_status_full_local_package(monkeypatch) -> None:
    fig06 = load_fig06_module()

    def phase_a(project_root=fig06.PROJECT_ROOT, *, require_local_records=True, **kwargs):
        return _result("passed")

    def appendix(project_root=fig06.PROJECT_ROOT, *, require_local_records=True):
        return _result("passed")

    monkeypatch.setattr(fig06, "validate_phase_a_completion", phase_a)
    monkeypatch.setattr(fig06, "validate_appendix_d5_completion", appendix)
    monkeypatch.setattr(fig06, "read_recorded_regression_status", lambda project_root=fig06.PROJECT_ROOT: "passed")

    assert fig06.publication_completion_status() == "Full publication package completed"


def test_publication_completion_status_clean_public_package(monkeypatch) -> None:
    fig06 = load_fig06_module()

    def phase_a(project_root=fig06.PROJECT_ROOT, *, require_local_records=True, **kwargs):
        return _result("failed" if require_local_records else "passed")

    def appendix(project_root=fig06.PROJECT_ROOT, *, require_local_records=True):
        return _result("failed" if require_local_records else "passed")

    monkeypatch.setattr(fig06, "validate_phase_a_completion", phase_a)
    monkeypatch.setattr(fig06, "validate_appendix_d5_completion", appendix)
    monkeypatch.setattr(fig06, "read_recorded_regression_status", lambda project_root=fig06.PROJECT_ROOT: "not run")

    assert (
        fig06.publication_completion_status()
        == "Core public Figure 6 and Appendix artifacts completed; local validation records are optional"
    )


def test_publication_completion_status_public_failure(monkeypatch) -> None:
    fig06 = load_fig06_module()

    def phase_a(project_root=fig06.PROJECT_ROOT, *, require_local_records=True, **kwargs):
        return _result("failed")

    def appendix(project_root=fig06.PROJECT_ROOT, *, require_local_records=True):
        return _result("passed")

    monkeypatch.setattr(fig06, "validate_phase_a_completion", phase_a)
    monkeypatch.setattr(fig06, "validate_appendix_d5_completion", appendix)
    monkeypatch.setattr(fig06, "read_recorded_regression_status", lambda project_root=fig06.PROJECT_ROOT: "passed")

    assert fig06.publication_completion_status() == "Public Figure 6 or Appendix artifacts failed validation"


def test_appendix_d5_artifacts_exist_uses_public_validation(monkeypatch) -> None:
    fig06 = load_fig06_module()
    calls: list[bool] = []

    def appendix(project_root=fig06.PROJECT_ROOT, *, require_local_records=True):
        calls.append(require_local_records)
        return _result("passed" if not require_local_records else "failed")

    monkeypatch.setattr(fig06, "validate_appendix_d5_completion", appendix)

    assert fig06.appendix_d5_artifacts_exist() is True
    assert calls == [False]
