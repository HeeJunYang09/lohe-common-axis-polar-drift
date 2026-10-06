"""Read-only checks of the additional numerical data and portable scripts."""
import ast
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, BASE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hitting():
    return load("additional_hitting", "multi_target_hitting_time/scripts/run_multi_target.py")


@pytest.fixture(scope="module")
def initial():
    return load("additional_initial", "initial_condition_sensitivity/scripts/run_initial_conditions.py")


def test_scripts_compile():
    for path in BASE.glob("*/scripts/*.py"):
        compile(path.read_text(), str(path), "exec")
        ast.parse(path.read_text())


def test_hitting_diagnostics(hitting):
    rows = hitting.validate()
    assert len(rows) == 125
    assert all(r["T_eta_sim"] >= r["t_f_num"] for r in rows)


def test_synthetic_hitting_law(hitting):
    hitting.self_check()


def test_initial_condition_diagnostics(initial):
    rows, regressions = initial.validate()
    assert sum(r["dimension"] == 3 for r in rows) == 153
    assert sum(r["dimension"] == 5 for r in rows) == 68
    assert all(c["passed"] for c in regressions)


def test_initialization_self_check(initial, tmp_path):
    old = initial.OUT
    try:
        initial.OUT = tmp_path
        initial.self_check()
    finally:
        initial.OUT = old


def test_compact_archives_are_numeric_or_json():
    for path in BASE.glob("*/data/**/*.npz"):
        with np.load(path, allow_pickle=False) as z:
            assert "states" not in z.files
            for key in z.files:
                assert z[key].dtype.kind != "O"


def test_initial_recompute_export_dispatch(initial, tmp_path, monkeypatch):
    source = initial.OUT
    def saved(dim, k, sigma, group, amp, seed):
        key = f"d{dim}_K{k:g}_s{sigma:.2f}_{group}_a{amp:.2f}_seed{seed}"
        return initial.load_diagnostics(source / "data/diagnostics" / (key + ".npz"))
    monkeypatch.setattr(initial, "run3", lambda k, s, g, a, seed: saved(3, k, s, g, a, seed))
    monkeypatch.setattr(initial, "run5", lambda k, g, a, seed: saved(5, k, 0., g, a, seed))
    monkeypatch.setattr(sys, "argv", ["run", "--recompute", "--stage", "sanity", "--output-dir", str(tmp_path)])
    try:
        initial.main()
    finally:
        initial.OUT = source
    assert len(list((tmp_path / "data/diagnostics").glob("*.npz"))) == 6


def test_relocated_copy_is_self_contained(tmp_path):
    root = BASE.parent
    clone = tmp_path / "relocated"
    for folder in ("src", "scripts", "data/cache"):
        shutil.copytree(root / folder, clone / folder)
    shutil.copytree(BASE, clone / "additional_experiments",
                    ignore=shutil.ignore_patterns("figures", "_work", "__pycache__"))
    for relative in ("multi_target_hitting_time/scripts/run_multi_target.py",
                     "initial_condition_sensitivity/scripts/run_initial_conditions.py"):
        command = [sys.executable, "-B", str(clone / "additional_experiments" / relative)]
        completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "PASS" in completed.stdout
