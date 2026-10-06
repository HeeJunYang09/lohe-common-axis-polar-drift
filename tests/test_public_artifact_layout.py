"""Public reproduction uses README mappings and CSV tables, not TeX fragments."""

import csv
import importlib.util
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    spec = importlib.util.spec_from_file_location(
        "layout_" + stem, ROOT / "scripts" / (stem + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_figure_mapping():
    regression = load_script("regression_fig1_4")
    assert regression.parse_figure_mapping() == {
        item["manuscript_figure"]: item["script"] for item in regression.FIGURES
    }


def test_duplicate_readme_mapping_rejected(tmp_path, monkeypatch):
    regression = load_script("regression_fig1_4")
    row = "| Figure 1 | `scripts/fig01_fast_locking.py` | output |\n"
    (tmp_path / "README.md").write_text(row + row)
    monkeypatch.setattr(regression, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Duplicate figure"):
        regression.parse_figure_mapping()


def test_appendix_csv_export_without_tex(tmp_path, monkeypatch):
    appendix = load_script("appendix_d5_diagnostics")
    output = tmp_path / "data/processed"
    output.mkdir(parents=True)
    monkeypatch.setattr(appendix, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(appendix, "PROCESSED_DIR", output)
    # Read the existing cache, and write only the regenerated table in isolation.
    with (ROOT / "data/processed/appendix_d5_vector_law.csv").open(newline="") as stream:
        vector_rows = list(csv.DictReader(stream))
    rows = appendix.write_appendix_table(vector_rows)
    assert len(rows) == 4
    generated = output / "appendix_tableA1_d5_diagnostics.csv"
    original = ROOT / "data/processed" / generated.name
    with original.open(newline="") as a, generated.open(newline="") as b:
        assert list(csv.DictReader(a)) == list(csv.DictReader(b))
    assert not (tmp_path / "paper").exists()
    assert not list(tmp_path.rglob("*.tex"))


def test_public_appendix_requires_csv_but_not_tex(tmp_path):
    figure = load_script("fig06_d5_block_selection")
    for directory in ("data", "figures"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    assert not (tmp_path / "paper").exists()
    result = figure.validate_appendix_d5_completion(tmp_path, require_local_records=False)
    assert result["status"] in {"passed", "warning"}, result
    table = tmp_path / "data/processed/appendix_tableA1_d5_diagnostics.csv"
    table.unlink()
    result = figure.validate_appendix_d5_completion(tmp_path, require_local_records=False)
    assert result["status"] == "failed"
    assert any(table.name in error for error in result["errors"])
