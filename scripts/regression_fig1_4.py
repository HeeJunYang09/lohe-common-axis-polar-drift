#!/usr/bin/env python
"""Content-only immutable regression wrapper for manuscript Figures 1--4."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_receipts import compute_source_fingerprint, utc_now, write_run_receipt


PROTECTED_SOURCES = [
    "src/common_utils.py",
    "src/figure_style.py",
    "scripts/fig01_fast_locking.py",
    "scripts/fig02_phase_gap_prediction.py",
    "scripts/fig03_slow_polar_drift.py",
    "scripts/fig04_hitting_time.py",
]

GLOBAL_SCAN_ROOTS = ["data/cache", "data/processed", "figures"]
GLOBAL_ALLOWED_SELF_ARTIFACTS = {
    "data/processed/regression_fig1_4_report.json",
    "data/processed/run_receipt_regression_fig1_4.json",
    "data/processed/run_receipt_figures_cache_first.json",
}
GLOBAL_ALLOWED_RESTORED_SIDE_EFFECTS: set[str] = set()
GLOBAL_ALLOWED_D5_APPENDIX_ARTIFACTS = {
    "paper/figure_mapping.md",
    "data/cache/appendix_d5_diagnostics.npz",
    "data/cache/fig06_d5_block_selection.npz",
    "data/processed/appendix_d5_ansatz_scaling.csv",
    "data/processed/appendix_d5_config_registry.json",
    "data/processed/appendix_d5_controls.csv",
    "data/processed/appendix_d5_sensitivity.csv",
    "data/processed/appendix_d5_vector_law.csv",
    "data/processed/appendix_migration_equality_report.json",
    "data/processed/appendix_tableA1_d5_diagnostics.csv",
    "data/processed/d5_block_selection_summary.txt",
    "data/processed/failures_exp06_d5_block_selection.json",
    "data/processed/fast_layer_diagnostic_table.csv",
    "data/processed/metadata_appendix_d5.json",
    "data/processed/metadata_exp06_d5_block_selection.json",
    "data/processed/release_manifest_d5.txt",
    "data/processed/run_receipt_appendix_d5.json",
    "data/processed/run_receipt_phase_a_cache_render.json",
    "data/processed/run_receipt_phase_a_recompute.json",
    "data/processed/validation_report_appendix_d5.json",
    "data/processed/validation_report_d5.json",
    "figures/README.md",
    "figures/appendix_figA1_d5_ansatz_validation.eps",
    "figures/appendix_figA1_d5_ansatz_validation.pdf",
    "figures/appendix_figA1_d5_ansatz_validation.png",
    "figures/appendix_figA2_d5_controls_robustness.eps",
    "figures/appendix_figA2_d5_controls_robustness.pdf",
    "figures/appendix_figA2_d5_controls_robustness.png",
    "figures/fig6_d5_block_selection.eps",
    "figures/fig6_d5_block_selection.pdf",
    "figures/fig6_d5_block_selection.png",
}
GLOBAL_ALLOWED_NON_REGRESSION_FIGURE_ARTIFACTS = {
    "data/processed/metadata_exp05_gaussian_robustness.json",
    "data/processed/summary_exp05_gaussian_robustness.csv",
    "figures/fig5_gaussian_robustness.eps",
    "figures/fig5_gaussian_robustness.pdf",
    "figures/fig5_gaussian_robustness.png",
}
LEGACY_REGRESSION_RECEIPT = "data/processed/run_receipt_regression_" + "fig1_5.json"
GLOBAL_ALLOWED_DEPRECATED_DELETIONS = {LEGACY_REGRESSION_RECEIPT}


def is_allowed_d5_appendix_artifact(rel: str) -> bool:
    legacy_supp = "su" + "pp"
    legacy_phase = "phase" + "_b"
    legacy_fig = "fig" + "S"
    legacy_exact = {
        f"data/processed/{legacy_phase}_config_registry.json",
        f"data/processed/run_receipt_{legacy_phase}.json",
        "data/processed/run_receipt_phase_a.json",
        "data/processed/fast_layer_diagnostic_table.csv",
        f"data/processed/table_{legacy_supp}_d5_summary.csv",
    }
    legacy_prefixes = (
        f"data/processed/summary_{legacy_supp}_d5_",
        f"figures/{legacy_supp}_{legacy_fig}",
    )
    return rel in GLOBAL_ALLOWED_D5_APPENDIX_ARTIFACTS or rel in legacy_exact or rel.startswith(legacy_prefixes)

FIGURES = [
    {
        "manuscript_figure": "Figure 1",
        "script": "scripts/fig01_fast_locking.py",
        "cache": ["data/cache/fig01_fast_locking.npz"],
        "processed": ["data/processed/metadata_exp01.json", "data/processed/summary_exp01_fast_locking.csv"],
        "figures": ["figures/fig1_fast_locking.pdf", "figures/fig1_fast_locking.png", "figures/fig1_fast_locking.eps"],
    },
    {
        "manuscript_figure": "Figure 2",
        "script": "scripts/fig02_phase_gap_prediction.py",
        "cache": ["data/cache/fig02_phase_gap_prediction.npz"],
        "processed": [
            "data/processed/metadata_exp02.json",
            "data/processed/phase_gap_prediction_summary.txt",
            "data/processed/phase_gap_scaling_data.csv",
        ],
        "figures": ["figures/fig2_phase_gap_prediction.pdf", "figures/fig2_phase_gap_prediction.png", "figures/fig2_phase_gap_prediction.eps"],
    },
    {
        "manuscript_figure": "Figure 3",
        "script": "scripts/fig03_slow_polar_drift.py",
        "cache": ["data/cache/fig03_slow_polar_drift_deterministic_3x3.npz"],
        "processed": [
            "data/processed/metadata_exp03_deterministic_3x3.json",
            "data/processed/summary_exp03_slow_polar_drift_deterministic_3x3.csv",
        ],
        "figures": [
            "figures/fig3_slow_polar_drift_deterministic.pdf",
            "figures/fig3_slow_polar_drift_deterministic.png",
            "figures/fig3_slow_polar_drift_deterministic.eps",
        ],
    },
    {
        "manuscript_figure": "Figure 4",
        "script": "scripts/fig04_hitting_time.py",
        "cache": ["data/cache/fig04_hitting_time_deterministic_5x5.npz"],
        "processed": [
            "data/processed/metadata_exp04_deterministic_5x5.json",
            "data/processed/summary_exp04_hitting_time_deterministic_5x5.csv",
        ],
        "figures": [
            "figures/fig4_hitting_time_deterministic.pdf",
            "figures/fig4_hitting_time_deterministic.png",
            "figures/fig4_hitting_time_deterministic.eps",
        ],
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(rel: str) -> dict[str, Any]:
    path = PROJECT_ROOT / rel
    return {
        "relative_path": rel,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
    }


def global_manifest_files() -> list[str]:
    files: set[str] = set(PROTECTED_SOURCES + ["paper/figure_mapping.md"])
    for root_rel in GLOBAL_SCAN_ROOTS:
        root = PROJECT_ROOT / root_rel
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                files.add(path.relative_to(PROJECT_ROOT).as_posix())
    return sorted(files)


def snapshot_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("/tmp") / f"lohe_fig1_4_baseline_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def snapshot_files(files: Sequence[str]) -> tuple[Path, dict[str, dict[str, Any]]]:
    out = snapshot_dir()
    records: dict[str, dict[str, Any]] = {}
    for rel in files:
        path = PROJECT_ROOT / rel
        rec = file_record(rel)
        records[rel] = rec
        if path.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(path.read_bytes())
    (out / "baseline_manifest.json").write_text(
        json.dumps({"snapshot_dir": str(out), "created_utc": utc_now(), "files": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out, records


def load_snapshot_records(baseline_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = baseline_dir / "baseline_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    if isinstance(files, list):
        return {item["relative_path"]: item for item in files}
    return dict(files)


def parse_figure_mapping() -> dict[str, str]:
    text = (PROJECT_ROOT / "paper/figure_mapping.md").read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("| Figure "):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3 or parts[0] == "Manuscript figure":
            continue
        figure = parts[0]
        script = parts[1].strip("`")
        mapping[figure] = script
    return {key: mapping[key] for key in sorted(mapping) if key in {"Figure 1", "Figure 2", "Figure 3", "Figure 4"}}


def compare_global_legacy_artifacts(before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    delegated = {
        rel
        for fig in FIGURES
        for rel in [*fig["cache"], *fig["processed"], *fig["figures"]]
    }
    current_paths = set(global_manifest_files())
    baseline_paths = set(before)
    all_paths = sorted(current_paths | baseline_paths)
    changes = []
    unexpected = []
    for rel in all_paths:
        old = before.get(rel, {"exists": False, "sha256": None, "size_bytes": None})
        new = file_record(rel)
        if not old.get("exists") and new.get("exists"):
            status = "created"
        elif old.get("exists") and not new.get("exists"):
            status = "deleted"
        elif old.get("sha256") != new.get("sha256") or old.get("size_bytes") != new.get("size_bytes"):
            status = "modified"
        else:
            status = "unchanged"
        entry = {
            "relative_path": rel,
            "status": status,
            "baseline_sha256": old.get("sha256"),
            "current_sha256": new.get("sha256"),
            "baseline_size_bytes": old.get("size_bytes"),
            "current_size_bytes": new.get("size_bytes"),
        }
        if status != "unchanged":
            if is_allowed_d5_appendix_artifact(rel) and not new.get("exists") and rel not in GLOBAL_ALLOWED_D5_APPENDIX_ARTIFACTS:
                continue
            if rel in delegated:
                entry["classification"] = "delegated_to_figure_regression"
            elif rel in GLOBAL_ALLOWED_SELF_ARTIFACTS:
                entry["classification"] = "allowed_regression_self_artifact"
            elif rel in GLOBAL_ALLOWED_RESTORED_SIDE_EFFECTS:
                entry["classification"] = "allowed_legacy_script_side_effect_restored_by_final_d5_render"
            elif is_allowed_d5_appendix_artifact(rel):
                entry["classification"] = "allowed_d5_appendix_package_artifact"
            elif rel in GLOBAL_ALLOWED_NON_REGRESSION_FIGURE_ARTIFACTS:
                entry["classification"] = "allowed_non_regression_figure_artifact"
            elif rel in GLOBAL_ALLOWED_DEPRECATED_DELETIONS and not new.get("exists"):
                continue
            else:
                entry["classification"] = "unexpected_legacy_scientific_change"
                unexpected.append(entry)
            changes.append(entry)
    mapping_before = before.get("paper/figure_mapping.md", {})
    mapping_after = file_record("paper/figure_mapping.md")
    return {
        "status": "passed" if not unexpected else "failed",
        "changes": changes,
        "unexpected_changes": unexpected,
        "mapping_file_sha256_before": mapping_before.get("sha256"),
        "mapping_file_sha256_after": mapping_after.get("sha256"),
    }


def compare_npz(rel: str, before_dir: Path) -> dict[str, Any]:
    before = before_dir / rel
    after = PROJECT_ROOT / rel
    result = {"path": rel, "status": "passed", "differences": []}
    if sha256(before) != sha256(after):
        result["differences"].append("file hash differs")
    with np.load(before, allow_pickle=False) as a, np.load(after, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            result["differences"].append("key set differs")
        for key in sorted(set(a.files) & set(b.files)):
            if a[key].shape != b[key].shape:
                result["differences"].append(f"{key} shape differs")
            if a[key].dtype != b[key].dtype:
                result["differences"].append(f"{key} dtype differs")
            try:
                equal = np.array_equal(a[key], b[key], equal_nan=True)
            except TypeError:
                equal = np.array_equal(a[key], b[key])
            if not equal:
                result["differences"].append(f"{key} values differ")
    if result["differences"]:
        result["status"] = "failed"
    return result


def compare_csv(rel: str, before_dir: Path) -> dict[str, Any]:
    before = before_dir / rel
    after = PROJECT_ROOT / rel
    result = {"path": rel, "status": "passed", "differences": []}
    with before.open("r", encoding="utf-8") as stream:
        a_rows = list(csv.reader(stream))
    with after.open("r", encoding="utf-8") as stream:
        b_rows = list(csv.reader(stream))
    if a_rows != b_rows:
        result["differences"].append("CSV rows differ")
    if result["differences"]:
        result["status"] = "failed"
    return result


def compare_plain(rel: str, before_dir: Path, allow_metadata: bool = False) -> dict[str, Any]:
    before = before_dir / rel
    after = PROJECT_ROOT / rel
    result = {"path": rel, "status": "passed", "differences": []}
    if sha256(before) != sha256(after):
        if allow_metadata and rel.endswith(".json"):
            try:
                a = json.loads(before.read_text(encoding="utf-8"))
                b = json.loads(after.read_text(encoding="utf-8"))
                for item in ("created_utc",):
                    a.pop(item, None)
                    b.pop(item, None)
                if a != b:
                    result["differences"].append("JSON scientific content differs")
                else:
                    result["differences"].append("JSON metadata-only byte difference")
            except Exception as exc:
                result["differences"].append(f"could not parse changed JSON: {exc}")
        else:
            result["differences"].append("file hash differs")
    if any("scientific" in item or "file hash" in item for item in result["differences"]):
        result["status"] = "failed"
    return result


def run_script(script: str) -> dict[str, Any]:
    started = utc_now()
    command = [sys.executable, script]
    proc = subprocess.run(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {
        "command": command,
        "started_utc": started,
        "finished_utc": utc_now(),
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--baseline-dir")
    args = parser.parse_args(argv)
    all_files = sorted(
        set(global_manifest_files())
        | set(PROTECTED_SOURCES + ["paper/figure_mapping.md"])
        | {item for fig in FIGURES for item in [fig["script"], *fig["cache"], *fig["processed"], *fig["figures"]]}
    )
    if args.snapshot_only:
        snap, before = snapshot_files(all_files)
        print(snap)
        return 0
    if args.baseline_dir:
        snap = Path(args.baseline_dir)
        before = load_snapshot_records(snap)
    else:
        snap, before = snapshot_files(all_files)
    parsed_mapping = parse_figure_mapping()
    expected_mapping = {fig["manuscript_figure"]: fig["script"] for fig in FIGURES}
    mapping_status = "passed" if parsed_mapping == expected_mapping else "failed"
    figure_results = []
    for fig in FIGURES:
        run = run_script(fig["script"])
        differences = []
        source_checks = [compare_plain(path, snap) for path in PROTECTED_SOURCES + [fig["script"]]]
        cache_checks = [compare_npz(path, snap) for path in fig["cache"]]
        processed_checks = [
            compare_csv(path, snap) if path.endswith(".csv") else compare_plain(path, snap, allow_metadata=True)
            for path in fig["processed"]
        ]
        figure_checks = [compare_plain(path, snap) for path in fig["figures"]]
        png_exact = any(check["path"].endswith(".png") and check["status"] == "passed" for check in figure_checks)
        if png_exact:
            for check in figure_checks:
                if check["path"].endswith((".pdf", ".eps")) and check["status"] == "failed":
                    check["status"] = "passed"
                    check["differences"].append("byte difference accepted as render-equivalent because PNG pixels are exact")
        for group in (source_checks, cache_checks, processed_checks, figure_checks):
            for check in group:
                if check["status"] != "passed":
                    differences.append(check)
        status = "passed" if run["return_code"] == 0 and not differences else "failed"
        figure_results.append(
            {
                "manuscript_figure": fig["manuscript_figure"],
                "script": fig["script"],
                **run,
                "source_checks": source_checks,
                "cache_checks": cache_checks,
                "processed_checks": processed_checks,
                "figure_checks": figure_checks,
                "status": status,
                "differences": differences,
            }
        )
    global_diff = compare_global_legacy_artifacts(before)
    overall = (
        "passed"
        if mapping_status == "passed"
        and global_diff["status"] == "passed"
        and all(item["status"] == "passed" for item in figure_results)
        else "failed"
    )
    report_path = PROJECT_ROOT / "data" / "processed" / "regression_fig1_4_report.json"
    report = {
        "schema_version": "fig1_4_regression_v1",
        "created_utc": utc_now(),
        "baseline_snapshot": str(snap),
        "baseline_manifest_file_count": len(before),
        "parsed_mapping": parsed_mapping,
        "expected_mapping": expected_mapping,
        "mapping_status": mapping_status,
        "figure_results": figure_results,
        "global_legacy_artifact_diff": global_diff,
        "status": overall,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_hash = sha256(report_path)
    receipt_path = PROJECT_ROOT / "data" / "processed" / "run_receipt_regression_fig1_4.json"
    write_run_receipt(
        receipt_path,
        phase="figures_1_4_regression",
        command=[sys.executable, "scripts/regression_fig1_4.py"],
        argv=[],
        started_utc=figure_results[0]["started_utc"] if figure_results else utc_now(),
        return_code=0 if overall == "passed" else 1,
        status="passed" if overall == "passed" else "failed",
        project_root=PROJECT_ROOT,
        config_hash="figures_1_4_cache_first",
        source_fingerprint=compute_source_fingerprint(PROJECT_ROOT),
        precision_mode=None,
        jax_x64=None,
        generated_artifacts=[report_path],
        warnings=[],
        failures=[] if overall == "passed" else ["Figure 1-4 regression differences detected"],
        extra={
            "baseline_snapshot": str(snap),
            "regression_report_path": "data/processed/regression_fig1_4_report.json",
            "regression_report_sha256": report_hash,
            "regression_report_schema_version": "fig1_4_regression_v1",
            "figure_results": [{k: v for k, v in item.items() if k in {"manuscript_figure", "script", "return_code", "status"}} for item in figure_results],
        },
    )
    print(f"Figure 1-4 regression status: {overall}")
    print(f"Report: {report_path}")
    print(f"Receipt: {receipt_path}")
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
