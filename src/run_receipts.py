"""Machine-readable execution receipts for the numerical figure scripts."""

from __future__ import annotations

import json
import platform
import subprocess
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTENT_FINGERPRINT_FILES = [
    "src/high_dimensional_utils.py",
    "src/run_receipts.py",
    "scripts/fig06_d5_block_selection.py",
    "scripts/appendix_d5_diagnostics.py",
    "scripts/regression_fig1_4.py",
    "tests/test_high_dimensional_utils.py",
    "requirements.txt",
    "requirements-lock.txt",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_metadata(project_root: Path) -> dict[str, Any]:
    commit = "unavailable"
    dirty = None
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        commit = out.stdout.strip() or "unavailable"
    except Exception:
        commit = "unavailable"
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), "status", "--short"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        dirty = bool(out.stdout.strip())
    except Exception:
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python_version": platform.python_version()}
    for name, key in [
        ("numpy", "numpy_version"),
        ("scipy", "scipy_version"),
        ("jax", "jax_version"),
        ("jaxlib", "jaxlib_version"),
        ("diffrax", "diffrax_version"),
        ("matplotlib", "matplotlib_version"),
    ]:
        try:
            module = __import__(name)
            versions[key] = getattr(module, "__version__", "unavailable")
        except Exception:
            versions[key] = "unavailable"
    return versions


def compute_source_fingerprint(
    project_root: Path,
    relative_paths: Sequence[str] = CONTENT_FINGERPRINT_FILES,
) -> str:
    """Return a deterministic SHA-256 fingerprint of content-critical sources."""
    digest = hashlib.sha256()
    for rel in sorted(relative_paths):
        path = project_root / rel
        if not path.exists():
            raise FileNotFoundError(f"source fingerprint input is missing: {rel}")
        data = path.read_bytes()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_digest_record(project_root: Path, path: Path | str) -> dict[str, Any]:
    """Return a repository-relative path, size, and SHA-256 record."""
    item = Path(path)
    if item.is_absolute():
        try:
            rel = item.relative_to(project_root)
        except ValueError:
            rel = item
    else:
        rel = item
        item = project_root / item
    if not item.exists():
        raise FileNotFoundError(f"generated artifact does not exist: {rel}")
    data = item.read_bytes()
    return {"path": str(rel), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def artifact_digest_records(
    project_root: Path,
    artifacts: Sequence[Path | str],
    *,
    receipt_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return sorted artifact digest records, excluding the receipt itself."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    receipt_resolved = receipt_path.resolve() if receipt_path is not None else None
    for artifact in artifacts:
        path = Path(artifact)
        abs_path = path if path.is_absolute() else project_root / path
        if receipt_resolved is not None and abs_path.resolve() == receipt_resolved:
            continue
        record = file_digest_record(project_root, abs_path)
        if record["path"] in seen:
            raise ValueError(f"duplicate generated artifact in receipt: {record['path']}")
        seen.add(record["path"])
        records.append(record)
    return sorted(records, key=lambda item: item["path"])


def parse_utc_timestamp(value: str | None, *, allow_none: bool = False) -> datetime | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError("timestamp is missing")
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp is timezone-naive")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp is not UTC")
    return parsed


def validate_receipt_timestamps(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        start = parse_utc_timestamp(receipt.get("started_utc"))
    except Exception as exc:
        errors.append(f"invalid started_utc: {exc}")
        start = None
    allow_no_finish = receipt.get("status") == "running"
    try:
        finish = parse_utc_timestamp(receipt.get("finished_utc"), allow_none=allow_no_finish)
    except Exception as exc:
        errors.append(f"invalid finished_utc: {exc}")
        finish = None
    if start is not None and finish is not None and finish < start:
        errors.append("finished_utc precedes started_utc")
    return errors


def write_run_receipt(
    path: Path,
    *,
    phase: str,
    command: Sequence[str],
    argv: Sequence[str],
    started_utc: str,
    finished_utc: str | None = None,
    return_code: int = 0,
    status: str = "passed",
    project_root: Path,
    config_hash: str | None = None,
    phase_a_source_config_hash: str | None = None,
    appendix_package_config_hash: str | None = None,
    source_fingerprint: str | None = None,
    precision_mode: str | None = None,
    jax_x64: bool | None = None,
    generated_artifacts: Sequence[Path | str] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    git = git_metadata(project_root)
    artifact_records = artifact_digest_records(project_root, generated_artifacts, receipt_path=path)
    receipt = {
        "schema_version": "run_receipt_v2",
        "phase": phase,
        "command": list(command),
        "argv": list(argv),
        "started_utc": started_utc,
        "finished_utc": finished_utc or utc_now(),
        "return_code": int(return_code),
        "status": status,
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "config_hash": config_hash or "",
        "phase_a_source_config_hash": phase_a_source_config_hash,
        "appendix_package_config_hash": appendix_package_config_hash,
        "source_fingerprint": source_fingerprint or compute_source_fingerprint(project_root),
        "precision_mode": precision_mode,
        "jax_x64": jax_x64,
        **package_versions(),
        "generated_artifacts": artifact_records,
        "warnings": list(warnings),
        "failures": list(failures),
    }
    if extra:
        receipt.update(dict(extra))
    _atomic_json_write(path, receipt)
    return receipt


def write_failed_receipt(
    path: Path,
    *,
    phase: str,
    command: Sequence[str],
    argv: Sequence[str],
    started_utc: str,
    project_root: Path,
    config_hash: str | None,
    phase_a_source_config_hash: str | None = None,
    appendix_package_config_hash: str | None = None,
    exc: BaseException,
    failure_stage: str,
    traceback_excerpt: str,
    generated_artifacts: Sequence[Path | str] = (),
) -> dict[str, Any]:
    return write_run_receipt(
        path,
        phase=phase,
        command=command,
        argv=argv,
        started_utc=started_utc,
        return_code=1,
        status="failed",
        project_root=project_root,
        config_hash=config_hash,
        phase_a_source_config_hash=phase_a_source_config_hash,
        appendix_package_config_hash=appendix_package_config_hash,
        generated_artifacts=generated_artifacts,
        warnings=[],
        failures=[str(exc)],
        extra={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "failure_stage": failure_stage,
            "traceback_excerpt": traceback_excerpt,
        },
    )


def read_receipt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_receipt(
    path: Path,
    *,
    phase: str,
    project_root: Path,
    expected_config_hash: str | None = None,
    expected_phase_a_source_config_hash: str | None = None,
    expected_appendix_package_config_hash: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_precision_mode: str | None = None,
    expected_jax_x64: bool | None = None,
    required_argv_tokens: Sequence[str] = (),
    required_artifacts: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return [f"missing receipt {path}"], warnings
    try:
        receipt = read_receipt(path)
    except Exception as exc:
        return [f"invalid receipt {path}: {exc}"], warnings
    required = [
        "schema_version",
        "phase",
        "command",
        "argv",
        "started_utc",
        "finished_utc",
        "return_code",
        "status",
        "git_commit",
        "git_dirty",
        "config_hash",
        "phase_a_source_config_hash",
        "appendix_package_config_hash",
        "source_fingerprint",
        "precision_mode",
        "jax_x64",
        "python_version",
        "numpy_version",
        "scipy_version",
        "jax_version",
        "jaxlib_version",
        "diffrax_version",
        "matplotlib_version",
        "generated_artifacts",
        "warnings",
        "failures",
    ]
    for key in required:
        if key not in receipt:
            errors.append(f"receipt missing {key}")
    if receipt.get("schema_version") != "run_receipt_v2":
        errors.append("receipt schema_version mismatch")
    if receipt.get("phase") != phase:
        errors.append(f"receipt phase {receipt.get('phase')!r} != {phase!r}")
    if receipt.get("return_code") != 0:
        errors.append("receipt return_code is nonzero")
    if receipt.get("status") not in {"passed", "completed with warnings"}:
        errors.append(f"receipt status is {receipt.get('status')!r}")
    if expected_config_hash is not None and receipt.get("config_hash") != expected_config_hash:
        errors.append("receipt config_hash mismatch")
    if expected_phase_a_source_config_hash is not None and receipt.get("phase_a_source_config_hash") != expected_phase_a_source_config_hash:
        errors.append("receipt phase_a_source_config_hash mismatch")
    if expected_appendix_package_config_hash is not None and receipt.get("appendix_package_config_hash") != expected_appendix_package_config_hash:
        errors.append("receipt appendix_package_config_hash mismatch")
    if expected_precision_mode is not None and receipt.get("precision_mode") != expected_precision_mode:
        errors.append("receipt precision_mode mismatch")
    if expected_jax_x64 is not None and receipt.get("jax_x64") is not expected_jax_x64:
        errors.append("receipt jax_x64 mismatch")
    errors.extend(validate_receipt_timestamps(receipt))
    expected_fingerprint = expected_source_fingerprint
    if expected_fingerprint is None:
        try:
            expected_fingerprint = compute_source_fingerprint(project_root)
        except Exception as exc:
            errors.append(f"could not compute source fingerprint: {exc}")
            expected_fingerprint = None
    if expected_fingerprint is not None and receipt.get("source_fingerprint") != expected_fingerprint:
        errors.append("receipt source_fingerprint mismatch")
    argv_text = " ".join(str(item) for item in receipt.get("argv", []) + receipt.get("command", []))
    for token in required_argv_tokens:
        if token not in argv_text:
            errors.append(f"receipt command does not include {token}")
    artifacts = receipt.get("generated_artifacts", [])
    if not artifacts:
        errors.append("receipt generated_artifacts is empty")
    found_paths: set[str] = set()
    for record in artifacts:
        if not isinstance(record, dict):
            errors.append("receipt artifact record is not an object")
            continue
        rel = record.get("path")
        found_paths.add(str(rel))
        artifact = project_root / str(rel)
        if not artifact.exists() or artifact.stat().st_size == 0:
            errors.append(f"receipt artifact missing or empty: {rel}")
            continue
        current = file_digest_record(project_root, artifact)
        if current["size_bytes"] != record.get("size_bytes"):
            errors.append(f"receipt artifact size mismatch: {rel}")
        if current["sha256"] != record.get("sha256"):
            errors.append(f"receipt artifact sha256 mismatch: {rel}")
    for rel in required_artifacts:
        if rel not in found_paths:
            errors.append(f"receipt missing required artifact: {rel}")
    return errors, warnings


def record_figure_regression(
    project_root: Path,
    *,
    figure_script: str,
    generated_artifacts: Sequence[Path | str],
) -> None:
    receipt_path = project_root / "data" / "processed" / "run_receipt_figures_cache_first.json"
    started = utc_now()
    previous: list[dict[str, Any]] = []
    if receipt_path.exists():
        try:
            payload = read_receipt(receipt_path)
            previous = list(payload.get("figure_results", []))
        except Exception:
            previous = []
    previous = [item for item in previous if item.get("script") != figure_script]
    previous.append(
        {
            "script": figure_script,
            "return_code": 0,
            "status": "passed",
            "generated_artifacts": [
                str(Path(path).relative_to(project_root)) if Path(path).is_absolute() else str(path)
                for path in generated_artifacts
            ],
            "finished_utc": utc_now(),
        }
    )
    expected = [
        "fig01_fast_locking.py",
        "fig02_phase_gap_prediction.py",
        "fig03_slow_polar_drift.py",
        "fig04_hitting_time.py",
        "fig05_gaussian_robustness.py",
    ]
    missing = [name for name in expected if name not in {item.get("script") for item in previous}]
    status = "passed" if not missing else "completed with warnings"
    write_run_receipt(
        receipt_path,
        phase="figures_cache_first_render",
        command=["python", f"scripts/{figure_script}"],
        argv=[],
        started_utc=started,
        return_code=0,
        status=status,
        project_root=project_root,
        config_hash="figures_cache_first",
        generated_artifacts=[artifact for item in previous for artifact in item["generated_artifacts"] if Path(project_root / artifact).exists()],
        warnings=[f"missing regression script {name}" for name in missing],
        failures=[],
        extra={"figure_results": previous},
    )
