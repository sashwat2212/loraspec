"""Deterministic JSON writers for AuditReport and RunManifest."""

from __future__ import annotations

from pathlib import Path

from loraspec.core.schemas import AuditReport, RunManifest


def write_report_json(report: AuditReport, out_path: Path) -> Path:
    """Serialize AuditReport deterministically to formatted JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out_path


def write_manifest_json(manifest: RunManifest, out_path: Path) -> Path:
    """Serialize RunManifest deterministically to formatted JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out_path
