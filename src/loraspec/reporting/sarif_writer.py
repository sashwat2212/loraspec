"""OASIS SARIF 2.1.0 writer mapping AuditFindings into GitHub Code Scanning results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loraspec import __version__
from loraspec.core.schemas import AuditFinding, AuditReport, Severity

SARIF_SCHEMA_URI = "https://json.schemastore.org/sarif-2.1.0.json"
LORASPEC_INFO_URI = "https://github.com/sashwat2212/loraspec"

_SARIF_LEVEL_MAP: dict[Severity, str] = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}


def _find_git_root(start: Path) -> Path | None:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in (cur, *cur.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _normalize_uri(
    file_path: str | None,
    root_paths: list[Path],
    git_root: Path | None,
) -> str:
    if not file_path:
        if git_root is not None and root_paths:
            try:
                rel_dir = root_paths[0].resolve().relative_to(git_root)
                rel_str = rel_dir.as_posix()
                return rel_str if rel_str else "."
            except ValueError:
                pass
        return "."

    raw = file_path.strip()
    if not raw or raw in {".", "./"}:
        return "."

    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if git_root is not None:
            try:
                return resolved.relative_to(git_root).as_posix()
            except ValueError:
                pass
        for root in root_paths:
            try:
                return resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
        return resolved.as_posix().lstrip("/") or "."

    cleaned = raw
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.lstrip("/")
    return cleaned or "."


def build_sarif_dict(report: AuditReport) -> dict[str, Any]:
    """Build a valid SARIF 2.1.0 dictionary from an AuditReport."""
    root_paths = [spec.root_path for spec in report.adapters]
    git_root = _find_git_root(root_paths[0]) if root_paths else _find_git_root(Path.cwd())

    rules_by_id: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    for finding in report.findings:
        _register_rule(rules_by_id, finding)
        uri = _normalize_uri(finding.file_path, root_paths, git_root)
        region: dict[str, Any] | None = (
            {"startLine": finding.line_number}
            if finding.line_number is not None and finding.line_number >= 1
            else None
        )
        phys_loc: dict[str, Any] = {"artifactLocation": {"uri": uri}}
        if region is not None:
            phys_loc["region"] = region

        results.append(
            {
                "ruleId": finding.rule_id,
                "level": _SARIF_LEVEL_MAP[finding.severity],
                "message": {"text": f"{finding.title} — {finding.evidence}"},
                "locations": [{"physicalLocation": phys_loc}],
                "properties": {
                    "severity": finding.severity.value,
                    "tier": finding.tier,
                    "source": finding.source,
                    "remediation": finding.remediation,
                },
            }
        )

    return {
        "$schema": SARIF_SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "loraspec",
                        "semanticVersion": __version__,
                        "informationUri": LORASPEC_INFO_URI,
                        "rules": list(rules_by_id.values()),
                    }
                },
                "results": results,
                "properties": {
                    "run_id": report.manifest.run_id,
                    "grade": report.grade,
                    "merge_risk_score": report.merge_risk_score,
                    "recommendations": [r.value for r in report.recommendations],
                },
            }
        ],
    }


def _register_rule(rules_by_id: dict[str, dict[str, Any]], finding: AuditFinding) -> None:
    if finding.rule_id in rules_by_id:
        return
    rules_by_id[finding.rule_id] = {
        "id": finding.rule_id,
        "name": finding.rule_id,
        "shortDescription": {"text": finding.title},
        "fullDescription": {"text": finding.remediation},
        "help": {
            "text": finding.remediation,
            "markdown": f"**Remediation:** {finding.remediation}",
        },
        "defaultConfiguration": {"level": _SARIF_LEVEL_MAP[finding.severity]},
    }


def write_sarif(report: AuditReport, out_path: Path) -> Path:
    """Serialize AuditReport as SARIF 2.1.0 JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_sarif_dict(report)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
