"""CLI smoke tests for loraspec --version, doctor, inspect, audit, report, and Stage 6 stubs."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from loraspec import __version__
from loraspec.cli.app import app
from loraspec.core.errors import EXIT_AUDIT_FAIL, EXIT_CONFIG_ERROR, EXIT_PASS
from loraspec.core.schemas import AuditReport, LoRAAdapterSpec

runner = CliRunner()


def test_cli_version_and_doctor() -> None:
    ver_res = runner.invoke(app, ["--version"])
    assert ver_res.exit_code == EXIT_PASS
    assert f"loraspec {__version__}" in ver_res.stdout

    doc_res = runner.invoke(app, ["doctor"])
    assert doc_res.exit_code == EXIT_PASS
    assert "Python >= 3.11" in doc_res.stdout
    assert "Safetensors Parser" in doc_res.stdout


def test_cli_inspect_valid_and_invalid(
    orthogonal_adapter_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    path_a, _ = orthogonal_adapter_pair
    res = runner.invoke(app, ["inspect", str(path_a), "--json"])
    assert res.exit_code == EXIT_PASS, res.stderr
    spec = LoRAAdapterSpec.model_validate_json(res.stdout)
    assert spec.adapter_id == "sql_lora"
    assert spec.rank == 4
    assert len(spec.layers) == 4

    human_res = runner.invoke(app, ["inspect", str(path_a)])
    assert human_res.exit_code == EXIT_PASS
    assert "sql_lora" in human_res.stdout

    missing_res = runner.invoke(app, ["inspect", str(tmp_path / "does_not_exist")])
    assert missing_res.exit_code == EXIT_CONFIG_ERROR


def test_cli_audit_pass_fail_sarif_and_report(
    orthogonal_adapter_pair: tuple[Path, Path],
    colliding_conflicting_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    # 1. Clean orthogonal pair passes Grade A (exit 0)
    clean_out = tmp_path / "clean_out"
    clean_sarif = clean_out / "report.sarif"
    ok_res = runner.invoke(
        app,
        [
            "audit",
            str(orthogonal_adapter_pair[0]),
            str(orthogonal_adapter_pair[1]),
            "--out-dir",
            str(clean_out),
            "--sarif",
            str(clean_sarif),
        ],
    )
    assert ok_res.exit_code == EXIT_PASS, ok_res.stdout + ok_res.stderr
    clean_rep = AuditReport.model_validate_json(
        (clean_out / "report.json").read_text(encoding="utf-8")
    )
    assert clean_rep.grade == "A"
    assert clean_rep.merge_risk_score == 0.0
    assert (clean_out / "manifest.json").is_file()
    assert json.loads(clean_sarif.read_text(encoding="utf-8"))["version"] == "2.1.0"

    # 2. Colliding conflicting pair fails Grade F (exit 1)
    vuln_out = tmp_path / "vuln_out"
    vuln_sarif = vuln_out / "report.sarif"
    fail_res = runner.invoke(
        app,
        [
            "audit",
            str(colliding_conflicting_pair[0]),
            str(colliding_conflicting_pair[1]),
            "--out-dir",
            str(vuln_out),
            "--sarif",
            str(vuln_sarif),
        ],
    )
    assert fail_res.exit_code == EXIT_AUDIT_FAIL
    assert "RULE-SPEC-004" in fail_res.stdout

    # 3. Duplicate path deduplication doesn't silently skip self-collision
    dup_res = runner.invoke(
        app,
        [
            "audit",
            str(orthogonal_adapter_pair[0]),
            str(orthogonal_adapter_pair[0]),
            "--fail-on",
            "critical",
        ],
    )
    # Same weights -> chordal 1.0, sign_conflict 0.0 -> HIGH RULE-SPEC-004 (passes critical gate)
    assert dup_res.exit_code == EXIT_PASS
    assert "RULE-SPEC-004" in dup_res.stdout

    # 4. `loraspec report` validates and re-renders SARIF cleanly
    rerender_sarif = tmp_path / "rerender.sarif"
    rep_res = runner.invoke(
        app,
        ["report", str(vuln_out / "report.json"), "--sarif", str(rerender_sarif)],
    )
    assert rep_res.exit_code == EXIT_PASS
    assert json.loads(rerender_sarif.read_text(encoding="utf-8"))["runs"][0]["results"]

    bad_rep = runner.invoke(app, ["report", str(tmp_path / "missing.json")])
    assert bad_rep.exit_code == EXIT_CONFIG_ERROR


def test_cli_stage6_stubs_exit_deterministically(
    orthogonal_adapter_pair: tuple[Path, Path],
) -> None:
    path_a, path_b = orthogonal_adapter_pair
    for args in (
        ["merge", str(path_a), str(path_b)],
        ["bench", str(path_a)],
        ["serve"],
        ["replay", "trace.jsonl"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == EXIT_CONFIG_ERROR
        assert "Stage 6" in res.stderr
