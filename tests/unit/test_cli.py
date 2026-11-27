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
    # 1. Clean orthogonal pair passes Grade A (exit 0) + writes HTML card
    clean_out = tmp_path / "clean_out"
    clean_sarif = clean_out / "report.sarif"
    clean_html = clean_out / "report.html"
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
            "--html",
            str(clean_html),
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
    assert clean_html.is_file()
    assert "Grade A" in clean_html.read_text(encoding="utf-8") or "SAFE" in clean_html.read_text(
        encoding="utf-8"
    )

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

    # 4. `loraspec report` validates and re-renders SARIF and HTML cleanly
    rerender_sarif = tmp_path / "rerender.sarif"
    rerender_html = tmp_path / "rerender.html"
    rep_res = runner.invoke(
        app,
        [
            "report",
            str(vuln_out / "report.json"),
            "--sarif",
            str(rerender_sarif),
            "--html",
            str(rerender_html),
        ],
    )
    assert rep_res.exit_code == EXIT_PASS
    assert json.loads(rerender_sarif.read_text(encoding="utf-8"))["runs"][0]["results"]
    assert rerender_html.is_file()
    assert "RULE-SPEC-004" in rerender_html.read_text(encoding="utf-8")

    bad_rep = runner.invoke(app, ["report", str(tmp_path / "missing.json")])
    assert bad_rep.exit_code == EXIT_CONFIG_ERROR

    # 5. --tiers 0 runs config probes cleanly and skips Tier 1; invalid --tiers exits 2
    tier0_res = runner.invoke(
        app,
        [
            "audit",
            str(colliding_conflicting_pair[0]),
            str(colliding_conflicting_pair[1]),
            "--tiers",
            "0",
        ],
    )
    assert tier0_res.exit_code == EXIT_PASS  # Collisions only detected at Tier 1

    bad_tier = runner.invoke(app, ["audit", str(orthogonal_adapter_pair[0]), "--tiers", "9"])
    assert bad_tier.exit_code == EXIT_CONFIG_ERROR

    # 6. Rich markup escaping & duplicate id deduplication across foo / foo / foo_1
    d1 = tmp_path / "d1" / "foo_[red]"
    d2 = tmp_path / "d2" / "foo_[red]"
    d3 = tmp_path / "d3" / "foo_[red]_1"
    for d in (d1, d2, d3):
        d.mkdir(parents=True)
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            (d / fname).write_bytes((orthogonal_adapter_pair[0] / fname).read_bytes())

    insp_markup = runner.invoke(app, ["inspect", str(d1)])
    assert insp_markup.exit_code == EXIT_PASS
    assert "foo_[red]" in insp_markup.stdout

    dedup_out = tmp_path / "dedup_out"
    dedup_res = runner.invoke(
        app,
        ["audit", str(d1), str(d2), str(d3), "--out-dir", str(dedup_out), "--fail-on", "critical"],
    )
    assert dedup_res.exit_code == EXIT_PASS
    rep_dedup = AuditReport.model_validate_json(
        (dedup_out / "report.json").read_text(encoding="utf-8")
    )
    assert len({a.adapter_id for a in rep_dedup.adapters}) == 3
    assert len(rep_dedup.manifest.adapter_sha256s) == 3

    # 7. Unwritable --out-dir (pointing at existing file) exits deterministically with code 2
    file_target = tmp_path / "not_a_dir.txt"
    file_target.write_text("x", encoding="utf-8")
    bad_out = runner.invoke(
        app, ["audit", str(orthogonal_adapter_pair[0]), "--out-dir", str(file_target)]
    )
    assert bad_out.exit_code == EXIT_CONFIG_ERROR

    # 8. --fail-on info passes on Grade A (0 findings) and --help lists all 5 severities
    info_res = runner.invoke(
        app,
        [
            "audit",
            str(orthogonal_adapter_pair[0]),
            str(orthogonal_adapter_pair[1]),
            "--fail-on",
            "info",
        ],
    )
    assert info_res.exit_code == EXIT_PASS
    help_res = runner.invoke(app, ["audit", "--help"], env={"COLUMNS": "200"})
    assert help_res.exit_code == EXIT_PASS
    assert "critical|high|medium|low|info" in help_res.stdout


def test_cli_stage6_stubs_exit_deterministically(
    orthogonal_adapter_pair: tuple[Path, Path],
) -> None:
    path_a, _ = orthogonal_adapter_pair
    for args in (
        ["serve"],
        ["replay", "trace.jsonl"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == EXIT_CONFIG_ERROR
        assert "Stage 6" in res.stderr

    # `bench` is active in M2: single adapter with default --top-k 2 raises ConfigError (exit 2)
    single_bench = runner.invoke(app, ["bench", str(path_a)])
    assert single_bench.exit_code == EXIT_CONFIG_ERROR
    assert "Stage 6" not in single_bench.stderr

    syn_bench = runner.invoke(app, ["bench", "--synthetic", "--smoke"])
    assert syn_bench.exit_code == EXIT_PASS, syn_bench.stdout + syn_bench.stderr


def test_cli_merge_validation_errors(
    orthogonal_adapter_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    path_a, path_b = orthogonal_adapter_pair

    # Invalid --weights count exits 2
    bad_weights = runner.invoke(
        app,
        [
            "merge",
            str(path_a),
            str(path_b),
            "--weights",
            "1.0",
            "--out-dir",
            str(tmp_path / "m1"),
        ],
    )
    assert bad_weights.exit_code == EXIT_CONFIG_ERROR

    # Invalid --weights non-numeric exits 2
    bad_weights_str = runner.invoke(
        app,
        [
            "merge",
            str(path_a),
            str(path_b),
            "--weights",
            "abc,def",
            "--out-dir",
            str(tmp_path / "m2"),
        ],
    )
    assert bad_weights_str.exit_code == EXIT_CONFIG_ERROR

    # Unwritable --out-dir (regular file) exits 2
    bad_file = tmp_path / "file_target.txt"
    bad_file.write_text("x", encoding="utf-8")
    bad_out = runner.invoke(
        app,
        ["merge", str(path_a), str(path_b), "--out-dir", str(bad_file)],
    )
    assert bad_out.exit_code == EXIT_CONFIG_ERROR

    # Invalid --dtype or --layer-policy exits 2
    bad_dtype = runner.invoke(
        app,
        ["merge", str(path_a), str(path_b), "--dtype", "int8", "--out-dir", str(tmp_path / "m3")],
    )
    assert bad_dtype.exit_code == EXIT_CONFIG_ERROR

    bad_policy = runner.invoke(
        app,
        [
            "merge",
            str(path_a),
            str(path_b),
            "--layer-policy",
            "outer",
            "--out-dir",
            str(tmp_path / "m4"),
        ],
    )
    assert bad_policy.exit_code == EXIT_CONFIG_ERROR
