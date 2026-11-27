"""Unit tests for Pydantic schemas and strict `loraspec.toml` loading."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from loraspec.core.config import LoraspecConfig, load_config, resolve_math_backend
from loraspec.core.errors import EXIT_CONFIG_ERROR, ConfigError
from loraspec.core.schemas import AuditFinding, Severity, meets_fail_threshold


def test_default_and_valid_config(tmp_path: Path) -> None:
    default_cfg = load_config(None)
    assert isinstance(default_cfg, LoraspecConfig)
    assert default_cfg.project.fail_on == Severity.HIGH
    assert default_cfg.budget.max_cost_inr == 2500.0

    cfg_path = tmp_path / "loraspec.toml"
    cfg_path.write_text(
        '[project]\nfail_on = "critical"\n\n[budget]\nmax_cost_inr = 1800.0\n',
        encoding="utf-8",
    )
    loaded = load_config(cfg_path)
    assert loaded.project.fail_on == Severity.CRITICAL
    assert loaded.budget.max_cost_inr == 1800.0


def test_invalid_config_rejects_unknown_fields(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(ConfigError) as exc_info:
        load_config(missing)
    assert exc_info.value.exit_code == EXIT_CONFIG_ERROR

    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("[project]\nunknown_key = true\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_toml)

    syntax_bad = tmp_path / "syntax.toml"
    syntax_bad.write_text("[project\nfail_on = 'high'\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(syntax_bad)

    inverted_thresh = tmp_path / "inverted.toml"
    inverted_thresh.write_text(
        "[audit]\ncollision_high_chordal = 0.80\ncollision_crit_chordal = 0.40\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(inverted_thresh)


def test_meets_fail_threshold() -> None:
    findings = [
        AuditFinding(
            rule_id="RULE-CONF-004",
            title="Mixed ranks",
            severity=Severity.MEDIUM,
            tier=0,
            source="config_probe",
            evidence="8 vs 16",
            remediation="Use KnOTS.",
        )
    ]
    assert meets_fail_threshold(findings, Severity.MEDIUM) is True
    assert meets_fail_threshold(findings, Severity.HIGH) is False
    assert meets_fail_threshold([], Severity.LOW) is False


def test_schemas_ast_parity_with_frd_prd() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    schemas_src = (repo_root / "src" / "loraspec" / "core" / "schemas.py").read_text(
        encoding="utf-8"
    )
    frd_text = (repo_root / "docs" / "02-FRD-PRD.md").read_text(encoding="utf-8")

    match = re.search(
        r"#### A\. Stage 5 Implemented & Frozen Core Schemas.*?```python\n(.*?)\n```",
        frd_text,
        flags=re.DOTALL,
    )
    assert match is not None, "Could not locate §5.1A schema block in docs/02-FRD-PRD.md"
    frd_schemas_src = match.group(1)

    schemas_ast = ast.parse(schemas_src)
    # Strip optional leading module docstring in schemas.py before AST comparison
    if (
        schemas_ast.body
        and isinstance(schemas_ast.body[0], ast.Expr)
        and isinstance(schemas_ast.body[0].value, ast.Constant)
        and isinstance(schemas_ast.body[0].value.value, str)
    ):
        schemas_ast.body = schemas_ast.body[1:]

    frd_ast = ast.parse(frd_schemas_src)
    assert ast.dump(schemas_ast) == ast.dump(frd_ast)


def test_resolve_math_backend_mappings() -> None:
    assert resolve_math_backend("auto") == "auto"
    assert resolve_math_backend("pure_python") == "pure_python"
    assert resolve_math_backend("numpy") == "numpy"
    assert resolve_math_backend("triton_cuda") == "auto"
    assert resolve_math_backend("torch_mps") == "auto"
    assert resolve_math_backend("torch_cpu") == "auto"
    with pytest.raises(ConfigError):
        resolve_math_backend("unknown_backend")
