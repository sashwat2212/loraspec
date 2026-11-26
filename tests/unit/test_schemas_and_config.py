"""Unit tests for Pydantic schemas and strict `loraspec.toml` loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from loraspec.core.config import LoraspecConfig, load_config
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
