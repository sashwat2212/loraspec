"""End-to-end integration tests for KnOTS merging, PEFT export, and CLI round-trip."""

from __future__ import annotations

import json
import math
from pathlib import Path

from typer.testing import CliRunner

from loraspec.cli.app import app
from loraspec.core.errors import EXIT_AUDIT_FAIL, EXIT_CONFIG_ERROR, EXIT_PASS
from loraspec.core.schemas import AuditReport, KnOTSMergeSummary, LoRAAdapterSpec
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import load_peft_adapter, write_safetensors
from loraspec.knots import merge_knots_adapters

runner = CliRunner()


def _create_peft_dir(
    root: Path,
    *,
    rank: int,
    alpha: float,
    use_rslora: bool = False,
    seed_shift: int = 0,
    d: int = 24,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    prefixes = [
        "base_model.model.layers.0.self_attn.q_proj",
        "base_model.model.layers.0.mlp.gate_proj",
        "base_model.model.layers.0.mlp.up_proj",
        "base_model.model.layers.0.mlp.down_proj",
    ]
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": use_rslora,
        "bias": "none",
        "target_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"],
    }
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    tensors: dict[str, tuple[Matrix, str]] = {}
    for lidx, prefix in enumerate(prefixes):
        # Diagonal-like orthogonal offsets so effective rank is well-conditioned
        a_mat = [
            [
                1.0 if j == ((i + seed_shift + lidx) % d) else 0.05 * math.cos(i + j + seed_shift)
                for j in range(d)
            ]
            for i in range(rank)
        ]
        b_mat = [
            [
                1.5 if i == ((j + seed_shift + lidx) % d) else 0.05 * math.sin(i + j + seed_shift)
                for j in range(rank)
            ]
            for i in range(d)
        ]
        tensors[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
        tensors[f"{prefix}.lora_B.weight"] = (b_mat, "F32")

    write_safetensors(root / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    return root


def test_cli_merge_roundtrips_into_inspect_and_audit(tmp_path: Path) -> None:
    sql_dir = _create_peft_dir(tmp_path / "sql_r8", rank=8, alpha=8.0, seed_shift=0)
    rust_dir = _create_peft_dir(
        tmp_path / "rust_r16", rank=16, alpha=16.0, use_rslora=True, seed_shift=8
    )
    gen_dir = _create_peft_dir(tmp_path / "gen_r4", rank=4, alpha=4.0, seed_shift=16)

    merged_dir = tmp_path / "merged_out"
    res = runner.invoke(
        app,
        [
            "merge",
            str(sql_dir),
            str(rust_dir),
            "--out-dir",
            str(merged_dir),
            "--method",
            "knots_dare_ties",
            "--target-rank",
            "8",
            "--weights",
            "0.6,0.4",
            "--ties-density",
            "0.75",
            "--dare-drop-rate",
            "0.15",
            "--layer-policy",
            "intersection",
            "--gks-adapter",
            str(gen_dir),
            "--gks-lambda",
            "0.5",
            "--dtype",
            "f32",
            "--seed",
            "123",
        ],
    )
    assert res.exit_code == EXIT_PASS, res.stdout + res.stderr
    assert (merged_dir / "adapter_model.safetensors").is_file()
    assert (merged_dir / "adapter_config.json").is_file()
    manifest_file = merged_dir / "knots_merge_manifest.json"
    assert manifest_file.is_file()

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    summary = KnOTSMergeSummary.model_validate(manifest_data["summary"])
    assert summary.method == "knots_dare_ties"
    assert summary.target_rank == 8
    assert summary.adapters_merged == ["sql_r8", "rust_r16"]

    # Round-trip merged adapter into `loraspec inspect`
    insp = runner.invoke(app, ["inspect", str(merged_dir), "--json"])
    assert insp.exit_code == EXIT_PASS, insp.stderr
    spec = LoRAAdapterSpec.model_validate_json(insp.stdout)
    assert spec.rank == 8
    assert spec.lora_alpha == 8.0
    assert spec.scaling == 1.0
    assert len(spec.layers) == 4

    # Round-trip merged adapter into `loraspec audit`
    audit_out = tmp_path / "post_merge_audit"
    aud = runner.invoke(
        app,
        [
            "audit",
            str(merged_dir),
            "--out-dir",
            str(audit_out),
            "--html",
            str(audit_out / "report.html"),
        ],
    )
    assert aud.exit_code == EXIT_PASS, aud.stdout + aud.stderr
    rep = AuditReport.model_validate_json((audit_out / "report.json").read_text(encoding="utf-8"))
    assert rep.grade == "A"
    assert rep.merge_risk_score == 0.0
    assert (audit_out / "report.html").is_file()


def test_knots_export_peft_schema_compatibility(tmp_path: Path) -> None:
    a_dir = _create_peft_dir(tmp_path / "a_r4", rank=4, alpha=4.0, seed_shift=0)
    b_dir = _create_peft_dir(tmp_path / "b_r8", rank=8, alpha=8.0, seed_shift=6)

    loaded = [load_peft_adapter(a_dir), load_peft_adapter(b_dir)]
    for dtype_code in ("F32", "F16", "BF16"):
        out_dir = tmp_path / f"export_{dtype_code}"
        summary, _ = merge_knots_adapters(
            loaded,
            out_dir=out_dir,
            method="knots_ties",
            target_rank=8,
            dtype=dtype_code,
        )
        cfg = json.loads((out_dir / "adapter_config.json").read_text(encoding="utf-8"))
        assert cfg["peft_type"] == "LORA"
        assert cfg["r"] == 8
        assert cfg["lora_alpha"] == 8.0
        assert cfg["use_rslora"] is False
        assert cfg["bias"] == "none"
        assert cfg["base_model_name_or_path"] == "Qwen/Qwen2.5-3B-Instruct"
        assert sorted(cfg["target_modules"]) == ["down_proj", "gate_proj", "q_proj", "up_proj"]

        reloaded = load_peft_adapter(out_dir)
        assert reloaded.spec.weights_sha256 == summary.output_sha256
        assert all(ls.dtype == dtype_code for ls in reloaded.spec.layers)

    # Pre-merge audit failure on NaN adapter exits 1 cleanly
    nan_dir = tmp_path / "nan_adapter"
    nan_dir.mkdir(parents=True)
    (nan_dir / "adapter_config.json").write_text(
        (a_dir / "adapter_config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_safetensors(
        nan_dir / "adapter_model.safetensors",
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": (
                [[1.0 if j == i else 0.0 for j in range(24)] for i in range(4)],
                "F32",
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": (
                [[math.nan if (i == 0 and j == 0) else 0.0 for j in range(4)] for i in range(24)],
                "F32",
            ),
        },
    )
    nan_merge = runner.invoke(
        app, ["merge", str(a_dir), str(nan_dir), "--out-dir", str(tmp_path / "nan_out")]
    )
    assert nan_merge.exit_code == EXIT_AUDIT_FAIL

    # Invalid CLI flags / single path exit 2 cleanly
    bad_single = runner.invoke(app, ["merge", str(a_dir)])
    assert bad_single.exit_code == EXIT_CONFIG_ERROR
    bad_method = runner.invoke(app, ["merge", str(a_dir), str(b_dir), "--method", "unknown_method"])
    assert bad_method.exit_code == EXIT_CONFIG_ERROR


def test_knots_union_heterogeneous_rank_pattern_and_rule_knots_001_cli(tmp_path: Path) -> None:
    d = 24
    # Adapter A has r=4 on shared + exclusive layer; Adapter B has r=8 on shared only
    a_dir = _create_peft_dir(tmp_path / "u_a", rank=4, alpha=4.0, seed_shift=0, d=d)
    # Add an exclusive layer to Adapter A
    a_loaded = load_peft_adapter(a_dir)
    ex_tensors: dict[str, tuple[Matrix, str]] = {}
    for lspec in a_loaded.spec.layers:
        a_m, b_m = a_loaded.layer_factors[lspec.layer_name]
        ex_tensors[lspec.a_key] = (a_m, "F32")
        ex_tensors[lspec.b_key] = (b_m, "F32")
    ex_layer = "base_model.model.layers.1.mlp.up_proj"
    ex_tensors[f"{ex_layer}.lora_A.weight"] = (
        [[1.0 if j == i else 0.0 for j in range(d)] for i in range(4)],
        "F32",
    )
    ex_tensors[f"{ex_layer}.lora_B.weight"] = (
        [[1.0 if i == j else 0.0 for j in range(4)] for i in range(d)],
        "F32",
    )
    write_safetensors(a_dir / "adapter_model.safetensors", ex_tensors, metadata={"format": "pt"})

    b_dir = _create_peft_dir(tmp_path / "u_b", rank=8, alpha=8.0, seed_shift=6, d=d)
    merged_union = tmp_path / "merged_union"
    res = runner.invoke(
        app,
        [
            "merge",
            str(a_dir),
            str(b_dir),
            "--out-dir",
            str(merged_union),
            "--target-rank",
            "8",
            "--layer-policy",
            "union",
        ],
    )
    assert res.exit_code == EXIT_PASS, res.stdout + res.stderr
    cfg_union = json.loads((merged_union / "adapter_config.json").read_text(encoding="utf-8"))
    assert "rank_pattern" in cfg_union
    assert cfg_union["rank_pattern"][ex_layer] == 4

    # Round-trip audit must NOT emit RULE-CONF-002 on rank_pattern
    aud_union = runner.invoke(
        app, ["audit", str(merged_union), "--out-dir", str(tmp_path / "aud_u")]
    )
    assert aud_union.exit_code == EXIT_PASS, aud_union.stdout + aud_union.stderr
    rep_u = AuditReport.model_validate_json(
        (tmp_path / "aud_u" / "report.json").read_text(encoding="utf-8")
    )
    assert not any(f.rule_id == "RULE-CONF-002" for f in rep_u.findings)

    # Compressing to target_rank=1 triggers RULE-KNOTS-001 (MEDIUM):
    # exits 0 under default fail_on=high, exits 1 when fail_on="medium" in loraspec.toml
    toml_med = tmp_path / "med.toml"
    toml_med.write_text('[project]\nfail_on = "medium"\n', encoding="utf-8")
    low_rank_ok = runner.invoke(
        app,
        [
            "merge",
            str(a_dir),
            str(b_dir),
            "--out-dir",
            str(tmp_path / "low_ok"),
            "--target-rank",
            "1",
        ],
    )
    assert low_rank_ok.exit_code == EXIT_PASS
    assert "RULE-KNOTS-001" in low_rank_ok.stdout

    low_rank_fail = runner.invoke(
        app,
        [
            "merge",
            str(a_dir),
            str(b_dir),
            "--out-dir",
            str(tmp_path / "low_fail"),
            "--target-rank",
            "1",
            "--config",
            str(toml_med),
        ],
    )
    assert low_rank_fail.exit_code == EXIT_AUDIT_FAIL
