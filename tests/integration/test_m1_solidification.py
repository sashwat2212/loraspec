"""Comprehensive Milestone 1 (M1) production-solidification & scale integration suite."""

from __future__ import annotations

import asyncio
import json
import math
import re
import struct
import subprocess
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
import yaml
from benchmarks.planted_collisions.generate_corpus import (
    evaluate_planted_recall,
    generate_planted_corpus,
)
from typer.testing import CliRunner

from loraspec.auditor.probes import run_spectral_and_collision_audit
from loraspec.cli.app import app
from loraspec.core.budget import GpuBudgetGuard
from loraspec.core.config import AuditConfig, LoraspecConfig, ProjectConfig
from loraspec.core.errors import (
    EXIT_AUDIT_FAIL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONFIG_ERROR,
    EXIT_PARITY_ERROR,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
    BudgetExceededError,
    KernelParityError,
    RuntimeExecutionError,
)
from loraspec.core.schemas import AuditReport
from loraspec.core.spectral_math import (
    Matrix,
    compute_layer_spectral_analysis,
    compute_pairwise_collision,
)
from loraspec.inspect.safetensors_reader import (
    load_peft_adapter,
    write_safetensors,
)
from loraspec.knots import (
    align_subspaces_knots,
    knots_dare_ties,
    knots_linear,
    knots_ties,
    merge_knots_adapters,
)
from loraspec.reporting.sarif_writer import SARIF_SCHEMA_URI

runner = CliRunner()

SCALE_8_LAYERS: tuple[str, ...] = (
    "base_model.model.layers.0.self_attn.q_proj",
    "base_model.model.layers.0.mlp.gate_proj",
    "base_model.model.layers.0.mlp.up_proj",
    "base_model.model.layers.0.mlp.down_proj",
    "base_model.model.layers.1.self_attn.q_proj",
    "base_model.model.layers.1.mlp.gate_proj",
    "base_model.model.layers.1.mlp.up_proj",
    "base_model.model.layers.1.mlp.down_proj",
)


def _write_adapter(
    root: Path,
    *,
    rank: int,
    alpha: float,
    use_rslora: bool = False,
    use_dora: bool = False,
    rank_pattern: dict[str, int] | None = None,
    alpha_pattern: dict[str, float] | None = None,
    tensors: dict[str, tuple[Matrix, str]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": use_rslora,
        "use_dora": use_dora,
        "bias": "none",
        "target_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"],
    }
    if rank_pattern is not None:
        cfg["rank_pattern"] = rank_pattern
    if alpha_pattern is not None:
        cfg["alpha_pattern"] = alpha_pattern
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_safetensors(root / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    return root


def _build_orthogonal_scale_tensors(
    *,
    d: int,
    rank: int,
    col_offset: int,
    layers: tuple[str, ...],
) -> dict[str, tuple[Matrix, str]]:
    tensors: dict[str, tuple[Matrix, str]] = {}
    strong_modes = max(4, rank // 4)
    for lidx, prefix in enumerate(layers):
        shift = (col_offset + lidx * 4) % d
        a_mat: Matrix = [[0.0] * d for _ in range(rank)]
        b_mat: Matrix = [[0.0] * rank for _ in range(d)]
        for i in range(rank):
            idx = (shift + i) % d
            a_mat[i][idx] = 1.0
            # Keep adapter rank_utilization >= 0.28 while top-32 merged rank retains >= 94% energy
            b_mat[idx][i] = 2.0 if i < strong_modes else 0.4
        tensors[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
        tensors[f"{prefix}.lora_B.weight"] = (b_mat, "F32")
    return tensors


def test_scale_4_adapters_8_layers_d2048_under_10s_and_bounded_ram(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """KPI-ENG-01 / NFR-02: 4 adapters x 8 layers x d=2048 (r=16..32) audit + KnOTS < 10s."""
    pytest.importorskip("numpy")
    d = 2048
    specs = [
        ("sql_r16", 16, 16.0, False, 0),
        ("rust_r24", 24, 24.0, True, 64),
        ("json_r32", 32, 32.0, False, 128),
        ("math_r16", 16, 16.0, True, 192),
    ]
    adapter_dirs: list[Path] = []
    for name, rank, alpha, use_rs, offset in specs:
        tensors = _build_orthogonal_scale_tensors(
            d=d, rank=rank, col_offset=offset, layers=SCALE_8_LAYERS
        )
        adapter_dirs.append(
            _write_adapter(
                tmp_path / name,
                rank=rank,
                alpha=alpha,
                use_rslora=use_rs,
                tensors=tensors,
            )
        )

    merged_dir = tmp_path / "merged_scale"
    t0 = time.perf_counter()
    loaded = [load_peft_adapter(p) for p in adapter_dirs]
    metrics, collisions, findings = run_spectral_and_collision_audit(loaded, AuditConfig())
    summary, merge_findings = merge_knots_adapters(
        loaded,
        out_dir=merged_dir,
        method="knots_ties",
        target_rank=32,
        ties_density=0.5,
    )
    elapsed = time.perf_counter() - t0
    assert len(metrics) == 32
    assert len(collisions) == 48  # 6 pairs * 8 layers
    assert {f.rule_id for f in findings} == {"RULE-CONF-004"}  # mixed ranks & rsLoRA
    assert merge_findings == []
    assert summary.energy_retained >= 0.90
    assert elapsed < 10.0, f"4-adapter d={d} audit+merge exceeded 10s SLA: {elapsed:.3f}s"

    del loaded, metrics, collisions, findings

    # Separately measure traced Python heap peak without biasing wall-clock SLA
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        loaded_tm = [load_peft_adapter(p) for p in adapter_dirs]
        run_spectral_and_collision_audit(loaded_tm, AuditConfig())
        merge_knots_adapters(
            loaded_tm,
            out_dir=tmp_path / "merged_scale_tm",
            method="knots_ties",
            target_rank=32,
            ties_density=0.5,
        )
        _cur, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    with capsys.disabled():
        print(
            f"\n[M1 Scale KPI-ENG-01/NFR-02] 4 adapters x 8 layers x d={d} (r=16..32): "
            f"elapsed={elapsed:.3f}s, peak_ram={peak_bytes / (1024 * 1024):.2f} MiB"
        )

    assert peak_bytes < 250 * 1024 * 1024, (
        f"Peak traced RAM {peak_bytes / (1024 * 1024):.2f} MiB exceeded 250 MiB NFR-01 bound"
    )

    re_audit_out = tmp_path / "re_audit"
    res = runner.invoke(
        app,
        [
            "audit",
            str(merged_dir),
            "--out-dir",
            str(re_audit_out),
            "--html",
            str(re_audit_out / "card.html"),
        ],
    )
    assert res.exit_code == EXIT_PASS, res.stdout + res.stderr
    rep = AuditReport.model_validate_json((re_audit_out / "report.json").read_text("utf-8"))
    assert rep.knots_summary == summary
    assert "KnOTS Joint-SVD Merge Diagnostics" in (re_audit_out / "card.html").read_text("utf-8")


def test_cross_backend_full_parity_and_cli_config_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify full numpy vs pure_python parity and cfg.project.default_backend CLI wiring."""
    pytest.importorskip("numpy")

    d_out, d_in = 14, 12
    layers_a: dict[str, tuple[Matrix, str]] = {}
    layers_b: dict[str, tuple[Matrix, str]] = {}
    for lidx, prefix in enumerate(SCALE_8_LAYERS[:4]):
        a1 = [
            [math.cos(0.17 * (i + 1) * (j + 1 + lidx)) * 0.4 for j in range(d_in)] for i in range(4)
        ]
        b1 = [
            [math.sin(0.21 * (i + 1 + lidx) * (j + 1)) * 0.5 for j in range(4)]
            for i in range(d_out)
        ]
        a2 = [
            [math.sin(0.19 * (i + 2) * (j + 1 + lidx)) * 0.35 for j in range(d_in)]
            for i in range(6)
        ]
        b2 = [
            [math.cos(0.23 * (i + 1) * (j + 2 + lidx)) * 0.45 for j in range(6)]
            for i in range(d_out)
        ]
        layers_a[f"{prefix}.lora_A.weight"] = (a1, "F32")
        layers_a[f"{prefix}.lora_B.weight"] = (b1, "F32")
        layers_b[f"{prefix}.lora_A.weight"] = (a2, "F32")
        layers_b[f"{prefix}.lora_B.weight"] = (b2, "F32")

    pa = _write_adapter(tmp_path / "pa", rank=4, alpha=8.0, use_rslora=False, tensors=layers_a)
    pb = _write_adapter(tmp_path / "pb", rank=6, alpha=12.0, use_rslora=True, tensors=layers_b)

    la_np, lb_np = load_peft_adapter(pa, backend="numpy"), load_peft_adapter(pb, backend="numpy")
    la_py = load_peft_adapter(pa, backend="pure_python")
    lb_py = load_peft_adapter(pb, backend="pure_python")

    m_np, c_np, f_np = run_spectral_and_collision_audit(
        [la_np, lb_np], AuditConfig(), backend="numpy"
    )
    m_py, c_py, f_py = run_spectral_and_collision_audit(
        [la_py, lb_py], AuditConfig(), backend="pure_python"
    )
    assert [f.rule_id for f in f_np] == [f.rule_id for f in f_py]
    for x, y in zip(m_np, m_py, strict=True):
        assert x.spectral_norm == pytest.approx(y.spectral_norm, abs=1e-7)
        assert x.frobenius_norm == pytest.approx(y.frobenius_norm, abs=1e-7)
        assert x.effective_stable_rank == pytest.approx(y.effective_stable_rank, abs=1e-5)
    for ca, cb in zip(c_np, c_py, strict=True):
        assert ca.grassmann_chordal_similarity == pytest.approx(
            cb.grassmann_chordal_similarity, abs=1e-6
        )
        assert ca.sign_conflict_rate == pytest.approx(cb.sign_conflict_rate, abs=1e-6)

    # Standalone spectral_math + knots_align (including wide d_out < R_sum and d_in < r_m)
    ls_small = la_np.spec.layers[0].model_copy(
        update={
            "in_features": 2,
            "out_features": 3,
            "rank": 4,
            "a_shape": (4, 2),
            "b_shape": (3, 4),
        }
    )
    a_small = [[0.7, -0.3], [0.2, 0.9], [-0.4, 0.5], [0.1, -0.6]]
    b_small = [[0.3, -0.2, 0.4, 0.1], [-0.5, 0.8, -0.1, 0.3], [0.2, 0.4, -0.7, 0.6]]
    ms_np, fs_np = compute_layer_spectral_analysis(
        adapter_id="s",
        layer_spec=ls_small,
        a_mat=a_small,
        b_mat=b_small,
        scaling=1.0,
        backend="numpy",
    )
    ms_py, fs_py = compute_layer_spectral_analysis(
        adapter_id="s",
        layer_spec=ls_small,
        a_mat=a_small,
        b_mat=b_small,
        scaling=1.0,
        backend="pure_python",
    )
    assert ms_np.spectral_norm == pytest.approx(ms_py.spectral_norm, abs=1e-8)
    assert compute_pairwise_collision(
        fs_np, fs_np, backend="numpy"
    ).grassmann_chordal_similarity == (
        pytest.approx(
            compute_pairwise_collision(
                fs_py, fs_py, backend="pure_python"
            ).grassmann_chordal_similarity,
            abs=1e-8,
        )
    )

    al_np = align_subspaces_knots(
        layer_name="small",
        adapter_ids=["s1", "s2"],
        scaled_bs=[b_small, [[-v for v in row] for row in b_small]],
        a_mats=[a_small, a_small],
        backend="numpy",
    )
    al_py = align_subspaces_knots(
        layer_name="small",
        adapter_ids=["s1", "s2"],
        scaled_bs=[b_small, [[-v for v in row] for row in b_small]],
        a_mats=[a_small, a_small],
        backend="pure_python",
    )
    for sn, sp in zip(al_np.sigmas, al_py.sigmas, strict=True):
        assert sn == pytest.approx(sp, abs=1e-8)
    assert al_np.pre_merge_sign_conflict == pytest.approx(al_py.pre_merge_sign_conflict, abs=1e-6)

    # Operators parity
    v1 = [[0.2 * (i + 1) - 0.05 * j for j in range(10)] for i in range(4)]
    v2 = [[-0.15 * (i + 1) + 0.08 * j for j in range(10)] for i in range(4)]
    sigs = [8.0, 3.0, 1.0, 0.0]
    for r_np, r_py in zip(
        knots_linear([v1, v2], backend="numpy"),
        knots_linear([v1, v2], backend="pure_python"),
        strict=True,
    ):
        assert r_np == pytest.approx(r_py, abs=1e-12)
    for r_np, r_py in zip(
        knots_ties([v1, v2], sigmas=sigs, density=0.5, backend="numpy"),
        knots_ties([v1, v2], sigmas=sigs, density=0.5, backend="pure_python"),
        strict=True,
    ):
        assert r_np == pytest.approx(r_py, abs=1e-12)
    for r_np, r_py in zip(
        knots_dare_ties(
            [v1, v2], sigmas=sigs, density=0.5, drop_rate=0.25, rng_seed=42, backend="numpy"
        ),
        knots_dare_ties(
            [v1, v2], sigmas=sigs, density=0.5, drop_rate=0.25, rng_seed=42, backend="pure_python"
        ),
        strict=True,
    ):
        assert r_np == pytest.approx(r_py, abs=1e-12)

    # End-to-end merge parity across all 3 methods
    for method in ("knots_linear", "knots_ties", "knots_dare_ties"):
        sum_np, _ = merge_knots_adapters(
            [la_np, lb_np],
            out_dir=tmp_path / f"m_np_{method}",
            method=method,
            target_rank=6,
            rng_seed=42,
            backend="numpy",
        )
        sum_py, _ = merge_knots_adapters(
            [la_py, lb_py],
            out_dir=tmp_path / f"m_py_{method}",
            method=method,
            target_rank=6,
            rng_seed=42,
            backend="pure_python",
        )
        assert sum_np.energy_retained == pytest.approx(sum_py.energy_retained, abs=1e-5)
        assert sum_np.mean_reconstruction_rmse == pytest.approx(
            sum_py.mean_reconstruction_rmse, abs=1e-5
        )
        assert sum_np.sign_conflict_reduction_pct == pytest.approx(
            sum_py.sign_conflict_reduction_pct, abs=1e-3
        )

    # CLI wiring: cfg.project.default_backend = "pure_python" bypasses numpy completely
    import loraspec.core.spectral_math as sm_mod
    import loraspec.inspect.safetensors_reader as sr_mod
    import loraspec.knots.aligner as al_mod
    import loraspec.knots.operators as op_mod
    import loraspec.knots.serializer as ser_mod

    def _forbid_numpy() -> None:
        raise AssertionError("numpy path called despite default_backend='pure_python'")

    monkeypatch.setattr(sm_mod, "_optional_numpy", _forbid_numpy)
    monkeypatch.setattr(sr_mod, "_optional_numpy", _forbid_numpy)
    monkeypatch.setattr(al_mod, "optional_numpy", _forbid_numpy)
    monkeypatch.setattr(op_mod, "optional_numpy", _forbid_numpy)
    monkeypatch.setattr(ser_mod, "optional_numpy", _forbid_numpy)

    pure_toml = tmp_path / "pure.toml"
    pure_toml.write_text(
        '[project]\nfail_on = "critical"\ndefault_backend = "pure_python"\n',
        encoding="utf-8",
    )
    cli_aud = runner.invoke(
        app,
        [
            "audit",
            str(pa),
            str(pb),
            "--config",
            str(pure_toml),
            "--out-dir",
            str(tmp_path / "aud_py"),
        ],
    )
    assert cli_aud.exit_code == EXIT_PASS, cli_aud.stdout + cli_aud.stderr

    cli_mrg = runner.invoke(
        app,
        [
            "merge",
            str(pa),
            str(pb),
            "--config",
            str(pure_toml),
            "--method",
            "knots_dare_ties",
            "--target-rank",
            "6",
            "--out-dir",
            str(tmp_path / "cli_m_py"),
        ],
    )
    assert cli_mrg.exit_code == EXIT_PASS, cli_mrg.stdout + cli_mrg.stderr


def test_heterogeneous_rank_pattern_roundtrip_and_zero_adapters(tmp_path: Path) -> None:
    """Test heterogeneous rank_pattern/alpha_pattern roundtrip and zero/near-zero adapters."""
    d = 16
    # Adapter A has rank_pattern + alpha_pattern + use_rslora=True
    t_a: dict[str, tuple[Matrix, str]] = {}
    for prefix in SCALE_8_LAYERS[:4]:
        rk = 2 if prefix.endswith("q_proj") else 4
        a_m = [[1.0 if j == i else 0.0 for j in range(d)] for i in range(rk)]
        b_m = [[1.5 if i == j else 0.0 for j in range(rk)] for i in range(d)]
        t_a[f"{prefix}.lora_A.weight"] = (a_m, "F32")
        t_a[f"{prefix}.lora_B.weight"] = (b_m, "F32")

    pa = _write_adapter(
        tmp_path / "pat_a",
        rank=4,
        alpha=8.0,
        use_rslora=True,
        rank_pattern={"layers.0.self_attn.q_proj": 2},
        alpha_pattern={"layers.0.self_attn.q_proj": 6.0},
        tensors=t_a,
    )

    # Adapter B lacks layer 0 q_proj and has rank 6 on remaining MLP layers
    t_b: dict[str, tuple[Matrix, str]] = {}
    for prefix in SCALE_8_LAYERS[1:4]:
        a_m = [[1.0 if j == (i + 4) else 0.0 for j in range(d)] for i in range(6)]
        b_m = [[1.2 if i == (j + 4) else 0.0 for j in range(6)] for i in range(d)]
        t_b[f"{prefix}.lora_A.weight"] = (a_m, "F32")
        t_b[f"{prefix}.lora_B.weight"] = (b_m, "F32")

    pb = _write_adapter(tmp_path / "uni_b", rank=6, alpha=6.0, tensors=t_b)

    merged_dir = tmp_path / "merged_pat"
    m_res = runner.invoke(
        app,
        [
            "merge",
            str(pa),
            str(pb),
            "--target-rank",
            "8",
            "--layer-policy",
            "union",
            "--out-dir",
            str(merged_dir),
        ],
    )
    assert m_res.exit_code == EXIT_PASS, m_res.stdout + m_res.stderr

    merged_cfg = json.loads((merged_dir / "adapter_config.json").read_text("utf-8"))
    assert merged_cfg["rank_pattern"] == {"layers.0.self_attn.q_proj": 2}
    assert merged_cfg["alpha_pattern"] == {"layers.0.self_attn.q_proj": 2.0}

    # Re-audit merged adapter -> Grade A, 0 findings, knots_summary populated & rendered
    aud_dir = tmp_path / "aud_pat"
    a_res = runner.invoke(
        app,
        [
            "audit",
            str(merged_dir),
            "--out-dir",
            str(aud_dir),
            "--html",
            str(aud_dir / "card.html"),
            "--sarif",
            str(aud_dir / "card.sarif"),
        ],
    )
    assert a_res.exit_code == EXIT_PASS, a_res.stdout + a_res.stderr
    rep = AuditReport.model_validate_json((aud_dir / "report.json").read_text("utf-8"))
    assert rep.grade == "A"
    assert rep.findings == []
    assert rep.knots_summary is not None

    # Check q_proj preserved exact alpha/sqrt(r) scaling (6 / sqrt(2) * 1.5)
    q_metric = next(m for m in rep.layer_metrics if m.module_type == "q_proj")
    assert q_metric.spectral_norm == pytest.approx(1.5 * (6.0 / math.sqrt(2.0)), abs=1e-4)

    # Near-zero (4e-9) adapter passes RULE-SPEC-001; exact zero fails pre-merge blocker
    tiny_t = {
        f"{p}.lora_A.weight": (
            [[1.0 if j == i else 0.0 for j in range(d)] for i in range(2)],
            "F32",
        )
        for p in SCALE_8_LAYERS[:4]
    }
    for p in SCALE_8_LAYERS[:4]:
        tiny_t[f"{p}.lora_B.weight"] = (
            [
                [4e-9 if (i == 0 and j == 0) else (1.0 if i == j else 0.0) for j in range(2)]
                for i in range(d)
            ],
            "F32",
        )
    tiny_t[f"{SCALE_8_LAYERS[0]}.lora_B.weight"] = (
        [[4e-9 if (i == 0 and j == 0) else 0.0 for j in range(2)] for i in range(d)],
        "F32",
    )
    p_tiny = _write_adapter(tmp_path / "tiny", rank=2, alpha=2.0, tensors=tiny_t)
    m_tiny, _, f_tiny = run_spectral_and_collision_audit([load_peft_adapter(p_tiny)], AuditConfig())
    assert all(m.frobenius_norm > 1e-12 for m in m_tiny)
    assert not any(f.rule_id == "RULE-SPEC-001" for f in f_tiny)


def test_dora_blocking_audit_and_merge_cli(tmp_path: Path) -> None:
    """DoRA (use_dora=True and lora_magnitude_vector) triggers RULE-CONF-002 and blocks merge."""
    d = 8
    t_std: dict[str, tuple[Matrix, str]] = {}
    for prefix in SCALE_8_LAYERS[:4]:
        t_std[f"{prefix}.lora_A.weight"] = (
            [[1.0 if j == i else 0.0 for j in range(d)] for i in range(2)],
            "F32",
        )
        t_std[f"{prefix}.lora_B.weight"] = (
            [[1.0 if i == j else 0.0 for j in range(2)] for i in range(d)],
            "F32",
        )
    p_std = _write_adapter(tmp_path / "std", rank=2, alpha=2.0, tensors=t_std)
    p_dora_cfg = _write_adapter(
        tmp_path / "dora_cfg", rank=2, alpha=2.0, use_dora=True, tensors=t_std
    )

    # Also construct a .safetensors with 1D lora_magnitude_vector and use_dora=False in config
    p_dora_mag = tmp_path / "dora_mag"
    p_dora_mag.mkdir()
    (p_dora_mag / "adapter_config.json").write_text(
        (p_std / "adapter_config.json").read_text("utf-8"), encoding="utf-8"
    )
    hdr = json.dumps(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            },
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [16, 32],
            },
            "base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [32, 40],
            },
        }
    ).encode("utf-8")
    (p_dora_mag / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(hdr))
        + hdr
        + struct.pack("<10f", 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    )

    for dora_path in (p_dora_cfg, p_dora_mag):
        aud = runner.invoke(app, ["audit", str(dora_path)])
        assert aud.exit_code == EXIT_AUDIT_FAIL
        assert "RULE-CONF-002" in aud.stdout

        out_dir = tmp_path / f"blocked_{dora_path.name}"
        mrg = runner.invoke(app, ["merge", str(p_std), str(dora_path), "--out-dir", str(out_dir)])
        assert mrg.exit_code == EXIT_CONFIG_ERROR
        assert "DoRA" in mrg.stderr
        assert not out_dir.exists()


def test_cli_exit_codes_0_to_5_and_artifacts_offline_sarif(tmp_path: Path) -> None:
    """Verify exit codes 0..5, offline zero-CDN HTML, and OASIS SARIF 2.1.0 schema compliance."""
    d = 12
    t1 = {
        f"{p}.lora_A.weight": (
            [[1.0 if j == i else 0.0 for j in range(d)] for i in range(2)],
            "F32",
        )
        for p in SCALE_8_LAYERS[:4]
    }
    t2 = {
        f"{p}.lora_A.weight": (
            [[1.0 if j == i else 0.0 for j in range(d)] for i in range(2)],
            "F32",
        )
        for p in SCALE_8_LAYERS[:4]
    }
    for p in SCALE_8_LAYERS[:4]:
        t1[f"{p}.lora_B.weight"] = (
            [[1.0 if i == j else 0.0 for j in range(2)] for i in range(d)],
            "F32",
        )
        t2[f"{p}.lora_B.weight"] = (
            [[-1.0 if i == j else 0.0 for j in range(2)] for i in range(d)],
            "F32",
        )
    p1 = _write_adapter(tmp_path / "c1", rank=2, alpha=2.0, tensors=t1)
    p2 = _write_adapter(tmp_path / "c2", rank=2, alpha=2.0, tensors=t2)

    # Exit 0 (doctor)
    assert runner.invoke(app, ["doctor"]).exit_code == EXIT_PASS
    # Exit 1 (colliding pair audit)
    out_fail = tmp_path / "out_fail"
    res1 = runner.invoke(
        app,
        [
            "audit",
            str(p1),
            str(p2),
            "--out-dir",
            str(out_fail),
            "--sarif",
            str(out_fail / "report.sarif"),
            "--html",
            str(out_fail / "report.html"),
        ],
    )
    assert res1.exit_code == EXIT_AUDIT_FAIL
    # Exit 2 (invalid CLI / Stage 6 stub)
    assert runner.invoke(app, ["audit", str(p1), "--tiers", "7"]).exit_code == EXIT_CONFIG_ERROR
    assert runner.invoke(app, ["bench", str(p1)]).exit_code == EXIT_CONFIG_ERROR

    # Exit 3 (BudgetExceededError via GpuBudgetGuard)
    guard = GpuBudgetGuard(max_cost_inr=2500.0, smoke=True, smoke_cap_inr=10.0)
    with pytest.raises(BudgetExceededError) as exc_b:
        asyncio.run(guard.reserve_or_raise("e2e_l4_24gb", estimated_seconds=3600.0))
    assert exc_b.value.exit_code == EXIT_BUDGET_EXCEEDED == 3

    # Exit 4 & 5 domain exit codes
    assert RuntimeExecutionError("cuda missing").exit_code == EXIT_RUNTIME_ERROR == 4
    assert KernelParityError("atol exceeded").exit_code == EXIT_PARITY_ERROR == 5

    # Validate offline HTML & SARIF 2.1.0 fields
    html_text = (out_fail / "report.html").read_text("utf-8")
    assert len(html_text.encode("utf-8")) < 250 * 1024
    assert not re.search(r"https?://", html_text, flags=re.IGNORECASE)
    assert not re.search(r"<script[^>]+src=", html_text, flags=re.IGNORECASE)
    assert not re.search(r"<link[^>]+href=", html_text, flags=re.IGNORECASE)
    assert "@import" not in html_text.lower()
    assert "RULE-SPEC-004" in html_text

    sarif = json.loads((out_fail / "report.sarif").read_text("utf-8"))
    assert sarif["$schema"] == SARIF_SCHEMA_URI
    assert sarif["version"] == "2.1.0"
    run0 = sarif["runs"][0]
    assert run0["tool"]["driver"]["name"] == "loraspec"
    assert run0["properties"]["grade"] == "F"
    assert any(r["ruleId"] == "RULE-SPEC-004" for r in run0["results"])


def test_planted_corpus_both_backends_100pct_recall_0pct_far(tmp_path: Path) -> None:
    """KPI-RES-01: 7 faults + 3 controls achieve 100% recall & 0.0% FAR on both backends."""
    corpus_dir = generate_planted_corpus(tmp_path / "corpus", seed=42)
    for backend_name in ("auto", "pure_python"):
        cfg = LoraspecConfig(project=ProjectConfig(default_backend=backend_name))
        ev = evaluate_planted_recall(corpus_dir, cfg=cfg)
        assert ev.total_faults == 7
        assert ev.total_controls == 3
        assert ev.recall == 1.0
        assert ev.false_alarm_rate == 0.0
        assert all(cr.passed_expectation for cr in ev.case_results)


def test_action_yml_paths_and_adapter_paths_alias_shell_logic() -> None:
    """Test exact shell/Python path parsing extracted from root action.yml."""
    repo_root = Path(__file__).resolve().parents[2]
    action_data = yaml.safe_load((repo_root / "action.yml").read_text(encoding="utf-8"))
    assert "paths" in action_data["inputs"]
    assert "adapter-paths" in action_data["inputs"]

    run_script: str = action_data["runs"]["steps"][1]["run"]
    match = re.search(r"python3 -c '(\n.*?)'\)", run_script, flags=re.DOTALL)
    assert match is not None, "Could not locate python3 -c path parser in action.yml"
    py_snippet = match.group(1)

    def _run_snippet(paths_val: str, alias_val: str) -> list[str]:
        bash_cmd = (
            "adapter_paths=()\n"
            "while IFS= read -r -d '' entry; do\n"
            '  adapter_paths+=("$entry")\n'
            "done < <(python3 -c '" + py_snippet + "')\n"
            "printf '%s\\n' \"${adapter_paths[@]}\""
        )
        proc = subprocess.run(
            ["bash", "-c", bash_cmd],
            env={"LORASPEC_PATHS": paths_val, "LORASPEC_ADAPTER_PATHS": alias_val},
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in proc.stdout.splitlines() if line]

    assert _run_snippet(".", "") == ["."]
    assert _run_snippet("./a ./b", "") == ["./a", "./b"]
    assert _run_snippet(".", "./alias_a\n./alias_b") == ["./alias_a", "./alias_b"]
    assert _run_snippet("", "./alias_only") == ["./alias_only"]
    assert _run_snippet("./explicit_a", "./ignored_alias") == ["./explicit_a"]
    assert _run_snippet('"." "./dir with spaces/a"', "") == [".", "./dir with spaces/a"]
