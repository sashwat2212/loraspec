"""Deterministic synthetic planted subspace fault & clean control corpus (KPI-RES-01)."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from loraspec.auditor.grading import assign_grade, compute_merge_risk_score
from loraspec.auditor.probes import run_spectral_and_collision_audit
from loraspec.core.config import AuditConfig, LoraspecConfig, resolve_math_backend
from loraspec.core.schemas import Severity, meets_fail_threshold
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import load_peft_adapter, write_safetensors
from loraspec.knots import merge_knots_adapters

ALL_LINEAR_PREFIXES: tuple[str, ...] = (
    "base_model.model.layers.0.self_attn.q_proj",
    "base_model.model.layers.0.mlp.gate_proj",
    "base_model.model.layers.0.mlp.up_proj",
    "base_model.model.layers.0.mlp.down_proj",
)

ALL_LINEAR_8_PREFIXES: tuple[str, ...] = (
    *ALL_LINEAR_PREFIXES,
    "base_model.model.layers.1.self_attn.q_proj",
    "base_model.model.layers.1.mlp.gate_proj",
    "base_model.model.layers.1.mlp.up_proj",
    "base_model.model.layers.1.mlp.down_proj",
)


@dataclass(frozen=True)
class PlantedCaseSpec:
    case_id: str
    category: Literal["fault", "control"]
    adapter_rel_paths: list[str]
    expected_rules: list[str]
    description: str


@dataclass(frozen=True)
class PlantedCaseResult:
    case_id: str
    category: Literal["fault", "control"]
    detected_rules: list[str]
    expected_rules: list[str]
    grade: str
    merge_risk_score: float
    failed_gate: bool
    passed_expectation: bool


@dataclass(frozen=True)
class CorpusEvaluationResult:
    total_faults: int
    total_controls: int
    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int
    recall: float
    false_alarm_rate: float
    case_results: list[PlantedCaseResult]


def _eye_block(rows: int, cols: int, offset: int = 0, scale: float = 1.0) -> Matrix:
    mat: Matrix = [[0.0] * cols for _ in range(rows)]
    for c in range(cols):
        r = offset + c
        if 0 <= r < rows:
            mat[r][c] = scale
    return mat


def _row_eye_block(rows: int, cols: int, offset: int = 0, scale: float = 1.0) -> Matrix:
    mat: Matrix = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        c = offset + r
        if 0 <= c < cols:
            mat[r][c] = scale
    return mat


def _write_peft_dir(
    root: Path,
    *,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    rank: int,
    alpha: float,
    use_rslora: bool = False,
    target_modules: list[str],
    tensors: dict[str, tuple[Matrix, str]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": use_rslora,
        "use_dora": False,
        "bias": "none",
        "target_modules": target_modules,
    }
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    write_safetensors(root / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    return root


def generate_planted_corpus(out_dir: Path, *, seed: int = 42) -> Path:
    """Generate 7 planted subspace fault cases and 3 clean controls into `out_dir`."""
    _ = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    faults_dir = out_dir / "faults"
    controls_dir = out_dir / "controls"
    for sub in (faults_dir, controls_dir):
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True, exist_ok=True)

    d = 24
    cases: list[PlantedCaseSpec] = []
    all_linear_mods = ["q_proj", "gate_proj", "up_proj", "down_proj"]

    # Fault 1: 01_nan_weights (RULE-SPEC-001)
    t_nan: dict[str, tuple[Matrix, str]] = {}
    for idx, prefix in enumerate(ALL_LINEAR_PREFIXES):
        a_m = _row_eye_block(4, d, offset=0, scale=1.0)
        b_m = _eye_block(d, 4, offset=0, scale=1.0)
        if idx == 0:
            b_m[0][0] = math.nan
        t_nan[f"{prefix}.lora_A.weight"] = (a_m, "F32")
        t_nan[f"{prefix}.lora_B.weight"] = (b_m, "F32")
    _write_peft_dir(
        faults_dir / "01_nan_weights" / "adapter",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_nan,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="01_nan_weights",
            category="fault",
            adapter_rel_paths=["faults/01_nan_weights/adapter"],
            expected_rules=["RULE-SPEC-001"],
            description="NaN-poisoned lora_B weights on layer 0 q_proj.",
        )
    )

    # Fault 2: 02_dead_zero_layer (RULE-SPEC-001)
    t_dead: dict[str, tuple[Matrix, str]] = {}
    for idx, prefix in enumerate(ALL_LINEAR_PREFIXES):
        a_m = _row_eye_block(4, d, offset=0, scale=1.0)
        b_m = [[0.0] * 4 for _ in range(d)] if idx == 2 else _eye_block(d, 4, offset=0, scale=1.0)
        t_dead[f"{prefix}.lora_A.weight"] = (a_m, "F32")
        t_dead[f"{prefix}.lora_B.weight"] = (b_m, "F32")
    _write_peft_dir(
        faults_dir / "02_dead_zero_layer" / "adapter",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_dead,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="02_dead_zero_layer",
            category="fault",
            adapter_rel_paths=["faults/02_dead_zero_layer/adapter"],
            expected_rules=["RULE-SPEC-001"],
            description="Dead zero up_proj layer (Frobenius norm == 0).",
        )
    )

    # Fault 3: 03_rank_collapse_r16 (RULE-SPEC-002)
    t_col: dict[str, tuple[Matrix, str]] = {}
    for prefix in ALL_LINEAR_PREFIXES:
        a_m = [[1.0 if (i == 0 and j == 0) else 0.0 for j in range(d)] for i in range(16)]
        b_m = [[3.0 if (i == 0 and j == 0) else 0.0 for j in range(16)] for i in range(d)]
        t_col[f"{prefix}.lora_A.weight"] = (a_m, "F32")
        t_col[f"{prefix}.lora_B.weight"] = (b_m, "F32")
    _write_peft_dir(
        faults_dir / "03_rank_collapse_r16" / "adapter",
        rank=16,
        alpha=16.0,
        target_modules=all_linear_mods,
        tensors=t_col,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="03_rank_collapse_r16",
            category="fault",
            adapter_rel_paths=["faults/03_rank_collapse_r16/adapter"],
            expected_rules=["RULE-SPEC-002"],
            description="Rank-16 adapter collapsed to effective rank r_eff=1.0.",
        )
    )

    # Fault 4: 04_spectral_norm_spike (RULE-SPEC-003)
    t_spike: dict[str, tuple[Matrix, str]] = {}
    for idx, prefix in enumerate(ALL_LINEAR_8_PREFIXES):
        scale = 30.0 if idx == len(ALL_LINEAR_8_PREFIXES) - 1 else 1.0
        t_spike[f"{prefix}.lora_A.weight"] = (_row_eye_block(4, d, offset=0, scale=1.0), "F32")
        t_spike[f"{prefix}.lora_B.weight"] = (_eye_block(d, 4, offset=0, scale=scale), "F32")
    _write_peft_dir(
        faults_dir / "04_spectral_norm_spike" / "adapter",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_spike,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="04_spectral_norm_spike",
            category="fault",
            adapter_rel_paths=["faults/04_spectral_norm_spike/adapter"],
            expected_rules=["RULE-SPEC-003"],
            description="30x spectral norm spike on layer 1 down_proj.",
        )
    )

    # Fault 5: 05_base_model_mismatch (RULE-CONF-001)
    t_base_a: dict[str, tuple[Matrix, str]] = {}
    t_base_b: dict[str, tuple[Matrix, str]] = {}
    for prefix in ALL_LINEAR_PREFIXES:
        t_base_a[f"{prefix}.lora_A.weight"] = (_row_eye_block(4, d, offset=0, scale=1.0), "F32")
        t_base_a[f"{prefix}.lora_B.weight"] = (_eye_block(d, 4, offset=0, scale=1.0), "F32")
        t_base_b[f"{prefix}.lora_A.weight"] = (_row_eye_block(4, d, offset=4, scale=1.0), "F32")
        t_base_b[f"{prefix}.lora_B.weight"] = (_eye_block(d, 4, offset=4, scale=1.0), "F32")
    _write_peft_dir(
        faults_dir / "05_base_model_mismatch" / "adapter_a",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_base_a,
    )
    _write_peft_dir(
        faults_dir / "05_base_model_mismatch" / "adapter_b",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_base_b,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="05_base_model_mismatch",
            category="fault",
            adapter_rel_paths=[
                "faults/05_base_model_mismatch/adapter_a",
                "faults/05_base_model_mismatch/adapter_b",
            ],
            expected_rules=["RULE-CONF-001"],
            description="Conflicting base_model_name_or_path across fleet.",
        )
    )

    # Fault 6: 06_attention_only_no_mlp (RULE-CONF-003)
    attn_prefixes = [
        "base_model.model.layers.0.self_attn.q_proj",
        "base_model.model.layers.0.self_attn.v_proj",
    ]
    t_attn: dict[str, tuple[Matrix, str]] = {}
    for prefix in attn_prefixes:
        t_attn[f"{prefix}.lora_A.weight"] = (_row_eye_block(4, d, offset=0, scale=1.0), "F32")
        t_attn[f"{prefix}.lora_B.weight"] = (_eye_block(d, 4, offset=0, scale=1.0), "F32")
    _write_peft_dir(
        faults_dir / "06_attention_only_no_mlp" / "adapter",
        rank=4,
        alpha=4.0,
        target_modules=["q_proj", "v_proj"],
        tensors=t_attn,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="06_attention_only_no_mlp",
            category="fault",
            adapter_rel_paths=["faults/06_attention_only_no_mlp/adapter"],
            expected_rules=["RULE-CONF-003"],
            description="Attention-only adapter omitting MLP projections.",
        )
    )

    # Fault 7: 07_colliding_sign_conflict_pair (RULE-SPEC-004)
    t_coll_a: dict[str, tuple[Matrix, str]] = {}
    t_coll_b: dict[str, tuple[Matrix, str]] = {}
    for prefix in ALL_LINEAR_PREFIXES:
        a_m = _row_eye_block(4, d, offset=0, scale=1.0)
        t_coll_a[f"{prefix}.lora_A.weight"] = (a_m, "F32")
        t_coll_a[f"{prefix}.lora_B.weight"] = (_eye_block(d, 4, offset=0, scale=1.5), "F32")
        t_coll_b[f"{prefix}.lora_A.weight"] = (a_m, "F32")
        t_coll_b[f"{prefix}.lora_B.weight"] = (_eye_block(d, 4, offset=0, scale=-1.5), "F32")
    _write_peft_dir(
        faults_dir / "07_colliding_sign_conflict_pair" / "adapter_a",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_coll_a,
    )
    _write_peft_dir(
        faults_dir / "07_colliding_sign_conflict_pair" / "adapter_b",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_coll_b,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="07_colliding_sign_conflict_pair",
            category="fault",
            adapter_rel_paths=[
                "faults/07_colliding_sign_conflict_pair/adapter_a",
                "faults/07_colliding_sign_conflict_pair/adapter_b",
            ],
            expected_rules=["RULE-SPEC-004"],
            description="Identical subspace with opposite directional signs (chordal=1, conflict=1).",
        )
    )

    # Control 1: 01_orthogonal_all_linear_pair
    _write_peft_dir(
        controls_dir / "01_orthogonal_all_linear_pair" / "adapter_a",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_base_a,
    )
    _write_peft_dir(
        controls_dir / "01_orthogonal_all_linear_pair" / "adapter_b",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_base_b,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="01_orthogonal_all_linear_pair",
            category="control",
            adapter_rel_paths=[
                "controls/01_orthogonal_all_linear_pair/adapter_a",
                "controls/01_orthogonal_all_linear_pair/adapter_b",
            ],
            expected_rules=[],
            description="Disjoint orthogonal all-linear pair (Grade A).",
        )
    )

    # Control 2: 02_rslora_orthogonal_pair
    t_rs_a: dict[str, tuple[Matrix, str]] = {}
    t_rs_b: dict[str, tuple[Matrix, str]] = {}
    for prefix in ALL_LINEAR_PREFIXES:
        t_rs_a[f"{prefix}.lora_A.weight"] = (_row_eye_block(4, d, offset=8, scale=1.0), "F32")
        t_rs_a[f"{prefix}.lora_B.weight"] = (_eye_block(d, 4, offset=8, scale=1.0), "F32")
        t_rs_b[f"{prefix}.lora_A.weight"] = (_row_eye_block(4, d, offset=12, scale=1.0), "F32")
        t_rs_b[f"{prefix}.lora_B.weight"] = (_eye_block(d, 4, offset=12, scale=1.0), "F32")
    _write_peft_dir(
        controls_dir / "02_rslora_orthogonal_pair" / "adapter_a",
        rank=4,
        alpha=8.0,
        use_rslora=True,
        target_modules=all_linear_mods,
        tensors=t_rs_a,
    )
    _write_peft_dir(
        controls_dir / "02_rslora_orthogonal_pair" / "adapter_b",
        rank=4,
        alpha=8.0,
        use_rslora=True,
        target_modules=all_linear_mods,
        tensors=t_rs_b,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="02_rslora_orthogonal_pair",
            category="control",
            adapter_rel_paths=[
                "controls/02_rslora_orthogonal_pair/adapter_a",
                "controls/02_rslora_orthogonal_pair/adapter_b",
            ],
            expected_rules=[],
            description="Clean orthogonal rsLoRA pair (Grade A).",
        )
    )

    # Control 3: 03_knots_aligned_post_merge (KnOTS merge of rotated conflicting sources)
    c3_root = controls_dir / "03_knots_aligned_post_merge"
    t_rot_a: dict[str, tuple[Matrix, str]] = {}
    t_rot_b: dict[str, tuple[Matrix, str]] = {}
    theta = 0.7
    ct, st = math.cos(theta), math.sin(theta)
    neg_q = [
        [-ct, st, 0.0, 0.0],
        [-st, -ct, 0.0, 0.0],
        [0.0, 0.0, -ct, -st],
        [0.0, 0.0, st, -ct],
    ]
    for prefix in ALL_LINEAR_PREFIXES:
        a1 = _row_eye_block(4, d, offset=0, scale=1.0)
        b1 = _eye_block(d, 4, offset=0, scale=1.5)
        # B2 = B1 @ (-Q), A2 = (-Q)^T @ A1
        b2: Matrix = [
            [sum(b1[i][k] * neg_q[k][j] for k in range(4)) for j in range(4)] for i in range(d)
        ]
        a2: Matrix = [
            [sum(neg_q[k][i] * a1[k][j] for k in range(4)) for j in range(d)] for i in range(4)
        ]
        t_rot_a[f"{prefix}.lora_A.weight"] = (a1, "F32")
        t_rot_a[f"{prefix}.lora_B.weight"] = (b1, "F32")
        t_rot_b[f"{prefix}.lora_A.weight"] = (a2, "F32")
        t_rot_b[f"{prefix}.lora_B.weight"] = (b2, "F32")

    src_a = _write_peft_dir(
        c3_root / "sources" / "adapter_a",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_rot_a,
    )
    src_b = _write_peft_dir(
        c3_root / "sources" / "adapter_b",
        rank=4,
        alpha=4.0,
        target_modules=all_linear_mods,
        tensors=t_rot_b,
    )
    merged_ctrl = c3_root / "merged_adapter"
    merge_knots_adapters(
        [load_peft_adapter(src_a), load_peft_adapter(src_b)],
        out_dir=merged_ctrl,
        method="knots_ties",
        target_rank=4,
        ties_density=1.0,
        rng_seed=seed,
    )
    cases.append(
        PlantedCaseSpec(
            case_id="03_knots_aligned_post_merge",
            category="control",
            adapter_rel_paths=["controls/03_knots_aligned_post_merge/merged_adapter"],
            expected_rules=[],
            description="KnOTS-aligned post-merge adapter from rotated conflicting sources.",
        )
    )

    manifest_data = {
        "schema_version": "1.0.0",
        "seed": seed,
        "cases": [asdict(c) for c in cases],
    }
    (out_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


def evaluate_planted_recall(
    corpus_dir: Path,
    *,
    cfg: LoraspecConfig | None = None,
) -> CorpusEvaluationResult:
    """Audit all planted faults and clean controls and return KPI-RES-01 recall & FAR."""
    manifest_path = corpus_dir / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Planted corpus manifest not found: {manifest_path}")

    raw_manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases: list[dict[str, Any]] = raw_manifest["cases"]
    audit_cfg = cfg.audit if cfg is not None else AuditConfig()
    fail_on = cfg.project.fail_on if cfg is not None else Severity.HIGH
    backend = resolve_math_backend(cfg.project.default_backend) if cfg is not None else "auto"

    results: list[PlantedCaseResult] = []
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for item in raw_cases:
        case_id = str(item["case_id"])
        category_raw = str(item["category"])
        category: Literal["fault", "control"] = "fault" if category_raw == "fault" else "control"
        rel_paths = [str(p) for p in item["adapter_rel_paths"]]
        expected_rules = [str(r) for r in item["expected_rules"]]

        loaded = [load_peft_adapter(corpus_dir / rel, backend=backend) for rel in rel_paths]
        layer_metrics, collisions, findings = run_spectral_and_collision_audit(
            loaded, audit_cfg, run_tier1=True, backend=backend
        )
        risk = compute_merge_risk_score(
            findings=findings, layer_metrics=layer_metrics, collisions=collisions
        )
        grade = assign_grade(findings=findings, merge_risk_score=risk)
        failed_gate = meets_fail_threshold(findings, fail_on)
        detected_rules = sorted({f.rule_id for f in findings})

        if category == "fault":
            passed = all(er in detected_rules for er in expected_rules) and failed_gate
            if passed:
                tp += 1
            else:
                fn += 1
        else:
            passed = len(findings) == 0 and grade == "A" and not failed_gate
            if passed:
                tn += 1
            else:
                fp += 1

        results.append(
            PlantedCaseResult(
                case_id=case_id,
                category=category,
                detected_rules=detected_rules,
                expected_rules=expected_rules,
                grade=grade,
                merge_risk_score=risk,
                failed_gate=failed_gate,
                passed_expectation=passed,
            )
        )

    total_faults = tp + fn
    total_controls = fp + tn
    recall = tp / float(total_faults) if total_faults > 0 else 0.0
    far = fp / float(total_controls) if total_controls > 0 else 0.0

    return CorpusEvaluationResult(
        total_faults=total_faults,
        total_controls=total_controls,
        true_positives=tp,
        false_negatives=fn,
        false_positives=fp,
        true_negatives=tn,
        recall=recall,
        false_alarm_rate=far,
        case_results=results,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and optionally verify the loraspec planted subspace fault corpus."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/loraspec-corpus"),
        help="Target directory for generated planted corpus.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic generation seed.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run evaluate_planted_recall and assert KPI-RES-01 (100% recall, <= 5% FAR).",
    )
    args = parser.parse_args(argv)

    corpus_path = generate_planted_corpus(args.out_dir, seed=args.seed)
    print(f"Generated planted corpus at {corpus_path}")

    if args.verify:
        ev = evaluate_planted_recall(corpus_path)
        for cr in ev.case_results:
            status = "PASS" if cr.passed_expectation else "FAIL"
            print(
                f"  [{status}] {cr.category.upper():7s} {cr.case_id:34s} "
                f"grade={cr.grade} risk={cr.merge_risk_score:5.1f} rules={cr.detected_rules}"
            )
        print(
            f"Recall: {ev.true_positives}/{ev.total_faults} ({ev.recall * 100:.1f}%) | "
            f"FAR: {ev.false_positives}/{ev.total_controls} ({ev.false_alarm_rate * 100:.1f}%)"
        )
        if ev.recall < 1.0 or ev.false_alarm_rate > 0.05:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

