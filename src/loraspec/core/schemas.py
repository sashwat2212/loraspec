"""Frozen Pydantic v2 domain schemas per docs/02-FRD-PRD.md."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def meets_fail_threshold(findings: Sequence[AuditFinding], fail_on: Severity) -> bool:
    """Return True if any finding meets or exceeds the configured fail_on severity."""
    threshold = SEVERITY_RANK[fail_on]
    return any(SEVERITY_RANK[f.severity] >= threshold for f in findings)


class AdapterFormat(StrEnum):
    PEFT_SAFETENSORS = "peft_safetensors"
    RAW_SAFETENSORS = "raw_safetensors"


class DeploymentRecommendation(StrEnum):
    SAFE_LINEAR_MERGE = "safe_linear_merge"
    USE_KNOTS_ALIGNMENT = "use_knots_alignment"
    USE_TOP_K_MOLORA = "use_top_k_molora"
    RETRAIN_COLLAPSED_LORA = "retrain_collapsed_lora"
    QUARANTINE_CORRUPT_ADAPTER = "quarantine_corrupt_adapter"


class LoRALayerSpec(BaseModel):
    layer_name: str
    module_type: str
    in_features: int
    out_features: int
    rank: int
    lora_alpha: float | None = None
    scaling: float | None = None
    dtype: str
    a_key: str
    b_key: str
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    param_count: int


class LoRAAdapterSpec(BaseModel):
    adapter_id: str
    adapter_format: AdapterFormat
    root_path: Path
    weights_path: Path
    config_path: Path | None = None
    base_model_name_or_path: str | None = None
    peft_type: str = "LORA"
    rank: int
    lora_alpha: float
    scaling: float
    lora_dropout: float = 0.0
    use_rslora: bool = False
    use_dora: bool = False
    bias: str = "none"
    target_modules: list[str] = Field(default_factory=list[str])
    total_lora_params: int
    weights_sha256: str
    layers: list[LoRALayerSpec] = Field(default_factory=list[LoRALayerSpec])
    metadata: dict[str, Any] = Field(default_factory=dict)


class LayerSpectralMetrics(BaseModel):
    adapter_id: str
    layer_name: str
    module_type: str
    rank: int
    singular_values: list[float]
    spectral_norm: float
    frobenius_norm: float
    effective_stable_rank: float
    rank_utilization: float
    top1_energy_ratio: float
    condition_number: float
    has_non_finite: bool = False


class PairwiseCollisionMetrics(BaseModel):
    adapter_a: str
    adapter_b: str
    layer_name: str
    active_rank_a: int
    active_rank_b: int
    principal_angles_deg: list[float]
    cos_principal_angle_max: float
    grassmann_chordal_similarity: float
    grassmann_geodesic_dist: float
    row_chordal_similarity: float
    sign_conflict_rate: float


class AuditFinding(BaseModel):
    rule_id: str
    title: str
    severity: Severity
    tier: Literal[0, 1, 2, 3]
    source: Literal["loraspec", "config_probe"]
    file_path: str | None = None
    line_number: int | None = None
    evidence: str
    remediation: str


class KnOTSMergeSummary(BaseModel):
    method: Literal["knots_ties", "knots_dare_ties", "knots_linear"]
    adapters_merged: list[str]
    target_rank: int
    energy_retained: float
    mean_reconstruction_rmse: float
    sign_conflict_reduction_pct: float
    output_sha256: str


class MoLoRARouteTelemetry(BaseModel):
    token_index: int
    token_text: str
    top_k_adapters: list[str]
    top_k_weights: list[float]
    router_entropy: float
    itl_ms: float


class RunManifest(BaseModel):
    run_id: str
    loraspec_version: str
    timestamp_utc: str
    git_sha: str | None
    host_os: str
    host_arch: str
    accelerator_backend: str
    adapter_sha256s: dict[str, str]
    rng_seed: int
    total_gpu_cost_inr: float = 0.0
    cli_args: list[str]


class AuditReport(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    manifest: RunManifest
    adapters: list[LoRAAdapterSpec] = Field(default_factory=list[LoRAAdapterSpec])
    grade: Literal["A", "B", "C", "D", "F"]
    merge_risk_score: float = 0.0
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])
    layer_metrics: list[LayerSpectralMetrics] = Field(default_factory=list[LayerSpectralMetrics])
    pairwise_collisions: list[PairwiseCollisionMetrics] = Field(
        default_factory=list[PairwiseCollisionMetrics]
    )
    knots_summary: KnOTSMergeSummary | None = None
    recommendations: list[DeploymentRecommendation] = Field(
        default_factory=list[DeploymentRecommendation]
    )


class MoLoRARouteTelemetryBatch(BaseModel):
    layer_name: str
    num_tokens: int
    top_k: int
    expert_indices: list[list[int]]
    routing_weights: list[list[float]]
    router_entropy: list[float]
    backend: Literal["triton_cuda", "torch_mps", "torch_cpu", "pure_python"]


class KernelBenchmarkPoint(BaseModel):
    hardware_profile: str
    arm_name: Literal[
        "base",
        "peft_linear",
        "peft_ties",
        "peft_eager_arrow",
        "vllm_static",
        "knots_ties",
        "triton_arrow_sgmv",
    ]
    batch_size: int
    seq_len: int
    dtype: Literal["fp16", "bf16", "fp32"]
    latency_ms: float
    achieved_tflops: float
    hbm_bandwidth_gb_s: float
    peak_vram_mb: float
    max_abs_err_vs_ref: float = 0.0


class BenchmarkSuiteSummary(BaseModel):
    manifest: RunManifest
    points: list[KernelBenchmarkPoint]
    speedup_vs_peft_eager: float
    vram_savings_pct: float
    arm_accuracy_pass_at_1: dict[str, float] = Field(default_factory=dict)
