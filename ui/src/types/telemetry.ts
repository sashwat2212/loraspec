export type Severity = "critical" | "high" | "medium" | "low" | "info";
export type KnotsMethod = "knots_ties" | "knots_dare_ties" | "knots_linear";

export interface HealthStatus {
  status: string;
  demo_replay: boolean;
  backend: string;
  loaded_adapters: number;
  version: string;
}

export interface LoRALayerSpec {
  layer_name: string;
  module_type: string;
  in_features: number;
  out_features: number;
  rank: number;
  lora_alpha: number | null;
  scaling: number | null;
  dtype: string;
  param_count: number;
}

export interface LoRAAdapterSpec {
  adapter_id: string;
  adapter_format: string;
  root_path: string;
  weights_path: string;
  base_model_name_or_path: string | null;
  peft_type: string;
  rank: number;
  lora_alpha: number;
  scaling: number;
  use_rslora: boolean;
  use_dora: boolean;
  target_modules: string[];
  total_lora_params: number;
  weights_sha256: string;
  layers: LoRALayerSpec[];
}

export interface LayerSpectralMetrics {
  adapter_id: string;
  layer_name: string;
  module_type: string;
  rank: number;
  singular_values: number[];
  spectral_norm: number;
  frobenius_norm: number;
  effective_stable_rank: number;
  rank_utilization: number;
  top1_energy_ratio: number;
  condition_number: number;
  has_non_finite: boolean;
}

export interface PairwiseCollisionMetrics {
  adapter_a: string;
  adapter_b: string;
  layer_name: string;
  active_rank_a: number;
  active_rank_b: number;
  principal_angles_deg: number[];
  cos_principal_angle_max: number;
  grassmann_chordal_similarity: number;
  grassmann_geodesic_dist: number;
  row_chordal_similarity: number;
  sign_conflict_rate: number;
}

export interface AuditFinding {
  rule_id: string;
  title: string;
  severity: Severity;
  tier: number;
  source: string;
  file_path: string | null;
  line_number: number | null;
  evidence: string;
  remediation: string;
}

export interface KnOTSMergeSummary {
  method: KnotsMethod;
  adapters_merged: string[];
  target_rank: number;
  energy_retained: number;
  mean_reconstruction_rmse: number;
  sign_conflict_reduction_pct: number;
  output_sha256: string;
}

export interface RunManifest {
  run_id: string;
  loraspec_version: string;
  timestamp_utc: string;
  git_sha: string | null;
  host_os: string;
  host_arch: string;
  accelerator_backend: string;
  adapter_sha256s: Record<string, string>;
  rng_seed: number;
  total_gpu_cost_inr: number;
  cli_args: string[];
}

export interface AuditReport {
  schema_version: string;
  manifest: RunManifest;
  adapters: LoRAAdapterSpec[];
  grade: "A" | "B" | "C" | "D" | "F";
  merge_risk_score: number;
  findings: AuditFinding[];
  layer_metrics: LayerSpectralMetrics[];
  pairwise_collisions: PairwiseCollisionMetrics[];
  knots_summary: KnOTSMergeSummary | null;
  recommendations: string[];
}

export interface KnOTSPreviewRequest {
  adapter_paths: string[];
  method: KnotsMethod;
  target_rank: number;
  ties_density: number;
  dare_drop_rate: number;
  use_gks: boolean;
  gks_adapter_path?: string | null;
  rng_seed: number;
}

export interface KnOTSPreviewResponse {
  summary: KnOTSMergeSummary;
  pre_knots_sign_conflict_rate: number;
  post_knots_sign_conflict_rate: number;
  findings: AuditFinding[];
}

export interface MoLoRARouteTelemetry {
  token_index: number;
  token_text: string;
  top_k_adapters: string[];
  top_k_weights: number[];
  router_entropy: number;
  itl_ms: number;
}

export interface ReplayScenarioSummary {
  replay_id: string;
  title: string;
  prompt: string;
  model_name: string;
  hardware_profile: string;
  num_tokens: number;
  ttft_ms: number;
  mean_itl_ms: number;
  peak_vram_mb: number;
}

export interface KernelBenchmarkPoint {
  hardware_profile: string;
  arm_name: string;
  batch_size: number;
  seq_len: number;
  dtype: string;
  latency_ms: number;
  achieved_tflops: number;
  hbm_bandwidth_gb_s: number;
  peak_vram_mb: number;
  max_abs_err_vs_ref: number;
}

export interface BenchmarkSuiteSummary {
  manifest: RunManifest;
  points: KernelBenchmarkPoint[];
  speedup_vs_peft_eager: number;
  vram_savings_pct: number;
  arm_accuracy_pass_at_1: Record<string, number>;
}

export type WsTelemetryEvent =
  | {
      type: "gpu_stats";
      replay_id: string;
      hardware_profile: string;
      ttft_ms: number;
      itl_ms: number;
      vram_mb: number;
    }
  | {
      type: "token";
      replay_id: string;
      routing: MoLoRARouteTelemetry;
    }
  | {
      type: "done";
      replay_id: string;
      total_tokens: number;
      ttft_ms: number;
      mean_itl_ms: number;
      peak_vram_mb: number;
    }
  | {
      type: "error";
      detail: string;
    };

