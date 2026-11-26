# Stage 4 / SDD-01: Bounds-Safe Safetensors Reader & Spectral Subspace Auditor — `loraspec`

> [!NOTE]
> **Subsystem**: `SDD-01` (`ARCH-COMP-03` Reader + `ARCH-COMP-04` Spectral/Grassmann Engine + `ARCH-COMP-10` Scorer)
> **Linked Features**: `FEAT-01`, `FEAT-02`, `FEAT-03`, `FEAT-04`, `FEAT-05`
> **Upstream Docs**: [02-FRD-PRD.md](./02-FRD-PRD.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/loraspec/inspect/safetensors_reader.py`, `src/loraspec/core/spectral_math.py`, `src/loraspec/auditor/probes.py`, `src/loraspec/auditor/grading.py`

---

## 1. Bounds-Safe `.safetensors` Binary Reader & PEFT Config Parser (`ARCH-COMP-03`)

To guarantee `$0` zero-dependency auditing and immunity against `pickle` code execution or OOM header attacks (`T1`, `T2`), `safetensors_reader.py` parses `.safetensors` directly from binary headers:

```text
Byte Layout of .safetensors:
[0..8)            uint64 little-endian header_len (N)
[8..8+N)          UTF-8 JSON tensor index + optional "__metadata__"
[8+N..file_size)  Contiguous little-endian IEEE-754 tensor payload bytes
```

### 1.1 Validation Rules (Raising `AdapterParseError` / `EXIT_CONFIG_ERROR = 2`)
1. `file_size >= 8`, `0 < N <= 104_857_600` (100 MB header ceiling), and `8 + N <= file_size`.
2. Every non-`__metadata__` tensor entry must declare `dtype` in `{"F32", "F16", "BF16"}` (plus scalar metadata dtypes if present), positive non-empty integer `shape`, non-overlapping byte slices `[begin, end]` satisfying `0 <= begin <= end <= file_size - (8 + N)`, and `end - begin == prod(shape) * dtype_bytes`; LoRA factor tensors must have strictly positive rank and feature dimensions (`rank > 0, in_features > 0, out_features > 0`).
3. **PEFT Layer Key Normalization**: Regex/suffix matcher groups by `(adapter_namespace, layer_prefix)` across `...lora_A[.adapter_name].weight` and `...lora_B[.adapter_name].weight` (including unprefixed `lora_A.weight`/`lora_B.weight`), preventing silent multi-adapter overwrites and validating inner rank dimension consistency (`lora_A.shape[0] == lora_B.shape[1] == r`).
4. **Dtype Decoding**: Decodes `F32` (`<f`), `F16` (`<e`), and `BF16` (`uint16 << 16 -> <f`) cleanly in pure Python batch unpacking (or vectorized `numpy.frombuffer` when `numpy` is available).

---

## 2. Factorized Rank-$r$ Spectral & Grassmann Math (`ARCH-COMP-04`)

### 2.1 Scaling & Exact Factorized SVD
For layer $\ell$ of adapter $m$ with `lora_alpha` $\alpha_m$, rank $r_m$, and `use_rslora`:

$$s_m = \begin{cases} \alpha_m / \sqrt{r_m}, & \text{if } \texttt{use\_rslora} = \text{True} \\ \alpha_m / r_m, & \text{otherwise} \end{cases}$$

Let $\tilde{B} = s_m B_m \in \mathbb{R}^{d_{\text{out}} \times r_m}$ and $A_m^T \in \mathbb{R}^{d_{\text{in}} \times r_m}$:
1. Thin QR: $\tilde{B} = Q_B R_B$ ($Q_B \in \mathbb{R}^{d_{\text{out}} \times r_m}$), $A_m^T = Q_A R_A$ ($Q_A \in \mathbb{R}^{d_{\text{in}} \times r_m}$).
2. Core SVD: $R_B R_A^T = U_c \Sigma V_c^T$ on tiny $r_m \times r_m$ matrix ($\Sigma = \text{diag}(\sigma_1, \dots, \sigma_{r_m})$).
3. **Zero $O(d_{\text{out}} d_{\text{in}})$ Allocation**: Active column basis $U_m = Q_B U_c[:, \mathcal{I}]$ and row basis $V_m = Q_A V_c[:, \mathcal{I}]$ retain active singular directions with $\sigma_i > \max(10^{-9},\; 10^{-6}\sigma_1)$ (re-orthonormalized via synchronized thin QR if $d < r$).

### 2.2 Per-Layer Spectral Health Metrics (`LayerSpectralMetrics`)
- **Spectral Norm**: $\|\Delta W\|_2 = \sigma_1$
- **Frobenius Norm**: $\|\Delta W\|_F = \sqrt{\sum_{i=1}^{r_m} \sigma_i^2}$
- **Effective Stable Rank**: $r_{\text{eff}} = \|\Delta W\|_F^2 / \sigma_1^2 \in [1, r_m]$ (`0.0` if $\sigma_1 \le 10^{-12}$)
- **Rank Utilization**: $r_{\text{eff}} / r_m \in [0, 1]$
- **Top-1 Energy Ratio**: $\sigma_1^2 / \sum_{i=1}^{r_m} \sigma_i^2$
- **Condition Ratio**: $\sigma_1 / \max(\sigma_{r_m}, 10^{-12})$

### 2.3 Pairwise Grassmann Subspace Collision (`PairwiseCollisionMetrics`)
For adapter pair $(a, b)$ at shared layer $\ell$ with orthonormal column bases $U_a \in \mathbb{R}^{d_{\text{out}} \times k_a}$, $U_b \in \mathbb{R}^{d_{\text{out}} \times k_b}$ and row bases $V_a \in \mathbb{R}^{d_{\text{in}} \times k_{va}}$, $V_b \in \mathbb{R}^{d_{\text{in}} \times k_{vb}}$:
- Principal cosines (column space): $c_i^{\text{col}} = \text{clip}(\sigma_i(M_U), 0, 1)$ where $M_U = U_a^T U_b$, for $i = 1 \dots p$ ($p = \min(k_a, k_b)$).
- Principal angles: $\theta_i = \arccos(c_i^{\text{col}})$ (reported in degrees).
- **Column-Space Chordal Similarity** (`grassmann_chordal_similarity`): $S_{\text{chord}}^{\text{col}} = \frac{1}{p} \sum_{i=1}^p (c_i^{\text{col}})^2 \in [0, 1]$, alongside **Row-Space Chordal Similarity** (`row_chordal_similarity`): $S_{\text{chord}}^{\text{row}} = \frac{1}{p_{\text{row}}} \sum_{i=1}^{p_{\text{row}}} (c_i^{\text{row}})^2$ from $M_V = V_a^T V_b$.
- **Geodesic Distance**: $d_{\text{geod}} = \sqrt{\sum_{i=1}^p \theta_i^2}$.
- **Gauge-Invariant Directional Sign Conflict Rate** (`sign_conflict_rate`): Magnitude-weighted Frobenius cancellation ratio over active singular triplets $(i, j)$ with $\text{signed\_overlap}_{i,j} = \sigma_{a,i}\,\sigma_{b,j}\,(M_U)_{i,j}\,(M_V)_{i,j}$, invariant under $(B, A) \mapsto (-B, -A)$:
  $$\text{sign\_conflict\_rate} = \frac{\sum_{\text{signed\_overlap} < 0} |\text{signed\_overlap}_{i,j}|}{\sum_{i,j} |\text{signed\_overlap}_{i,j}|}$$

---

## 3. Audit Rule Catalog & Grading Rubric

### 3.1 Configuration & Spectral Rules
| Rule ID | Severity | Trigger Condition | Remediation |
|---|---|---|---|
| **`RULE-CONF-001`** | `CRITICAL` | Adapters in fleet specify different `base_model_name_or_path` or mismatching layer input/output dimensions. | Re-train or filter adapters targeting identical base model checkpoint. |
| **`RULE-CONF-002`** | `HIGH` | Missing `adapter_config.json`, missing/non-positive `lora_alpha`, unsupported `peft_type` (`!= "LORA"`), or `r` disagreeing with `.safetensors` weights (without `rank_pattern`). | Fix PEFT `adapter_config.json` and verify non-zero scaling and matching rank. |
| **`RULE-CONF-003`** | `HIGH` | `target_modules` omits MLP projections (`gate_proj`, `up_proj`, `down_proj`). | Apply LoRA to all linear layers (attention + MLP) per *LoRA Without Regret*. |
| **`RULE-CONF-004`** | `MEDIUM` | Heterogeneous `rank` or mixed `use_rslora` values across adapters in fleet. | Use `loraspec merge` (`KnOTS`) or dynamic Top-$k$ `Arrow-SGMV` instead of naive linear merge. |
| **`RULE-SPEC-001`** | `CRITICAL` | Non-finite (`NaN`/`Inf`) weights or dead zero layer ($\|\Delta W\|_F \le 10^{-12}$). | Quarantine/remove corrupted checkpoint before merge or serving. |
| **`RULE-SPEC-002`** | `HIGH` | Mean `rank_utilization < 0.20` OR $\ge 30\%$ non-dead finite layers have `top1_energy_ratio > 0.85` (when $r \ge 4$). | Truncate redundant rank via SVD compression or adjust LR/rank training schedule. |
| **`RULE-SPEC-003`** | `HIGH` | Layer $\|\Delta W\|_2 > \max(10.0,\; 5.0 \cdot \text{median}(\|\Delta W\|_2))$. | Inspect learning rate / gradient clipping or clamp spectral norm prior to merge. |
| **`RULE-SPEC-004`** | `CRITICAL` / `HIGH` | `CRITICAL` if `grassmann_chordal_similarity >= 0.70` & `sign_conflict_rate >= 0.35` on $\ge 20\%$ shared layers; `HIGH` if `grassmann_chordal_similarity >= 0.45` (or `cos_principal_angle_max >= 0.85` & `sign_conflict_rate >= 0.25`). | Switch from naive merge to **`loraspec merge` (`KnOTS` joint-SVD)** or **Per-Token Top-$k$ `MoLoRA`**. |

### 3.2 `MergeRiskScore (0.0–100.0)` & Grade (`A–F`)
$$\text{MergeRiskScore} = \min\!\Big(100.0,\; 35 \cdot I_{\text{crit\_conf\_or\_nan}} + 30 \cdot C_{\text{collision}} + 20 \cdot R_{\text{collapse}} + 15 \cdot S_{\text{spike\_or\_config}}\Big)$$
- **Grade `A`**: `0` findings AND `MergeRiskScore == 0.0` (`SAFE_LINEAR_MERGE`).
- **Grade `B`**: `0` CRITICAL/HIGH findings AND `MergeRiskScore <= 15.0` (only `MEDIUM`/`LOW`/`INFO` advisories such as heterogeneous rank).
- **Grade `C`**: `0` CRITICAL/HIGH findings AND `15.0 < MergeRiskScore <= 35.0` (`USE_KNOTS_ALIGNMENT` advised).
- **Grade `D`**: Any `HIGH` finding (e.g., `HIGH` collision, rank collapse, spectral spike, or missing/unsupported config) OR `35.0 < MergeRiskScore <= 60.0` (`USE_KNOTS_ALIGNMENT` or `USE_TOP_K_MOLORA` required).
- **Grade `F`**: Any `CRITICAL` finding (`RULE-CONF-001`, `RULE-SPEC-001`, `CRITICAL RULE-SPEC-004`) OR `MergeRiskScore > 60.0`.

---

## 4. Automated Verification Matrix (`SDD-01`)
1. Binary `.safetensors` roundtrip tests (`F32`, `F16`, `BF16`) + oversized-header/truncated-payload security rejection tests.
2. Analytic SVD & Grassmann tests: identity subspaces $\to \theta = 0^\circ, S_{\text{chord}} = 1.0$; orthogonal subspaces $\to \theta = 90^\circ, S_{\text{chord}} = 0.0$; uniform rank-$r$ orthogonal factors $\to r_{\text{eff}} = r$; rank-1 outer product $\to r_{\text{eff}} = 1.0$; `use_rslora=True` scaling check.
3. Full audit rule & Grade `A–F` transition tests (`RULE-CONF-001..004`, `RULE-SPEC-001..004`).
