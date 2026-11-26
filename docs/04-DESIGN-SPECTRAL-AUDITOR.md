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
2. Every non-`__metadata__` tensor entry must declare `dtype` in `{"F32", "F16", "BF16"}` (plus scalar metadata dtypes if present), positive non-empty integer `shape`, and offsets `[begin, end]` satisfying `0 <= begin <= end <= file_size - (8 + N)` and `end - begin == prod(shape) * dtype_bytes`.
3. **PEFT Layer Key Normalization**: Regex/suffix matcher pairs `...lora_A[.adapter_name].weight` with `...lora_B[.adapter_name].weight`, validating inner rank dimension consistency (`lora_A.shape[0] == lora_B.shape[1] == r`).
4. **Dtype Decoding**: Decodes `F32` (`<f`), `F16` (`<e`), and `BF16` (`uint16 << 16 -> <f`) cleanly in pure Python (or vectorized `numpy.frombuffer` when `[spectral]` is installed).

---

## 2. Factorized Rank-$r$ Spectral & Grassmann Math (`ARCH-COMP-04`)

### 2.1 Scaling & Exact Factorized SVD
For layer $\ell$ of adapter $m$ with `lora_alpha` $\alpha_m$, rank $r_m$, and `use_rslora`:

$$s_m = \begin{cases} \alpha_m / \sqrt{r_m}, & \text{if } \texttt{use\_rslora} = \text{True} \\ \alpha_m / r_m, & \text{otherwise} \end{cases}$$

Let $\tilde{B} = s_m B_m \in \mathbb{R}^{d_{\text{out}} \times r_m}$ and $A_m^T \in \mathbb{R}^{d_{\text{in}} \times r_m}$:
1. Thin QR: $\tilde{B} = Q_B R_B$ ($Q_B \in \mathbb{R}^{d_{\text{out}} \times r_m}$), $A_m^T = Q_A R_A$ ($Q_A \in \mathbb{R}^{d_{\text{in}} \times r_m}$).
2. Core SVD: $R_B R_A^T = U_c \Sigma V_c^T$ on tiny $r_m \times r_m$ matrix ($\Sigma = \text{diag}(\sigma_1, \dots, \sigma_{r_m})$).
3. **Zero $O(d_{\text{out}} d_{\text{in}})$ Allocation**: Active column basis $U_m = Q_B U_c[:, :k_m]$ and row basis $V_m = Q_A V_c[:, :k_m]$ retain directions with $\sigma_i > 10^{-6} \sigma_1$.

### 2.2 Per-Layer Spectral Health Metrics (`LayerSpectralMetrics`)
- **Spectral Norm**: $\|\Delta W\|_2 = \sigma_1$
- **Frobenius Norm**: $\|\Delta W\|_F = \sqrt{\sum_{i=1}^{r_m} \sigma_i^2}$
- **Effective Stable Rank**: $r_{\text{eff}} = \|\Delta W\|_F^2 / \sigma_1^2 \in [1, r_m]$ (`0.0` if $\sigma_1 == 0$)
- **Rank Utilization**: $r_{\text{eff}} / r_m \in [0, 1]$
- **Top-1 Energy Ratio**: $\sigma_1^2 / \sum_{i=1}^{r_m} \sigma_i^2$
- **Condition Ratio**: $\sigma_1 / \max(\sigma_{r_m}, 10^{-12})$

### 2.3 Pairwise Grassmann Subspace Collision (`PairwiseCollisionMetrics`)
For adapter pair $(a, b)$ at shared layer $\ell$ with orthonormal column bases $U_a \in \mathbb{R}^{d_{\text{out}} \times k_a}$, $U_b \in \mathbb{R}^{d_{\text{out}} \times k_b}$ and row bases $V_a \in \mathbb{R}^{d_{\text{in}} \times k_a}$, $V_b \in \mathbb{R}^{d_{\text{in}} \times k_b}$:
- Principal cosines (column space): $c_i^{\text{col}} = \text{clip}(\sigma_i(U_a^T U_b), 0, 1)$ for $i = 1 \dots p$ ($p = \min(k_a, k_b)$).
- Principal angles: $\theta_i = \arccos(c_i)$ (reported in degrees).
- **Chordal Similarity**: $S_{\text{chord}} = \frac{1}{p} \sum_{i=1}^p (c_i^{\text{col}})^2 \in [0, 1]$ (and symmetric combined column/row overlap $\max(S_{\text{chord}}^{\text{col}}, S_{\text{chord}}^{\text{row}})$ or joint bilateral product overlap).
- **Geodesic Distance**: $d_{\text{geod}} = \sqrt{\sum_{i=1}^p \theta_i^2}$.
- **Directional Sign Conflict Rate**: Magnitude-weighted fraction of active basis coordinates across matched principal directions ($|u_{a,i} \cdot u_{b,j}|$ signed pairs) or flattened factor alignment exhibiting opposing signs (`sign(x_a) != sign(x_b)`).

---

## 3. Audit Rule Catalog & Grading Rubric

### 3.1 Configuration & Spectral Rules
| Rule ID | Severity | Trigger Condition | Remediation |
|---|---|---|---|
| **`RULE-CONF-001`** | `CRITICAL` | Adapters in fleet specify different `base_model_name_or_path` or mismatching layer input/output dimensions. | Re-train or filter adapters targeting identical base model checkpoint. |
| **`RULE-CONF-002`** | `HIGH` | Missing `adapter_config.json`, `lora_alpha <= 0`, or unsupported `peft_type`. | Fix PEFT `adapter_config.json` and verify non-zero scaling. |
| **`RULE-CONF-003`** | `HIGH` | `target_modules` omits MLP projections (`gate_proj`, `up_proj`, `down_proj`). | Apply LoRA to all linear layers (attention + MLP) per *LoRA Without Regret*. |
| **`RULE-CONF-004`** | `MEDIUM` | Heterogeneous `rank` or mixed `use_rslora` values across adapters in fleet. | Use `loraspec merge` (`KnOTS`) or dynamic Top-$k$ `Arrow-SGMV` instead of naive linear merge. |
| **`RULE-SPEC-001`** | `CRITICAL` | Non-finite (`NaN`/`Inf`) weights or dead zero layer ($\|\Delta W\|_F \le 10^{-12}$). | Quarantine/remove corrupted checkpoint before merge or serving. |
| **`RULE-SPEC-002`** | `HIGH` | Mean `rank_utilization < 0.20` OR $\ge 30\%$ layers have `energy_top_1 > 0.85` (when $r \ge 4$). | Truncate redundant rank via SVD compression or adjust LR/rank training schedule. |
| **`RULE-SPEC-003`** | `HIGH` | Layer $\|\Delta W\|_2 > \max(10.0,\; 5.0 \cdot \text{median}(\|\Delta W\|_2))$. | Inspect learning rate / gradient clipping or clamp spectral norm prior to merge. |
| **`RULE-SPEC-004`** | `CRITICAL` / `HIGH` | `CRITICAL` if `chordal_sim >= 0.70` & `sign_conflict >= 0.35` on $\ge 20\%$ shared layers; `HIGH` if `chordal_sim >= 0.45` (or `cos_max >= 0.85` & `conflict >= 0.25`). | Switch from naive merge to **`loraspec merge` (`KnOTS` joint-SVD)** or **Per-Token Top-$k$ `MoLoRA`**. |

### 3.2 `MergeRiskScore (0.0–100.0)` & Grade (`A–F`)
$$\text{MergeRiskScore} = \min\!\Big(100.0,\; 35 \cdot I_{\text{crit\_conf\_or\_nan}} + 30 \cdot C_{\text{collision}} + 20 \cdot R_{\text{collapse}} + 15 \cdot S_{\text{spike\_or\_config}}\Big)$$
- **Grade `A`**: `0` CRITICAL/HIGH findings AND `MergeRiskScore == 0.0` (`SAFE_LINEAR_MERGE`).
- **Grade `B`**: `0` CRITICAL findings, `MergeRiskScore <= 15.0` (only `MEDIUM`/`LOW` advisories like heterogeneous rank).
- **Grade `C`**: `0` CRITICAL findings, `15.0 < MergeRiskScore <= 35.0` (`USE_KNOTS_ALIGNMENT` advised).
- **Grade `D`**: `HIGH` collision or rank collapse (`35.0 < MergeRiskScore <= 60.0`; `USE_KNOTS_ALIGNMENT` or `USE_TOP_K_MOLORA` required).
- **Grade `F`**: Any `CRITICAL` finding (`RULE-CONF-001`, `RULE-SPEC-001`, `CRITICAL RULE-SPEC-004`) OR `MergeRiskScore > 60.0`.

---

## 4. Automated Verification Matrix (`SDD-01`)
1. Binary `.safetensors` roundtrip tests (`F32`, `F16`, `BF16`) + oversized-header/truncated-payload security rejection tests.
2. Analytic SVD & Grassmann tests: identity subspaces $\to \theta = 0^\circ, S_{\text{chord}} = 1.0$; orthogonal subspaces $\to \theta = 90^\circ, S_{\text{chord}} = 0.0$; uniform rank-$r$ orthogonal factors $\to r_{\text{eff}} = r$; rank-1 outer product $\to r_{\text{eff}} = 1.0$; `use_rslora=True` scaling check.
3. Full audit rule & Grade `A–F` transition tests (`RULE-CONF-001..004`, `RULE-SPEC-001..004`).
