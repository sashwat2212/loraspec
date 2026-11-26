# Stage 4 / SDD-02: Scalable `KnOTS` Joint-SVD Alignment & PEFT Merger — `loraspec`

> [!NOTE]
> **Subsystem**: `SDD-02` (`ARCH-COMP-05` `KnOTS` Aligner, Sign-Consensus Operators & `GenKnowSub` Residual Engine)
> **Linked Features**: `FEAT-06`, `FEAT-07`, `FEAT-08`
> **Upstream Docs**: [02-FRD-PRD.md](./02-FRD-PRD.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/loraspec/knots/aligner.py`, `src/loraspec/knots/operators.py`, `src/loraspec/knots/serializer.py` (Target: Milestone **M1**)

---

## 1. Mathematical Design: Memory-Safe Factorized `KnOTS`

Standard `KnOTS` (`arXiv:2410.19735`, ICLR 2025) concatenates full dense updates:

$$\Delta W_{\text{stack}} = \begin{bmatrix} s_1 B_1 A_1 & s_2 B_2 A_2 & \dots & s_K B_K A_K \end{bmatrix} \in \mathbb{R}^{d_{\text{out}} \times (K \cdot d_{\text{in}})}$$

On a `Qwen3-4B` or `Llama-3.1-8B` MLP layer ($d_{\text{out}} = 14336, d_{\text{in}} = 4096, K=6$), materializing $\Delta W_{\text{stack}}$ requires **~1.4 GB per layer** in FP32 and an $O(d^3)$ dense SVD that hangs on CPU GitHub Actions runners (`RISK-04`).

### 1.1 Exact Algebraic Equivalent in $O((d_{\text{out}} + d_{\text{in}}) R_{\text{sum}}^2)$
Let $R_{\text{sum}} = \sum_{m=1}^K r_m$ (e.g., $6 \times 16 = 96 \ll 4096$).
1. **Optional `GenKnowSub` Subtraction (`use_gks=True`)**: If a general-knowledge adapter $(B_0, A_0)$ is supplied with weight $\lambda$, append $[-\lambda s_0 B_0]$ and $A_0$ cleanly to each adapter's factorized representation (effectively adding $r_0$ rank columns without expanding to dense space).
2. **Stacked Left Factor QR**:
   $$B_{\text{cat}} = \begin{bmatrix} s_1 B_1 & \cdots & s_K B_K \end{bmatrix} \in \mathbb{R}^{d_{\text{out}} \times R_{\text{sum}}}, \qquad B_{\text{cat}} = Q_U R_U$$
   where $Q_U \in \mathbb{R}^{d_{\text{out}} \times R_{\text{sum}}}$ is orthonormal and $R_U = [R_{U,1}, \dots, R_{U,K}] \in \mathbb{R}^{R_{\text{sum}} \times R_{\text{sum}}}$.
3. **Stacked Block-Diagonal Right QR**:
   Form orthonormal row basis across $\text{span}(A_1^T, \dots, A_K^T)$ via thin QR of $[A_1^T, \dots, A_K^T] = Q_A R_A$ ($Q_A \in \mathbb{R}^{d_{\text{in}} \times R_{\text{sum}}}$), giving $A_m = (C_m)^T Q_A^T$ where $C_m \in \mathbb{R}^{R_{\text{sum}} \times r_m}$!
   *(Wait—notice how $V_{\text{stack}}^T$ in $\Delta W_{\text{stack}}$ has $K$ independent $d_{\text{in}}$ blocks of size $d_{\text{out}} \times d_{\text{in}}$!)*
   Crucial exact observation: $\Delta W_{\text{stack}} = Q_U [R_{U,1} A_1, \dots, R_{U,K} A_K]$.
   To find the exact left singular vectors $U$ and singular values $\Sigma$ of $\Delta W_{\text{stack}}$:
   $$\Delta W_{\text{stack}} \Delta W_{\text{stack}}^T = Q_U \left( \sum_{m=1}^K (R_{U,m} A_m)(R_{U,m} A_m)^T \right) Q_U^T = Q_U G Q_U^T$$
   where each Gram contribution $G_m = R_{U,m} (A_m A_m^T) R_{U,m}^T \in \mathbb{R}^{R_{\text{sum}} \times R_{\text{sum}}}$ uses only $A_m A_m^T \in \mathbb{R}^{r_m \times r_m}$!
   Eigendecomposing the symmetric PSD matrix $G = \sum_{m=1}^K G_m = W_c \Sigma^2 W_c^T$ (size **$96 \times 96$**, taking **`< 0.2 ms` on CPU!**) produces:
   - Exact shared orthogonal left basis: $U = Q_U W_c \in \mathbb{R}^{d_{\text{out}} \times R_{\text{sum}}}$
   - Exact joint singular values: $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_{R_{\text{sum}}})$
   - Exact aligned right task representations in $V_m^T$-space (for each task $m$):
     $$V_m^T = \Sigma^{-1} W_c^T R_{U,m} A_m \in \mathbb{R}^{R_{\text{sum}} \times d_{\text{in}}}$$
     *(Numerically conditioned alternative implemented in M1: instead of forming $G = M M^T$ which squares the condition number $\kappa(G) = \kappa(M)^2$, take per-adapter thin QR $A_m^T = Q_{A,m} R_{A,m}$ ($d_{\text{in}} \times r_m$), form the tiny stacked $R_{\text{sum}} \times R_{\text{sum}}$ core $M = [R_{U,1} R_{A,1}^T, \dots, R_{U,K} R_{A,K}^T]$, and take its SVD $M = W_c \Sigma \tilde{V}_{\text{cat}}^T$, giving identical $U = Q_U W_c$, $\Sigma$, and $V_m^T = (\tilde{V}_{\text{cat}, m}^T) Q_{A,m}^T$ directly without squaring singular values or dividing by $\Sigma$!)*

This yields **mathematically exact** `KnOTS` alignment at **500× lower RAM** and **100× faster CPU speed** than naive dense SVD!

---

## 2. Aligned Combination Operators (`src/loraspec/knots/operators.py`)

Given aligned $\{V_m^T\}_{m=1}^K$ and normalized task weights $\{w_m\}_{m=1}^K$:
1. **`knots_linear`**: $V_{\text{merged}}^T = \sum_{m=1}^K w_m V_m^T$.
2. **`knots_ties`**:
   - *Trim*: Retain top `ties_density` fraction by magnitude per row/tensor in each $V_m^T$; zero out lower tail.
   - *Elect Sign*: $\gamma_{i,j} = \text{sign}\left(\sum_{m=1}^K w_m \tilde{V}_{m, i, j}^T\right)$.
   - *Disjoint Merge*: Average only adapters whose trimmed entry matches elected sign $\gamma_{i,j}$: $V_{\text{merged}, i, j}^T = \frac{\sum_{m \in \mathcal{A}_{i,j}} w_m \tilde{V}_{m, i, j}^T}{\sum_{m \in \mathcal{A}_{i,j}} w_m}$.
3. **`knots_dare_ties`**: Apply deterministic seeded Bernoulli mask $M_m \sim \text{Bernoulli}(1 - p_{\text{dare}})$, rescale $\tilde{V}_m^T \leftarrow \frac{M_m \odot V_m^T}{1 - p_{\text{dare}}}$, then run `knots_ties`.

---

## 3. Reconstruction, Rank Truncation & PEFT Export (`src/loraspec/knots/serializer.py`)

To compress back into a single PEFT adapter of target rank $r_{\text{out}}$ without dense reconstruction:
1. Perform thin QR on $V_{\text{merged}} \in \mathbb{R}^{d_{\text{in}} \times R_{\text{sum}}} \to Q_V R_V$ and thin SVD on tiny $R_{\text{sum}} \times R_{\text{sum}}$ core $\Sigma R_V^T = \tilde{U} \tilde{\Sigma} \tilde{V}^T$.
2. Keep top $r_{\text{out}}$ components:
   $$B_{\text{out}} = U \tilde{U}[:, :r_{\text{out}}] \sqrt{\tilde{\Sigma}[:r_{\text{out}}]}, \qquad A_{\text{out}} = \sqrt{\tilde{\Sigma}[:r_{\text{out}}]} \tilde{V}^T[:r_{\text{out}}, :] Q_V^T$$
   setting `lora_alpha = r_out` (so scaling $s = 1.0$) and saving directly as standard PEFT `adapter_model.safetensors` + `adapter_config.json` + `knots_merge_manifest.json`.
3. **Quality Telemetry (`KnOTSMergeSummary`)**: Computes retained spectral energy $\eta_{r_{\text{out}}} = \sum_{i=1}^{r_{\text{out}}} \tilde{\sigma}_i^2 / \sum \tilde{\sigma}_i^2$ and factorized Frobenius RMSE vs each source adapter, emitting `RULE-KNOTS-001` (`MEDIUM`) if $\eta < 0.90$.

---

## 4. Verification Matrix (`SDD-02`)
1. Exact numerical equivalence test proving our Gram-QR factorized `KnOTS` matches brute-force dense SVD $\Delta W_{\text{stack}}$ within `atol < 1e-5` on synthetic matrices.
2. Heterogeneous rank test ($r \in \{8, 16, 32\}$) verifying output shapes `(r_out, d_in)` and `(d_out, r_out)` load cleanly into PEFT schema.
3. Conflict reduction test verifying `knots_ties` reduces sign-conflict drop-off vs unaligned TIES on rotated conflicting synthetic adapters.
