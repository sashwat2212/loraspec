# Multi-LoRA Subspace Interference, `KnOTS` Alignment & Per-Token `MoLoRA` Kernels: Complete Research & Systems Dossier (2025–2026)

> [!NOTE]
> **Purpose**: Primary-source systems, linear-algebra, and ecosystem research dossier grounding **`loraspec`** (**LoRA Spec**tral Subspace Auditor, `KnOTS` Aligner & Per-Token Top-$k$ `MoLoRA` Triton Studio). Companion documents: [00-CHARTER.md](./00-CHARTER.md), [01-BRD.md](./01-BRD.md), [02-FRD-PRD.md](./02-FRD-PRD.md), [03-ARCHITECTURE.md](./03-ARCHITECTURE.md), and [05-IMPLEMENTATION-PLAN.md](./05-IMPLEMENTATION-PLAN.md).

---

## 1. The Multi-LoRA Production Bottleneck

As post-training teams specialize open-weight models (`Qwen2.5`, `Qwen3`, `Llama-3.x`, `Gemma-3`) across narrow domains—SQL generation, Rust systems coding, strict JSON schema extraction, mathematical reasoning, tool calling, and compliance—organizations accumulate **libraries of dozens of LoRA adapters** ($\Delta W_m = s_m B_m A_m$, where $s_m = \alpha_m / r_m$ or $\alpha_m / \sqrt{r_m}$ under rsLoRA).

Deploying $K$ adapters in production forces teams into an unhappy trilemma:

| Deployment Strategy | Throughput & VRAM | Compositional Multi-Domain Quality | Verified Failure Mode in 2025–2026 |
|---|---|---|---|
| **A. Naive Static Merging** (`linear`, `TIES`, `DARE` directly in parameter space) | Best ($1\times$ base GEMM, zero extra serving overhead) | **Degrades sharply** across heterogeneous tasks | LoRA factorizations $(B_m, A_m)$ are non-unique up to invertible $r \times r$ transforms ($B_m G \cdot G^{-1} A_m$); independent training runs occupy misaligned subspaces, causing destructive sign interference and spectral cancellation (`arXiv:2410.19735`). |
| **B. Eager Unbatched Dynamic Routing** (`peft.create_arrow_model`) | **Stalls production serving** (Python per-adapter loops, unbatched gather/scatter) | High (per-token specialization) | Implemented in Hugging Face `peft` as unbatched eager PyTorch loops; lacks continuous-batched GPU kernels (`@triton.jit`) and has zero integration with `vLLM V1` paged attention / continuous batching. |
| **C. `vLLM` Punica Static Multi-LoRA Serving** (`LoRARequest`) | Fast batched SGMV / BGMV | **Cannot mix skills inside one prompt** | `vLLM`'s `PunicaWrapper` (`token_lora_indices`) pins an entire request sequence to **one** static `LoRARequest`. Mixed prompts (*"Write a Rust parser that emits JSON validated against a SQL warehouse query"*) cannot route SQL tokens to `sql-lora`, Rust tokens to `rust-lora`, and JSON tokens to `json-lora` within one decode stream. |

---

## 2. Linear-Algebraic Foundations & Literature Basis

### 2.1 Subspace Geometry & Pre-Merge Diagnostics
Let adapter $m \in \{1, \dots, K\}$ at linear projection layer $\ell$ produce a scaled low-rank weight update:

$$\Delta W_m^{(\ell)} = s_m B_m^{(\ell)} A_m^{(\ell)} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}, \qquad B_m^{(\ell)} \in \mathbb{R}^{d_{\text{out}} \times r_m}, \quad A_m^{(\ell)} \in \mathbb{R}^{r_m \times d_{\text{in}}}$$

with $s_m = \alpha_m / r_m$ (standard LoRA) or $s_m = \alpha_m / \sqrt{r_m}$ (*Rank-Stabilized LoRA*, `rsLoRA`, Kalajdzievski 2023).

1. **Factorized Rank-$r$ SVD Without $O(d^2)$ Materialization**:
   Never allocate $\Delta W_m \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$! Compute thin QR decompositions:
   $$s_m B_m = Q_B R_B \;\; (Q_B \in \mathbb{R}^{d_{\text{out}} \times r_m},\; R_B \in \mathbb{R}^{r_m \times r_m}), \qquad A_m^T = Q_A R_A \;\; (Q_A \in \mathbb{R}^{d_{\text{in}} \times r_m},\; R_A \in \mathbb{R}^{r_m \times r_m})$$
   Then take the SVD of the tiny $r_m \times r_m$ core matrix $C = R_B R_A^T = U_c \Sigma V_c^T$. The exact non-zero singular triplets of $\Delta W_m$ are:
   $$U_m = Q_B U_c, \qquad \Sigma_m = \Sigma, \qquad V_m = Q_A V_c$$
   This slashes memory from $O(d_{\text{out}} d_{\text{in}})$ to $O((d_{\text{out}} + d_{\text{in}}) r_m)$ and compute from $O(d^3)$ to $O((d_{\text{out}} + d_{\text{in}}) r_m^2 + r_m^3)$—allowing CPU auditing of full 7B/32B adapter safetensors in seconds.

2. **Effective Stable Rank & Rank Collapse**:
   $$r_{\text{eff}}(\Delta W_m) = \frac{\|\Delta W_m\|_F^2}{\|\Delta W_m\|_2^2} = \frac{\sum_{i=1}^{r_m} \sigma_i^2}{\sigma_1^2} \in [1, r_m], \qquad \text{Rank Utilization} = \frac{r_{\text{eff}}(\Delta W_m)}{r_m}$$
   When a rank-64 adapter exhibits $r_{\text{eff}} \approx 1.3$ ($\text{utilization} < 0.20$), 90%+ of its VRAM and bandwidth are wasted on redundant singular directions—a clear signal to compress rank or retrain with better learning-rate/all-linear placement (*LoRA Without Regret*, Thinking Machines Lab, Sept 2025).

3. **Grassmann Principal Angles & Subspace Collision**:
   Given active orthonormal column (or row) subspaces $U_a \in \mathbb{R}^{d \times k_a}$ and $U_b \in \mathbb{R}^{d \times k_b}$ spanning the Grassmannians $\text{Gr}(k_a, d)$ and $\text{Gr}(k_b, d)$, the cosines of the **principal (canonical) angles** $0 \le \theta_1 \le \dots \le \theta_p \le \pi/2$ ($p = \min(k_a, k_b)$) are the singular values of $U_a^T U_b$:
   $$c_i = \sigma_i(U_a^T U_b) = \cos\theta_i, \qquad \text{Chordal Similarity} = \frac{1}{p} \sum_{i=1}^{p} \cos^2 \theta_i = \frac{\|U_a^T U_b\|_F^2}{p} \in [0, 1]$$
   $$\text{Geodesic Distance} = \sqrt{\sum_{i=1}^{p} \theta_i^2}$$
   - **Orthogonal Subspaces ($\theta_i \approx 90^\circ$, chordal $\approx 0$)**: Adapters modify disjoint directions; linear summation preserves both cleanly.
   - **Colliding + Sign-Conflicting Subspaces ($\theta_1 \approx 0^\circ$, high chordal similarity + opposite singular directions / entry signs)**: Naive merging cancels out task signal. Requires **`KnOTS` joint-SVD alignment** or **per-token dynamic `MoLoRA` routing**.

### 2.2 `KnOTS`: Joint-SVD Subspace Alignment (ICLR 2025)
- **Primary Source**: George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, Judy Hoffman — *"Model Merging with SVD to Tie the KnOTS"* (`arXiv:2410.19735`, ICLR 2025; reference repo `gstoica27/KnOTS`).
- **Key Insight**: LoRAs trained independently lack permutation/basis alignment across updates. `KnOTS` concatenates layer updates column-wise across all $K$ tasks:
  $$\Delta W_{\text{stack}}^{(\ell)} = \begin{bmatrix} \Delta W_1^{(\ell)} & \Delta W_2^{(\ell)} & \cdots & \Delta W_K^{(\ell)} \end{bmatrix} \in \mathbb{R}^{d_{\text{out}} \times (K \cdot d_{\text{in}})}$$
  and performs a joint SVD $\Delta W_{\text{stack}}^{(\ell)} = U \Sigma [V_1^T, V_2^T, \dots, V_K^T]$, projecting every task into a **shared orthogonal basis $U\Sigma$**. Standard merging operators (sign election, magnitude trimming, TIES/DARE) are then applied safely across the aligned $\{V_m^T\}_{m=1}^K$ matrices before reconstructing $\Delta W_{\text{merged}}^{(\ell)} = U \Sigma V_{\text{merged}}^T$ and factorizing back to LoRA $(B_{\text{out}}, A_{\text{out}})$ at target rank $r_{\text{out}}$.
- **Scalable Factorized Variant (Built into `loraspec`)**: Instead of instantiating $\Delta W_{\text{stack}} \in \mathbb{R}^{d_{\text{out}} \times K d_{\text{in}}}$, we stack $s_m B_m$ across $m=1\dots K$, run one thin QR $Q_{\text{all}} R_{\text{all}}$ on $[s_1 B_1, \dots, s_K B_K]$ (size $d_{\text{out}} \times \sum r_m$), and run thin SVD on the compressed core—yielding mathematically identical $U, \Sigma, \{V_m^T\}$ in **millisecond CPU runtime and bounded $O(d \sum r_m)$ RAM**.

### 2.3 `Arrow` Zero-Shot Spectral Routing & `GenKnowSub`
- **Primary Sources**:
  - Oleksiy Ostapenko et al. — *"Towards Modular LLMs by Building and Reusing a Library of LoRAs"* (`ICML 2024`, `Arrow` routing + Model-Based Clustering; integrated as eager PyTorch `ArrowConfig` / `create_arrow_model` in `huggingface/peft`).
  - Bagherifard et al. — *"GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction"* (`ACL 2025`, `arXiv:2505.10939`).
  - Related Multi-LoRA Composition: `MoLoRA` (*Composable Specialization via Per-Token Adapter Routing*) and `LoRASpace` (`arXiv:2610.27444`).
- **How `Arrow` Routes Without Retraining**: For each adapter $m$ and layer $\ell$, extract the dominant singular direction (prototype vector $p_{m}^{(\ell)} \in \mathbb{R}^{d_{\text{in}}}$, normalized $\|p_m^{(\ell)}\|_2 = 1$). Because LoRA representations $(B_m, A_m)$ are sign-ambiguous ($B_m A_m = (-B_m)(-A_m)$), `Arrow` routes token representation $x_t \in \mathbb{R}^{d_{\text{in}}}$ using absolute dot-product / cosine alignment:
  $$z_{t, m} = \left| \frac{x_t^T}{\|x_t\|_2} \, p_m^{(\ell)} \right|, \qquad g_t = \text{softmax}\!\left(\frac{\text{TopK}(z_t, k)}{\tau}\right)$$
- **`GenKnowSub` Purification (`use_gks=True`)**: Subtracting a shared general-domain adapter $\Delta W_{\text{gen}}$ from each specialist $\Delta W_m \leftarrow \Delta W_m - \lambda \Delta W_{\text{gen}}$ removes generic chat/formatting drift from the prototypes $p_m^{(\ell)}$, sharpening cross-adapter cosine separation.

### 2.4 Custom `@triton.jit` Per-Token Top-$k$ `Arrow-SGMV` Kernel & `vLLM V1`
- **Why Standard `vLLM` Punica Is Not Enough**: `vLLM`'s `PunicaWrapper` dispatches `SGMV` (decode) and `BGMV` (prefill) assuming **1 adapter ID per request** (`lora_ids` tensor of length `batch_size`, expanded per sequence). Supporting **per-token Top-$k$ routing across heterogeneous ranks ($r_m \in \{8, 16, 32, 64\}$)** inside continuous batches requires:
  1. Computing Top-$k$ prototype similarities $z_{t,m} = |\hat{x}_t \cdot p_m|$ per token slot $t \in [0, B\cdot S)$.
  2. Sorting/bucketing active `(token_id, slot_k)` pairs by selected `expert_id` into contiguous segments (`sorted_token_ids`, `expert_offsets`, `expert_weights`) so SRAM tiles coalesce reads across tokens routed to the same adapter.
  3. **Two-Stage Fused `@triton.jit` `Arrow-SGMV`**:
     - **Shrink Kernel**: Accumulates $h_{t,k} = x_t A_{e(t,k)}$ in FP32 registers with per-expert rank masking (`mask_r = offs_r < rank_e`), handling mixed ranks without padding every adapter to `max_rank` in compute!
     - **Expand + Weighted Reduction Kernel**: Computes $\sum_{k=1}^K g_{t,k} s_{e(t,k)} (h_{t,k} B_{e(t,k)})$ in FP32 and writes directly to $Y_t$, eliminating the $K$-separate kernel launches and Python loops of eager `peft`.
- **Zero-Fork `vLLM V1` Integration**: Registered via `[project.entry-points."vllm.general_plugins"]` so standard `pip install loraspec` + `vLLM` activates per-token Top-$k$ `MoLoRA` without maintaining a fork of `vLLM`.

---

## 3. Verified 2026 Ecosystem Whitespace & Competitive Audit

| Tool / Paper | Verified Capabilities | Verified Whitespace Owned by `loraspec` |
|---|---|---|
| **`huggingface/peft`** | Static `add_weighted_adapter` (`linear`, `ties`, `ties_svd`, `dare_ties`); eager unbatched `create_arrow_model` + `use_gks`. | No pre-merge Grassmann collision & rank-collapse auditor; no SARIF PR CI gate; no scalable `KnOTS` joint-SVD alignment workflow with reconstruction diagnostics; `Arrow` has no batched Triton kernel or `vLLM` plugin. |
| **`vLLM V1` (Punica)** | High-speed `SGMV`/`BGMV` multi-LoRA serving locked to **one static `LoRARequest` per sequence**. | Cannot switch or blend Top-$k$ adapters **token-by-token** inside a single generation; no spectral pre-flight checks; no token-level routing observatory. |
| **`gstoica27/KnOTS`** | Research reference implementation for joint-SVD LoRA alignment (`arXiv:2410.19735`). | Research scripts only—not a production `.safetensors` CLI/SDK, lacks factorized memory-safe QR+SVD auditing, SARIF CI integration, `MoLoRA` Triton serving, or interactive UI. |
| **`LoRASpace` (`arXiv:2610.27444`)** | Academic paper on pool-wide shared subspaces for multi-LoRA composition. | Not packaged as a drop-in PEFT+vLLM production auditor, CI Action, `@triton.jit` Top-$k$ SGMV runtime, and token-colorized Web Studio. |
| **`loraspec` (This Project)** | **End-to-End Multi-LoRA Lifecycle**: Pre-flight Grassmann/SVD SARIF Auditor → Scalable `KnOTS` Joint-SVD Aligner → `@triton.jit` Top-$k$ `Arrow-SGMV` Kernel → `vLLM V1` Plugin → React 19 Token-Colorized Studio (`DEMO_REPLAY=true`). | Bridges `peft` ↔ `vLLM` with from-scratch linear algebra, custom GPU kernels, `$0` CPU GitHub Action, and controlled L4/A100 empirical validation. |

---

## 4. Brand & Namespace Collision Audit (Naming Gate)

Before initializing the repository, candidate project/PyPI names were audited against **PyPI**, **GitHub**, and **arXiv** to guarantee clean trademark/namespace ownership and instant search recall:

| Candidate Name | PyPI / GitHub / arXiv Audit Result | Verdict |
|---|---|---|
| `VeLoRA` | Collides with existing arXiv/ML literature (*VeLoRA*: Variance-reduced / Vector LoRA papers). | **REJECTED** |
| `PRISM-LoRA` / `prismlora` | Collides with existing PRISM-LoRA academic/OSS references. | **REJECTED** |
| `SubLoRA` | Collides with prior SubLoRA / subspace LoRA papers on arXiv. | **REJECTED** |
| `LoRASpace` | Active NeurIPS paper & GitHub repo (`Thamirawaran/loraspace`, `arXiv:2610.27444`). | **REJECTED** |
| `LoraScope` | Existing Stable Diffusion/Civitai `.safetensors` metadata explorer utility. | **REJECTED** |
| `LSAR` / `lsa` | Generic acronyms colliding with existing PyPI packages (`lsa`) and Latent Semantic Analysis tooling. | **REJECTED** |
| **`loraspec`** | **Zero collisions** on PyPI, GitHub AI infrastructure, and arXiv; directly reads as **"LoRA Spectral / Specification Auditor"**; concise 8-character CLI (`uvx loraspec`, `pip install loraspec`). | **SELECTED (LOCKED)** |

---

## 5. Hardware & Frugal GPU Compute Economics (`₹2,000–₹3,000` Ceiling)

| Tier / Environment | Hardware & Unit Cost (Incl. 18% GST) | Workload & Cost Allocation |
|---|---|---|
| **Local Dev & CI** | Apple M5 (16 GB Unified Memory, PyTorch MPS) + GitHub `ubuntu-latest` (`₹0`) | Pure-Python/NumPy/MPS spectral math, `.safetensors` auditing, `KnOTS` alignment tests, SARIF/HTML reporting, FastAPI + React 19 UI, `--smoke` dry runs (**₹0**). |
| **Cloud L4 Development & Ablation** | E2E Networks 1× NVIDIA L4 24 GB (`₹49/hr + 18% GST = ₹57.82/hr`) | Adapter fine-tuning/verification (`6` domain LoRAs on `Qwen2.5-3B` / `Qwen3-4B`), Triton `Arrow-SGMV` kernel parity & autotuning, 4-arm accuracy + latency ablation, recorded studio telemetry (~20–26 hrs = **₹1,156–₹1,503**). |
| **Cloud A100 Roofline Pass** | E2E Networks 1× NVIDIA A100 80 GB (`₹189/hr + 18% GST = ₹223.02/hr`) | Strictly capped 1.5–2.0 hr final run measuring L4 (`300 GB/s`) vs A100 (`~2.0 TB/s`) `Arrow-SGMV` HBM bandwidth scaling & high-concurrency `vLLM V1` curves (**₹335–₹446**). |
| **Contingency Reserve** | Re-runs, seed variance check, HF artifact upload | **₹350–₹550** buffer keeping total spend strictly inside **`₹2,000–₹3,000`**. |
