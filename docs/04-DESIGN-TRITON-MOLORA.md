# Stage 4 / SDD-03: `Arrow` Spectral Router & `@triton.jit` Top-$k$ `Arrow-SGMV` Kernel — `loraspec`

> [!NOTE]
> **Subsystem**: `SDD-03` (`ARCH-COMP-06` `Arrow` Router + `ARCH-COMP-07` Triton `Arrow-SGMV` & PyTorch Reference Dispatcher)
> **Linked Features**: `FEAT-09`, `FEAT-10`, `FEAT-11`, `FEAT-12`
> **Upstream Docs**: [02-FRD-PRD.md](./02-FRD-PRD.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/loraspec/kernels/router.py`, `src/loraspec/kernels/arrow_sgmv.py`, `src/loraspec/kernels/reference.py` (Target: Milestone **M2**)

---

## 1. Zero-Shot `Arrow` Prototype Extraction & Per-Token Router (`ARCH-COMP-06`)

For each target linear projection layer $\ell$ across $K$ loaded adapters ($m \in \{0, \dots, K-1\}$):
1. **Prototype Extraction** (at load time, `$0` training):
   From factorized SVD (after optional `GenKnowSub` subtraction $\Delta W_m - \lambda \Delta W_{\text{gen}}$), extract the top right singular vector $p_m^{(\ell)} = v_1^{(m, \ell)} \in \mathbb{R}^{d_{\text{in}}}$ normalized to $\|p_m^{(\ell)}\|_2 = 1$, stacked into $P^{(\ell)} \in \mathbb{R}^{K \times d_{\text{in}}}$.
2. **Per-Token Sign-Invariant Top-$k$ Gate**:
   Given flattened token activations $X \in \mathbb{R}^{T \times d_{\text{in}}}$:
   $$\hat{X} = \frac{X}{\max(\|X\|_2, \epsilon)}, \qquad Z = \left| \hat{X} \left(P^{(\ell)}\right)^T \right| \in \mathbb{R}^{T \times K}$$
   Select `top_k` expert indices $E \in \mathbb{Z}^{T \times k}$ and raw scores $Z_{\text{top}} \in \mathbb{R}^{T \times k}$; normalize:
   $$G = \text{softmax}\!\left(\frac{Z_{\text{top}}}{\tau}\right) \in \mathbb{R}^{T \times k}, \qquad H_t = -\sum_{j=0}^{k-1} G_{t,j} \log(G_{t,j} + \epsilon)$$
   Notice that because `Arrow` takes $|\hat{x}_t^T p_m|$, sign flips $(B_m, A_m) \mapsto (-B_m, -A_m)$ leave routing identically invariant.

---

## 2. Two-Stage `@triton.jit` Top-$k$ Segmented Gather-MatMul (`Arrow-SGMV`) (`ARCH-COMP-07`)

### 2.1 Why Standard `vLLM` Punica Is Insufficient
Standard `vLLM` `SGMV` maps `sequence_id -> single adapter_id`, forcing homogeneous per-request routing. `Arrow-SGMV` must evaluate **up to $k$ distinct heterogeneous-rank adapters per token** ($r_m \in \{8, 16, 32, 64\}$) in a single fused pipeline without Python loops.

### 2.2 Token-to-Expert Segment Permutation Layout
Before launching the Triton kernels on $T$ tokens with $k$ slots ($N = T \cdot k$ assignments):
1. Flatten `expert_ids = E.reshape(-1)` (`int32`, length $N$) and `slot_weights = G.reshape(-1)` (`float32`, length $N$).
2. Compute `perm = torch.argsort(expert_ids, stable=True)` (`int32`).
3. Derive contiguous GPU buffer descriptors:
   - `sorted_token_ids = perm // k` (`int32[N]`) — original token index in `X`
   - `sorted_slots = perm % k` (`int32[N]`) — Top-$k$ slot index (`0..k-1`)
   - `sorted_weights = slot_weights[perm]` (`float32[N]`)
   - `expert_counts = torch.bincount(expert_ids, minlength=K)`
   - `expert_offsets` (`int32[K+1]`, prefix sum of `expert_counts`)
   - `adapter_ranks` (`int32[K]`), `adapter_scalings` (`float32[K]`)
   - Packed adapter weight buffers `A_packed` (`[K, r_max, d_in]`) and `B_packed` (`[K, d_out, r_max]`) or pointer tables (`a_ptrs`, `b_ptrs`) avoiding wasted compute via rank masking!

### 2.3 Kernel Stage 1: `_arrow_sgmv_shrink_kernel` (`@triton.jit`)
- **Grid**: `(K, cdiv(max_seg_tokens, BLOCK_M), cdiv(r_max, BLOCK_R))` (skipping empty experts cleanly via `if seg_len == 0: return`).
- **Inner Loop**: Tiles along `d_in` (`BLOCK_K = 64`):
  - Loads token vector tile `X[token_ids, offs_k]` (`mask = (offs_m < seg_len) & (offs_k < d_in)`).
  - Loads adapter shrink tile `A_e[offs_r, offs_k]` masked by **`offs_r < rank_e`** (zero wasted work beyond rank $r_e$ when $r_e < r_{\max}$).
  - Accumulates `acc += tl.dot(x_tile, tl.trans(a_tile))` strictly in **`tl.float32`**.
- **Output**: Writes intermediate low-rank activations `H_sorted[seg_start + offs_m, offs_r]` (`float32`, shape `[N, r_max]`).
  *(Memory footprint is tiny: for $T=2048, k=2, r_{\max}=32$, `H_sorted` is only `512 KB`!)*

### 2.4 Kernel Stage 2: `_arrow_sgmv_expand_kernel` + Deterministic Slot Reduction
To avoid floating-point non-determinism and slow global `atomic_add` contention across experts writing to the same token $t$:
- **`_arrow_sgmv_expand_kernel` (`@triton.jit`)**: Computes `delta_sorted[n, :] = (sorted_weights[n] * scaling_e) * (H_sorted[n, :r_e] @ B_e^T)` strictly in `tl.float32`, masked by `offs_r < rank_e` and `offs_n < d_out`, writing directly into a slot-indexed buffer `Y_slots[token_id, slot_idx, :]` (`[T, k, d_out]`, or using a fused reduction tile when `T*k*d_out` fits comfortably, e.g., $k=2$).
- **Deterministic Final Sum**: `Y_out = Y_base + Y_slots.sum(dim=1).to(Y_base.dtype)`.
  - Zero cross-expert atomic contention!
  - Bitwise deterministic reduction across slots `0..k-1`!
  - Works seamlessly across both `Prefill` ($T \gg 1$) and `Decode` ($T = \text{batch\_size}$).

---

## 3. Unified Dispatcher & Cross-Platform Fallback (`FEAT-11`)

```python
class ArrowMoLoRADispatcher:
    def forward(
        self,
        x: torch.Tensor,              # [B, S, d_in] or [T, d_in]
        base_out: torch.Tensor,       # W_0 x already computed by cuBLAS/base layer
        prototypes: torch.Tensor,     # [K, d_in]
        a_weights: torch.Tensor,      # [K, r_max, d_in]
        b_weights: torch.Tensor,      # [K, d_out, r_max]
        ranks: torch.Tensor,          # [K]
        scalings: torch.Tensor,       # [K]
        top_k: int = 2,
        temperature: float = 1.0,
        backend: Literal["auto", "triton_cuda", "torch_reference"] = "auto",
    ) -> tuple[torch.Tensor, MoLoRARouteTelemetryBatch]: ...
```
- On CUDA (`torch.cuda.is_available()` and `triton` importable): executes `@triton.autotune`-configured `_arrow_sgmv_shrink_kernel` + `_arrow_sgmv_expand_kernel`.
- On Apple Silicon M5 (`mps`) or CPU CI (`backend="auto"`): executes exact PyTorch reference `reference.arrow_molora_reference()` with identical math and telemetry output.

---

## 4. Property-Based Equivalence & Roofline Model (`SDD-03`)

1. **Equivalence Contract**: Randomized property test suite (`tests/gpu/test_triton_parity.py`) checks:
   - Token counts $T \in \{1, 3, 17, 128, 1025, 2048\}$
   - Mixed heterogeneous ranks $r \in \{8, 16, 32, 64\}$ simultaneously in one pool
   - $k \in \{1, 2, 3, 4\}$
   - Skewed routing where $\ge 1$ expert receives `0` tokens
   - Assertions: `max_abs_err <= 1e-3` (FP16) and `<= 5e-3` (BF16) against `FP32` reference.
2. **Roofline Instrumentation (`loraspec bench`)**: Computes total moved bytes (HBM reads of $X, A, B, H$ + writes of $H, Y$) vs arithmetic FLOPs, comparing achieved `GB/s` on **NVIDIA L4 (`300 GB/s` peak)** and **NVIDIA A100 (`2039 GB/s` peak)** alongside `peft` eager Arrow speedup and peak VRAM savings.
