# Security Policy

## Scope & Threat Model

`loraspec` inspects `.safetensors` LoRA adapter checkpoints, computes factorized spectral diagnostics, executes `KnOTS` alignment, and (in Stage 6 M2–M3) dispatches custom `@triton.jit` kernels and a FastAPI + React 19 telemetry gateway. All security invariants are specified in [docs/03-ARCHITECTURE.md](./docs/03-ARCHITECTURE.md) and [docs/04-DESIGN-SPECTRAL-AUDITOR.md](./docs/04-DESIGN-SPECTRAL-AUDITOR.md).

### Mandatory Security Controls
- **Zero-Pickle Binary Parsing**: `.safetensors` files are parsed strictly from JSON headers + bounded IEEE-754 slices without `pickle.load`, `torch.load(..., weights_only=False)`, or arbitrary code execution.
- **Header & Offset Bounds Guards**: Any `.safetensors` file with header length `<= 0`, `> 100 MB`, truncated payload, or out-of-bounds tensor offsets fails fast with `AdapterParseError`.
- **XSS-Safe Reporting & Streaming**: Offline HTML reports use `Jinja2` `autoescape=True` with zero external CDN requests; the React 19 Studio renders token text via escaped DOM nodes without `dangerouslySetInnerHTML`.
- **Cost Circuit Breakers**: Cloud GPU benchmark scripts route time/spend accounting through `GpuBudgetGuard` (`--smoke` `< ₹10`, hard ceiling `<= ₹3,000`).

## Reporting a Vulnerability

Please **do not** open public issues with exploit details for vulnerabilities in `loraspec` (e.g., malformed `.safetensors` parser DoS, SARIF/HTML XSS injection, or path traversal). Report privately via GitHub Security Advisories.
