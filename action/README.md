# `loraspec` GitHub Marketplace Action (`action.yml`)

Scheduled for **Stage 6 Milestone M1** (per [docs/04-DESIGN-CLI-CI-BENCHMARKS.md](../docs/04-DESIGN-CLI-CI-BENCHMARKS.md) and [docs/05-IMPLEMENTATION-PLAN.md](../docs/05-IMPLEMENTATION-PLAN.md)):

- Runs **CPU-natively at `$0`** on standard `ubuntu-latest` PR runners in `< 20s`.
- Audits changed `.safetensors` LoRA checkpoints and `adapter_config.json` files for rank collapse, spectral spikes, base-model SHA drift, and Grassmann subspace collisions.
- Emits inline **SARIF 2.1.0** annotations via `github/codeql-action/upload-sarif@v3` and uploads the zero-CDN offline HTML Certification Card (`loraspec.html`).
