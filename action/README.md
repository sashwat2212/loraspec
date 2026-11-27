# `loraspec` GitHub Marketplace Action (`sashwat2212/loraspec@v1`)

CPU-native **`$0`** composite GitHub Action (defined in root [`action.yml`](../action.yml)) that audits PEFT LoRA `.safetensors` checkpoints and `adapter_config.json` files in `< 20s` on `ubuntu-latest`, posting **OASIS SARIF 2.1.0** PR annotations and uploading a zero-CDN **HTML Certification Card**.

## 10-Line Workflow Quickstart

```yaml
name: LoRA Subspace Gate
on: [pull_request]

permissions:
  contents: read
  actions: read
  security-events: write

jobs:
  loraspec-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: sashwat2212/loraspec@v1
        with:
          paths: |
            ./adapters/sql_lora
            ./adapters/rust_lora
          fail-on: high
```

> [!NOTE]
> `permissions.security-events: write` is required when `upload-sarif: "true"` so `github/codeql-action/upload-sarif@v3` can publish Code Scanning annotations.

## Inputs

| Input | Default | Description |
|---|---|---|
| `paths` | `"."` | Space- or newline-separated paths to adapter directories or `.safetensors` files. |
| `adapter-paths` | `""` | Optional alias for `paths` (used when `paths` is left at default `"."`). |
| `config` | `""` | Optional path to `loraspec.toml`. |
| `tiers` | `"0,1"` | Comma-separated audit tiers (`"0"` or `"0,1"`). |
| `fail-on` | `"high"` | Minimum finding severity to fail the check (`critical\|high\|medium\|low\|info`). |
| `out-dir` | `"loraspec-artifacts"` | Directory where `report.json` and `manifest.json` are written. |
| `sarif-file` | `"loraspec-artifacts/loraspec.sarif"` | Output path for OASIS SARIF 2.1.0 report. |
| `html-file` | `"loraspec-artifacts/loraspec.html"` | Output path for zero-CDN offline HTML Certification Card. |
| `upload-sarif` | `"true"` | Upload SARIF 2.1.0 results via `github/codeql-action/upload-sarif@v3`. |
| `upload-html-artifact` | `"true"` | Upload HTML Certification Card, SARIF, and JSON via `actions/upload-artifact@v4`. |
| `loraspec-package` | `"loraspec"` | Package spec or local path passed to `uvx --from`. |
| `python-version` | `"3.12"` | Python version installed via `astral-sh/setup-uv@v4`. |

## Outputs

| Output | Description |
|---|---|
| `grade` | Deterministic certification grade (`A`–`F`) extracted from `report.json`. |
| `merge-risk-score` | Bounded `MergeRiskScore` (`0.00`–`100.00`) extracted from `report.json`. |
| `sarif-file` | Path to emitted SARIF 2.1.0 file. |
| `html-file` | Path to emitted zero-CDN HTML Certification Card. |
| `report-json` | Path to emitted `AuditReport` JSON file. |
| `exit-code` | Raw exit code returned by `loraspec audit` (`0`–`5`). |
