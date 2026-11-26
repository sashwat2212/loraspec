# Contributing to `loraspec`

Thank you for your interest in contributing to **`loraspec`** (*LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio*).

## Development Setup

`loraspec` uses [`uv`](https://docs.astral.sh/uv/) for deterministic Python `>= 3.11` dependency management.

```bash
git clone https://github.com/sashwat2212/loraspec.git
cd loraspec
uv sync --dev
uv run loraspec doctor
```

## Quality Gate (Required Before Opening a PR)

Run the exact same checks enforced in `.github/workflows/ci.yml`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -v
```

- **Formatting & Linting**: `ruff` with `line-length = 100`.
- **Static Typing**: Strict `pyright` across `src/` and `tests/` (`0` errors).
- **Testing**: Unit tests cover `.safetensors` binary edge cases, factorized QR/SVD & Grassmann principal angles, audit rules, SARIF 2.1.0, `GpuBudgetGuard`, and CLI contracts.

## Code of Conduct & Security

- Contributors agree to abide by [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).
- Report security issues privately per [SECURITY.md](./SECURITY.md).
