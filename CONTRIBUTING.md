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
- **Testing**: Unit & integration tests cover `.safetensors` binary edge cases, factorized QR/SVD & Grassmann principal angles, KnOTS subspace alignment, DoRA/rank-pattern validation, cross-backend parity, SARIF 2.1.0, offline HTML cards, `GpuBudgetGuard`, scale benchmarks, and CLI contracts.

## Hard Milestone Solidification & Zero-Defect Gate Rule

Per [`docs/00-CHARTER.md §5.1`](./docs/00-CHARTER.md#51-governance--sign-off-rules), `loraspec` enforces a **strict sequential milestone solidification policy**:

- **Zero Premature Progression**: No transition to Milestone $k+1$ (e.g., `M2` Triton/SGMV) is permitted until Milestone $k$ passes **100%** of unit, integration, scale (`KPI-ENG-01`/`NFR-02`), cross-backend numerical parity (`numpy` vs `pure_python`), CLI exit-code, CI/Action, offline HTML/SARIF security, and documentation/AST-schema parity gates with **zero open defects, silent fallbacks, or placeholders**.
- **Required Pre-Transition Evidence**: Clean `ruff check .`, `ruff format --check .`, strict `pyright`, full `pytest -v` pass, planted corpus verification (`100%` recall, `<= 5%` FAR), AST schema parity check (`src/loraspec/core/schemas.py` vs `docs/02-FRD-PRD.md §5.1A`), and clean `uv build` wheel install smoke tests (`doctor`, `inspect`, `merge`, `audit --html --sarif`, `report --html`).

## Code of Conduct & Security

- Contributors agree to abide by [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).
- Report security issues privately per [SECURITY.md](./SECURITY.md).
