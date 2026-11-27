"""Frozen `loraspec` CLI surface (`docs/02-FRD-PRD.md` & `docs/04-DESIGN-CLI-CI-BENCHMARKS.md`)."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.table import Table

from loraspec import __version__
from loraspec.auditor.grading import (
    assign_grade,
    compute_merge_risk_score,
    synthesize_recommendations,
)
from loraspec.auditor.probes import run_spectral_and_collision_audit
from loraspec.core.config import load_config
from loraspec.core.errors import (
    EXIT_AUDIT_FAIL,
    EXIT_CONFIG_ERROR,
    EXIT_PASS,
    ConfigError,
    LoraspecError,
)
from loraspec.core.manifest import build_run_manifest, detect_accelerator_backend
from loraspec.core.schemas import AuditReport, Severity, meets_fail_threshold
from loraspec.inspect.safetensors_reader import LoadedLoRAWeights, load_peft_adapter
from loraspec.knots import merge_knots_adapters
from loraspec.knots.aligner import LayerPolicy
from loraspec.knots.operators import KnotsMethod
from loraspec.knots.serializer import ExportDtype
from loraspec.reporting.html_renderer import write_certification_html
from loraspec.reporting.json_writer import write_manifest_json, write_report_json
from loraspec.reporting.sarif_writer import write_sarif

app = typer.Typer(
    name="loraspec",
    help=(
        "LoRA Subspace Auditor (KnOTS) & Per-Token Top-k MoLoRA Triton Studio — "
        "pre-flight Grassmann/SVD collision auditor, KnOTS merger, and Arrow-SGMV serving."
    ),
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"loraspec {__version__}")
        raise typer.Exit(code=EXIT_PASS)


@app.callback()
def _root_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show loraspec version and exit.",
        ),
    ] = False,
) -> None:
    _ = version


def _optional_pkg_status(pkg: str) -> str:
    return "READY" if importlib.util.find_spec(pkg) is not None else "ABSENT"


@app.command()
def doctor() -> None:
    """Verify Python runtime, compute backends (CPU/MPS/CUDA/Triton), and optional stacks."""
    py_ok = sys.version_info >= (3, 11)
    backend = detect_accelerator_backend()

    table = Table(title=f"loraspec doctor (v{__version__})")
    table.add_column("Check", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Details")

    table.add_row(
        "Python >= 3.11",
        "PASS" if py_ok else "FAIL",
        f"{platform.python_version()} ({platform.system().lower()}/{platform.machine().lower()})",
    )
    table.add_row("Active Backend", "READY", backend)
    table.add_row("Safetensors Parser", "READY", "Built-in bounds-safe binary F32/F16/BF16 reader")
    table.add_row(
        "Extra: spectral",
        _optional_pkg_status("numpy"),
        f"numpy={_optional_pkg_status('numpy')}, safetensors={_optional_pkg_status('safetensors')}",
    )
    table.add_row(
        "Extra: kernels",
        _optional_pkg_status("torch"),
        f"torch={_optional_pkg_status('torch')}, triton={_optional_pkg_status('triton')}",
    )
    table.add_row(
        "Extra: serving",
        _optional_pkg_status("vllm"),
        f"vllm={_optional_pkg_status('vllm')}, peft={_optional_pkg_status('peft')}",
    )
    table.add_row(
        "Extra: studio",
        _optional_pkg_status("fastapi"),
        f"fastapi={_optional_pkg_status('fastapi')}, uvicorn={_optional_pkg_status('uvicorn')}",
    )
    console.print(table)
    if not py_ok:
        raise typer.Exit(code=EXIT_CONFIG_ERROR)


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="Path to PEFT adapter directory or .safetensors.")],
    json_out: Annotated[
        bool, typer.Option("--json", help="Print canonical LoRAAdapterSpec JSON.")
    ] = False,
) -> None:
    """Safely parse and display canonical LoRAAdapterSpec metadata (zero-pickle)."""
    try:
        loaded = load_peft_adapter(path)
    except LoraspecError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    spec = loaded.spec
    if json_out:
        typer.echo(spec.model_dump_json(indent=2))
        return

    table = Table(title=f"LoRAAdapterSpec: {rich_escape(spec.adapter_id)}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Adapter Format", rich_escape(spec.adapter_format.value))
    table.add_row("Base Model", rich_escape(spec.base_model_name_or_path or "unspecified"))
    table.add_row(
        "Rank / Alpha", f"r={spec.rank}, alpha={spec.lora_alpha} (scale={spec.scaling:.4f})"
    )
    table.add_row("rsLoRA / DoRA", f"rslora={spec.use_rslora}, dora={spec.use_dora}")
    table.add_row("Target Modules", rich_escape(", ".join(spec.target_modules) or "-"))
    table.add_row("Paired Layers", str(len(spec.layers)))
    table.add_row("Total LoRA Params", f"{spec.total_lora_params:,}")
    table.add_row("Weights SHA-256", spec.weights_sha256[:16] + "...")
    console.print(table)


def _parse_audit_tiers(raw_tiers: str) -> set[int]:
    parts = [p.strip() for p in raw_tiers.split(",") if p.strip()]
    if not parts:
        raise ConfigError("At least one audit tier must be specified (e.g. '0' or '0,1').")
    parsed: set[int] = set()
    for part in parts:
        if part not in {"0", "1"}:
            raise ConfigError(
                f"Unsupported audit tier '{part}' in --tiers='{raw_tiers}'. "
                "Supported Tier 0-1 values: 0, 1."
            )
        parsed.add(int(part))
    return parsed


@app.command()
def audit(
    paths: Annotated[
        list[Path],
        typer.Argument(help="One or more PEFT adapter directories or .safetensors paths."),
    ],
    config: Annotated[
        Path | None, typer.Option("--config", help="Optional path to loraspec.toml.")
    ] = None,
    tiers: Annotated[
        str, typer.Option("--tiers", help="Audit tiers to run (e.g. '0' or '0,1').")
    ] = "0,1",
    fail_on: Annotated[
        Severity | None,
        typer.Option("--fail-on", help="Severity threshold (critical|high|medium|low|info)."),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Write report.json and manifest.json to this directory."),
    ] = None,
    sarif: Annotated[
        Path | None, typer.Option("--sarif", help="Write OASIS SARIF 2.1.0 report to this path.")
    ] = None,
    html: Annotated[
        Path | None,
        typer.Option("--html", help="Write zero-CDN offline HTML Certification Card to this path."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic RNG seed for manifest.")] = 42,
) -> None:
    """Run $0 CPU spectral rank-collapse & pairwise Grassmann subspace collision audit."""
    try:
        cfg = load_config(config)
        parsed_tiers = _parse_audit_tiers(tiers)
        threshold = fail_on if fail_on is not None else cfg.project.fail_on

        loaded_list: list[LoadedLoRAWeights] = []
        used_ids: set[str] = set()
        for p in paths:
            raw = load_peft_adapter(p)
            base_id = raw.spec.adapter_id
            unique_id = base_id
            if unique_id in used_ids:
                suffix = 1
                while f"{base_id}_{suffix}" in used_ids:
                    suffix += 1
                unique_id = f"{base_id}_{suffix}"
                raw = LoadedLoRAWeights(
                    spec=raw.spec.model_copy(update={"adapter_id": unique_id}),
                    layer_factors=raw.layer_factors,
                )
            used_ids.add(unique_id)
            loaded_list.append(raw)
    except LoraspecError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    run_tier1 = 1 in parsed_tiers
    layer_metrics, collisions, findings = run_spectral_and_collision_audit(
        loaded_list,
        cfg.audit,
        run_tier1=run_tier1,
    )
    risk_score = compute_merge_risk_score(
        findings=findings,
        layer_metrics=layer_metrics,
        collisions=collisions,
    )
    grade = assign_grade(findings=findings, merge_risk_score=risk_score)
    recommendations = synthesize_recommendations(findings=findings, collisions=collisions)

    sha_map = {item.spec.adapter_id: item.spec.weights_sha256 for item in loaded_list}
    manifest = build_run_manifest(
        adapter_sha256s=sha_map,
        root_hint=loaded_list[0].spec.root_path if loaded_list else Path.cwd(),
        rng_seed=seed,
        total_gpu_cost_inr=0.0,
        cli_args=sys.argv[1:],
    )
    report_obj = AuditReport(
        manifest=manifest,
        adapters=[item.spec for item in loaded_list],
        grade=grade,
        merge_risk_score=risk_score,
        findings=findings,
        layer_metrics=layer_metrics,
        pairwise_collisions=collisions,
        recommendations=recommendations,
    )

    try:
        if out_dir is not None:
            write_report_json(report_obj, out_dir / "report.json")
            write_manifest_json(manifest, out_dir / "manifest.json")
        if sarif is not None:
            write_sarif(report_obj, sarif)
        if html is not None:
            write_certification_html(report_obj, html)
    except OSError as exc:
        typer.echo(f"Error writing output artifact: {exc}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from exc

    rec_str = ", ".join(r.value for r in recommendations)
    console.print(
        f"[bold]Grade {grade}[/bold] | MergeRiskScore={risk_score:.2f}/100.0 | "
        f"Adapters={len(loaded_list)} | Findings={len(findings)} | Policy={rich_escape(rec_str)}"
    )

    if findings:
        ftable = Table(title="Audit Findings")
        ftable.add_column("Rule", style="bold")
        ftable.add_column("Severity")
        ftable.add_column("Location")
        ftable.add_column("Title")
        for f in findings:
            loc = (
                f"{f.file_path}:{f.line_number}"
                if (f.file_path and f.line_number)
                else (f.file_path or "-")
            )
            ftable.add_row(
                rich_escape(f.rule_id),
                rich_escape(f.severity.value.upper()),
                rich_escape(loc),
                rich_escape(f.title),
            )
        console.print(ftable)

    if meets_fail_threshold(findings, threshold):
        raise typer.Exit(code=EXIT_AUDIT_FAIL)


@app.command()
def report(
    report_json: Annotated[Path, typer.Argument(help="Path to an existing report.json file.")],
    sarif: Annotated[
        Path | None,
        typer.Option("--sarif", help="Write SARIF 2.1.0 report to this path."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json-out", help="Re-serialize validated AuditReport JSON to this path."),
    ] = None,
    html: Annotated[
        Path | None,
        typer.Option("--html", help="Render zero-CDN offline HTML Certification Card."),
    ] = None,
) -> None:
    """Re-render SARIF 2.1.0, HTML Certification Card, or validated JSON from report.json."""
    if not report_json.is_file():
        typer.echo(f"Error: Report file not found: {report_json}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR)
    try:
        loaded = AuditReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.echo(f"Error: Invalid AuditReport JSON '{report_json}': {exc}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from exc

    try:
        if sarif is not None:
            write_sarif(loaded, sarif)
            typer.echo(f"Wrote SARIF 2.1.0 report to {sarif}")
        if json_out is not None:
            write_report_json(loaded, json_out)
            typer.echo(f"Wrote AuditReport JSON to {json_out}")
        if html is not None:
            write_certification_html(loaded, html)
            typer.echo(f"Wrote HTML Certification Card to {html}")
    except OSError as exc:
        typer.echo(f"Error writing output artifact: {exc}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from exc
    if sarif is None and json_out is None and html is None:
        adapter_names = ", ".join(a.adapter_id for a in loaded.adapters) or "none"
        typer.echo(
            f"Validated AuditReport [{adapter_names}] "
            f"(Grade {loaded.grade}, MergeRiskScore={loaded.merge_risk_score:.2f}, "
            f"{len(loaded.findings)} finding(s))."
        )


def _stage6_stub(command_name: str, milestone: str) -> None:
    typer.echo(
        f"`loraspec {command_name}` is scheduled for Stage 6 ({milestone}). "
        "Available in M1: `doctor`, `inspect`, `audit`, `merge`, `report`.",
        err=True,
    )
    raise typer.Exit(code=EXIT_CONFIG_ERROR)


def _parse_merge_weights(raw_weights: str | None) -> list[float] | None:
    if raw_weights is None or not raw_weights.strip():
        return None
    parts = [p.strip() for p in raw_weights.split(",") if p.strip()]
    if not parts:
        raise ConfigError("--weights cannot be empty.")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ConfigError(f"Invalid numeric value in --weights='{raw_weights}': {exc}") from exc


@app.command()
def merge(
    paths: Annotated[list[Path], typer.Argument(help="Two or more LoRA adapter directories.")],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Output PEFT adapter directory.")
    ] = Path("./merged_knots_lora"),
    method: Annotated[
        str | None, typer.Option("--method", help="knots_ties | knots_dare_ties | knots_linear")
    ] = None,
    target_rank: Annotated[
        int, typer.Option("--target-rank", help="Target output LoRA rank.")
    ] = 16,
    weights: Annotated[
        str | None,
        typer.Option("--weights", help="Comma-separated merge weights (default uniform)."),
    ] = None,
    ties_density: Annotated[
        float | None,
        typer.Option("--ties-density", help="Top-magnitude retention fraction in TIES (0..1]."),
    ] = None,
    dare_drop_rate: Annotated[
        float | None,
        typer.Option("--dare-drop-rate", help="DARE Bernoulli drop rate [0..1)."),
    ] = None,
    layer_policy: Annotated[
        str,
        typer.Option("--layer-policy", help="Disjoint layer handling: union | intersection."),
    ] = "union",
    gks_adapter: Annotated[
        Path | None,
        typer.Option("--gks-adapter", help="Optional general-knowledge LoRA for GenKnowSub."),
    ] = None,
    gks_lambda: Annotated[
        float, typer.Option("--gks-lambda", help="GenKnowSub subtraction coefficient lambda.")
    ] = 1.0,
    dtype: Annotated[
        str, typer.Option("--dtype", help="Export tensor dtype: f16 | bf16 | f32.")
    ] = "f32",
    seed: Annotated[
        int | None, typer.Option("--seed", help="Deterministic RNG seed for DARE and manifest.")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Optional path to loraspec.toml.")
    ] = None,
) -> None:
    """Align & merge heterogeneous LoRAs via factorized KnOTS joint-SVD ($0 CPU)."""
    try:
        cfg = load_config(config)
        if len(paths) < 2:
            raise ConfigError(f"`loraspec merge` requires at least 2 adapters, got {len(paths)}.")
        if target_rank < 1:
            raise ConfigError(f"--target-rank must be >= 1, got {target_rank}.")

        resolved_method_str = method if method is not None else cfg.knots.default_method
        if resolved_method_str not in {"knots_ties", "knots_dare_ties", "knots_linear"}:
            raise ConfigError(
                f"Unsupported --method='{resolved_method_str}'. "
                "Supported: knots_ties, knots_dare_ties, knots_linear."
            )
        resolved_method = cast(KnotsMethod, resolved_method_str)

        if layer_policy not in {"union", "intersection"}:
            raise ConfigError(
                f"Unsupported --layer-policy='{layer_policy}'. Supported: union, intersection."
            )
        resolved_policy = cast(LayerPolicy, layer_policy)

        dtype_norm = dtype.strip().upper()
        if dtype_norm not in {"F16", "BF16", "F32"}:
            raise ConfigError(f"Unsupported --dtype='{dtype}'. Supported: f16, bf16, f32.")
        resolved_dtype = cast(ExportDtype, dtype_norm)

        parsed_weights = _parse_merge_weights(weights)
        resolved_density = ties_density if ties_density is not None else cfg.knots.ties_density
        resolved_drop = dare_drop_rate if dare_drop_rate is not None else cfg.knots.dare_drop_rate
        resolved_seed = seed if seed is not None else cfg.knots.rng_seed

        loaded_list: list[LoadedLoRAWeights] = []
        used_ids: set[str] = set()
        for p in paths:
            raw = load_peft_adapter(p)
            base_id = raw.spec.adapter_id
            unique_id = base_id
            if unique_id in used_ids:
                suffix = 1
                while f"{base_id}_{suffix}" in used_ids:
                    suffix += 1
                unique_id = f"{base_id}_{suffix}"
                raw = LoadedLoRAWeights(
                    spec=raw.spec.model_copy(update={"adapter_id": unique_id}),
                    layer_factors=raw.layer_factors,
                )
            used_ids.add(unique_id)
            loaded_list.append(raw)

        loaded_gks = load_peft_adapter(gks_adapter) if gks_adapter is not None else None
    except LoraspecError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    # Pre-merge critical integrity gate (base mismatch or NaN/dead weights)
    _, _, pre_findings = run_spectral_and_collision_audit(
        loaded_list,
        cfg.audit,
        run_tier1=True,
    )
    blockers = [
        f
        for f in pre_findings
        if f.rule_id in {"RULE-CONF-001", "RULE-SPEC-001"} and f.severity == Severity.CRITICAL
    ]
    if blockers:
        for blk in blockers:
            typer.echo(f"Pre-merge blocker [{blk.rule_id}]: {blk.evidence}", err=True)
        raise typer.Exit(code=EXIT_AUDIT_FAIL)

    try:
        summary, findings = merge_knots_adapters(
            loaded_list,
            out_dir=out_dir,
            method=resolved_method,
            target_rank=target_rank,
            weights=parsed_weights,
            ties_density=resolved_density,
            dare_drop_rate=resolved_drop,
            layer_policy=resolved_policy,
            gks_adapter=loaded_gks,
            gks_lambda=gks_lambda,
            dtype=resolved_dtype,
            rng_seed=resolved_seed,
            min_energy_retention=cfg.knots.min_energy_retention,
            cli_args=sys.argv[1:],
        )
    except LoraspecError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc
    except OSError as exc:
        typer.echo(f"Error writing merged adapter output: {exc}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from exc

    table = Table(title=f"KnOTS Merge Complete ({rich_escape(summary.method)})")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Adapters Merged", rich_escape(", ".join(summary.adapters_merged)))
    table.add_row("Target Rank / Dtype", f"r={summary.target_rank} ({resolved_dtype})")
    table.add_row("Singular Energy Retained", f"{summary.energy_retained * 100.0:.2f}%")
    table.add_row("Mean Reconstruction RMSE", f"{summary.mean_reconstruction_rmse:.6f}")
    table.add_row("Sign Conflict Reduction", f"{summary.sign_conflict_reduction_pct:.2f}%")
    table.add_row("Output SHA-256", summary.output_sha256[:16] + "...")
    table.add_row("Output Directory", rich_escape(str(out_dir)))
    console.print(table)

    if findings:
        for f in findings:
            console.print(
                f"[yellow]{rich_escape(f.rule_id)} ({rich_escape(f.severity.value.upper())}): "
                f"{rich_escape(f.evidence)}[/yellow]"
            )

    if meets_fail_threshold(findings, cfg.project.fail_on):
        raise typer.Exit(code=EXIT_AUDIT_FAIL)


@app.command()
def bench(
    paths: Annotated[list[Path], typer.Argument(help="LoRA adapter paths to benchmark.")],
    top_k: Annotated[int, typer.Option("--top-k", help="Top-k active adapters per token.")] = 2,
    smoke: Annotated[bool, typer.Option("--smoke", help="Enforce < ₹10 smoke budget cap.")] = True,
) -> None:
    """Benchmark Arrow-SGMV Triton kernel parity, latency, and HBM bandwidth (Stage 6 M2)."""
    _ = (paths, top_k, smoke)
    _stage6_stub("bench", "M2")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8000,
    demo_replay: Annotated[
        bool, typer.Option("--demo-replay", help="Serve recorded telemetry at ₹0.")
    ] = True,
) -> None:
    """Launch FastAPI gateway + WebSocket telemetry + React 19 Studio (Stage 6 M3)."""
    _ = (host, port, demo_replay)
    _stage6_stub("serve", "M3")


@app.command()
def replay(
    trace_path: Annotated[Path, typer.Argument(help="Recorded MoLoRA telemetry JSONL trace.")],
    speed: Annotated[float, typer.Option("--speed", help="Replay speed multiplier.")] = 1.0,
) -> None:
    """Stream a recorded L4/A100 MoLoRA token-routing session (Stage 6 M3)."""
    _ = (trace_path, speed)
    _stage6_stub("replay", "M3")


def main() -> None:
    """Console entry point (`loraspec = loraspec.cli.app:main`)."""
    app()
