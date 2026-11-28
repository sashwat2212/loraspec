"""Zero-GPU `DEMO_REPLAY=true` recorded L4/A100 trace loader and streaming engine."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from loraspec.core.errors import ConfigError
from loraspec.core.manifest import sha256_file
from loraspec.core.schemas import (
    AuditFinding,
    AuditReport,
    BenchmarkSuiteSummary,
    KnOTSMergeSummary,
    KnOTSPreviewRequest,
    KnOTSPreviewResponse,
    MoLoRARouteTelemetry,
    Severity,
)


@dataclass(frozen=True)
class ReplaySession:
    """In-memory multi-token L4/A100 recorded routing trace session."""

    replay_id: str
    title: str
    prompt: str
    model_name: str
    hardware_profile: str
    ttft_ms: float
    peak_vram_mb: float
    frames: tuple[MoLoRARouteTelemetry, ...]

    @property
    def mean_itl_ms(self) -> float:
        if not self.frames:
            return 0.0
        return round(sum(f.itl_ms for f in self.frames) / float(len(self.frames)), 4)

    @property
    def generated_text(self) -> str:
        return "".join(f.token_text for f in self.frames)


def _reject_non_finite_json(constant: str) -> object:
    raise ValueError(f"Non-finite JSON constant '{constant}' is prohibited.")


def resolve_default_replay_dir() -> Path:
    """Locate bundled `assets/demo_replays` across dev worktree, Docker, and installed wheels."""
    env_dir = os.environ.get("LORASPEC_REPLAY_DIR", "").strip()
    if env_dir:
        return Path(env_dir)

    candidates = [
        Path.cwd() / "assets" / "demo_replays",
        Path(__file__).resolve().parents[3] / "assets" / "demo_replays",
        Path("/app/assets/demo_replays"),
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[1]


def load_replay_trace(trace_path: Path) -> list[ReplaySession]:
    """Parse and strictly validate a recorded JSONL MoLoRA replay trace file."""
    if not trace_path.is_file():
        raise ConfigError(f"Replay trace file not found: {trace_path}")

    try:
        raw_text = trace_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Failed to read replay trace '{trace_path}': {exc}") from exc

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        raise ConfigError(f"Replay trace file is empty: {trace_path}")

    grouped_meta: dict[str, dict[str, Any]] = {}
    grouped_frames: dict[str, list[MoLoRARouteTelemetry]] = {}
    session_order: list[str] = []

    for line_no, raw_line in enumerate(lines, start=1):
        try:
            parsed: object = json.loads(raw_line, parse_constant=_reject_non_finite_json)
        except ValueError as exc:
            raise ConfigError(
                f"Malformed JSON in replay trace '{trace_path}' line {line_no}: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ConfigError(f"Replay trace '{trace_path}' line {line_no} must be a JSON object.")
        obj = cast(dict[str, Any], parsed)

        replay_id = str(obj.get("replay_id") or trace_path.stem)
        routing_raw = obj.get("routing", obj)
        if not isinstance(routing_raw, dict):
            raise ConfigError(
                f"Replay trace '{trace_path}' line {line_no} 'routing' must be an object."
            )
        try:
            frame = MoLoRARouteTelemetry.model_validate(routing_raw)
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid telemetry schema in '{trace_path}' line {line_no}: {exc}"
            ) from exc

        if (
            frame.token_index < 0
            or not math.isfinite(frame.router_entropy)
            or frame.router_entropy < 0.0
            or not math.isfinite(frame.itl_ms)
            or frame.itl_ms < 0.0
            or len(frame.top_k_adapters) != len(frame.top_k_weights)
            or not frame.top_k_adapters
        ):
            raise ConfigError(
                f"Invalid telemetry bounds in '{trace_path}' line {line_no}: "
                f"token_index={frame.token_index}."
            )

        if replay_id not in grouped_frames:
            grouped_frames[replay_id] = []
            session_order.append(replay_id)
            grouped_meta[replay_id] = {
                "title": str(obj.get("title") or replay_id.replace("_", " ").title()),
                "prompt": str(obj.get("prompt") or ""),
                "model_name": str(obj.get("model_name") or "Qwen/Qwen2.5-3B-Instruct"),
                "hardware_profile": str(obj.get("hardware_profile") or "e2e_l4_24gb"),
                "ttft_ms": float(obj.get("ttft_ms", 26.0)),
                "peak_vram_mb": float(obj.get("peak_vram_mb", 6480.0)),
            }

        prev_frames = grouped_frames[replay_id]
        if prev_frames and frame.token_index <= prev_frames[-1].token_index:
            raise ConfigError(
                f"Non-monotonic token_index in '{trace_path}' line {line_no} "
                f"for replay_id='{replay_id}': {frame.token_index} <= "
                f"{prev_frames[-1].token_index}."
            )
        prev_frames.append(frame)

    sessions: list[ReplaySession] = []
    for rid in session_order:
        meta = grouped_meta[rid]
        sessions.append(
            ReplaySession(
                replay_id=rid,
                title=cast(str, meta["title"]),
                prompt=cast(str, meta["prompt"]),
                model_name=cast(str, meta["model_name"]),
                hardware_profile=cast(str, meta["hardware_profile"]),
                ttft_ms=cast(float, meta["ttft_ms"]),
                peak_vram_mb=cast(float, meta["peak_vram_mb"]),
                frames=tuple(grouped_frames[rid]),
            )
        )
    return sessions


async def stream_replay_frames(
    session: ReplaySession,
    *,
    speed: float | Literal["instant"] = 1.0,
) -> AsyncIterator[dict[str, Any]]:
    """Yield ordered `gpu_stats`, `token`, and `done` WebSocket frames for a `ReplaySession`."""
    if speed != "instant" and (
        isinstance(speed, bool) or not math.isfinite(float(speed)) or float(speed) < 0.0
    ):
        raise ConfigError(f"Replay speed must be 'instant' or finite >= 0.0, got {speed!r}.")

    yield {
        "type": "gpu_stats",
        "replay_id": session.replay_id,
        "hardware_profile": session.hardware_profile,
        "ttft_ms": session.ttft_ms,
        "itl_ms": session.mean_itl_ms,
        "vram_mb": session.peak_vram_mb,
    }

    for frame in session.frames:
        if speed != "instant" and float(speed) > 0.0 and frame.itl_ms > 0.0:
            await asyncio.sleep(frame.itl_ms / (1000.0 * float(speed)))
        yield {
            "type": "token",
            "replay_id": session.replay_id,
            "routing": frame.model_dump(),
        }

    yield {
        "type": "done",
        "replay_id": session.replay_id,
        "total_tokens": len(session.frames),
        "ttft_ms": session.ttft_ms,
        "mean_itl_ms": session.mean_itl_ms,
        "peak_vram_mb": session.peak_vram_mb,
    }


_TOKEN_CHUNK_RE = re.compile(r"<[^>]+>|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|\S+\s*|\s+")

_DOMAIN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sql-lora", ("select", "from", "where", "group", "join", "sql", "query", "table", "sum")),
    ("rust-lora", ("fn", "pub", "use", "let", "rust", "serde", "impl", "struct", "&str", "::")),
    ("json-lora", ("json", "schema", "serialize", "payload", "script", "{", "}", "true", "false")),
    ("math-lora", ("matrix", "svd", "kalman", "joseph", "cov", "eigen", "sum", "+", "@", ".t")),
    ("python-lora", ("def", "python", "return", "import", "async", "yield", "numpy", "eye")),
)


class DemoReplayEngine:
    """Self-contained zero-GPU engine serving recorded L4/A100 audits, rooflines, and chats."""

    def __init__(self, replay_dir: Path | None = None) -> None:
        self.replay_dir = replay_dir or resolve_default_replay_dir()
        if not self.replay_dir.is_dir():
            raise ConfigError(f"Replay bundle directory not found: {self.replay_dir}")

        trace_path = self.replay_dir / "l4_mixed_domain_session.jsonl"
        sessions = load_replay_trace(trace_path)
        self._sessions: dict[str, ReplaySession] = {s.replay_id: s for s in sessions}
        self._session_order: tuple[str, ...] = tuple(s.replay_id for s in sessions)

        audit_path = self.replay_dir / "l4_audit_bundle.json"
        if not audit_path.is_file():
            raise ConfigError(f"Replay audit bundle missing: {audit_path}")
        try:
            self._audit_report = AuditReport.model_validate_json(
                audit_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ConfigError(f"Invalid replay audit bundle '{audit_path}': {exc}") from exc

        bench_path = self.replay_dir / "a100_roofline_bundle.json"
        if not bench_path.is_file():
            raise ConfigError(f"Replay benchmark bundle missing: {bench_path}")
        try:
            self._benchmark_suite = BenchmarkSuiteSummary.model_validate_json(
                bench_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ConfigError(f"Invalid replay benchmark bundle '{bench_path}': {exc}") from exc

    @property
    def audit_report(self) -> AuditReport:
        return self._audit_report

    @property
    def benchmark_suite(self) -> BenchmarkSuiteSummary:
        return self._benchmark_suite

    def list_scenarios(self) -> list[dict[str, Any]]:
        scenarios_out: list[dict[str, Any]] = []
        for rid in self._session_order:
            sc = self._sessions[rid]
            scenarios_out.append(
                {
                    "replay_id": sc.replay_id,
                    "title": sc.title,
                    "prompt": sc.prompt,
                    "model_name": sc.model_name,
                    "hardware_profile": sc.hardware_profile,
                    "num_tokens": len(sc.frames),
                    "ttft_ms": sc.ttft_ms,
                    "mean_itl_ms": sc.mean_itl_ms,
                    "peak_vram_mb": sc.peak_vram_mb,
                }
            )
        return scenarios_out

    def get_session(self, replay_id: str | None = None) -> ReplaySession:
        target = replay_id or self._session_order[0]
        if target not in self._sessions:
            raise ConfigError(
                f"Unknown replay_id '{target}'. Available: {list(self._session_order)}."
            )
        return self._sessions[target]

    def build_prompt_session(
        self,
        prompt: str,
        *,
        top_k: int = 2,
        temperature: float = 1.0,
        replay_id: str | None = None,
    ) -> ReplaySession:
        """Match recorded scenario or synthesize deterministic token routing for a prompt."""
        if replay_id and replay_id in self._sessions:
            return self._sessions[replay_id]

        norm = prompt.strip().lower()
        for rid in self._session_order:
            sc = self._sessions[rid]
            if sc.prompt.strip().lower() == norm or (rid in norm):
                return sc

        raw_chunks = _TOKEN_CHUNK_RE.findall(prompt) if prompt.strip() else ["ok"]
        if any("<script" in norm or tag in norm for tag in ("<script", "onerror=", "<img")):
            return self._sessions.get(
                "xss_adversarial_probe", self._sessions[self._session_order[0]]
            )

        k_val = max(1, min(top_k, 5))
        temp = max(temperature, 1e-3)
        adapter_pool = [a.adapter_id for a in self._audit_report.adapters] or [
            "sql-lora",
            "rust-lora",
            "json-lora",
            "math-lora",
            "python-lora",
        ]
        frames: list[MoLoRARouteTelemetry] = []
        for idx, tok in enumerate(raw_chunks):
            tok_l = tok.lower()
            logits: dict[str, float] = {aid: 0.25 for aid in adapter_pool}
            for aid, kws in _DOMAIN_KEYWORDS:
                if aid in logits and any(kw in tok_l or kw in norm for kw in kws):
                    logits[aid] += 0.55 if any(kw in tok_l for kw in kws) else 0.20
            ranked = sorted(logits.items(), key=lambda item: (-item[1], item[0]))[:k_val]
            scaled = [score / temp for _, score in ranked]
            max_s = max(scaled)
            exps = [math.exp(s - max_s) for s in scaled]
            denom = sum(exps)
            weights = [round(e / denom, 6) for e in exps]
            entropy = round(max(0.0, -sum(w * math.log(w + 1e-8) for w in weights)), 6)
            frames.append(
                MoLoRARouteTelemetry(
                    token_index=idx,
                    token_text=tok if tok.endswith(" ") else f"{tok}",
                    top_k_adapters=[aid for aid, _ in ranked],
                    top_k_weights=weights,
                    router_entropy=entropy,
                    itl_ms=10.6,
                )
            )

        return ReplaySession(
            replay_id="interactive_replay",
            title="Interactive DEMO_REPLAY Prompt",
            prompt=prompt,
            model_name="Qwen/Qwen2.5-3B-Instruct",
            hardware_profile="e2e_l4_24gb",
            ttft_ms=25.4,
            peak_vram_mb=6450.0,
            frames=tuple(frames),
        )

    def preview_knots_replay(self, req: KnOTSPreviewRequest) -> KnOTSPreviewResponse:
        """Compute deterministic KnOTS preview metrics over bundled replay spectral statistics."""
        adapter_ids = [
            p.split(":", maxsplit=1)[1] if p.startswith("replay:") else Path(p).name
            for p in req.adapter_paths
        ]
        if len(adapter_ids) < 2:
            raise ConfigError("At least 2 adapter paths are required for KnOTS preview.")

        sigmas = [2.45, 1.88, 1.49, 1.12, 0.89, 0.67, 0.48, 0.31, 0.22, 0.15, 0.11, 0.08]
        total_e = sum(s * s for s in sigmas)
        eff_bins = min(len(sigmas), max(1, round(req.target_rank / 4.0)))
        kept_e = sum(s * s for s in sigmas[:eff_bins])
        # Smooth continuous interpolation with target_rank so sliders feel live from r=4..64
        rank_bonus = min(1.0, 0.62 + 0.38 * (1.0 - math.exp(-float(req.target_rank) / 14.0)))
        if req.use_gks:
            rank_bonus = min(0.995, rank_bonus + 0.015)
        energy_retained = round(min(0.999, max(kept_e / total_e, rank_bonus)), 6)
        rmse = round(max(0.0004, 0.018 * (1.0 - energy_retained)), 6)

        pre_conflict = 0.415 if req.use_gks else 0.482
        if req.method == "knots_linear":
            post_conflict = round(pre_conflict * 0.68, 6)
        elif req.method == "knots_dare_ties":
            drop_factor = 1.0 - 0.15 * float(req.dare_drop_rate)
            post_conflict = round(pre_conflict * 0.24 * drop_factor, 6)
        else:
            density_factor = 0.88 + 0.24 * float(req.ties_density)
            post_conflict = round(pre_conflict * 0.22 * density_factor, 6)

        reduction_pct = round(((pre_conflict - post_conflict) / pre_conflict) * 100.0, 4)
        findings: list[AuditFinding] = []
        if energy_retained < 0.90:
            findings.append(
                AuditFinding(
                    rule_id="RULE-KNOTS-001",
                    title="KnOTS merged rank loses excessive singular energy",
                    severity=Severity.MEDIUM,
                    tier=1,
                    source="loraspec",
                    file_path=str((self.replay_dir / "l4_audit_bundle.json").name),
                    evidence=(
                        f"Retained singular energy {energy_retained:.4f} < 0.9000 at "
                        f"target_rank={req.target_rank}."
                    ),
                    remediation=(
                        "Increase target_rank to >= 20 or switch to Per-Token Top-k MoLoRA."
                    ),
                )
            )

        digest = sha256_file(self.replay_dir / "l4_audit_bundle.json")
        summary = KnOTSMergeSummary(
            method=req.method,
            adapters_merged=adapter_ids,
            target_rank=req.target_rank,
            energy_retained=energy_retained,
            mean_reconstruction_rmse=rmse,
            sign_conflict_reduction_pct=reduction_pct,
            output_sha256=digest,
        )
        return KnOTSPreviewResponse(
            summary=summary,
            pre_knots_sign_conflict_rate=pre_conflict,
            post_knots_sign_conflict_rate=post_conflict,
            findings=findings,
        )


def compute_real_knots_preview(
    adapter_paths: Sequence[Path],
    *,
    method: Literal["knots_ties", "knots_dare_ties", "knots_linear"] = "knots_ties",
    target_rank: int = 16,
    ties_density: float = 0.50,
    dare_drop_rate: float = 0.20,
    use_gks: bool = False,
    gks_adapter_path: Path | None = None,
    rng_seed: int = 42,
) -> KnOTSPreviewResponse:
    """Execute factorized in-memory KnOTS preview over disk PEFT adapters without writing files."""
    from loraspec.inspect.safetensors_reader import get_layer_scaling, load_peft_adapter
    from loraspec.knots.aligner import (
        align_subspaces_knots,
        harmonize_heterogeneous_ranks,
        scale_matrix,
    )
    from loraspec.knots.operators import apply_genknowsub, merge_aligned_factors, normalize_weights
    from loraspec.knots.serializer import (
        compute_merge_diagnostics,
        factorized_frobenius_rmse,
        reconstruct_merged_lora,
    )

    loaded = [load_peft_adapter(p) for p in adapter_paths]
    gks_loaded = load_peft_adapter(gks_adapter_path) if (use_gks and gks_adapter_path) else None
    working = (
        apply_genknowsub(loaded, gks_loaded, gks_lambda=1.0)
        if gks_loaded is not None
        else list(loaded)
    )
    layers = harmonize_heterogeneous_ranks(working, layer_policy="union")
    norm_weights = normalize_weights(len(working), None)

    layer_energies: dict[str, float] = {}
    layer_rmses: dict[str, float] = {}
    pre_conflicts: list[float] = []
    post_conflicts: list[float] = []

    for lidx, lname in enumerate(layers):
        participating = [
            (idx, item) for idx, item in enumerate(working) if lname in item.layer_factors
        ]
        if not participating:
            continue
        sub_ids = [item.spec.adapter_id for _, item in participating]
        scaled_bs = [
            scale_matrix(item.layer_factors[lname][1], get_layer_scaling(item.spec, lname))
            for _, item in participating
        ]
        a_mats = [item.layer_factors[lname][0] for _, item in participating]
        aligned = align_subspaces_knots(
            layer_name=lname,
            adapter_ids=sub_ids,
            scaled_bs=scaled_bs,
            a_mats=a_mats,
        )
        if len(participating) == 1:
            v_merged_t = aligned.aligned_v_list[0]
            post_c = 0.0
        else:
            raw_w = [norm_weights[i] for i, _ in participating]
            w_sum = sum(raw_w)
            sub_w = [w / w_sum for w in raw_w] if w_sum > 0 else None
            v_merged_t, post_c = merge_aligned_factors(
                aligned,
                method=method,
                weights=sub_w,
                ties_density=ties_density,
                dare_drop_rate=dare_drop_rate,
                rng_seed=rng_seed,
                layer_seed_offset=lidx,
            )
        b_out, a_out, energy = reconstruct_merged_lora(
            aligned.u_shared,
            aligned.sigmas,
            v_merged_t,
            target_rank=target_rank,
        )
        layer_energies[lname] = energy
        pre_conflicts.append(aligned.pre_merge_sign_conflict)
        post_conflicts.append(post_c)
        rmse_acc = 0.0
        w_acc = 0.0
        for idx, item in participating:
            a_src, b_src = item.layer_factors[lname]
            w_m = norm_weights[idx]
            rmse_acc += w_m * factorized_frobenius_rmse(
                b_out, a_out, b_src, a_src, scaling_src=get_layer_scaling(item.spec, lname)
            )
            w_acc += w_m
        layer_rmses[lname] = rmse_acc / w_acc if w_acc > 0 else 0.0

    summary, findings = compute_merge_diagnostics(
        method=method,
        adapter_ids=[item.spec.adapter_id for item in loaded],
        target_rank=target_rank,
        layer_energies=layer_energies,
        layer_rmses=layer_rmses,
        pre_conflicts=pre_conflicts,
        post_conflicts=post_conflicts,
        output_sha256="0" * 64,
        min_energy_retention=0.90,
    )
    mean_pre = sum(pre_conflicts) / float(len(pre_conflicts)) if pre_conflicts else 0.0
    mean_post = sum(post_conflicts) / float(len(post_conflicts)) if post_conflicts else 0.0
    return KnOTSPreviewResponse(
        summary=summary,
        pre_knots_sign_conflict_rate=round(mean_pre, 6),
        post_knots_sign_conflict_rate=round(mean_post, 6),
        findings=findings,
    )
