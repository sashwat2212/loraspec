"""FastAPI Studio REST, OpenAI SSE streaming & `/ws/telemetry` WebSocket gateway."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import math
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from loraspec import __version__
from loraspec.auditor.grading import (
    assign_grade,
    compute_merge_risk_score,
    synthesize_recommendations,
)
from loraspec.auditor.probes import run_spectral_and_collision_audit
from loraspec.core.config import AuditConfig
from loraspec.core.errors import LoraspecError, RuntimeExecutionError
from loraspec.core.manifest import build_run_manifest, detect_accelerator_backend
from loraspec.core.schemas import (
    AuditReport,
    BenchmarkSuiteSummary,
    KnOTSPreviewRequest,
    LoRAAdapterSpec,
    Severity,
)
from loraspec.inspect.safetensors_reader import LoadedLoRAWeights, load_peft_adapter
from loraspec.serving.demo_replay import (
    DemoReplayEngine,
    ReplaySession,
    compute_real_knots_preview,
    resolve_default_replay_dir,
    stream_replay_frames,
)
from loraspec.serving.vllm_plugin import get_registered_plugin_state


class AuditRequestPayload(BaseModel):
    adapter_paths: list[str]
    fail_on: Severity = Severity.HIGH
    tiers: str = "0,1"


class ChatMessageInput(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "loraspec-molora"
    messages: list[ChatMessageInput] = Field(default_factory=list[ChatMessageInput])
    top_k: int = Field(default=2, ge=1, le=16)
    temperature_router: float = Field(default=1.0, gt=0.0)
    stream: bool = True
    replay_id: str | None = None


class ConnectionManager:
    """Tracks active WebSocket clients and broadcasts live token telemetry frames."""

    def __init__(self) -> None:
        self._connections: set[Any] = set()
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: Any) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections)
        if not targets:
            return
        stale: list[Any] = []
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        if stale:
            async with self._lock:
                for ws in stale:
                    self._connections.discard(ws)


class GatewayState:
    """Holds gateway runtime state across REST, SSE, and WebSocket endpoints."""

    def __init__(
        self,
        *,
        demo_replay: bool,
        replay_dir: Path | None = None,
        adapter_paths: list[Path] | None = None,
        benchmark_summary: BenchmarkSuiteSummary | None = None,
    ) -> None:
        self.demo_replay = demo_replay
        self.replay_dir = replay_dir or resolve_default_replay_dir()
        self.replay_engine: DemoReplayEngine | None = (
            DemoReplayEngine(self.replay_dir) if demo_replay else None
        )
        self.mounted_paths: list[Path] = list(adapter_paths) if adapter_paths else []
        self.mounted_loaded: list[LoadedLoRAWeights] = [
            load_peft_adapter(p) for p in self.mounted_paths
        ]
        self.benchmark_summary = benchmark_summary
        self.ws_manager = ConnectionManager()


_default_state: GatewayState | None = None


def _get_active_state() -> GatewayState:
    global _default_state
    if _default_state is None:
        env_demo = os.environ.get("DEMO_REPLAY", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        _default_state = GatewayState(demo_replay=env_demo)
    return _default_state


def _require_studio_deps() -> tuple[Any, Any, Any, Any, Any]:
    try:
        fastapi_mod = importlib.import_module("fastapi")
        cors_mod = importlib.import_module("fastapi.middleware.cors")
        responses_mod = importlib.import_module("fastapi.responses")
        staticfiles_mod = importlib.import_module("fastapi.staticfiles")
    except ModuleNotFoundError as exc:
        raise RuntimeExecutionError(
            "FastAPI Studio gateway dependencies are missing. "
            "Install `loraspec[studio]` (`pip install 'loraspec[studio]'`)."
        ) from exc
    return fastapi_mod, cors_mod, responses_mod, staticfiles_mod, importlib


def _parse_ws_speed(raw_speed: str) -> float | Literal["instant"]:
    norm = raw_speed.strip().lower()
    if norm == "instant":
        return "instant"
    try:
        val = float(norm)
    except ValueError as exc:
        raise ValueError(f"Invalid WebSocket speed '{raw_speed}'.") from exc
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"WebSocket speed must be > 0.0 or 'instant', got '{raw_speed}'.")
    return val


async def health(state: GatewayState | None = None) -> dict[str, Any]:
    st = state or _get_active_state()
    if st.demo_replay:
        backend = "replay"
        adapters_count = (
            len(st.replay_engine.audit_report.adapters) if st.replay_engine is not None else 0
        )
    else:
        plugin_state = get_registered_plugin_state()
        backend = plugin_state.backend if plugin_state is not None else detect_accelerator_backend()
        adapters_count = (
            len(plugin_state.adapter_ids) if plugin_state is not None else len(st.mounted_loaded)
        )
    return {
        "status": "ok",
        "demo_replay": st.demo_replay,
        "backend": backend,
        "loaded_adapters": adapters_count,
        "version": __version__,
    }


async def list_adapters(state: GatewayState | None = None) -> list[dict[str, Any]]:
    st = state or _get_active_state()
    if st.mounted_loaded:
        specs: list[LoRAAdapterSpec] = [item.spec for item in st.mounted_loaded]
    elif st.demo_replay and st.replay_engine is not None:
        specs = st.replay_engine.audit_report.adapters
    else:
        specs = []
    return [spec.model_dump(mode="json") for spec in specs]


async def get_default_audit(state: GatewayState | None = None) -> dict[str, Any]:
    st = state or _get_active_state()
    if st.mounted_loaded:
        layer_metrics, collisions, findings = run_spectral_and_collision_audit(
            st.mounted_loaded, AuditConfig(), run_tier1=True
        )
        score = compute_merge_risk_score(
            findings=findings, layer_metrics=layer_metrics, collisions=collisions
        )
        grade = assign_grade(findings=findings, merge_risk_score=score)
        recs = synthesize_recommendations(findings=findings, collisions=collisions)
        manifest = build_run_manifest(
            adapter_sha256s={
                item.spec.adapter_id: item.spec.weights_sha256 for item in st.mounted_loaded
            },
            cli_args=["serve"],
        )
        report = AuditReport(
            manifest=manifest,
            adapters=[item.spec for item in st.mounted_loaded],
            grade=grade,
            merge_risk_score=score,
            findings=findings,
            layer_metrics=layer_metrics,
            pairwise_collisions=collisions,
            recommendations=recs,
        )
        return report.model_dump(mode="json")

    if st.demo_replay and st.replay_engine is not None:
        return st.replay_engine.audit_report.model_dump(mode="json")

    fastapi_mod, _, _, _, _ = _require_studio_deps()
    raise fastapi_mod.HTTPException(
        status_code=503,
        detail="No mounted adapters and DEMO_REPLAY=false. Mount adapters or POST /api/v1/audit.",
    )


async def run_audit_endpoint(
    payload: AuditRequestPayload,
    state: GatewayState | None = None,
) -> dict[str, Any]:
    st = state or _get_active_state()
    fastapi_mod, _, _, _, _ = _require_studio_deps()
    if not payload.adapter_paths:
        raise fastapi_mod.HTTPException(
            status_code=400, detail="adapter_paths must contain >= 1 adapter path."
        )

    if st.demo_replay and st.replay_engine is not None and payload.adapter_paths == ["@replay"]:
        return st.replay_engine.audit_report.model_dump(mode="json")

    tiers_parts = {p.strip() for p in payload.tiers.split(",") if p.strip()}
    if not tiers_parts or not tiers_parts.issubset({"0", "1"}):
        raise fastapi_mod.HTTPException(
            status_code=400, detail=f"Invalid tiers '{payload.tiers}'. Supported: 0, 1."
        )

    try:
        loaded_list: list[LoadedLoRAWeights] = []
        used_ids: set[str] = set()
        for raw_path in payload.adapter_paths:
            raw = load_peft_adapter(Path(raw_path))
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

        layer_metrics, collisions, findings = run_spectral_and_collision_audit(
            loaded_list,
            AuditConfig(),
            run_tier1=("1" in tiers_parts),
        )
        score = compute_merge_risk_score(
            findings=findings, layer_metrics=layer_metrics, collisions=collisions
        )
        grade = assign_grade(findings=findings, merge_risk_score=score)
        recs = synthesize_recommendations(findings=findings, collisions=collisions)
        manifest = build_run_manifest(
            adapter_sha256s={
                item.spec.adapter_id: item.spec.weights_sha256 for item in loaded_list
            },
            root_hint=loaded_list[0].spec.root_path,
            cli_args=["api", "audit"],
        )
        report = AuditReport(
            manifest=manifest,
            adapters=[item.spec for item in loaded_list],
            grade=grade,
            merge_risk_score=score,
            findings=findings,
            layer_metrics=layer_metrics,
            pairwise_collisions=collisions,
            recommendations=recs,
        )
        return report.model_dump(mode="json")
    except LoraspecError as exc:
        raise fastapi_mod.HTTPException(status_code=400, detail=str(exc)) from exc


async def knots_preview(
    payload: KnOTSPreviewRequest,
    state: GatewayState | None = None,
) -> dict[str, Any]:
    st = state or _get_active_state()
    fastapi_mod, _, _, _, _ = _require_studio_deps()
    if len(payload.adapter_paths) < 2:
        raise fastapi_mod.HTTPException(
            status_code=400, detail="KnOTS preview requires >= 2 adapters."
        )

    try:
        if (
            st.demo_replay
            and st.replay_engine is not None
            and all(p.startswith("replay:") for p in payload.adapter_paths)
        ):
            resp = st.replay_engine.preview_knots_replay(payload)
            return resp.model_dump(mode="json")

        real_resp = compute_real_knots_preview(
            [Path(p) for p in payload.adapter_paths],
            method=payload.method,
            target_rank=payload.target_rank,
            ties_density=payload.ties_density,
            dare_drop_rate=payload.dare_drop_rate,
            use_gks=payload.use_gks,
            gks_adapter_path=Path(payload.gks_adapter_path) if payload.gks_adapter_path else None,
            rng_seed=payload.rng_seed,
        )
        return real_resp.model_dump(mode="json")
    except LoraspecError as exc:
        raise fastapi_mod.HTTPException(status_code=400, detail=str(exc)) from exc


async def list_benchmarks(
    hardware: str = "all",
    state: GatewayState | None = None,
) -> dict[str, Any]:
    st = state or _get_active_state()
    fastapi_mod, _, _, _, _ = _require_studio_deps()

    suite: BenchmarkSuiteSummary | None = st.benchmark_summary
    if suite is None and st.demo_replay and st.replay_engine is not None:
        suite = st.replay_engine.benchmark_suite
    if suite is None:
        raise fastapi_mod.HTTPException(
            status_code=503,
            detail="No benchmark bundle available and DEMO_REPLAY=false.",
        )

    norm = hardware.strip().lower()
    if norm not in {"all", "l4", "a30", "a100"}:
        raise fastapi_mod.HTTPException(
            status_code=400,
            detail=f"Invalid hardware filter '{hardware}'. Expected all | l4 | a30 | a100.",
        )
    if norm == "all":
        return suite.model_dump(mode="json")

    token_map = {"l4": "l4", "a30": "a30", "a100": "a100"}
    needle = token_map[norm]
    filtered = [pt for pt in suite.points if needle in pt.hardware_profile.lower()]
    filtered_suite = suite.model_copy(update={"points": filtered})
    return filtered_suite.model_dump(mode="json")


async def list_replays(state: GatewayState | None = None) -> list[dict[str, Any]]:
    st = state or _get_active_state()
    if not st.demo_replay or st.replay_engine is None:
        return []
    return st.replay_engine.list_scenarios()


def _resolve_chat_session(payload: ChatCompletionRequest, st: GatewayState) -> ReplaySession:
    prompt_text = ""
    for msg in reversed(payload.messages):
        if msg.role == "user":
            prompt_text = msg.content
            break
    if not prompt_text and payload.messages:
        prompt_text = payload.messages[-1].content

    assert st.replay_engine is not None
    return st.replay_engine.build_prompt_session(
        prompt_text,
        top_k=payload.top_k,
        temperature=payload.temperature_router,
        replay_id=payload.replay_id,
    )


async def chat_completions(
    payload: ChatCompletionRequest,
    state: GatewayState | None = None,
) -> Any:
    st = state or _get_active_state()
    fastapi_mod, _, responses_mod, _, _ = _require_studio_deps()

    if not st.demo_replay or st.replay_engine is None:
        plugin_state = get_registered_plugin_state()
        if plugin_state is None or not plugin_state.enabled:
            raise fastapi_mod.HTTPException(
                status_code=503,
                detail=(
                    "Live MoLoRA GPU runtime is unavailable and DEMO_REPLAY=false. "
                    "Boot with DEMO_REPLAY=true or configure LORASPEC_MOLORA_ENABLE=1."
                ),
            )
        raise fastapi_mod.HTTPException(
            status_code=503,
            detail="Live text tokenizer stream requires an attached vLLM AsyncLLMEngine worker.",
        )

    session = _resolve_chat_session(payload, st)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if not payload.stream:
        routing_list = [f.model_dump(mode="json") for f in session.frames]
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": session.generated_text},
                    "finish_reason": "stop",
                }
            ],
            "loraspec_routing": routing_list,
            "loraspec_stats": {
                "replay_id": session.replay_id,
                "hardware_profile": session.hardware_profile,
                "ttft_ms": session.ttft_ms,
                "mean_itl_ms": session.mean_itl_ms,
                "peak_vram_mb": session.peak_vram_mb,
            },
        }

    async def _sse_generator() -> AsyncIterator[str]:
        await st.ws_manager.broadcast_json(
            {
                "type": "gpu_stats",
                "replay_id": session.replay_id,
                "hardware_profile": session.hardware_profile,
                "ttft_ms": session.ttft_ms,
                "itl_ms": session.mean_itl_ms,
                "vram_mb": session.peak_vram_mb,
            }
        )
        for frame in session.frames:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": frame.token_text},
                        "finish_reason": None,
                    }
                ],
                "loraspec_routing": frame.model_dump(mode="json"),
            }
            await st.ws_manager.broadcast_json(
                {
                    "type": "token",
                    "replay_id": session.replay_id,
                    "routing": frame.model_dump(mode="json"),
                }
            )
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)

        done_chunk: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": payload.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        await st.ws_manager.broadcast_json(
            {
                "type": "done",
                "replay_id": session.replay_id,
                "total_tokens": len(session.frames),
                "ttft_ms": session.ttft_ms,
                "mean_itl_ms": session.mean_itl_ms,
                "peak_vram_mb": session.peak_vram_mb,
            }
        )
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return responses_mod.StreamingResponse(_sse_generator(), media_type="text/event-stream")


async def telemetry_ws(
    websocket: Any,
    replay_id: str | None = None,
    speed: str = "1.0",
    state: GatewayState | None = None,
) -> None:
    st = state or _get_active_state()
    await st.ws_manager.connect(websocket)
    try:
        try:
            parsed_speed = _parse_ws_speed(speed)
        except ValueError as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await websocket.close(code=1008)
            return

        if not st.demo_replay or st.replay_engine is None:
            plugin_state = get_registered_plugin_state()
            if plugin_state is not None:
                frames = plugin_state.ring_buffer.drain()
                for f in frames:
                    await websocket.send_json(
                        {"type": "token", "replay_id": "live", "routing": f.model_dump(mode="json")}
                    )
                await websocket.send_json(
                    {"type": "done", "replay_id": "live", "total_tokens": len(frames)}
                )
                return
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": "Live runtime unavailable and DEMO_REPLAY=false.",
                }
            )
            await websocket.close(code=1008)
            return

        try:
            session = st.replay_engine.get_session(replay_id)
        except LoraspecError as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await websocket.close(code=1008)
            return

        async for msg in stream_replay_frames(session, speed=parsed_speed):
            await websocket.send_json(msg)
    finally:
        await st.ws_manager.disconnect(websocket)


def _resolve_ui_dist_dir(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_dir() else None
    env_dist = os.environ.get("LORASPEC_UI_DIST", "").strip()
    if env_dist:
        p = Path(env_dist)
        return p if p.is_dir() else None
    candidates = [
        Path.cwd() / "ui" / "dist",
        Path(__file__).resolve().parents[3] / "ui" / "dist",
        Path("/app/ui/dist"),
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "index.html").is_file():
            return cand
    return None


def create_app(
    *,
    demo_replay: bool | None = None,
    replay_dir: Path | None = None,
    adapter_paths: list[Path] | None = None,
    benchmark_summary: BenchmarkSuiteSummary | None = None,
    ui_dist_dir: Path | None = None,
) -> Any:
    """Create configured FastAPI Studio Gateway serving REST, SSE, WebSocket & built SPA."""
    global _default_state
    fastapi_mod, cors_mod, responses_mod, staticfiles_mod, _ = _require_studio_deps()

    resolved_demo = (
        demo_replay
        if demo_replay is not None
        else os.environ.get("DEMO_REPLAY", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    state = GatewayState(
        demo_replay=resolved_demo,
        replay_dir=replay_dir,
        adapter_paths=adapter_paths,
        benchmark_summary=benchmark_summary,
    )
    _default_state = state

    app_inst = fastapi_mod.FastAPI(
        title="loraspec Studio & MoLoRA Telemetry Gateway",
        version=__version__,
    )
    app_inst.state.gateway_state = state
    app_inst.add_middleware(
        cors_mod.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app_inst.get("/health")
    async def _health_route() -> dict[str, Any]:
        return await health(state=state)

    @app_inst.get("/api/v1/adapters")
    async def _adapters_route() -> list[dict[str, Any]]:
        return await list_adapters(state=state)

    @app_inst.get("/api/v1/audit")
    async def _get_audit_route() -> dict[str, Any]:
        return await get_default_audit(state=state)

    @app_inst.post("/api/v1/audit")
    async def _post_audit_route(payload: AuditRequestPayload) -> dict[str, Any]:
        return await run_audit_endpoint(payload, state=state)

    @app_inst.post("/api/v1/knots/preview")
    async def _knots_preview_route(payload: KnOTSPreviewRequest) -> dict[str, Any]:
        return await knots_preview(payload, state=state)

    @app_inst.get("/api/v1/benchmarks")
    async def _benchmarks_route(hardware: str = "all") -> dict[str, Any]:
        return await list_benchmarks(hardware=hardware, state=state)

    @app_inst.get("/api/v1/replays")
    async def _replays_route() -> list[dict[str, Any]]:
        return await list_replays(state=state)

    @app_inst.post("/v1/chat/completions")
    async def _chat_route(payload: ChatCompletionRequest) -> Any:
        return await chat_completions(payload, state=state)

    async def _ws_route(
        websocket: Any,
        replay_id: str | None = None,
        speed: str = "1.0",
    ) -> None:
        await telemetry_ws(websocket, replay_id=replay_id, speed=speed, state=state)

    _ws_route.__annotations__ = {
        "websocket": fastapi_mod.WebSocket,
        "replay_id": str | None,
        "speed": str,
        "return": None,
    }
    app_inst.add_api_websocket_route("/ws/telemetry", _ws_route)

    dist_path = _resolve_ui_dist_dir(ui_dist_dir)
    if dist_path is not None and (dist_path / "index.html").is_file():
        assets_dir = dist_path / "assets"
        if assets_dir.is_dir():
            app_inst.mount(
                "/assets",
                staticfiles_mod.StaticFiles(directory=str(assets_dir)),
                name="ui-assets",
            )

        @app_inst.get("/", include_in_schema=False)
        async def _serve_spa_root() -> Any:
            return responses_mod.FileResponse(dist_path / "index.html")

    return app_inst


app: Any = create_app() if importlib.util.find_spec("fastapi") is not None else None
