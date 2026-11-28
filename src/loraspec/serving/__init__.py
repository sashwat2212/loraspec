"""Milestone 3 (`M3`) serving plane: vLLM V1 hook, FastAPI gateway, and zero-GPU DEMO_REPLAY."""

from __future__ import annotations

from loraspec.serving.demo_replay import (
    DemoReplayEngine,
    ReplaySession,
    load_replay_trace,
    resolve_default_replay_dir,
    stream_replay_frames,
)
from loraspec.serving.gateway import ConnectionManager, app, create_app
from loraspec.serving.vllm_plugin import (
    MoLoRALinearAdapter,
    RegisteredPluginState,
    TelemetryRingBuffer,
    capture_decode_step_telemetry,
    get_registered_plugin_state,
    register,
    reset_plugin_state,
    wrap_model_with_molora,
)

__all__ = [
    "ConnectionManager",
    "DemoReplayEngine",
    "MoLoRALinearAdapter",
    "RegisteredPluginState",
    "ReplaySession",
    "TelemetryRingBuffer",
    "app",
    "capture_decode_step_telemetry",
    "create_app",
    "get_registered_plugin_state",
    "load_replay_trace",
    "register",
    "reset_plugin_state",
    "resolve_default_replay_dir",
    "stream_replay_frames",
    "wrap_model_with_molora",
]
