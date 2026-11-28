"""Integration tests for FastAPI Studio REST/SSE/WS gateway and `DEMO_REPLAY=true` mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from loraspec.cli.app import app as cli_app
from loraspec.core.errors import EXIT_CONFIG_ERROR, EXIT_PASS
from loraspec.core.schemas import (
    AuditReport,
    BenchmarkSuiteSummary,
    KnOTSPreviewResponse,
    MoLoRARouteTelemetry,
)
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import write_safetensors
from loraspec.serving.demo_replay import DemoReplayEngine, load_replay_trace

runner = CliRunner()


def _create_peft_pair(tmp_path: Path) -> tuple[Path, Path]:
    specs = [("sql_lora", 0, 8), ("rust_lora", 2, 16)]
    dirs: list[Path] = []
    for name, shift, rk in specs:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        cfg = {
            "peft_type": "LORA",
            "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            "r": rk,
            "lora_alpha": float(rk * 2),
            "lora_dropout": 0.0,
            "use_rslora": False,
            "bias": "none",
            "target_modules": ["q_proj"],
        }
        (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        a_mat: Matrix = [
            [1.0 if j == (shift + r) % 32 else 0.0 for j in range(32)] for r in range(rk)
        ]
        b_mat: Matrix = [
            [0.5 if i == (shift + c) % 32 else 0.0 for c in range(rk)] for i in range(32)
        ]
        write_safetensors(
            root / "adapter_model.safetensors",
            {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": (a_mat, "F32"),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": (b_mat, "F32"),
            },
            metadata={"format": "pt"},
        )
        dirs.append(root)
    return dirs[0], dirs[1]


def test_demo_replay_zero_gpu_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `DEMO_REPLAY` boots and loads authentic bundles without torch/vllm imports."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "vllm", None)

    engine = DemoReplayEngine()
    scenarios = engine.list_scenarios()
    assert len(scenarios) >= 3
    assert any(s["replay_id"] == "rust_sql_json_etl" for s in scenarios)
    assert any(s["replay_id"] == "xss_adversarial_probe" for s in scenarios)
    assert engine.audit_report.grade in {"A", "B", "C", "D", "F"}
    assert engine.benchmark_suite.speedup_vs_peft_eager >= 2.0


def test_cli_replay_jsonl_trace(tmp_path: Path) -> None:
    """Verify `loraspec replay` validates JSONL traces cleanly and rejects corrupt lines."""
    trace_file = Path("assets/demo_replays/l4_mixed_domain_session.jsonl")
    assert trace_file.is_file()
    sessions = load_replay_trace(trace_file)
    assert len(sessions) >= 3

    res_ok = runner.invoke(
        cli_app,
        ["replay", str(trace_file), "--speed", "0", "--json-stream"],
    )
    assert res_ok.exit_code == EXIT_PASS, res_ok.stdout + res_ok.stderr
    lines = [json.loads(line) for line in res_ok.stdout.strip().splitlines() if line.strip()]
    token_events = [item for item in lines if item.get("type") == "token"]
    assert len(token_events) >= 20
    first_routing = MoLoRARouteTelemetry.model_validate(token_events[0]["routing"])
    assert first_routing.token_index == 0

    bad_trace = tmp_path / "corrupt.jsonl"
    bad_trace.write_text("{broken json\n", encoding="utf-8")
    res_bad = runner.invoke(cli_app, ["replay", str(bad_trace)])
    assert res_bad.exit_code == EXIT_CONFIG_ERROR


def test_rest_endpoints_health_audit_knots_benchmarks(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    testclient_mod: Any = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient

    from loraspec.serving.gateway import create_app

    p_a, p_b = _create_peft_pair(tmp_path)
    app_replay = create_app(demo_replay=True)
    client = TestClient(app_replay)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    health_data = r_health.json()
    assert health_data["status"] == "ok"
    assert health_data["demo_replay"] is True
    assert health_data["backend"] == "replay"
    assert health_data["loaded_adapters"] >= 5

    r_adapters = client.get("/api/v1/adapters")
    assert r_adapters.status_code == 200
    assert len(r_adapters.json()) >= 5

    r_default_audit = client.get("/api/v1/audit")
    assert r_default_audit.status_code == 200
    AuditReport.model_validate(r_default_audit.json())

    r_post_audit = client.post(
        "/api/v1/audit",
        json={"adapter_paths": [str(p_a), str(p_b)], "fail_on": "high", "tiers": "0,1"},
    )
    assert r_post_audit.status_code == 200
    rep = AuditReport.model_validate(r_post_audit.json())
    assert len(rep.adapters) == 2

    r_bad_audit = client.post("/api/v1/audit", json={"adapter_paths": ["/nonexistent/adapter"]})
    assert r_bad_audit.status_code == 400

    # KnOTS preview over both replay virtual adapters and disk PEFT adapters
    r_knots_replay = client.post(
        "/api/v1/knots/preview",
        json={
            "adapter_paths": ["replay:sql-lora", "replay:rust-lora", "replay:json-lora"],
            "method": "knots_ties",
            "target_rank": 16,
        },
    )
    assert r_knots_replay.status_code == 200
    k_resp = KnOTSPreviewResponse.model_validate(r_knots_replay.json())
    assert 0.0 < k_resp.summary.energy_retained <= 1.0

    r_knots_real = client.post(
        "/api/v1/knots/preview",
        json={
            "adapter_paths": [str(p_a), str(p_b)],
            "method": "knots_ties",
            "target_rank": 8,
        },
    )
    assert r_knots_real.status_code == 200
    KnOTSPreviewResponse.model_validate(r_knots_real.json())

    r_bench = client.get("/api/v1/benchmarks?hardware=all")
    assert r_bench.status_code == 200
    suite = BenchmarkSuiteSummary.model_validate(r_bench.json())
    assert any("l4" in pt.hardware_profile for pt in suite.points)
    assert any("a100" in pt.hardware_profile for pt in suite.points)


def test_openai_sse_chat_completions_includes_routing() -> None:
    pytest.importorskip("fastapi")
    testclient_mod: Any = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient

    from loraspec.serving.gateway import create_app

    client = TestClient(create_app(demo_replay=True))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "loraspec-molora",
            "messages": [{"role": "user", "content": "SELECT * FROM telemetry"}],
            "top_k": 2,
            "temperature_router": 1.0,
            "stream": True,
            "replay_id": "rust_sql_json_etl",
        },
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        raw_lines = [ln for ln in resp.iter_lines() if ln.startswith("data: ")]

    assert raw_lines[-1] == "data: [DONE]"
    routing_chunks = [
        json.loads(ln.removeprefix("data: "))
        for ln in raw_lines[:-1]
        if "loraspec_routing" in json.loads(ln.removeprefix("data: "))
    ]
    assert len(routing_chunks) >= 10
    for ch in routing_chunks:
        MoLoRARouteTelemetry.model_validate(ch["loraspec_routing"])

    # With DEMO_REPLAY=false and no live GPU engine, POST /v1/chat/completions returns HTTP 503
    live_absent_client = TestClient(create_app(demo_replay=False))
    resp_503 = live_absent_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp_503.status_code == 503


def test_websocket_telemetry_stream_and_invalid_replay_close() -> None:
    pytest.importorskip("fastapi")
    testclient_mod: Any = pytest.importorskip("fastapi.testclient")
    starlette_ws: Any = pytest.importorskip("starlette.websockets")
    TestClient = testclient_mod.TestClient
    WebSocketDisconnect = starlette_ws.WebSocketDisconnect

    from loraspec.serving.gateway import create_app

    client = TestClient(create_app(demo_replay=True))
    with client.websocket_connect("/ws/telemetry?replay_id=rust_sql_json_etl&speed=instant") as ws:
        msgs: list[dict[str, Any]] = []
        while True:
            msg = ws.receive_json()
            msgs.append(msg)
            if msg["type"] == "done":
                break
    assert msgs[0]["type"] == "gpu_stats"
    tokens = [m for m in msgs if m["type"] == "token"]
    assert [t["routing"]["token_index"] for t in tokens] == list(range(len(tokens)))
    for t in tokens:
        MoLoRARouteTelemetry.model_validate(t["routing"])

    # Invalid replay_id closes cleanly with WebSocket code 1008
    with client.websocket_connect("/ws/telemetry?replay_id=unknown_id&speed=instant") as ws_bad:
        err_msg = ws_bad.receive_json()
        assert err_msg["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as exc_ws:
            ws_bad.receive_json()
        assert exc_ws.value.code == 1008


def test_spa_static_serving_and_zero_xss_sinks(tmp_path: Path) -> None:
    """Verify React UI sources avoid unsafe DOM sinks and gateway serves built SPA assets."""
    ui_src_dir = Path("ui/src")
    ts_files = sorted([*ui_src_dir.rglob("*.ts"), *ui_src_dir.rglob("*.tsx")])
    assert len(ts_files) >= 6
    forbidden_sinks = ("dangerouslySetInnerHTML", ".innerHTML =", "eval(")
    for ts_file in ts_files:
        text = ts_file.read_text(encoding="utf-8")
        for sink in forbidden_sinks:
            assert sink not in text, f"Forbidden XSS sink {sink!r} found in {ts_file}"

    pytest.importorskip("fastapi")
    testclient_mod: Any = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient

    from loraspec.serving.gateway import create_app

    tmp_dist = tmp_path / "dist"
    assets_dir = tmp_dist / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dist / "index.html").write_text(
        '<!doctype html><html><body><div id="root"></div></body></html>',
        encoding="utf-8",
    )
    (assets_dir / "main.js").write_text('console.log("studio");', encoding="utf-8")

    spa_client = TestClient(create_app(demo_replay=True, ui_dist_dir=tmp_dist))
    r_root = spa_client.get("/")
    assert r_root.status_code == 200
    assert '<div id="root"></div>' in r_root.text
    r_asset = spa_client.get("/assets/main.js")
    assert r_asset.status_code == 200
    assert 'console.log("studio");' in r_asset.text

    built_index = Path("ui/dist/index.html")
    if built_index.is_file():
        built_client = TestClient(create_app(demo_replay=True, ui_dist_dir=built_index.parent))
        r_built = built_client.get("/")
        assert r_built.status_code == 200
