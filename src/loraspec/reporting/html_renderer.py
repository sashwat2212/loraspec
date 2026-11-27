"""Zero-CDN offline HTML Certification Card renderer (FRD-REQ-32, Threat T5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, PackageLoader

from loraspec.core.schemas import AuditReport, PairwiseCollisionMetrics, Severity

_GRADE_STYLES: dict[str, tuple[str, str]] = {
    "A": ("#10b981", "SAFE"),
    "B": ("#84cc16", "ADVISORY"),
    "C": ("#f59e0b", "REVIEW"),
    "D": ("#f97316", "HIGH RISK"),
    "F": ("#ef4444", "BLOCKED"),
}

_GAUGE_CIRCUMFERENCE = 251.33  # 2 * pi * 40
_MAX_HEATMAP_ADAPTERS = 14
_MAX_RANK_BARS = 40
_MAX_COLLISION_ROWS = 24


@dataclass(frozen=True)
class _HeatmapLabel:
    short_id: str
    col_x: float
    row_y: float


@dataclass(frozen=True)
class _HeatmapCell:
    x: int
    y: int
    size: int
    fill: str
    value_label: str
    tooltip: str


@dataclass(frozen=True)
class _RankBar:
    x: int
    y: int
    width: int
    height: int
    color: str
    pct_label: str
    short_label: str
    tooltip: str


@dataclass(frozen=True)
class _SeverityCount:
    name: str
    count: int


def _build_jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader("loraspec.reporting", "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _risk_gauge_color(score: float) -> str:
    if score <= 0.0:
        return "#10b981"
    if score <= 15.0:
        return "#84cc16"
    if score <= 35.0:
        return "#f59e0b"
    if score <= 60.0:
        return "#f97316"
    return "#ef4444"


def _heatmap_cell_color(chordal: float, conflict: float, *, is_diag: bool) -> str:
    if is_diag:
        return "#1e293b"
    if chordal >= 0.70 and conflict >= 0.35:
        return "#991b1b"
    if chordal >= 0.70:
        return "#b45309"
    if chordal >= 0.45:
        return "#92400e"
    if chordal >= 0.20:
        return "#1e3a5f"
    return "#065f46"


def _build_heatmap(
    report: AuditReport,
) -> tuple[list[_HeatmapLabel], list[_HeatmapCell], int, int, int]:
    adapter_ids = [a.adapter_id for a in report.adapters]
    if not adapter_ids and report.pairwise_collisions:
        seen: list[str] = []
        for pc in report.pairwise_collisions:
            if pc.adapter_a not in seen:
                seen.append(pc.adapter_a)
            if pc.adapter_b not in seen:
                seen.append(pc.adapter_b)
        adapter_ids = seen

    if len(adapter_ids) < 2:
        return [], [], 0, 0, 0

    adapter_ids = adapter_ids[:_MAX_HEATMAP_ADAPTERS]
    pair_stats: dict[tuple[str, str], tuple[float, float]] = {}
    grouped: dict[tuple[str, str], list[PairwiseCollisionMetrics]] = {}
    for pc in report.pairwise_collisions:
        grouped.setdefault((pc.adapter_a, pc.adapter_b), []).append(pc)
        grouped.setdefault((pc.adapter_b, pc.adapter_a), []).append(pc)

    for key, items in grouped.items():
        max_c = max(it.grassmann_chordal_similarity for it in items)
        max_conf = max(it.sign_conflict_rate for it in items)
        pair_stats[key] = (max_c, max_conf)

    cell_size = 56
    label_pad = 128
    top_pad = 36
    labels: list[_HeatmapLabel] = []
    cells: list[_HeatmapCell] = []

    for idx, aid in enumerate(adapter_ids):
        short = aid if len(aid) <= 14 else (aid[:12] + "..")
        labels.append(
            _HeatmapLabel(
                short_id=short,
                col_x=label_pad + idx * cell_size + cell_size / 2.0,
                row_y=top_pad + idx * cell_size + cell_size / 2.0 + 4.0,
            )
        )

    for r_idx, row_id in enumerate(adapter_ids):
        for c_idx, col_id in enumerate(adapter_ids):
            is_diag = r_idx == c_idx
            if is_diag:
                val, conf = 1.0, 0.0
                tooltip = f"{row_id} (self)"
            else:
                val, conf = pair_stats.get((row_id, col_id), (0.0, 0.0))
                tooltip = (
                    f"{row_id} vs {col_id}: max chordal={val:.3f}, max sign_conflict={conf:.3f}"
                )
            cells.append(
                _HeatmapCell(
                    x=label_pad + c_idx * cell_size,
                    y=top_pad + r_idx * cell_size,
                    size=cell_size,
                    fill=_heatmap_cell_color(val, conf, is_diag=is_diag),
                    value_label=f"{val:.2f}",
                    tooltip=tooltip,
                )
            )

    n = len(adapter_ids)
    width = max(280, label_pad + n * cell_size + 20)
    height = max(160, top_pad + n * cell_size + 16)
    return labels, cells, width, height, label_pad


def _build_rank_bars(report: AuditReport) -> tuple[list[_RankBar], int, int]:
    metrics = report.layer_metrics
    if not metrics:
        return [], 300, 106

    if len(metrics) > _MAX_RANK_BARS:
        # Prioritize lowest rank-utilization layers plus head layers while staying bounded
        sorted_by_util = sorted(
            enumerate(metrics),
            key=lambda pair: (pair[1].rank_utilization, pair[0]),
        )
        selected_indices = sorted(idx for idx, _ in sorted_by_util[:_MAX_RANK_BARS])
        shown = [metrics[idx] for idx in selected_indices]
    else:
        shown = list(metrics)

    top_y = 26
    bottom_y = 126
    span_y = bottom_y - top_y
    threshold_y = int(round(bottom_y - 0.20 * span_y))
    bar_w = 24
    step_x = 34
    bars: list[_RankBar] = []

    for idx, m in enumerate(shown):
        util = max(0.0, min(1.0, float(m.rank_utilization)))
        bar_h = max(2, int(round(util * span_y)))
        color = "#10b981" if util >= 0.45 else ("#f59e0b" if util >= 0.20 else "#ef4444")
        short_mod = m.module_type[:5] if len(m.module_type) > 5 else m.module_type
        bars.append(
            _RankBar(
                x=28 + idx * step_x,
                y=bottom_y - bar_h,
                width=bar_w,
                height=bar_h,
                color=color,
                pct_label=f"{util * 100.0:.0f}%",
                short_label=short_mod,
                tooltip=(
                    f"{m.adapter_id} | {m.layer_name}: "
                    f"r_eff={m.effective_stable_rank:.2f}/{m.rank}, "
                    f"||dW||_2={m.spectral_norm:.4f}"
                ),
            )
        )

    width = max(320, 44 + len(bars) * step_x)
    return bars, width, threshold_y


def render_certification_html(report: AuditReport) -> str:
    """Render a zero-CDN, XSS-safe offline HTML Certification Card (< 250 KB)."""
    env = _build_jinja_env()
    template = env.get_template("certification_card.html.j2")

    grade_color, grade_label = _GRADE_STYLES.get(report.grade, ("#93a4bd", "UNKNOWN"))
    clamped_risk = max(0.0, min(100.0, float(report.merge_risk_score)))
    risk_dash = round((clamped_risk / 100.0) * _GAUGE_CIRCUMFERENCE, 2)

    adapter_names = [a.adapter_id for a in report.adapters]
    header_title = ", ".join(adapter_names) if adapter_names else "LoRA Audit Report"
    if len(header_title) > 120:
        header_title = header_title[:117] + "..."

    base_models = sorted(
        {
            a.base_model_name_or_path
            for a in report.adapters
            if a.base_model_name_or_path is not None
        }
    )
    base_models_display = ", ".join(base_models) if base_models else "unspecified"

    heatmap_labels, heatmap_cells, heatmap_w, heatmap_h, heatmap_pad = _build_heatmap(report)
    rank_bars, rank_svg_w, rank_threshold_y = _build_rank_bars(report)

    top_collisions = sorted(
        report.pairwise_collisions,
        key=lambda c: (c.grassmann_chordal_similarity, c.sign_conflict_rate),
        reverse=True,
    )[:_MAX_COLLISION_ROWS]

    severity_counts: list[_SeverityCount] = []
    for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO):
        cnt = sum(1 for f in report.findings if f.severity == sev)
        if cnt > 0:
            severity_counts.append(_SeverityCount(name=sev.value, count=cnt))

    return template.render(
        report=report,
        header_title=header_title,
        base_models_display=base_models_display,
        grade_color=grade_color,
        grade_label=grade_label,
        risk_score_display=f"{clamped_risk:.1f}",
        risk_gauge_dash=risk_dash,
        risk_gauge_color=_risk_gauge_color(clamped_risk),
        heatmap_labels=heatmap_labels,
        heatmap_cells=heatmap_cells,
        heatmap_svg_width=heatmap_w,
        heatmap_svg_height=heatmap_h,
        heatmap_label_pad=heatmap_pad,
        top_collisions=top_collisions,
        rank_bars=rank_bars,
        rank_svg_width=rank_svg_w,
        rank_threshold_y=rank_threshold_y,
        severity_counts=severity_counts,
    )


def write_certification_html(report: AuditReport, out_path: Path) -> Path:
    """Render and write the zero-CDN offline HTML Certification Card to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_certification_html(report), encoding="utf-8")
    return out_path
