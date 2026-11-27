"""Integration test verifying KPI-RES-01 planted fault recall (100%) and FAR (<= 5%)."""

from __future__ import annotations

from pathlib import Path

from benchmarks.planted_collisions.generate_corpus import (
    evaluate_planted_recall,
    generate_planted_corpus,
)


def test_planted_corpus_100pct_recall_and_low_far(tmp_path: Path) -> None:
    corpus_dir = generate_planted_corpus(tmp_path / "planted_corpus", seed=42)
    evaluation = evaluate_planted_recall(corpus_dir)

    assert evaluation.total_faults == 7
    assert evaluation.total_controls == 3
    assert evaluation.true_positives == 7
    assert evaluation.false_negatives == 0
    assert evaluation.false_positives == 0
    assert evaluation.recall == 1.0
    assert evaluation.false_alarm_rate <= 0.05
    assert evaluation.false_alarm_rate == 0.0

    for cr in evaluation.case_results:
        assert cr.passed_expectation, f"Unexpected result on planted case '{cr.case_id}': {cr}"
