from __future__ import annotations

import numpy as np

from intent_lab.evaluate import confusion_matrix, evaluate, expected_calibration_error

LABELS = ["a", "b", "c"]


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    r = evaluate(y, p, LABELS)
    assert r.accuracy == 1.0 and r.macro_f1 == 1.0


def test_confusion_matrix_counts_true_by_predicted():
    m = confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), 3)
    assert m[0, 0] == 1 and m[0, 1] == 1 and m[1, 1] == 1


def test_a_class_never_predicted_scores_zero_not_nan():
    y = np.array([0, 1])
    p = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    r = evaluate(y, p, LABELS)
    assert r.per_class["b"]["recall"] == 0.0
    assert r.per_class["c"]["precision"] == 0.0
    assert np.isfinite(r.macro_f1)


def test_top_confusions_are_ranked_and_exclude_the_diagonal():
    y = np.array([0, 0, 0, 1])
    p = np.array([[0.1, 0.9, 0.0]] * 3 + [[0.1, 0.9, 0.0]])
    r = evaluate(y, p, LABELS)
    assert r.top_confusions()[0] == ("a", "b", 3)
    assert all(t != pred for t, pred, _ in r.top_confusions())


def test_overconfidence_and_underconfidence_both_raise_ece():
    correct = np.array([1.0, 0.0])
    over, _ = expected_calibration_error(np.array([0.99, 0.99]), correct)
    calibrated, _ = expected_calibration_error(np.array([0.55, 0.45]), correct)
    assert over > calibrated


def test_a_perfectly_calibrated_set_has_near_zero_ece():
    # 10 predictions at 0.9 confidence, 9 of them correct
    conf = np.full(10, 0.9)
    correct = np.array([1.0] * 9 + [0.0])
    ece, bins = expected_calibration_error(conf, correct)
    assert ece < 0.02
    assert len(bins) == 1 and bins[0]["count"] == 10


def test_bins_report_direction_of_miscalibration():
    ece, bins = expected_calibration_error(np.array([0.95, 0.95]), np.array([0.0, 0.0]))
    assert bins[0]["confidence"] > bins[0]["accuracy"], "should read as overconfident"
    assert ece > 0.9


def test_summary_mentions_the_headline_metrics():
    r = evaluate(np.array([0, 1]), np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]]), LABELS)
    text = r.summary()
    assert "accuracy" in text and "macro-F1" in text and "ECE" in text
