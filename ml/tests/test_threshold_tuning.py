import numpy as np

from ml.evaluation.tune_thresholds import (
    fbeta_curve,
    select_threshold_from_curves,
)


def test_fbeta_favours_recall_for_high_beta():
    # Point A: high precision, low recall. Point B: low precision, high recall.
    p = np.array([0.9, 0.5])
    r = np.array([0.4, 0.9])
    f1 = fbeta_curve(p, r, beta=1.0)
    f2 = fbeta_curve(p, r, beta=2.0)
    # With beta=1, A and B are comparable; with beta=2, B (high recall) wins clearly.
    assert f2[1] > f2[0]


def test_select_picks_max_fbeta_threshold():
    px = np.linspace(0.0, 1.0, 11)
    # Recall high at low conf, precision high at high conf (classic trade-off).
    r = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0])
    p = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])

    # High beta (recall-favoring) should pick a LOWER threshold than low beta.
    low_beta = select_threshold_from_curves(px, p, r, beta=1.0)
    high_beta = select_threshold_from_curves(px, p, r, beta=3.0)
    assert high_beta["threshold"] <= low_beta["threshold"]


def test_precision_floor_excludes_low_precision_points():
    px = np.linspace(0.0, 1.0, 11)
    r = np.linspace(1.0, 0.0, 11)
    p = np.linspace(0.1, 1.0, 11)
    result = select_threshold_from_curves(px, p, r, beta=2.0, precision_floor=0.6)
    # Chosen point must satisfy the precision floor.
    assert result["precision"] is None or result["precision"] >= 0.6


def test_degenerate_curve_returns_default():
    px = np.linspace(0.0, 1.0, 11)
    zeros = np.zeros(11)
    result = select_threshold_from_curves(px, zeros, zeros, beta=1.0, default=0.25)
    assert result["threshold"] == 0.25
    assert "note" in result


def test_threshold_is_clipped():
    px = np.linspace(0.0, 1.0, 11)
    r = np.zeros(11)
    r[0] = 1.0  # best point at conf 0.0 -> must be clipped up to floor
    p = np.zeros(11)
    p[0] = 1.0
    result = select_threshold_from_curves(px, p, r, beta=1.0, clip=(0.05, 0.95))
    assert result["threshold"] >= 0.05
