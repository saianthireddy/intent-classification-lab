"""Metrics, confusion matrix, and calibration.

Accuracy alone hides two things that matter for a router: *which* classes get
confused (a billing/refund mix-up is cheap, a technical-issue/refund mix-up is
not), and whether the confidence attached to a prediction means anything. A
classifier that is 95% accurate but claims 0.99 on everything cannot support a
"escalate to a human below threshold X" policy, which is the whole reason to
classify intent in the first place.

So this module reports macro-F1 alongside accuracy, a full confusion matrix, and
expected calibration error with its reliability bins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Report:
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    labels: list[str]
    ece: float = 0.0
    bins: list[dict[str, float]] = field(default_factory=list)

    def top_confusions(self, k: int = 3) -> list[tuple[str, str, int]]:
        """The k most frequent (true, predicted) mistakes."""
        pairs = [
            (self.labels[i], self.labels[j], count)
            for i, row in enumerate(self.confusion)
            for j, count in enumerate(row)
            if i != j and count
        ]
        return sorted(pairs, key=lambda t: -t[2])[:k]

    def summary(self) -> str:
        lines = [f"accuracy {self.accuracy:.3f}   macro-F1 {self.macro_f1:.3f}   ECE {self.ece:.3f}"]
        for name, m in self.per_class.items():
            lines.append(
                f"  {name:<18} P {m['precision']:.2f}  R {m['recall']:.2f}  "
                f"F1 {m['f1']:.2f}  n={int(m['support'])}"
            )
        for true, pred, n in self.top_confusions():
            lines.append(f"  confused {true} -> {pred}: {n}")
        return "\n".join(lines)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[int(t), int(p)] += 1
    return matrix


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """ECE with equal-width bins, plus the bins themselves.

    Returning the bins matters: a single ECE number cannot tell you whether the
    model is over- or under-confident, and the direction is what you act on.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    ece = 0.0
    bins: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if not count:
            continue
        avg_conf = float(confidences[in_bin].mean())
        accuracy = float(correct[in_bin].mean())
        ece += (count / total) * abs(avg_conf - accuracy)
        bins.append({"lower": float(lo), "upper": float(hi), "count": count,
                     "confidence": avg_conf, "accuracy": accuracy})
    return float(ece), bins


def evaluate(
    y_true: np.ndarray, probabilities: np.ndarray, labels: list[str], n_bins: int = 10
) -> Report:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_pred = probabilities.argmax(axis=1)
    n_classes = len(labels)

    matrix = confusion_matrix(y_true, y_pred, n_classes)
    per_class: dict[str, dict[str, float]] = {}
    f1s = []
    for i, name in enumerate(labels):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": float(matrix[i, :].sum())}

    confidences = probabilities.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    ece, bins = expected_calibration_error(confidences, correct, n_bins)

    return Report(
        accuracy=float(correct.mean()),
        macro_f1=float(np.mean(f1s)),
        per_class=per_class,
        confusion=matrix.tolist(),
        labels=list(labels),
        ece=ece,
        bins=bins,
    )
