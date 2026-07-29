"""The pretrained path, covered without touching the Hugging Face hub.

A fake :class:`Backend` stands in for the network. That leaves the parts most
likely to be wrong — label ordering, encoding, split integrity, the metric
function, the missing-dependency guard — fully tested, while being honest that
the actual fine-tune is not run here.
"""
from __future__ import annotations

import numpy as np
import pytest

from intent_lab.data import INTENTS
from intent_lab.finetune import (
    FinetuneConfig,
    encode_dataset,
    label_maps,
    make_compute_metrics,
    report_from_logits,
    run_finetune,
    transformers_available,
)


class FakeTokenizer:
    """Mimics the slice of a HF tokenizer this project uses."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts, truncation=True, padding="max_length", max_length=32, return_tensors=None):
        self.calls += 1
        ids = [[(hash(t) % 900) + 100] * max_length for t in texts]
        return {
            "input_ids": ids,
            "attention_mask": [[1] * max_length for _ in texts],
        }


class FakeBackend:
    """Returns logits that are correct for every example, so wiring is testable
    without asserting anything about model quality."""

    def __init__(self) -> None:
        self.trained = False
        self.seen_config: FinetuneConfig | None = None

    def load_tokenizer(self, model_name):
        self.model_name = model_name
        return FakeTokenizer()

    def load_model(self, model_name, n_labels):
        self.n_labels = n_labels
        return {"name": model_name, "labels": n_labels}

    def train(self, model, encoded, config, compute_metrics):
        self.trained = True
        self.seen_config = config
        # exercise the metric fn the way Trainer would
        logits = np.eye(len(INTENTS))[encoded["val"]["labels"]] * 10.0
        self.val_metrics = compute_metrics((logits, np.array(encoded["val"]["labels"])))
        return model

    def predict_proba(self, model, encoded):
        return np.eye(len(INTENTS))[encoded["labels"]] * 10.0


def test_label_maps_match_the_projects_intent_order():
    label2id, id2label = label_maps()
    assert [id2label[i] for i in range(len(INTENTS))] == list(INTENTS)
    assert all(label2id[name] == i for i, name in enumerate(INTENTS))


def test_encoding_preserves_split_sizes_and_labels(dataset):
    encoded = encode_dataset(FakeTokenizer(), dataset, max_length=16)
    for split in ("train", "val", "test"):
        assert len(encoded[split]["labels"]) == len(getattr(dataset, split))
        assert encoded[split]["labels"] == [e.label for e in getattr(dataset, split)]
        assert len(encoded[split]["input_ids"][0]) == 16


def test_compute_metrics_returns_the_three_headline_numbers():
    logits = np.array([[9.0, 0, 0, 0, 0, 0], [0, 9.0, 0, 0, 0, 0]])
    metrics = make_compute_metrics()((logits, np.array([0, 1])))
    assert set(metrics) == {"accuracy", "macro_f1", "ece"}
    assert metrics["accuracy"] == 1.0


def test_softmax_is_overflow_safe_on_large_logits():
    """exp() of a raw large logit overflows; the max-shift is what prevents it."""
    logits = np.array([[1e4, 0, 0, 0, 0, 0], [0, 1e4, 0, 0, 0, 0]])
    metrics = make_compute_metrics()((logits, np.array([0, 1])))
    assert np.isfinite(metrics["ece"])
    assert metrics["accuracy"] == 1.0


def test_report_from_logits_matches_label_count():
    logits = np.eye(len(INTENTS)) * 5
    report = report_from_logits(logits, np.arange(len(INTENTS)))
    assert report.accuracy == 1.0
    assert report.labels == list(INTENTS)


def test_run_finetune_drives_the_whole_pipeline_through_the_seam(dataset):
    backend = FakeBackend()
    config = FinetuneConfig(model_name="fake/model", max_length=8)
    report = run_finetune(dataset, config, backend=backend)

    assert backend.trained
    assert backend.model_name == "fake/model"
    assert backend.n_labels == len(INTENTS)
    assert backend.seen_config is config
    assert set(backend.val_metrics) == {"accuracy", "macro_f1", "ece"}
    assert sum(sum(row) for row in report.confusion) == len(dataset.test)


def test_a_missing_transformers_install_fails_with_an_actionable_message(dataset, monkeypatch):
    monkeypatch.setattr("intent_lab.finetune.transformers_available", lambda: False)
    with pytest.raises(RuntimeError, match="requirements-finetune"):
        run_finetune(dataset)


def test_availability_probe_does_not_raise():
    assert isinstance(transformers_available(), bool)
