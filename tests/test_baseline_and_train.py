"""End-to-end: both models train, and the comparison is apples to apples."""
from __future__ import annotations

import numpy as np

from intent_lab.baseline import train_baseline
from intent_lab.data import INTENTS
from intent_lab.train import TrainConfig, evaluate_on_test, predict_proba, train


def test_baseline_trains_and_beats_chance(dataset):
    _, report = train_baseline(dataset)
    assert report.accuracy > 1.0 / len(INTENTS)
    assert set(report.per_class) == set(INTENTS)


def test_baseline_is_deterministic(dataset):
    _, first = train_baseline(dataset, seed=3)
    _, second = train_baseline(dataset, seed=3)
    assert first.accuracy == second.accuracy


def test_transformer_learns_the_training_data(small_dataset):
    """Fit on a tiny split with dropout off — if it cannot overfit, the training
    loop or the gradient path is broken, regardless of what test accuracy says."""
    config = TrainConfig(epochs=60, patience=60, dropout=0.0, seed=1)
    model, tokenizer, history = train(small_dataset, config)
    ids = np.array(tokenizer.encode_batch([e.text for e in small_dataset.train]))
    import torch

    probabilities = predict_proba(model, torch.tensor(ids))
    predicted = probabilities.argmax(axis=1)
    actual = np.array([e.label for e in small_dataset.train])
    assert (predicted == actual).mean() > 0.85
    assert history.train_loss[-1] < history.train_loss[0]


def test_early_stopping_restores_the_best_epoch_not_the_last(dataset):
    config = TrainConfig(epochs=200, patience=5, seed=2)
    _, _, history = train(dataset, config)
    if history.stopped_early:
        assert history.best_epoch <= len(history.val_macro_f1) - 1
        assert history.val_macro_f1[history.best_epoch] == max(history.val_macro_f1)


def test_training_is_reproducible_for_a_seed(small_dataset):
    config = TrainConfig(epochs=12, patience=12, seed=5)
    first = evaluate_on_test(*train(small_dataset, config)[:2], small_dataset)
    second = evaluate_on_test(*train(small_dataset, config)[:2], small_dataset)
    assert first.accuracy == second.accuracy
    assert first.confusion == second.confusion


def test_test_report_covers_every_test_example(dataset):
    model, tokenizer, _ = train(dataset, TrainConfig(epochs=8, patience=8))
    report = evaluate_on_test(model, tokenizer, dataset)
    assert sum(sum(row) for row in report.confusion) == len(dataset.test)
