#!/usr/bin/env python3
"""Train the baseline and the from-scratch Transformer on identical splits.

This is the script whose output is pasted into the README table, so the numbers
there are reproducible from a clean clone: `python scripts/run_comparison.py`.
"""
from __future__ import annotations

import argparse
import time

from intent_lab.baseline import train_baseline
from intent_lab.data import build_dataset, label_distribution
from intent_lab.tokenizer import Tokenizer
from intent_lab.train import TrainConfig, evaluate_on_test, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13, help="dataset seed")
    parser.add_argument("--variants", type=int, default=8, help="renderings per phrase")
    args = parser.parse_args()

    dataset = build_dataset(seed=args.seed, variants_per_phrase=args.variants)
    print("=" * 74)
    print("DATASET")
    print("=" * 74)
    print(f"  sizes            {dataset.sizes}")
    print(f"  train per class  {label_distribution(dataset.train)}")

    tokenizer = Tokenizer().fit([e.text for e in dataset.train])
    unk = tokenizer.unk_rate([e.text for e in dataset.test])
    print(f"  train vocabulary {tokenizer.vocab_size} tokens")
    print(f"  test OOV rate    {unk:.1%}  <- template families are held out, so this is high")

    print()
    print("=" * 74)
    print("BASELINE  TF-IDF (word 1-2 + char_wb 3-5) -> logistic regression")
    print("=" * 74)
    started = time.time()
    _, baseline_report = train_baseline(dataset)
    print(f"  fit in {time.time() - started:.2f}s")
    print(baseline_report.summary())

    print()
    print("=" * 74)
    print("FROM SCRATCH  Transformer encoder (attention written by hand)")
    print("=" * 74)
    started = time.time()
    model, tokenizer, history = train(dataset, TrainConfig())
    elapsed = time.time() - started
    transformer_report = evaluate_on_test(model, tokenizer, dataset)
    print(f"  {model.parameter_count():,} parameters, trained in {elapsed:.2f}s on CPU")
    print(f"  best epoch {history.best_epoch} of {len(history.val_macro_f1)}"
          f" (early stopped: {history.stopped_early})")
    print(f"  best val macro-F1 {max(history.val_macro_f1):.3f}")
    print(transformer_report.summary())

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    delta = transformer_report.macro_f1 - baseline_report.macro_f1
    winner = "Transformer" if delta > 0 else "baseline"
    print(f"  macro-F1  baseline {baseline_report.macro_f1:.3f}"
          f"  vs  transformer {transformer_report.macro_f1:.3f}   ({winner} wins by {abs(delta):.3f})")
    print(f"  ECE       baseline {baseline_report.ece:.3f}"
          f"  vs  transformer {transformer_report.ece:.3f}")
    print()
    print("  At this data scale the linear model is the better engineering choice.")
    print("  See the README for why that is the expected result, not a bug.")


if __name__ == "__main__":
    main()
