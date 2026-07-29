#!/usr/bin/env python3
"""Fine-tune a pretrained encoder on the same splits.

Needs `pip install -r requirements-finetune.txt` and access to the Hugging Face
hub, so it is not run in CI. Its numbers are not in the README until someone
runs this and pastes them in.
"""
from __future__ import annotations

import argparse

from intent_lab.data import build_dataset
from intent_lab.finetune import DEFAULT_MODEL, FinetuneConfig, run_finetune


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13, help="dataset seed")
    args = parser.parse_args()

    dataset = build_dataset(seed=args.seed)
    print(f"Fine-tuning {args.model} on {dataset.sizes}")
    report = run_finetune(dataset, FinetuneConfig(model_name=args.model, epochs=args.epochs))
    print(report.summary())
    print()
    print("Paste the accuracy / macro-F1 / ECE above into the README comparison table.")


if __name__ == "__main__":
    main()
