"""Fine-tuning a pretrained encoder on the same task and the same splits.

The from-scratch model in ``model.py`` is capped by its vocabulary: a third of
test tokens are words the training split never contained, and a randomly
initialised word embedding has nothing useful to say about them. That is the
argument for transfer learning, and this module is the experiment that tests it
against the identical splits and the identical metrics.

**This path is not exercised in CI and its numbers are not published in the
README.** It needs weights from the Hugging Face hub, and the environment these
files were written in cannot reach it, so claiming a fine-tuned score here would
be asserting something unverified. What *is* tested offline is everything around
the network: label mapping, encoding, the metric function, split integrity, and
the guard that fires when ``transformers`` is missing. The seam is
:class:`Backend` — tests inject a fake, ``run_finetune`` injects the real one.

To produce real numbers, run ``scripts/finetune_bert.py`` somewhere with hub
access and paste the output into the README table next to the other two rows.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from intent_lab.data import INTENTS, Dataset
from intent_lab.evaluate import Report, evaluate

DEFAULT_MODEL = "prajjwal1/bert-tiny"  # 4M params: a fine-tune that finishes on CPU


class Tokenizing(Protocol):
    """The slice of a Hugging Face tokenizer this module actually uses."""

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FinetuneConfig:
    model_name: str = DEFAULT_MODEL
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 5e-5
    max_length: int = 32
    seed: int = 7
    output_dir: str = "artifacts/bert-finetune"


class Backend(Protocol):
    """Everything that needs the network, behind one interface."""

    def load_tokenizer(self, model_name: str) -> Tokenizing: ...

    def load_model(self, model_name: str, n_labels: int) -> Any: ...

    def train(self, model: Any, encoded: dict[str, dict], config: FinetuneConfig,
              compute_metrics: Callable[[Any], dict[str, float]]) -> Any: ...

    def predict_proba(self, model: Any, encoded: dict) -> np.ndarray: ...


def label_maps() -> tuple[dict[str, int], dict[int, str]]:
    """id2label / label2id, in the order the rest of the project uses.

    Getting this wrong silently permutes every metric, so it comes from the same
    ``INTENTS`` tuple as the from-scratch model rather than being retyped.
    """
    label2id = {name: i for i, name in enumerate(INTENTS)}
    return label2id, {i: name for name, i in label2id.items()}


def encode_split(tokenizer: Tokenizing, examples: list, max_length: int) -> dict[str, Any]:
    encoded = tokenizer(
        [e.text for e in examples],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return {**dict(encoded), "labels": [e.label for e in examples]}


def encode_dataset(tokenizer: Tokenizing, dataset: Dataset, max_length: int) -> dict[str, dict]:
    return {
        split: encode_split(tokenizer, getattr(dataset, split), max_length)
        for split in ("train", "val", "test")
    }


def make_compute_metrics() -> Callable[[Any], dict[str, float]]:
    """Metric fn for the Trainer, using this project's own evaluate()."""

    def compute(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred[0], eval_pred[1]
        logits = np.asarray(logits, dtype=np.float64)
        # softmax over the class axis, max-shifted so large logits cannot overflow
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        report = evaluate(np.asarray(labels), probabilities, list(INTENTS))
        return {"accuracy": report.accuracy, "macro_f1": report.macro_f1, "ece": report.ece}

    return compute


def report_from_logits(logits: np.ndarray, labels: np.ndarray) -> Report:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return evaluate(np.asarray(labels), probabilities, list(INTENTS))


def transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


class HuggingFaceBackend:
    """The real backend. Imports are inside methods so importing this module
    never requires ``transformers`` to be installed."""

    def load_tokenizer(self, model_name: str) -> Tokenizing:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)

    def load_model(self, model_name: str, n_labels: int) -> Any:
        from transformers import AutoModelForSequenceClassification

        label2id, id2label = label_maps()
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_labels, id2label=id2label, label2id=label2id
        )

    def train(self, model, encoded, config, compute_metrics):  # pragma: no cover - needs hub
        import torch
        from transformers import Trainer, TrainingArguments

        class _Wrapped(torch.utils.data.Dataset):
            def __init__(self, split: dict) -> None:
                self.split = split
                self.labels = split["labels"]

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, i: int) -> dict:
                item = {k: v[i] for k, v in self.split.items() if k != "labels"}
                item["labels"] = torch.tensor(self.labels[i])
                return item

        args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            seed=config.seed,
            logging_steps=10,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=_Wrapped(encoded["train"]),
            eval_dataset=_Wrapped(encoded["val"]),
            compute_metrics=compute_metrics,
        )
        trainer.train()
        return trainer

    def predict_proba(self, trainer, encoded) -> np.ndarray:  # pragma: no cover - needs hub
        return trainer.predict(encoded).predictions


def run_finetune(
    dataset: Dataset, config: FinetuneConfig | None = None, backend: Backend | None = None
) -> Report:
    """Fine-tune and return a test report on the same splits as the other models."""
    config = config or FinetuneConfig()
    if backend is None:
        if not transformers_available():
            raise RuntimeError(
                "transformers is not installed. Install the extra with "
                "`pip install -r requirements-finetune.txt`, and note that this "
                "path needs access to the Hugging Face hub."
            )
        backend = HuggingFaceBackend()

    tokenizer = backend.load_tokenizer(config.model_name)
    encoded = encode_dataset(tokenizer, dataset, config.max_length)
    model = backend.load_model(config.model_name, len(INTENTS))
    trained = backend.train(model, encoded, config, make_compute_metrics())
    logits = backend.predict_proba(trained, encoded["test"])
    return report_from_logits(logits, np.array(encoded["test"]["labels"]))
