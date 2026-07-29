"""Training loop with early stopping on validation macro-F1.

Two rules this enforces rather than trusts:

* **The test split is touched exactly once**, after training is finished. Model
  selection uses validation only. Selecting on test is the most common way a
  reported number becomes meaningless, and it is invisible in the output.
* **Early stopping restores the best weights.** Stopping at the last epoch after
  patience has run out means reporting a model several epochs worse than the one
  actually selected — a bug that quietly costs accuracy and is easy to miss.

Selection is on macro-F1, not accuracy: the classes are mildly imbalanced, and
accuracy would let the model coast on the larger ones.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from intent_lab.data import INTENTS, Dataset, Example
from intent_lab.evaluate import Report, evaluate
from intent_lab.model import IntentTransformer
from intent_lab.tokenizer import Tokenizer


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    patience: int = 20
    label_smoothing: float = 0.05
    seed: int = 7
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.2


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_early: bool = False


def _tensors(tokenizer: Tokenizer, examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(tokenizer.encode_batch([e.text for e in examples]), dtype=torch.long)
    labels = torch.tensor([e.label for e in examples], dtype=torch.long)
    return ids, labels


@torch.no_grad()
def predict_proba(model: nn.Module, ids: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.softmax(model(ids), dim=-1).cpu().numpy()


def _epoch_loss(model: nn.Module, criterion: nn.Module, ids: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(criterion(model(ids), y))


def train(
    dataset: Dataset, config: TrainConfig | None = None
) -> tuple[IntentTransformer, Tokenizer, History]:
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    tokenizer = Tokenizer().fit([e.text for e in dataset.train])
    train_ids, train_y = _tensors(tokenizer, dataset.train)
    val_ids, val_y = _tensors(tokenizer, dataset.val)

    model = IntentTransformer(
        vocab_size=tokenizer.vocab_size,
        n_classes=len(INTENTS),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_length=tokenizer.max_length,
        dropout=config.dropout,
        pad_id=tokenizer.pad_id,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    history = History()
    best_f1, best_state, epochs_without_gain = -1.0, None, 0
    generator = torch.Generator().manual_seed(config.seed)

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(train_ids), generator=generator)
        batch_losses = []
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(train_ids[idx]), train_y[idx])
            loss.backward()
            # Small model, small data, high LR: clipping keeps a bad batch from
            # undoing several good epochs.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss))
        scheduler.step()

        val_report = evaluate(val_y.numpy(), predict_proba(model, val_ids), list(INTENTS))
        history.train_loss.append(float(np.mean(batch_losses)))
        history.val_loss.append(_epoch_loss(model, criterion, val_ids, val_y))
        history.val_macro_f1.append(val_report.macro_f1)

        if val_report.macro_f1 > best_f1:
            best_f1, epochs_without_gain = val_report.macro_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.patience:
                history.stopped_early = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # report the selected model, not the last one
    return model, tokenizer, history


def evaluate_on_test(model: IntentTransformer, tokenizer: Tokenizer, dataset: Dataset) -> Report:
    """Evaluate on test. Call once, after training."""
    ids, y = _tensors(tokenizer, dataset.test)
    return evaluate(y.numpy(), predict_proba(model, ids), list(INTENTS))
