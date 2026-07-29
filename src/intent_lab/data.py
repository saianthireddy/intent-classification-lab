"""A synthetic intent dataset built to be hard enough to be informative.

The temptation with generated data is to make each class lexically distinct,
which produces 1.00 accuracy and tells you nothing — the same failure mode as a
retrieval benchmark whose corpus is smaller than its top-k. So this generator
does three things on purpose:

* **Shared vocabulary across intents.** "reset", "account", "charge" and
  "order" each appear under more than one label, so a bag-of-words model cannot
  separate classes on single keywords alone.
* **Confusable pairs.** ``billing_question`` vs ``refund_request`` and
  ``password_reset`` vs ``account_access`` are deliberately close in surface
  form; most of the residual error lives there, which is the interesting part.
* **Paraphrases held out by split.** Templates are partitioned *before*
  rendering, so a phrasing seen in training never reappears verbatim in test.
  Splitting after rendering would leak, and the score would flatter the model.

Everything is seeded, so a reported number is reproducible from a clean clone.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

INTENTS = (
    "password_reset",
    "account_access",
    "billing_question",
    "refund_request",
    "order_status",
    "technical_issue",
)

# Templates are grouped so a whole phrasing family lands in exactly one split.
_TEMPLATES: dict[str, list[list[str]]] = {
    "password_reset": [
        ["i forgot my password", "i cannot remember my password", "forgot the password again"],
        ["how do i reset my password", "reset my password please", "need to reset the password"],
        ["the password reset email never arrived", "no reset email came through"],
        ["my password stopped working after the update"],
    ],
    "account_access": [
        ["i cannot log into my account", "unable to log in to my account", "cannot access my account"],
        ["my account is locked", "account got locked out", "locked out of the account"],
        ["it says my account does not exist", "the account is not recognised"],
        ["two factor is blocking me from my account"],
    ],
    "billing_question": [
        ["why was i charged twice", "there is a duplicate charge", "charged twice this month"],
        ["what is this charge on my card", "i do not recognise this charge"],
        ["can you explain my invoice", "the invoice amount looks wrong"],
        ["when does my subscription renew", "when will i be charged next"],
    ],
    "refund_request": [
        ["i want a refund for this charge", "please refund the duplicate charge"],
        ["how do i get my money back", "i would like my money back"],
        ["cancel my order and refund me", "refund the order i just placed"],
        ["this was charged by mistake, refund it"],
    ],
    "order_status": [
        ["where is my order", "my order has not arrived", "the order is still not here"],
        ["when will my order ship", "has my order shipped yet"],
        ["can you track my order", "i need a tracking number for the order"],
        ["my order says delivered but nothing arrived"],
    ],
    "technical_issue": [
        ["the app keeps crashing", "the app crashes when i open it"],
        ["i am getting an error code", "an error code appears on every upload"],
        ["the page will not load", "nothing loads on the dashboard"],
        ["the export button does nothing after the update"],
    ],
}

_PREFIXES = ("", "hi ", "hello ", "hey there ", "quick question ", "urgent ")
_SUFFIXES = ("", " please", " thanks", " asap", " any help appreciated", " can you help")


@dataclass(frozen=True)
class Example:
    text: str
    label: int

    @property
    def intent(self) -> str:
        return INTENTS[self.label]


@dataclass(frozen=True)
class Dataset:
    train: list[Example]
    val: list[Example]
    test: list[Example]

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _render(rng: random.Random, phrase: str, n: int) -> list[str]:
    out = set()
    # Bounded attempts: the prefix/suffix space is finite, so asking for more
    # variants than exist would otherwise spin forever.
    for _ in range(n * 12):
        if len(out) >= n:
            break
        out.add(f"{rng.choice(_PREFIXES)}{phrase}{rng.choice(_SUFFIXES)}".strip())
    return sorted(out)


def build_dataset(seed: int = 13, variants_per_phrase: int = 8) -> Dataset:
    """Split template *families* across train/val/test, then render variants."""
    rng = random.Random(seed)
    train: list[Example] = []
    val: list[Example] = []
    test: list[Example] = []

    for label, intent in enumerate(INTENTS):
        families = list(_TEMPLATES[intent])
        rng.shuffle(families)
        if len(families) < 3:  # pragma: no cover - guards a bad edit to _TEMPLATES
            raise ValueError(f"{intent} needs at least 3 template families to split")
        # One family each to val and test; the rest train. Held-out phrasings,
        # not held-out renderings of phrasings the model already saw.
        buckets: Sequence[tuple[list[list[str]], list[Example]]] = (
            (families[:-2], train),
            (families[-2:-1], val),
            (families[-1:], test),
        )
        for group, sink in buckets:
            for family in group:
                for phrase in family:
                    for text in _render(rng, phrase, variants_per_phrase):
                        sink.append(Example(text=text, label=label))

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return Dataset(train=train, val=val, test=test)


def label_distribution(examples: list[Example]) -> dict[str, int]:
    counts = dict.fromkeys(INTENTS, 0)
    for ex in examples:
        counts[ex.intent] += 1
    return counts


def vocabulary_overlap() -> dict[str, list[str]]:
    """Tokens that appear under more than one intent — the reason this is hard."""
    seen: dict[str, set[str]] = {}
    for intent, families in _TEMPLATES.items():
        for family in families:
            for phrase in family:
                for token in phrase.split():
                    seen.setdefault(token, set()).add(intent)
    return {tok: sorted(intents) for tok, intents in sorted(seen.items()) if len(intents) > 1}
