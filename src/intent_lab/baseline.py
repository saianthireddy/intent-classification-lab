"""TF-IDF + logistic regression baseline.

This exists to make the Transformer earn its place. On a few hundred short,
templated utterances a linear model over character and word n-grams is a strong
opponent — it has far fewer parameters to fit, and character n-grams degrade
gracefully on words the training split never contained, which is exactly the
failure mode a word-level model from scratch suffers here.

Reporting a from-scratch Transformer without this number would be the more
flattering choice and the less honest one.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_union

from intent_lab.data import INTENTS, Dataset
from intent_lab.evaluate import Report, evaluate


def build_baseline(seed: int = 7) -> Pipeline:
    features = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
        # Character n-grams are the reason this is competitive: an unseen word
        # still shares substrings with the training vocabulary.
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=seed)),
        ]
    )


def train_baseline(dataset: Dataset, seed: int = 7) -> tuple[Pipeline, Report]:
    """Fit on train (not train+val, so the comparison against the Transformer is fair)."""
    pipeline = build_baseline(seed)
    pipeline.fit([e.text for e in dataset.train], [e.label for e in dataset.train])
    probabilities = pipeline.predict_proba([e.text for e in dataset.test])
    y_true = np.array([e.label for e in dataset.test])
    return pipeline, evaluate(y_true, probabilities, list(INTENTS))
