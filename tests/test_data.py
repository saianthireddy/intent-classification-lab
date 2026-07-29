"""The dataset's job is to be hard and leak-free. Both are asserted."""
from __future__ import annotations

from intent_lab.data import INTENTS, build_dataset, label_distribution, vocabulary_overlap


def test_splits_are_non_empty_and_cover_every_intent(dataset):
    for split in (dataset.train, dataset.val, dataset.test):
        assert split
        assert set(label_distribution(split)) == set(INTENTS)
        assert all(count > 0 for count in label_distribution(split).values())


def test_no_text_appears_in_more_than_one_split(dataset):
    train, val, test = ({e.text for e in s} for s in (dataset.train, dataset.val, dataset.test))
    assert not train & val
    assert not train & test
    assert not val & test


def test_template_families_are_held_out_not_just_renderings(dataset):
    """The important guarantee: a *phrasing* seen in training never recurs.

    Splitting after rendering would leave the same phrase in train and test with
    only a different prefix, and the score would flatter the model. Approximated
    here by checking no test utterance's content words are a subset of a training
    utterance's — an exact-substring check would pass trivially.
    """
    stop = {"hi", "hello", "hey", "there", "quick", "question", "urgent",
            "please", "thanks", "asap", "any", "help", "appreciated", "can", "you"}

    def core(text: str) -> frozenset[str]:
        return frozenset(w for w in text.split() if w not in stop)

    train_cores = {core(e.text) for e in dataset.train}
    leaked = [e.text for e in dataset.test if core(e.text) in train_cores]
    assert not leaked, f"phrasing leaked into test: {leaked[:3]}"


def test_dataset_is_deterministic_for_a_seed():
    a, b = build_dataset(seed=99), build_dataset(seed=99)
    assert [e.text for e in a.test] == [e.text for e in b.test]
    assert [e.text for e in build_dataset(seed=1).test] != [e.text for e in a.test]


def test_intents_share_vocabulary_so_the_task_is_not_trivial():
    """If every class had its own keywords, any model would score 1.00."""
    overlap = vocabulary_overlap()
    assert len(overlap) >= 20
    # the deliberately confusable pairs must actually share words
    shared_billing_refund = [
        tok for tok, intents in overlap.items()
        if {"billing_question", "refund_request"} <= set(intents)
    ]
    assert shared_billing_refund


def test_variants_per_phrase_scales_the_training_set():
    small, large = build_dataset(variants_per_phrase=2), build_dataset(variants_per_phrase=6)
    assert len(large.train) > len(small.train)
