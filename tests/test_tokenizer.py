from __future__ import annotations

from intent_lab.tokenizer import PAD, UNK, Tokenizer, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Why WAS I charged, twice?!") == ["why", "was", "i", "charged", "twice"]


def test_pad_and_unk_occupy_the_first_two_ids():
    tk = Tokenizer().fit(["hello world"])
    assert tk.itos[0] == PAD and tk.itos[1] == UNK
    assert tk.pad_id == 0 and tk.unk_id == 1


def test_encode_pads_and_truncates_to_max_length():
    tk = Tokenizer(max_length=4).fit(["a b c d e f"])
    assert len(tk.encode("a b")) == 4
    assert tk.encode("a b")[2:] == [tk.pad_id, tk.pad_id]
    assert len(tk.encode("a b c d e f")) == 4


def test_unseen_words_map_to_unk_not_to_a_new_id():
    tk = Tokenizer().fit(["known words only"])
    ids = tk.encode("known mystery")
    assert ids[1] == tk.unk_id


def test_vocabulary_is_fitted_on_training_text_only(dataset):
    """Fitting on all splits is a silent leak; this is the guard."""
    tk = Tokenizer().fit([e.text for e in dataset.train])
    assert tk.unk_rate([e.text for e in dataset.train]) == 0.0
    assert tk.unk_rate([e.text for e in dataset.test]) > 0.0, (
        "held-out phrasings should introduce unseen words; if this is 0 the "
        "split is leaking or the templates stopped differing"
    )
