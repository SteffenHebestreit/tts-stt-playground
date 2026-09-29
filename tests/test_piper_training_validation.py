"""Unit tests for piper-training-service/validation.py (dependency-light helpers)."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from fastapi import HTTPException


def _load_validation():
    """Load the standalone validation module without importing the torch stack."""
    path = Path(__file__).resolve().parents[1] / "piper-training-service" / "validation.py"
    spec = spec_from_file_location("pt_validation", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validation = _load_validation()


# --- safe_name (path-traversal guard) ---

@pytest.mark.parametrize("value", ["luna", "my_voice", "Voice-1", "a1b2c3", "550e8400-e29b-41d4-a716-446655440000"])
def test_safe_name_accepts_normal_names(value):
    assert validation.safe_name(value) == value


def test_safe_name_strips_whitespace():
    assert validation.safe_name("  luna  ") == "luna"


@pytest.mark.parametrize("value", [
    "../etc", "..", ".", "", "   ",
    "a/b", "a\\b", "/abs", "C:\\x",
    "foo/../bar", "name\x00", "sub/dir",
])
def test_safe_name_rejects_traversal(value):
    with pytest.raises(HTTPException) as exc:
        validation.safe_name(value, field="model_name")
    assert exc.value.status_code == 400
    assert "model_name" in exc.value.detail


# --- coerce_resume_int ---

@pytest.mark.parametrize("value,default,expected", [
    (5, 1, 5),
    ("12", 1, 12),
    (0, 7, 7),       # non-positive -> default
    (-3, 7, 7),
    (None, 9, 9),
    ("nope", 9, 9),
    (2.9, 1, 2),     # int() truncates
])
def test_coerce_resume_int(value, default, expected):
    assert validation.coerce_resume_int(value, default) == expected


# --- coerce_resume_path ---

def test_coerce_resume_path_valid():
    p = validation.coerce_resume_path("checkpoints/job/ck.pt")
    assert p is not None
    assert p.name == "ck.pt"


@pytest.mark.parametrize("value", [None, "", "   ", 123, [], {}])
def test_coerce_resume_path_invalid(value):
    assert validation.coerce_resume_path(value) is None


# --- phoneme_id_map_from_entries (must match TTSDataset ordering) ---

def test_phoneme_map_ordering_and_specials():
    entries = [{"phonemes": "ab"}, {"phonemes": "ba c"}]
    result = validation.phoneme_id_map_from_entries(entries)
    # sorted(set("abc ") ∪ {<pad>,<unk>,<start>,<end>,' '})
    expected_keys = sorted({"a", "b", "c", " ", "<pad>", "<unk>", "<start>", "<end>"})
    assert list(result.keys()) == expected_keys
    # ids are a contiguous 0..n-1 range in sorted order
    assert list(result.values()) == list(range(len(expected_keys)))


def test_phoneme_map_falls_back_to_text_field():
    result = validation.phoneme_id_map_from_entries([{"text": "xy"}])
    assert "x" in result and "y" in result


def test_phoneme_map_specials_only_when_empty():
    result = validation.phoneme_id_map_from_entries([])
    assert set(result.keys()) == {"<pad>", "<unk>", "<start>", "<end>", " "}


def test_phoneme_map_ignores_non_dict_entries():
    result = validation.phoneme_id_map_from_entries(["not-a-dict", None, {"phonemes": "z"}])
    assert "z" in result


# --- validate_epochs ---
#
# The UI offers 100-5000; the API enforced nothing. `epochs=0` produced a job
# reporting 0/0, which divides by zero the moment anything computes a percentage
# — including the gateway's own job normaliser and the resume path's
# `round(saved_epoch / total_epochs * 100, 1)`.


@pytest.mark.parametrize("value", [1, 100, 1000, 5000, 100_000])
def test_validate_epochs_accepts_the_usable_range(value):
    assert validation.validate_epochs(value) == value


def test_validate_epochs_accepts_a_numeric_string():
    """Form fields arrive as strings when a client posts them by hand."""
    assert validation.validate_epochs("250") == 250


@pytest.mark.parametrize("value", [0, -1, -1000, 100_001])
def test_validate_epochs_rejects_values_outside_the_range(value):
    with pytest.raises(HTTPException) as exc:
        validation.validate_epochs(value)
    assert exc.value.status_code == 400
    assert "epochs" in str(exc.value.detail)


def test_validate_epochs_rejects_zero_specifically():
    """The one that produced a divide-by-zero rather than an obviously bad job."""
    with pytest.raises(HTTPException):
        validation.validate_epochs(0)


@pytest.mark.parametrize("value", ["lots", None, "", 3.7e400])
def test_validate_epochs_rejects_non_integers(value):
    with pytest.raises(HTTPException) as exc:
        validation.validate_epochs(value)
    assert exc.value.status_code == 400


def test_validate_epochs_names_the_field_it_rejected():
    """`extra_epochs` on /resume-training shares this validator."""
    with pytest.raises(HTTPException) as exc:
        validation.validate_epochs(0, "extra_epochs")
    assert "extra_epochs" in str(exc.value.detail)


# --- split_train_val ---
#
# The inline version used np.random.permutation with no seed, so retraining a
# voice validated the new model against a set it had partly trained on last
# time, and two runs' loss curves were not comparable. It also used
# max(1, int(n * 0.1)), which reserved a validation entry even when there was
# only one entry in total — leaving the training set empty and surfacing hours
# later as a bare "Dataset is empty".


def _entries(n: int) -> list:
    return [{"audio_path": f"audio/seg_{i:04d}.wav", "text": f"line {i}"} for i in range(n)]


def test_split_is_reproducible_across_calls():
    a_train, a_val = validation.split_train_val(_entries(50))
    b_train, b_val = validation.split_train_val(_entries(50))
    assert a_train == b_train
    assert a_val == b_val


def test_split_does_not_depend_on_input_order():
    """Segments are collected concurrently, so arrival order is arbitrary."""
    entries = _entries(50)
    shuffled = list(reversed(entries))
    a_train, a_val = validation.split_train_val(entries)
    b_train, b_val = validation.split_train_val(shuffled)
    assert a_val == b_val
    assert a_train == b_train


def test_split_partitions_without_overlap_or_loss():
    entries = _entries(37)
    train, val = validation.split_train_val(entries)
    assert len(train) + len(val) == len(entries)
    train_paths = {e["audio_path"] for e in train}
    val_paths = {e["audio_path"] for e in val}
    assert not (train_paths & val_paths), "an entry landed in both splits"
    assert train_paths | val_paths == {e["audio_path"] for e in entries}


@pytest.mark.parametrize("n,expected_val", [(2, 1), (10, 1), (20, 2), (100, 10)])
def test_split_reserves_roughly_a_tenth_for_validation(n, expected_val):
    _train, val = validation.split_train_val(_entries(n))
    assert len(val) == expected_val


def test_split_never_empties_the_training_set():
    """The old formula gave n=1 a validation entry and no training data."""
    train, val = validation.split_train_val(_entries(2))
    assert len(train) == 1 and len(val) == 1


@pytest.mark.parametrize("n", [0, 1])
def test_split_refuses_a_dataset_too_small_to_divide(n):
    with pytest.raises(HTTPException) as exc:
        validation.split_train_val(_entries(n))
    assert exc.value.status_code == 400
    assert "at least 2" in str(exc.value.detail)


def test_split_accepts_non_dict_entries():
    """The stable sort key must not assume a metadata dict."""
    train, val = validation.split_train_val(["c", "a", "b", "d"])
    assert sorted(train + val) == ["a", "b", "c", "d"]


def test_a_different_seed_gives_a_different_split():
    """Guards against the sort accidentally making the shuffle a no-op."""
    _t1, v1 = validation.split_train_val(_entries(100), seed=1)
    _t2, v2 = validation.split_train_val(_entries(100), seed=2)
    assert v1 != v2


# --- persisted vocabulary ---
#
# The vocabulary used to be rebuilt from train.json wherever it was needed: by the
# dataset when a run started and again by the exporter when it ended. Its ids are
# the row numbers of the trained embedding, so any dataset change in between
# re-numbered the symbols under the weights.

GOOD_VOCAB = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3, " ": 4, "a": 5}


def test_validate_phoneme_id_map_accepts_what_the_dataset_builds():
    built = validation.phoneme_id_map_from_entries([{"phonemes": "abc"}])
    assert validation.validate_phoneme_id_map(built, n_vocab=256) == built


@pytest.mark.parametrize("mapping,fragment", [
    ({}, "empty"),
    (None, "empty"),
    (["a"], "empty"),
    ({"<pad>": 0, "<unk>": 0}, "both"),                       # two symbols, one row
    ({"<pad>": 0, "<unk>": 1, "a": -1}, "invalid id"),
    ({"<pad>": 0, "<unk>": 1, "a": 1.5}, "invalid id"),
    ({"<pad>": 0, "<unk>": 1, "a": True}, "invalid id"),      # bool is an int subclass
    ({"<pad>": 0, "<unk>": 1, "": 2}, "invalid symbol"),
    ({"<pad>": 0, "a": 1}, "<unk>"),
    ({"<unk>": 1, "a": 2}, "<pad>"),
])
def test_validate_phoneme_id_map_rejects_what_an_embedding_cannot_use(mapping, fragment):
    with pytest.raises(ValueError, match=fragment):
        validation.validate_phoneme_id_map(mapping)


def test_validate_phoneme_id_map_checks_the_embedding_size():
    """An id at or above the row count is an out-of-range lookup, a device-side
    assert on GPU."""
    assert validation.validate_phoneme_id_map(GOOD_VOCAB, n_vocab=6) == GOOD_VOCAB
    with pytest.raises(ValueError, match="embedding"):
        validation.validate_phoneme_id_map(GOOD_VOCAB, n_vocab=5)


def test_save_and_load_vocab_round_trip_without_leaving_a_temp_file(tmp_path):
    path = tmp_path / validation.VOCAB_FILENAME
    vocab = {**GOOD_VOCAB, "ʃ": 6, "ç": 7}

    validation.save_vocab(path, vocab, n_vocab=256, boundary_tokens=True)

    assert validation.load_vocab(path, n_vocab=256) == vocab
    assert [p.name for p in tmp_path.iterdir()] == [validation.VOCAB_FILENAME]
    assert '"ʃ"' in path.read_text(encoding="utf-8"), "non-ASCII symbols are written as themselves"


def test_save_vocab_refuses_an_invalid_vocabulary_and_writes_nothing(tmp_path):
    path = tmp_path / validation.VOCAB_FILENAME
    with pytest.raises(ValueError):
        validation.save_vocab(path, {"a": 0})
    assert not path.exists()


@pytest.mark.parametrize("content", ["not json", "[]", '{"phoneme_id_map": []}', '{"other": 1}'])
def test_load_vocab_rejects_a_damaged_file(tmp_path, content):
    path = tmp_path / validation.VOCAB_FILENAME
    path.write_text(content)
    with pytest.raises(ValueError):
        validation.load_vocab(path)


def test_load_vocab_reports_a_missing_file_as_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        validation.load_vocab(tmp_path / "nope.json")


def test_vocab_for_checkpoint_prefers_the_embedded_map_over_the_file(tmp_path):
    validation.save_vocab(tmp_path / validation.VOCAB_FILENAME, {**GOOD_VOCAB, "z": 6})

    vocab, source = validation.vocab_for_checkpoint({"phoneme_to_id": GOOD_VOCAB}, tmp_path)

    assert (vocab, source) == (GOOD_VOCAB, "checkpoint")


def test_vocab_for_checkpoint_falls_back_to_the_file_beside_it(tmp_path):
    validation.save_vocab(tmp_path / validation.VOCAB_FILENAME, GOOD_VOCAB)

    vocab, source = validation.vocab_for_checkpoint({"model_state_dict": {}}, tmp_path)

    assert (vocab, source) == (GOOD_VOCAB, validation.VOCAB_FILENAME)


def test_vocab_for_checkpoint_reports_none_for_a_legacy_checkpoint(tmp_path):
    assert validation.vocab_for_checkpoint({"model_state_dict": {}}, tmp_path) == (None, "none")


def test_vocab_for_checkpoint_raises_on_an_invalid_embedded_map_instead_of_guessing(tmp_path):
    validation.save_vocab(tmp_path / validation.VOCAB_FILENAME, GOOD_VOCAB)
    with pytest.raises(ValueError):
        validation.vocab_for_checkpoint({"phoneme_to_id": {"a": 0}}, tmp_path)


# --- env_int ---


@pytest.mark.parametrize("raw,expected", [
    (None, 5), ("", 5), ("   ", 5),          # unset -> default
    ("3", 3), (" 7 ", 7), ("0", 0),
    ("five", 5), ("2.5", 5),                 # a typo must not stop the service
])
def test_env_int_reads_the_variable_or_falls_back(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("PT_TEST_KNOB", raising=False)
    else:
        monkeypatch.setenv("PT_TEST_KNOB", raw)
    assert validation.env_int("PT_TEST_KNOB", 5) == expected


def test_env_int_clamps_to_the_minimum(monkeypatch):
    monkeypatch.setenv("PT_TEST_KNOB", "0")
    assert validation.env_int("PT_TEST_KNOB", 5, minimum=1) == 1
    monkeypatch.setenv("PT_TEST_KNOB", "-4")
    assert validation.env_int("PT_TEST_KNOB", 5, minimum=0) == 0


# --- prune_old_checkpoints ---
#
# 10000 epochs at the default save interval is 2000 checkpoints of a few hundred
# MB each; nothing ever deleted one.


def _touch_checkpoints(directory, epochs, extra=("final_model.pt", "best_model.pt", "vocab.json", "job_state.json")):
    for epoch in epochs:
        (directory / f"checkpoint_epoch_{epoch}.pt").write_bytes(b"x")
    for name in extra:
        (directory / name).write_bytes(b"x")


def _remaining_epochs(directory):
    return sorted(
        int(p.stem.rsplit("_", 1)[1]) for p in directory.glob("checkpoint_epoch_*.pt")
        if p.stem.rsplit("_", 1)[1].isdigit()
    )


def test_prune_keeps_the_newest_n_by_epoch_number_not_by_name(tmp_path):
    """Sorted as text, checkpoint_epoch_100 comes before checkpoint_epoch_25."""
    _touch_checkpoints(tmp_path, [5, 10, 25, 100, 105])

    removed = validation.prune_old_checkpoints(tmp_path, keep_last=2)

    assert _remaining_epochs(tmp_path) == [100, 105]
    assert sorted(p.name for p in removed) == [
        "checkpoint_epoch_10.pt", "checkpoint_epoch_25.pt", "checkpoint_epoch_5.pt",
    ]


def test_prune_never_touches_final_best_vocab_or_state(tmp_path):
    _touch_checkpoints(tmp_path, range(5, 55, 5))

    validation.prune_old_checkpoints(tmp_path, keep_last=1)

    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.startswith("checkpoint_epoch_")) == [
        "best_model.pt", "final_model.pt", "job_state.json", "vocab.json",
    ]


def test_prune_honours_protected_paths(tmp_path):
    """The checkpoint a resume started from is still referenced until the first
    new one is written."""
    _touch_checkpoints(tmp_path, [5, 10, 15, 20])

    validation.prune_old_checkpoints(tmp_path, keep_last=1, protect=(tmp_path / "checkpoint_epoch_5.pt",))

    assert _remaining_epochs(tmp_path) == [5, 20]


def test_prune_with_fewer_files_than_the_limit_deletes_nothing(tmp_path):
    _touch_checkpoints(tmp_path, [5, 10])
    assert validation.prune_old_checkpoints(tmp_path, keep_last=5) == []
    assert _remaining_epochs(tmp_path) == [5, 10]


@pytest.mark.parametrize("keep_last", [0, -1])
def test_prune_disabled_keeps_everything(tmp_path, keep_last):
    _touch_checkpoints(tmp_path, [5, 10, 15])
    assert validation.prune_old_checkpoints(tmp_path, keep_last=keep_last) == []
    assert _remaining_epochs(tmp_path) == [5, 10, 15]


def test_prune_ignores_look_alike_names_and_a_missing_directory(tmp_path):
    for name in ("checkpoint_epoch_5.pt.tmp", "checkpoint_epoch_x.pt", "checkpoint_epoch_7.pt.bak"):
        (tmp_path / name).write_bytes(b"x")
    _touch_checkpoints(tmp_path, [5, 10, 15], extra=())

    validation.prune_old_checkpoints(tmp_path, keep_last=1)

    assert _remaining_epochs(tmp_path) == [15]
    assert (tmp_path / "checkpoint_epoch_5.pt.tmp").exists()
    assert validation.prune_old_checkpoints(tmp_path / "does-not-exist", keep_last=1) == []
