"""LocalAgreement on absolute-time words (stt-service/local_agreement.py).

The live decoder re-transcribes a trailing window every tick. The first version
compared ``words[i]`` with ``prev_words[i]`` from index 0 of the *current* window,
which only lines up while the window is anchored; once it slides, index 0 is
different audio in consecutive decodes and nothing is ever confirmed. These tests
play a model against a sliding window and check what the agreement logic makes of
it, with the old comparison run alongside as the control.
"""

import sys
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from test_stt_support import STT_DIR, Segment, Word


@pytest.fixture(scope="module")
def la():
    spec = spec_from_file_location("stt_local_agreement_under_test", STT_DIR / "local_agreement.py")
    module = module_from_spec(spec)
    sys.modules["stt_local_agreement_under_test"] = module  # dataclasses resolve types through it
    spec.loader.exec_module(module)
    return module


def _speech(n_words: int, sentence_every: int = 9):
    """Ground truth: (start, end, text) with a pause after each sentence."""
    truth, t = [], 0.0
    for i in range(n_words):
        dur = 0.25 + 0.05 * (i % 5)
        text = f"w{i}" + ("." if i % sentence_every == sentence_every - 1 else "")
        truth.append((t, t + dur, text))
        t += dur + 0.05 + (0.6 if i % sentence_every == sentence_every - 1 else 0.0)
    return truth


def _hear(la, truth, window_start, now, wobble=0.0):
    """What a decent model returns for the window [window_start, now].

    Words completely inside are returned; the word still being spoken at the edge
    is garbled differently on every call, as a cut-off word is. `wobble` shifts
    every timestamp by a different amount each call, the way alignment jitter does.
    """
    out = []
    for i, (start, end, text) in enumerate(truth):
        if start < window_start - 1e-6:
            continue
        shift = wobble * (((i * 7 + int(now * 10)) % 5) - 2) / 2.0
        if end <= now - 0.3:
            out.append(la.TimedWord(start + shift, end + shift, text))
        elif start < now:
            out.append(la.TimedWord(start, now, f"x{int(now * 10) % 7}{text}"))
    return out


def _run(la, truth, soft, hard, wobble=0.0, tick=0.5):
    agreement = la.LocalAgreement(soft, hard)
    now, previous, windows = 0.0, "", []
    end = truth[-1][1] + 3.0
    while now < end:
        now += tick
        start = agreement.window_start(now)
        windows.append(now - start)
        agreement.update(_hear(la, truth, start, now, wobble), now)
        assert agreement.confirmed_text.startswith(previous), "confirmed text changed after being shown"
        previous = agreement.confirmed_text
    return agreement, windows


def _legacy_confirmed_counts(la, truth, window_s, tick=0.5):
    """The comparison this module replaces: index 0 of a sliding tail, every tick."""
    prev, counts, now = [], [], 0.0
    while now < truth[-1][1] + 3.0:
        now += tick
        words = [w.text for w in _hear(la, truth, max(0.0, now - window_s), now)]
        agree = 0
        while agree < len(words) and agree < len(prev) and words[agree] == prev[agree]:
            agree += 1
        prev = words
        counts.append(agree)
    return counts


def test_the_index_zero_comparison_collapses_once_the_window_slides(la):
    """The control: why the old approach never confirmed anything on a long session."""
    truth = _speech(60)
    counts = _legacy_confirmed_counts(la, truth, window_s=4.0)
    rolled = counts[len(counts) // 2:]          # the second half is entirely past the first window
    assert sum(1 for c in rolled if c) <= len(rolled) // 4, (
        "the legacy comparison agreed on most ticks after the window slid; this control "
        "no longer demonstrates the defect"
    )


@pytest.mark.parametrize("soft,hard,wobble", [(8, 16, 0.0), (8, 16, 0.12), (2, 4, 0.05), (4, 8, 0.25)])
def test_a_long_session_is_committed_completely_and_in_order(la, soft, hard, wobble):
    """Every word ends up in `confirmed`, once, in order, however far the window slid."""
    truth = _speech(90)
    agreement, windows = _run(la, truth, soft, hard, wobble)

    assert agreement.confirmed_text.split() == [t for _, _, t in truth]
    assert max(windows) <= hard + 0.5, "the decode window outgrew its ceiling"
    # The session is far longer than one window, so this only holds if the
    # window really did slide.
    assert truth[-1][1] > 2 * hard


def test_two_agreeing_hypotheses_commit_their_shared_prefix(la):
    agreement = la.LocalAgreement(8.0)
    tw = la.TimedWord
    agreement.update([tw(0.0, 0.4, "the"), tw(0.5, 0.9, "quick"), tw(1.0, 1.4, "brown")], 1.5)
    assert agreement.confirmed_text == ""
    assert agreement.pending_text == "the quick brown"

    agreement.update([tw(0.0, 0.4, "the"), tw(0.5, 0.9, "quick"), tw(1.0, 1.4, "green"), tw(1.5, 1.9, "fox")], 2.0)
    assert agreement.confirmed_text == "the quick"
    assert agreement.pending_text == "green fox"


def test_punctuation_and_case_flips_do_not_block_agreement(la):
    """Whisper re-dresses the same speech between decodes ("Haus," / "haus")."""
    agreement = la.LocalAgreement(8.0)
    tw = la.TimedWord
    agreement.update([tw(0.0, 0.4, "Guten"), tw(0.5, 0.9, "Tag,")], 1.0)
    agreement.update([tw(0.0, 0.4, "guten"), tw(0.5, 0.9, "Tag."), tw(1.0, 1.4, "zusammen")], 1.5)
    assert agreement.confirmed_text == "guten Tag."   # the newer spelling is kept
    assert agreement.pending_text == "zusammen"


def test_a_committed_word_that_comes_back_with_shifted_times_is_not_duplicated(la):
    agreement = la.LocalAgreement(8.0)
    tw = la.TimedWord
    agreement.update([tw(0.0, 0.5, "Haus")], 0.6)
    agreement.update([tw(0.0, 0.5, "Haus"), tw(0.6, 1.0, "Boot")], 1.1)
    assert agreement.confirmed_text == "Haus"
    # Re-decoded with a start drifted past the committed end; text matching catches it.
    agreement.update([tw(0.45, 0.95, "Haus"), tw(0.6, 1.0, "Boot"), tw(1.1, 1.5, "Tür")], 1.6)
    assert agreement.confirmed_text.split() == ["Haus", "Boot"]
    assert agreement.pending_text == "Tür"


def test_a_genuine_repetition_is_kept(la):
    """"nein nein" is two words; the overlap match must not eat the second."""
    agreement = la.LocalAgreement(8.0)
    tw = la.TimedWord
    agreement.update([tw(0.0, 0.4, "nein")], 0.5)
    agreement.update([tw(0.0, 0.4, "nein"), tw(0.9, 1.3, "Nein.")], 1.4)
    assert agreement.confirmed_text == "nein"
    agreement.update([tw(0.0, 0.4, "nein"), tw(0.9, 1.3, "Nein.")], 1.5)
    assert agreement.confirmed_text == "nein Nein."


def test_words_the_window_moves_past_are_committed_not_dropped(la):
    """A hypothesis that never settles must not lose speech when the window slides on."""
    agreement = la.LocalAgreement(2.0, 4.0)
    tw = la.TimedWord
    for tick in range(1, 30):
        now = tick * 0.5
        start = agreement.window_start(now)
        # Same audio, a different reading every tick: no two decodes ever agree.
        heard = [
            tw(pos * 0.5, pos * 0.5 + 0.3, f"p{pos}v{tick}")
            for pos in range(int(now / 0.5) + 1)
            if pos * 0.5 >= start and pos * 0.5 + 0.3 <= now
        ]
        agreement.update(heard, now)
    committed = agreement.confirmed_text.split()
    positions = [int(w[1:].split("v")[0]) for w in committed]
    assert positions == sorted(set(positions)), "a position was committed twice or out of order"
    assert positions[0] == 0, "the earliest words were dropped instead of committed"
    # Everything the window has left behind is in the transcript, gap-free.
    assert positions == list(range(len(positions)))
    assert len(positions) >= int(agreement.window_start(14.5) / 0.5) - 1


def test_the_window_is_cut_at_a_committed_sentence_end(la):
    agreement = la.LocalAgreement(3.0, 12.0)
    tw = la.TimedWord
    hypothesis = [tw(0.0, 0.5, "Hallo"), tw(0.6, 1.0, "Welt."), tw(1.2, 1.6, "Wie"), tw(1.7, 2.1, "geht")]
    agreement.update(hypothesis, 4.0)
    assert agreement.trim_s == 0.0, "nothing is committed after one decode"
    agreement.update(hypothesis + [tw(2.2, 2.6, "es")], 4.5)
    assert agreement.confirmed_text.startswith("Hallo Welt.")
    assert agreement.trim_s == pytest.approx(1.0), "the window was not cut at the sentence end"
    assert agreement.window_start(4.6) == pytest.approx(1.0)


def test_the_window_never_starts_earlier_than_the_hard_ceiling(la):
    agreement = la.LocalAgreement(2.0, 5.0)
    assert agreement.window_start(60.0) == pytest.approx(55.0)
    assert agreement.window_start(3.0) == 0.0


def test_a_forgotten_tail_cannot_agree_with_the_next_hypothesis(la):
    agreement = la.LocalAgreement(8.0)
    tw = la.TimedWord
    agreement.update([tw(0.0, 0.4, "eins")], 0.5)
    agreement.drop_pending()
    agreement.update([tw(0.0, 0.4, "eins")], 0.6)
    assert agreement.confirmed_text == ""     # it had to be seen twice *after* the drop


def test_word_timestamps_are_shifted_onto_the_session_timeline(la):
    segment = Segment("hello world", words=[Word(0.1, 0.4, " hello"), Word(0.5, 0.9, " world")])
    words = la.words_from_segments([segment], offset_s=10.0)
    assert [(w.text, round(w.start, 2), round(w.end, 2)) for w in words] == [
        ("hello", 10.1, 10.4), ("world", 10.5, 10.9),
    ]


def test_segments_without_word_timings_are_spread_over_their_span(la):
    words = la.words_from_segments([Segment("ein kurzer Satz", start=1.0, end=4.0)], offset_s=100.0)
    assert [w.text for w in words] == ["ein", "kurzer", "Satz"]
    starts = [w.start for w in words]
    assert starts == sorted(starts) and starts[0] == pytest.approx(101.0)
    assert words[-1].end == pytest.approx(104.0)


def test_confirmed_never_shrinks_whatever_the_model_says(la):
    agreement = la.LocalAgreement(2.0, 4.0)
    tw = la.TimedWord
    seen = ""
    hypotheses = [
        [tw(0.0, 0.4, "a"), tw(0.5, 0.9, "b")],
        [tw(0.0, 0.4, "a"), tw(0.5, 0.9, "b"), tw(1.0, 1.4, "c")],
        [],                                      # all filtered
        [tw(3.0, 3.4, "zzz")],                   # something unrelated
        [tw(0.0, 0.4, "x")],                     # an out-of-order stale hypothesis
    ]
    for i, hypothesis in enumerate(hypotheses):
        agreement.update(hypothesis, 2.0 + i)
        assert agreement.confirmed_text.startswith(seen)
        seen = agreement.confirmed_text


def test_languages_written_without_spaces_are_joined_without_them(la):
    """Whisper reports Chinese and Japanese "words" one character at a time."""
    tw = la.TimedWord
    chars = [tw(i * 0.3, i * 0.3 + 0.25, c, lead=False) for i, c in enumerate("你好世界。")]
    agreement = la.LocalAgreement(8.0)
    agreement.update(chars, 1.6)
    agreement.update(chars + [tw(1.5, 1.75, "我", lead=False)], 2.0)
    assert agreement.confirmed_text == "你好世界。"
    assert agreement.pending_text == "我"
    assert " " not in agreement.confirmed_text


def test_segment_words_are_marked_unspaced_for_those_languages(la):
    segment = Segment("你好", words=[Word(0.0, 0.2, "你"), Word(0.2, 0.4, "好")])
    words = la.words_from_segments([segment], 0.0, spaced=la.UNSPACED_LANGUAGES.isdisjoint({"zh"}))
    assert [w.lead for w in words] == [False, False]
    assert la.join_words(words) == "你好"
    assert {"zh", "ja", "th"} <= la.UNSPACED_LANGUAGES and "de" not in la.UNSPACED_LANGUAGES


def test_a_sentence_end_in_fullwidth_punctuation_cuts_the_window(la):
    tw = la.TimedWord
    hypothesis = [tw(0.0, 0.4, "你好", lead=False), tw(0.5, 0.9, "。", lead=False), tw(1.0, 1.4, "再见", lead=False)]
    agreement = la.LocalAgreement(2.0, 8.0)
    agreement.update(hypothesis, 3.5)
    agreement.update(hypothesis + [tw(1.5, 1.9, "吧", lead=False)], 4.0)
    assert agreement.trim_s == pytest.approx(0.9)
