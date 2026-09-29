"""LocalAgreement-2 on absolute-time-tagged words, for the live transcription path.

The interim decoder re-transcribes a trailing window of the session on every tick
and needs to tell which words are settled. The first implementation compared
``words[i]`` with ``prev_words[i]`` from index 0 of the *current* window. That
only lines up while the window is anchored: once it slides, index 0 refers to
different audio in consecutive decodes, agreement collapses and ``confirmed``
stays empty for the rest of the session.

This follows Machacek et al. (whisper_streaming, arXiv 2307.14743). Every word
carries its position on the session timeline, so a hypothesis is compared with the
previous one only where they cover the same audio; a word is *committed* once two
consecutive hypotheses agree on it. Committed words never change again, they form
a monotonic session transcript, and the decode window is trimmed at committed
boundaries instead of at an arbitrary sample offset.

Kept free of torch, numpy and faster-whisper so it can be unit-tested directly.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

# Word timings jitter between two decodes of the same audio, so a word near the
# committed end cannot be classified by its time alone: it may be one already
# committed and re-decoded a little later, or a new word that came out a little
# early. Discarding by time would lose the second kind for good. So only words
# whose midpoint lies clearly (_CERTAIN_S) before the committed end are discarded
# on time; in the doubtful band up to _DOUBT_S after it a word is discarded only
# if it also repeats the committed text.
_CERTAIN_S = 0.35
_DOUBT_S = 0.3
_OVERLAP_MAX_NGRAM = 5
_SENTENCE_END = ".!?…。！？"
_CLOSERS = "\"'”’»)]}"
# Punctuation and case flip between decodes of the same speech ("Haus," / "haus");
# agreeing on the letters is what matters, not on how the token was dressed.
_EDGE_PUNCT = re.compile(r"^\W+|\W+$")


# Whisper writes these without spaces between words, and reports "words" that are
# single characters. Joining those with a space would put one between every pair.
UNSPACED_LANGUAGES = frozenset({"zh", "ja", "th", "lo", "my", "yue"})


@dataclass(frozen=True)
class TimedWord:
    """One word on the session timeline (seconds since the first audio frame)."""

    start: float
    end: float
    text: str
    # Whether a space belongs between this word and the one before it.
    lead: bool = True


def word_key(text: str) -> str:
    """Comparison key for agreement: casefolded, without edge punctuation."""
    stripped = _EDGE_PUNCT.sub("", text).casefold()
    return stripped or text.strip()


def words_from_segments(segments: Iterable, offset_s: float, spaced: bool = True) -> list[TimedWord]:
    """Flatten faster-whisper segments into session-timeline words.

    ``offset_s`` is where the decoded audio started on the session timeline, since
    faster-whisper reports times relative to the buffer it was handed. Segments
    without word timings (a model whose alignment failed, or a stub) get their
    words spread over the segment by character count: less precise, but still
    monotonic, which is all the agreement logic needs. ``spaced`` is False for
    languages written without spaces (see UNSPACED_LANGUAGES).
    """
    out: list[TimedWord] = []
    for segment in segments:
        timed = getattr(segment, "words", None) or []
        added = 0
        for word in timed:
            text = (getattr(word, "word", "") or "").strip()
            if not text:
                continue
            start = float(word.start) + offset_s
            out.append(TimedWord(start, max(start, float(word.end) + offset_s), text, spaced))
            added += 1
        if added:
            continue

        parts = (getattr(segment, "text", "") or "").split()
        if not parts:
            continue
        seg_start = float(getattr(segment, "start", 0.0) or 0.0)
        seg_end = float(getattr(segment, "end", seg_start) or seg_start)
        span = max(0.0, seg_end - seg_start)
        weights = [len(p) + 1 for p in parts]
        total = float(sum(weights))
        cursor = seg_start
        for part, weight in zip(parts, weights):
            step = span * weight / total
            out.append(TimedWord(cursor + offset_s, cursor + step + offset_s, part))
            cursor += step
    return out


def join_words(words: Iterable[TimedWord]) -> str:
    """The text of `words`, spaced the way each was written."""
    out = ""
    for word in words:
        out += f" {word.text}" if out and word.lead else word.text
    return out


class LocalAgreement:
    """Committed transcript plus tentative tail for one live session.

    Usage per interim tick::

        start = agreement.window_start(now_s)      # where to cut the audio
        words = words_from_segments(segments, start)
        agreement.update(words, now_s)
        agreement.confirmed_text, agreement.pending_text

    ``soft_window_s`` is the target window length: past it, the window is cut at
    the last committed sentence end, which keeps left context for as long as the
    speaker keeps talking in one sentence. ``hard_window_s`` is the ceiling (one
    Whisper window is 30 s): tentative words the window moves past are committed
    as they stand, so nothing is ever dropped from the transcript.
    """

    def __init__(self, soft_window_s: float = 8.0, hard_window_s: float | None = None):
        self.soft_window_s = max(0.0, float(soft_window_s))
        hard = 2.0 * self.soft_window_s if hard_window_s is None else float(hard_window_s)
        self.hard_window_s = max(self.soft_window_s, hard)
        # Start of the audio the next decode should cover.
        self.trim_s = 0.0
        self._confirmed = ""
        self._last_end: float | None = None
        self._recent: deque[TimedWord] = deque(maxlen=_OVERLAP_MAX_NGRAM)
        self._since_trim: list[TimedWord] = []
        self._tail: list[TimedWord] = []

    @property
    def confirmed_text(self) -> str:
        """Every committed word of the session, in order. Only ever grows."""
        return self._confirmed

    @property
    def pending_text(self) -> str:
        """The tentative tail of the latest hypothesis."""
        return join_words(self._tail)

    @property
    def pending_words(self) -> list[TimedWord]:
        """The tentative tail as timed words (a copy)."""
        return list(self._tail)

    def window_start(self, now_s: float) -> float:
        """Session time at which the next decode window should begin."""
        return max(self.trim_s, now_s - self.hard_window_s)

    def drop_pending(self) -> None:
        """Forget the tentative tail, e.g. after the decode language changed."""
        self._tail = []

    def update(self, words: Sequence[TimedWord], now_s: float) -> None:
        """Fold one hypothesis (words on the session timeline) into the transcript.

        ``words`` must come from the window that starts at ``window_start(now_s)``.
        """
        # Tentative words the window has moved past can never be confirmed by a
        # later hypothesis, and dropping them would silently lose speech. Commit
        # them as they stand.
        window_start = self.window_start(now_s)
        stale = 0
        while stale < len(self._tail) and _midpoint(self._tail[stale]) < window_start:
            stale += 1
        if stale:
            self._commit(self._tail[:stale])
            self._tail = self._tail[stale:]

        new = self._without_committed(words)
        agreed = 0
        while (
            agreed < len(new)
            and agreed < len(self._tail)
            and word_key(new[agreed].text) == word_key(self._tail[agreed].text)
        ):
            agreed += 1
        self._commit(new[:agreed])
        self._tail = new[agreed:]
        self._trim(now_s)

    def _without_committed(self, words: Sequence[TimedWord]) -> list[TimedWord]:
        """Drop the part of a hypothesis that re-transcribes already committed audio."""
        if self._last_end is None:
            return list(words)
        end = self._last_end
        new = [w for w in words if _midpoint(w) >= end - _CERTAIN_S]
        recent = list(self._recent)
        # Longest overlap first: the committed tail repeated at the head of the
        # hypothesis. Matching on text is what keeps a real repetition ("nein
        # nein") apart from a re-decoded word.
        for n in range(min(len(recent), len(new), _OVERLAP_MAX_NGRAM), 0, -1):
            head = new[:n]
            if all(_midpoint(w) < end + _DOUBT_S for w in head) and [
                word_key(w.text) for w in recent[-n:]
            ] == [word_key(w.text) for w in head]:
                return new[n:]
        return new

    def _commit(self, words: Sequence[TimedWord]) -> None:
        for word in words:
            if self._confirmed and word.lead:
                self._confirmed += " "
            self._confirmed += word.text
            self._recent.append(word)
            self._since_trim.append(word)
            self._last_end = word.end if self._last_end is None else max(self._last_end, word.end)

    def _trim(self, now_s: float) -> None:
        """Cut the window at the last committed sentence end once it outgrows the target."""
        # Words older than the hard window are unreachable for any future window.
        horizon = now_s - self.hard_window_s
        if self._since_trim and self._since_trim[0].end <= horizon:
            self._since_trim = [w for w in self._since_trim if w.end > horizon]

        if now_s - self.trim_s <= self.soft_window_s:
            return
        for word in reversed(self._since_trim):
            if _ends_sentence(word.text):
                if word.end > self.trim_s:
                    self.trim_s = word.end
                    self._since_trim = [w for w in self._since_trim if w.end > word.end]
                return


def _midpoint(word: TimedWord) -> float:
    return (word.start + word.end) / 2.0


def _ends_sentence(text: str) -> bool:
    core = text.rstrip(_CLOSERS)
    return bool(core) and core[-1] in _SENTENCE_END
