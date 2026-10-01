"""Sentence-aware text splitting for the Magpie TTS service (pure Python, no torch or NeMo).

Two splitters:

* ``split_for_synthesis`` (German, English, Spanish, French) uses chatterbox-tts-service's
  sentence rules: German ordinals and abbreviations, closing quotes, and a length ceiling
  that holds even for text without terminal punctuation. The copy exists only because
  each service builds its image from its own directory. tests/test_magpie_support.py
  fails as soon as the shared constants, ``_dot_ends_sentence``, ``_sentences`` or
  ``_split_sentences`` stop being the same code as chatterbox's (docstrings aside), and
  tests/test_chatterbox_chunking.py runs every one of its cases against both copies.
* ``split_cjk`` (Chinese, Japanese: no spaces between words) cuts at the scripts' own
  sentence marks, and an over-long sentence after a clause mark.

Why Magpie splits at all: ``MagpieTTSModel.do_tts`` decodes at most 500 frames (about
23 s of audio) per chunk of text and, above a per-language threshold (45 English words,
100 Chinese characters, ...), chunks the text itself. Measured on a 585-character German
text, that internal chunking left audible defects at the points where it ended a chunk
by force: a hesitation inserted ("Uh, jedes Jahr"), a repeated word with a stray one
after it ("dauern, dauern geht"), a stray trailing token. Chinese and Japanese are only
split by NeMo at 。？！… , so a long sentence joined by commas stayed one chunk and was
cut off at the frame limit. Groups that each fit one single-chunk generation avoid both,
and give the worker a place to stop between groups once the client's connection has
closed (a hang-up, or the gateway's read timeout).
"""

import re


_CLOSERS = "\"'\u201c\u201d\u2019\u00ab\u00bb)]"
_OPENERS = "\"'\u201e\u201c\u201d\u00ab\u00bb(["
_TERMINAL = ".!?;:"
_WHITESPACE = re.compile(r"\s+")

# Abbreviations that are (practically) never the last word of a sentence: a period
# after them is not a boundary, whatever follows. Lower case, without the dot.
_ABBREVIATIONS = frozenset("""
    dr prof nr str tel abb tab kap vgl ggf evtl inkl exkl zzgl max min bzw bzgl ca mio mrd
    std sek jh jhd geb gest hrsg dipl ing hr fr st mr mrs ms dt bsp ehem gem sog bes allg zt
""".split())
# Often sentence-final ("Äpfel, Birnen usw. Danach ...") but also mid-sentence
# ("... usw. sind gesund"): a boundary only if a capitalised word follows.
_AMBIGUOUS_ABBREVIATIONS = frozenset({"usw", "etc", "uvm", "usf"})
# Nouns that follow a written-out ordinal number ("im 100. Jahr"). Ordinals of one
# or two digits ("am 3. Mai", "der 21. Jahrgang") need no help: see below.
_ORDINAL_FOLLOWERS = frozenset({
    "jahr", "jahre", "jahrhundert", "jahrestag", "jubilaeum", "jubiläum", "geburtstag", "todestag",
    "auflage", "ausgabe", "wiederholung", "besucher", "kunde", "spiel", "platz", "mal", "folge",
})


def _dot_ends_sentence(text: str, dot: int, after: int) -> bool:
    """Whether the single '.' at *dot* ends a sentence; *after* is where the next word starts.

    German is full of periods that are not boundaries: ordinals ("am 3. Mai"),
    abbreviations ("z. B.", "Dr. Müller", "ca. 5", "Nr. 7"). Splitting there makes
    the TTS speak "am drei" and pause. The bias is deliberate: a missed boundary
    only makes a chunk longer (the length ceiling still applies), a false one
    changes how the text is read.

    Scans outwards from the dot instead of slicing or searching from the start of
    the text: the text is caller-controlled, and a regex like `(\\S+)$` over a long
    unbroken token is quadratic (23 s of event-loop time for 5000 characters).
    """
    begin = dot
    while begin > 0 and not text[begin - 1].isspace():
        begin -= 1
    token = text[begin:dot].lstrip(_OPENERS)
    following = text[after:after + 64].lstrip(_OPENERS)
    word = following.split(None, 1)[0] if following.strip() else ""
    lowered = token.lower()

    if not token:
        return True
    if token.isdigit():
        if len(token) <= 2:
            return False  # "3. Mai", "21. Jahrhundert"
        return not (word[:1].islower() or word.strip(".,;:!?").lower() in _ORDINAL_FOLLOWERS)
    if len(token) == 1 and token.isalpha():
        return False  # initials and the parts of "z. B.", "d. h.", "u. a.", "S. 5"
    if "." in token:
        parts = token.split(".")
        if all(p.isalpha() and len(p) <= 2 for p in parts if p):
            return False  # "z.B.", "d.h.", "i.d.R."
    if lowered in _ABBREVIATIONS:
        return False
    if lowered in _AMBIGUOUS_ABBREVIATIONS:
        return word[:1].isupper()
    return True


def _sentences(line: str) -> list:
    """Split one line at sentence boundaries (German-aware for periods)."""
    out, start = [], 0
    for space in _WHITESPACE.finditer(line):
        # Walk back from the whitespace: optional closing quotes, then the
        # terminal punctuation. Linear overall, unlike one regex for all of it.
        end = space.start()
        closers = end
        while closers > start and line[closers - 1] in _CLOSERS:
            closers -= 1
        first = closers
        while first > start and line[first - 1] in _TERMINAL:
            first -= 1
        if first == closers:
            continue  # no terminal punctuation before this whitespace
        if closers - first == 1 and line[first] == "." and not _dot_ends_sentence(line, first, space.end()):
            continue
        out.append(line[start:end])
        start = space.end()
    out.append(line[start:])
    return [s.strip() for s in out if s.strip()]


def _split_sentences(text: str, first_chunk_chars: int = 40,
                     min_chars: int = 60, max_chars: int = 180) -> list[str]:
    """Split text into chunks of whole sentences, merged up to a floor, never past a ceiling.

    This is chatterbox's streaming splitter, where the floors matter for
    time-to-first-audio: a small one for the first chunk, a larger one for the rest
    (tiny clips have poor prosody). Magpie calls it through ``split_for_synthesis``
    with equal floors. What matters here is the ceiling, which holds for every chunk:
    without it, text that has no terminal punctuation (very common in LLM output)
    collapses into a single chunk, and that is the one NeMo would split by force.
    """
    text = text.strip()
    if not text:
        return []

    raw = [s for line in re.split(r"\n+", text) for s in _sentences(line)]

    # Enforce the ceiling: re-split over-long pieces on commas, then whitespace.
    bounded: list[str] = []
    for part in raw:
        while len(part) > max_chars:
            cut = part.rfind(", ", 0, max_chars)
            # Keep the comma with the chunk it terminates: cutting before it
            # moved the pause to the START of the next chunk, which the model
            # then speaks as a leading ", ...".
            end = cut + 1 if cut >= min_chars else cut
            if cut < min_chars:
                cut = end = part.rfind(" ", 0, max_chars)
            if cut < min_chars:
                cut = end = max_chars  # a single unbroken token; hard-cut it
            bounded.append(part[:end].strip())
            part = part[end:].strip()
        if part:
            bounded.append(part)

    merged: list[str] = []
    buf = ""
    for part in bounded:
        # Flush before overflowing, or the merge step would undo the ceiling
        # the loop above just enforced — and chunk 0 is what sets TTFA.
        if buf and len(buf) + 1 + len(part) > max_chars:
            merged.append(buf)
            buf = ""
        buf = f"{buf} {part}".strip() if buf else part
        floor = first_chunk_chars if not merged else min_chars
        if len(buf) >= floor:
            merged.append(buf)
            buf = ""
    if buf:
        # A trailing fragment is appended only if that keeps us under the
        # ceiling; otherwise it stands alone rather than breaking the bound.
        if merged and len(merged[-1]) + len(buf) + 1 <= max_chars:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged or [text]


def split_for_synthesis(text: str, max_chars: int) -> list[str]:
    """Groups of whole sentences, each at most *max_chars* long (an over-long sentence is cut at a comma, then a space).

    No small first chunk and no floor beyond a modest one: nobody is streaming this, so
    the only goals are clean sentence ends and a ceiling the model can finish in one go.
    """
    max_chars = max(20, int(max_chars))
    floor = min(60, max_chars // 2)
    return _split_sentences(text, first_chunk_chars=floor, min_chars=floor, max_chars=max_chars)


# --- Text without spaces between words (Chinese, Japanese) -----------------------------

# A run of sentence marks ends a sentence ("？！", "……"), together with the closing marks
# right after it, so that 「……。」 stays whole.
_CJK_TERMINAL = "\u3002\uff01\uff1f\u2026!?"  # 。 ！ ？ … !?
_CJK_CLOSERS = "\u300d\u300f\uff09\u3011\u3015\u3009\u300b\u201d\u2019\"')]"  # 」 』 ） 】 〕 〉 》 ” ’ "')]
# Where a sentence longer than the ceiling may be cut.
_CJK_CLAUSE = "\uff0c\u3001\uff1b\uff1a,;:"  # ， 、 ； ： ,;:
_ASCII_CLAUSE = ",;:"


def _add_span(spans: list, text: str, start: int, end: int) -> None:
    """Append ``(start, end)`` narrowed to leave out surrounding whitespace, unless nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        spans.append((start, end))


def _cjk_sentence_spans(text: str) -> list:
    """``(start, end)`` of every sentence in *text*, in order; one pass over the text.

    A sentence ends after a run of 。！？…!? and the closing marks that follow it, at a
    '.' only before whitespace or the end of the text (so "3.5" stays whole), and at
    every line break.
    """
    spans: list = []
    start, i, n = 0, 0, len(text)
    while i < n:
        char = text[i]
        if char in "\r\n":
            following = i + 1
        elif char in _CJK_TERMINAL or (char == "." and (i + 1 == n or text[i + 1].isspace())):
            following = i + 1
            while following < n and (text[following] in _CJK_TERMINAL or text[following] in _CJK_CLOSERS):
                following += 1
        else:
            i += 1
            continue
        _add_span(spans, text, start, following)
        start = i = following
    _add_span(spans, text, start, n)
    return spans


def _cjk_cut(text: str, start: int, limit: int, floor: int) -> int:
    """Where a piece that begins at *start* ends: by *limit*, and at least *floor* characters on.

    After the last clause mark in that range (an ASCII one between two digits, as in
    "14:30" or "1,000", does not count), else at the last whitespace (a Latin word is
    not cut in two), else hard at *limit*. Looks at no more than ``limit - start``
    characters.
    """
    lowest = start + floor
    for i in range(limit - 1, lowest - 2, -1):
        if text[i] in _CJK_CLAUSE and not (
            text[i] in _ASCII_CLAUSE and i > start and text[i - 1].isdigit() and text[i + 1].isdigit()
        ):
            return i + 1
    for i in range(limit, lowest - 1, -1):
        if text[i].isspace():
            return i
    return limit


def split_cjk(text: str, max_chars: int) -> list[str]:
    """Groups of whole sentences for text without spaces between words, each at most *max_chars* long.

    A sentence longer than *max_chars* is cut after its last clause mark (，、；：,;:)
    that keeps at least a third of *max_chars* before the cut; failing that at
    whitespace, failing that hard at *max_chars*. Neighbouring pieces are then joined
    while the group stays within *max_chars*: directly where they touched in the text,
    with one space where whitespace separated them. Pieces lose whitespace only at
    their ends, so text without whitespace comes back whole (``"".join(groups) == text``).

    Linear in the length of the text, which is caller-controlled: every cut looks at
    no more than *max_chars* characters and moves on by at least a third of that.
    """
    max_chars = max(1, int(max_chars))
    floor = max(1, max_chars // 3)
    pieces: list = []
    for start, end in _cjk_sentence_spans(text):
        while end - start > max_chars:
            cut = _cjk_cut(text, start, start + max_chars, floor)
            _add_span(pieces, text, start, cut)
            start = cut
            while text[start].isspace():  # stops at the sentence's last character at the latest
                start += 1
        _add_span(pieces, text, start, end)

    groups: list[str] = []
    parts: list = []
    size = 0
    previous_end = None
    for start, end in pieces:
        joint = "" if previous_end == start else " "
        if parts and size + len(joint) + end - start > max_chars:
            groups.append("".join(parts))
            parts, size = [], 0
        if parts:
            parts.append(joint)
            size += len(joint)
        parts.append(text[start:end])
        size += end - start
        previous_end = end
    if parts:
        groups.append("".join(parts))
    return groups
