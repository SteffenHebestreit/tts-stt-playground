"""Sentence-aware text splitting for the Magpie TTS service (pure Python, no torch or NeMo).

The sentence rules (German ordinals and abbreviations, closing quotes, a length ceiling
that is applied even to text without terminal punctuation) are the ones
chatterbox-tts-service uses and tests/test_chatterbox_chunking.py pins; this copy is
kept separate because each service builds its image from its own directory.

Why Magpie splits at all: ``MagpieTTSModel.do_tts`` has a decoder budget of 500 steps
(about 23 s of audio) per chunk and, above a per-language word threshold, chunks the
text itself. Measured on a 585-character German text, that internal chunking left
audible defects at the points where it ended a chunk by force: a hesitation inserted
("Uh, jedes Jahr"), a repeated word with a stray one after it ("dauern, dauern geht"),
a stray trailing token. Splitting at sentence ends into groups that each fit one
single-chunk generation avoids those boundaries, and gives the worker a place to stop
between groups when the client has gone away.
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
    """Split text into chunks for incremental generation.

    Three properties matter for time-to-first-audio:
    - the *first* chunk gets a small floor, because it alone sets TTFA;
    - later chunks get a larger floor, because tiny clips have poor prosody;
    - every chunk gets a ceiling. Without one, text that has no terminal
      punctuation (very common in LLM output) collapses into a single chunk and
      TTFA silently reverts to full-text latency — the exact failure this
      endpoint exists to avoid.
    The same ceiling keeps every chunk of /tts and /clone far below the library's
    1000-token (~40 s) limit per generate() call.
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
