"""Dependency-light parsing helpers for the Canary ASR service.

Extracted from app.py so the NeMo-hypothesis parsing can be unit-tested without
importing torch / nemo_toolkit / librosa.
"""


def parse_hypothesis(hyp) -> tuple:
    """Convert one NeMo hypothesis (or plain string) into ``(text, segments)``.

    Handles both the plain-string and Hypothesis-object shapes NeMo may return,
    and defends against malformed timestamp entries.
    """
    if isinstance(hyp, str):
        return hyp.strip(), []

    text = (getattr(hyp, "text", "") or "").strip()
    timestamp = getattr(hyp, "timestamp", None) or {}
    raw_segments = timestamp.get("segment", []) if isinstance(timestamp, dict) else []

    segments = []
    for seg in raw_segments:
        try:
            segments.append({
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "text": (seg.get("segment") or seg.get("text") or "").strip(),
            })
        except (TypeError, ValueError, AttributeError):
            continue
    return text, segments
