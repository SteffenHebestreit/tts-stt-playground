"""Where /prepare-dataset is allowed to read audio from.

``audio_path`` is client input. It used to be opened as-is when it looked like a
path, and fetched with an unbounded, redirect-following GET when it started
with "http": an SSRF primitive against whatever network the container sits on,
and a way to read any file the process can (the resulting error text came back
in the response). Both are now closed by default.

* A local path must resolve (symlinks followed) to somewhere inside the data
  directory or a directory the operator listed in TRAINING_ALLOWED_AUDIO_DIRS.
* An http(s) URL is refused unless its host is listed in
  TRAINING_ALLOWED_URL_HOSTS. The list is empty by default.
* Any other scheme is refused.

Dependency-free on purpose, so the rules can be tested without the audio stack.
"""

import logging
import math
import os
from pathlib import Path
from typing import List, Tuple, Union
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# A dataset is hours of audio, so this is far above what a single clip needs; it
# exists to stop a request writing (or a URL streaming) without limit.
DEFAULT_MAX_UPLOAD_MB = 500.0


class AudioSourceError(ValueError):
    """The requested audio source is not permitted."""


def max_upload_bytes() -> int:
    """MAX_UPLOAD_MB as bytes: the most one request may upload, or one URL may serve."""
    raw = os.getenv("MAX_UPLOAD_MB", "").strip()
    megabytes = DEFAULT_MAX_UPLOAD_MB
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if math.isfinite(value) and value > 0:
            megabytes = value
        else:
            logger.warning("MAX_UPLOAD_MB=%r is not a positive number; using %s", raw, DEFAULT_MAX_UPLOAD_MB)
    return int(megabytes * 1024 * 1024)


def data_root() -> Path:
    """The dataset directory. Relative, like every other ``data/...`` path here."""
    return Path("data")


def allowed_audio_roots() -> List[Path]:
    """Resolved directories a local audio_path may live in."""
    roots = [data_root()]
    extra = os.getenv("TRAINING_ALLOWED_AUDIO_DIRS", "")
    roots.extend(Path(p.strip()) for p in extra.split(",") if p.strip())
    return [root.resolve() for root in roots]


def allowed_url_hosts() -> frozenset:
    """Lower-cased hostnames a URL audio_path may point at."""
    raw = os.getenv("TRAINING_ALLOWED_URL_HOSTS", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def resolve_audio_source(value: str) -> Tuple[str, Union[Path, str]]:
    """Classify and vet an ``audio_path``: ``("file", Path)`` or ``("url", str)``.

    Raises ``AudioSourceError`` for anything not permitted. The file is not
    required to exist yet; a missing file is a per-segment skip, not a policy
    violation.
    """
    text = (value or "").strip()
    if not text:
        raise AudioSourceError("audio_path is empty")
    if "\x00" in text:
        raise AudioSourceError("audio_path contains a NUL byte")

    parts = urlsplit(text)
    if parts.scheme.lower() in ("http", "https"):
        # .hostname drops userinfo and port, so "http://allowed@evil/" is judged
        # by "evil", which is what the request would actually connect to.
        host = (parts.hostname or "").lower()
        if host and host in allowed_url_hosts():
            return "url", text
        raise AudioSourceError(
            "fetching audio from a URL is disabled. List the host in "
            "TRAINING_ALLOWED_URL_HOSTS to allow it, or upload the file and use its path."
        )
    if "://" in text:
        raise AudioSourceError(f"unsupported audio_path scheme: {text.split('://', 1)[0]!r}")

    path = Path(text)
    if ".." in path.parts:
        raise AudioSourceError("audio_path must not contain '..'")
    absolute = path if path.is_absolute() else Path.cwd() / path
    resolved = absolute.resolve()
    roots = allowed_audio_roots()
    if not any(resolved.is_relative_to(root) for root in roots):
        raise AudioSourceError(
            "audio_path must be inside the data directory "
            f"({', '.join(str(r) for r in roots)}); got {text!r}"
        )
    return "file", resolved
