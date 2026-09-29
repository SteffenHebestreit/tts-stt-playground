"""Stand-ins for heavy packages that exist only while a service module is imported.

The chatterbox, qwen3-tts and stt-service apps import torch, soundfile, uvicorn (and
faster-whisper) at module level, none of which the unit-test job installs. A test module
that put its stub into ``sys.modules`` and left it there did two harmful things: it made
every later ``import uvicorn`` in the suite succeed for the wrong reason (which is how a
missing ``uvicorn`` in tests/requirements.txt went unnoticed: the piper-tts tests only
passed when one of these modules had run first), and it made a module pass or fail
depending on which other modules had run before it.

``stubbed_modules`` installs the stand-ins for the duration of the ``with`` block and then
puts back exactly what was there, including "nothing". The app keeps working afterwards
because it bound the names it needs (``torch``, ``sf``, ...) at import.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Callable, Iterable, Mapping
from types import ModuleType

_MISSING = object()


def is_stub(module) -> bool:
    """True for a hand-made module (no file behind it), False for a real package."""
    return getattr(module, "__file__", None) is None


@contextmanager
def stubbed_modules(factories: Mapping[str, Callable[[], ModuleType]], force: Iterable[str] = ()):
    """Install ``factories[name]()`` as ``sys.modules[name]`` for the block, then restore.

    A name is stubbed when it is absent or already a stub. A real package that was
    imported earlier is left alone, except for names in ``force`` (a test that needs its
    own recording double regardless).
    """
    forced = set(force)
    saved = {name: sys.modules.get(name, _MISSING) for name in factories}
    try:
        for name, make in factories.items():
            current = saved[name]
            if name in forced or current is _MISSING or current is None or is_stub(current):
                sys.modules[name] = make()
        yield
    finally:
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
