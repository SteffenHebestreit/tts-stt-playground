"""Is a name a public suffix? A stdlib-only matcher for the vendored Public Suffix List.

The Settings page refuses a TRUSTED_HOSTS wildcard whose base anyone can register a
name under: `*.de`, `*.co.uk`, `*.github.io`, `*.ts.net`. Which names those are is what
the Public Suffix List records, so this module answers from a snapshot of it,
`public_suffix_list.dat` next to this file: both the ICANN and the PRIVATE section,
vendored unchanged with its MPL-2.0 header (its `VERSION:` line says when it was taken).
To refresh it:

    curl -fsSL https://publicsuffix.org/list/public_suffix_list.dat \
        -o frontend-service/public_suffix_list.dat

API
    is_public_suffix(name) -> bool         against the vendored list, parsed once on first use
    default_list() -> PublicSuffixList     that list; raises ListUnavailable when it cannot be read
    PublicSuffixList.from_lines(lines)     a list of your own (tests)
    PublicSuffixList.load(path)
    ListUnavailable                        the data file is missing, unreadable or implausibly short

Matching follows https://github.com/publicsuffix/list/wiki/Format: normal rules
(`co.uk`), wildcard rules (`*.ck`: every name directly under ck is a public suffix) and
exception rules (`!www.ck`: except www.ck). Only a leading `*.` wildcard is understood,
which is the only form the list uses; a rule with a `*` anywhere else is skipped. Names
and rules are compared in their ASCII form (punycode for an internationalised label), so
`xn--55qx5d.cn` and its Unicode spelling give the same answer.

One deliberate difference from the format's implicit `*` rule ("an unlisted top-level
label is a public suffix"): it is applied only to top-level labels the list knows (`de`,
and `ck`, which appears only as `*.ck`). A single label the list has never heard of
(`k2o`, `corp`, `home`) is a private name on someone's own DNS. That is exactly the case
the Settings page accepts with a warning (`*.k2o`), so it is reported as not public here.

Nothing is read at import; the data file is read on the first call.
"""

from __future__ import annotations

import os
import threading
from typing import Iterable, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public_suffix_list.dat")

# A list with fewer rules than this is a truncated or wrong file, not the PSL (the real
# one has more than 10,000). Accepting it would let every wildcard through.
_MIN_RULES = 1000


class ListUnavailable(RuntimeError):
    """The vendored list cannot be read, or is too short to be the real one."""


def _ascii_label(label: str) -> str:
    label = label.lower()
    if label.isascii():
        return label
    return "xn--" + label.encode("punycode").decode("ascii")


def _ascii_name(name: str) -> Optional[str]:
    """The lower-case ASCII form of a dotted name, or None when it is not one."""
    if not isinstance(name, str):
        return None
    text = name.strip()
    if text.endswith("."):
        text = text[:-1]
    if not text:
        return None
    labels = text.split(".")
    if any(not label for label in labels):
        return None
    try:
        return ".".join(_ascii_label(label) for label in labels)
    except UnicodeError:
        return None


class PublicSuffixList:
    """Parsed rules of one list; immutable after construction."""

    def __init__(self, rules: Iterable[str], wildcards: Iterable[str], exceptions: Iterable[str],
                 top_labels: Iterable[str]):
        self._rules = frozenset(rules)            # "co.uk"
        self._wildcards = frozenset(wildcards)    # "ck" for the rule "*.ck"
        self._exceptions = frozenset(exceptions)  # "www.ck" for the rule "!www.ck"
        self._top_labels = frozenset(top_labels)  # every top-level label a rule ends in

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "PublicSuffixList":
        rules: set[str] = set()
        wildcards: set[str] = set()
        exceptions: set[str] = set()
        top_labels: set[str] = set()
        for line in lines:
            text = line.strip()
            if not text or text.startswith("//"):
                continue
            # A rule is the line up to its first whitespace.
            rule = text.split()[0]
            exception = rule.startswith("!")
            if exception:
                rule = rule[1:]
            wildcard = rule.startswith("*.")
            if wildcard:
                rule = rule[2:]
            if "*" in rule or "!" in rule:
                continue
            name = _ascii_name(rule)
            if name is None:
                continue
            if exception:
                exceptions.add(name)
            elif wildcard:
                wildcards.add(name)
            else:
                rules.add(name)
            top_labels.add(name.rsplit(".", 1)[-1])
        return cls(rules, wildcards, exceptions, top_labels)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "PublicSuffixList":
        with open(path, encoding="utf-8") as handle:
            return cls.from_lines(handle)

    def __len__(self) -> int:
        return len(self._rules) + len(self._wildcards) + len(self._exceptions)

    def is_public_suffix(self, name: str) -> bool:
        """True when `name` itself is a public suffix (`de`, `co.uk`, `foo.ck`), False otherwise.

        A name that is not a syntactically valid dotted name is not a public suffix.
        """
        ascii_name = _ascii_name(name)
        if ascii_name is None:
            return False
        labels = ascii_name.split(".")
        if len(labels) == 1:
            return ascii_name in self._top_labels
        # An exception rule that matches the name or one of its parents makes the public
        # suffix that rule minus its first label: always shorter than the name.
        for start in range(len(labels)):
            if ".".join(labels[start:]) in self._exceptions:
                return False
        # Otherwise the name is a public suffix exactly when a rule matches all of it.
        return ascii_name in self._rules or ".".join(labels[1:]) in self._wildcards


_default: Optional[PublicSuffixList] = None
_default_lock = threading.Lock()


def default_list() -> PublicSuffixList:
    """The vendored list, read and parsed on first use (a failure is not cached)."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                try:
                    loaded = PublicSuffixList.load(DEFAULT_PATH)
                except (OSError, UnicodeError) as exc:
                    raise ListUnavailable(f"cannot read the public suffix list ({type(exc).__name__})") from exc
                if len(loaded) < _MIN_RULES:
                    raise ListUnavailable("the public suffix list is truncated")
                _default = loaded
    return _default


def is_public_suffix(name: str) -> bool:
    """`default_list().is_public_suffix(name)`; raises ListUnavailable without the data file."""
    return default_list().is_public_suffix(name)
