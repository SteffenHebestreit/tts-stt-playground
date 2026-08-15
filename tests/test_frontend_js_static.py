"""Static invariants of the browser layer (`app.js` + `index.html`).

Two groups: escaping rules, and the DOM contracts that couple the script to the
template. There is no JS test runner in this repo — the only browser test is the
Playwright live-mic one — so these read the source.

Escaping
--------

`app.js` already had `escapeHtml` and used it correctly in the transcription
renderers. Three other places did not, and one of them could not have been fixed
by escaping at all:

    onclick="resumeTraining('${voiceName}')"

A voice name is whatever the operator typed at training time. The training
service's `safe_name()` rejects path separators and `..` — it is a filesystem
guard, not an HTML one — so quotes and angle brackets pass straight through into
the jobs table. And escaping does not save this pattern: the HTML parser decodes
entities in an attribute value *before* the result is handed to the JavaScript
parser, so `&#39;` becomes `'` and still closes the argument string.

The same values also went unescaped into `<td>${voiceName}</td>`, and every
error path rendered a backend `detail` — which routinely carries an uploaded
filename — through `showStatus`, which built its node with innerHTML.

Script/template contracts
-------------------------
`showTab()` used to find the button to highlight by substring-matching each
button's own `onclick` text, so the script depended on the *spelling of a
handler call* in the template. It now looks up `<tabId>-button` by id, which is
a contract the template can be checked against — and is what the last test here
does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_JS = REPO_ROOT / "frontend-service" / "static" / "js" / "app.js"
INDEX_HTML = REPO_ROOT / "frontend-service" / "templates" / "index.html"


@pytest.fixture(scope="module")
def source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _strip_block_comments(text: str) -> str:
    """Drop /* ... */ blocks so documentation of a bad pattern is not a finding."""
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def test_no_inline_handler_interpolates_a_value(source: str):
    """`on*="fn('${x}')"` is a double context: HTML first, then JavaScript.

    Entity decoding happens between the two, so no HTML escaping can keep the
    value inside the argument string. Use data-* attributes plus bindActions().
    """
    offenders = []
    for match in re.finditer(r'\bon[a-z]+\s*=\s*"([^"]*)"', _strip_block_comments(source)):
        if "${" in match.group(1):
            line = source[:match.start()].count("\n") + 1
            offenders.append(f"line {line}: {match.group(0)[:90]}")

    assert not offenders, (
        "these inline handlers interpolate a value into JavaScript through an "
        "HTML attribute:\n  " + "\n  ".join(offenders)
        + "\nEscaping cannot fix this — the parser decodes entities before the "
          "JS is parsed. Emit `data-action` plus `data-*` values and wire them "
          "with bindActions()."
    )


def test_status_renderers_use_text_not_markup(source: str):
    """Both take a plain string from callers, most often a backend `detail`."""
    for name in ("showStatus", "setStatusBox"):
        match = re.search(rf"function {name}\([^)]*\)\s*\{{(.*?)\n\}}", source, re.S)
        assert match, f"{name}() not found — did it move or get renamed?"
        body = match.group(1)
        assert "innerHTML" not in body, (
            f"{name}() builds its node with innerHTML. Its callers pass backend "
            f"error details, which carry uploaded filenames and voice names."
        )
        assert "textContent" in body, f"{name}() no longer sets textContent"


# Values that originate outside the page: a backend response, an uploaded
# filename, or a name the operator typed. Every interpolation of one into an
# innerHTML template must go through escapeHtml.
UNTRUSTED = (
    "voiceName",
    "deploymentLabel",
    "createdAtLabel",
    "job.job_id",
    "job.status",
    "voice.id",
    "voice.name",
    "voice.language",
    "voice.quality",
    "error.message",
)


def _innerhtml_assignments(source: str) -> list[tuple[int, str]]:
    """(line, text) for each `x.innerHTML = ...` statement, template included."""
    out = []
    for match in re.finditer(r"\.innerHTML\s*(?:\+)?=\s*", source):
        start = match.end()
        # Take the rest of the statement: balance backticks/parens crudely by
        # scanning to the first `;` at depth zero.
        depth = 0
        i = start
        while i < len(source):
            ch = source[i]
            if ch == "`":
                depth ^= 1
            elif ch == ";" and depth == 0:
                break
            i += 1
        out.append((source[:match.start()].count("\n") + 1, source[start:i]))
    return out


def test_untrusted_values_are_escaped_in_every_innerhtml_template(source: str):
    """The convention exists; these are the places that skipped it."""
    offenders = []
    for line, statement in _innerhtml_assignments(source):
        for name in UNTRUSTED:
            # Bare `${name}` — not wrapped in escapeHtml(...) or a method call.
            pattern = r"\$\{\s*" + re.escape(name) + r"\s*(?:\|\||\})"
            if re.search(pattern, statement):
                offenders.append(f"line {line}: ${{{name}}}")

    assert not offenders, (
        "these values come from a backend response or from operator input and "
        "are interpolated into innerHTML unescaped:\n  " + "\n  ".join(sorted(set(offenders)))
        + "\nWrap them in escapeHtml()."
    )


def test_html_building_helpers_that_take_a_selector_escape_it(source: str):
    """A voice id is operator-chosen, so it is not a safe CSS selector."""
    for match in re.finditer(r"querySelector(?:All)?\(`([^`]*)`\)", source):
        selector = match.group(1)
        if "${" not in selector:
            continue
        line = source[:match.start()].count("\n") + 1
        assert "CSS.escape" in selector, (
            f"line {line}: querySelector builds a selector from an interpolated "
            f"value without CSS.escape — {selector}"
        )


def test_escapehtml_still_covers_the_attribute_delimiters(source: str):
    """Values now land in data-* attributes, so quote escaping is load-bearing."""
    body = re.search(r"function escapeHtml\([^)]*\)\s*\{(.*?)\n\}", source, re.S)
    assert body, "escapeHtml() not found"
    for char in ("&", "<", ">", '"', "'"):
        assert f"/{char}/g" in body.group(1) or f"/\\{char}/g" in body.group(1), (
            f"escapeHtml no longer escapes {char!r}, which data-* attribute "
            f"values depend on"
        )


def test_bindactions_exists_and_is_used_by_every_rebuilt_list(source: str):
    """Rebuilding a list with innerHTML detaches its listeners; each renderer
    that does so must re-bind, or its buttons silently stop working."""
    assert "function bindActions(" in source, "bindActions() helper is missing"
    for renderer in ("refreshCustomVoices", "refreshTrainedModels", "refreshTrainingJobs"):
        match = re.search(rf"async function {renderer}\(\)\s*\{{(.*?)\n\}}", source, re.S)
        if not match:
            continue
        assert "bindActions(" in match.group(1), (
            f"{renderer}() rebuilds its container with innerHTML but never calls "
            f"bindActions(), so its action buttons do nothing"
        )


def test_template_does_not_reintroduce_interpolated_inline_handlers():
    """Jinja renders provider metadata into the page; same rule applies."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    offenders = [
        match.group(0)[:90]
        for match in re.finditer(r'\bon[a-z]+\s*=\s*"([^"]*)"', html)
        if "{{" in match.group(1) or "{%" in match.group(1)
    ]
    assert not offenders, (
        "index.html interpolates template data into an inline handler:\n  "
        + "\n  ".join(offenders)
    )


# --- script/template DOM contracts -------------------------------------------


def test_every_tab_has_the_panel_and_button_id_showtab_looks_up():
    """`showTab(x)` reveals `#x` and highlights `#x-button`.

    Both are looked up by id and both fail silently when absent — a mistyped tab
    id shows nothing and highlights nothing, with no console error. Cheap to
    assert, invisible to catch by hand.
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    tab_ids = sorted(set(re.findall(r"showTab\('([^']+)'\)", html)))
    assert tab_ids, "no showTab() calls found in index.html — did the tabs move?"

    ids = set(re.findall(r'\bid="([^"]+)"', html))
    missing = [
        f"{tab}: " + ", ".join(
            part for part, present in (
                (f"#{tab}", tab in ids), (f"#{tab}-button", f"{tab}-button" in ids))
            if not present
        )
        for tab in tab_ids
        if tab not in ids or f"{tab}-button" not in ids
    ]
    assert not missing, (
        "these tabs are missing the ids showTab() resolves:\n  " + "\n  ".join(missing)
    )


def test_showtab_does_not_identify_buttons_by_their_handler_text(source: str):
    """Matching on `onclick*=` made the script depend on how the template spells
    a function call, and matched any handler *containing* the id."""
    match = re.search(r"function showTab\([^)]*\)\s*\{(.*?)\n\}", source, re.S)
    assert match, "showTab() not found"
    # Comments explain why the old selector was wrong; only code counts.
    body = re.sub(r"//[^\n]*", "", _strip_block_comments(match.group(1)))
    assert "onclick" not in body, (
        "showTab() selects the active button by its onclick text again. A tab id "
        "that is a suffix of another then highlights whichever button comes "
        "first in document order. Look up `${tabId}-button` by id."
    )
