"""Static invariants of the browser layer (`app.js` + `index.html`).

Two groups: escaping rules, and the DOM contracts that couple the script to the
template. These read the source; what the script *does* is exercised by
`test_frontend_ui_logic.py` (under Node) and `test_frontend_ui_browser.py` (in
Chromium).

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
import shutil
import subprocess
from html.parser import HTMLParser
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


def test_request_urls_encode_every_interpolated_segment(source: str):
    """An id with `/`, `?` or `#` in it changes which route a request addresses.

    Behaviour is covered in test_frontend_ui_logic.py; this is the tripwire for
    the next fetch() someone adds.
    """
    offenders = []
    for match in re.finditer(r"(?:fetch|getProviderApiPath|addModule)\(([^;]*?)`([^`]*)`", source):
        template = match.group(2)
        for expression in re.findall(r"\$\{([^}]*)\}", template):
            # `path` is getProviderApiPath's own already-built suffix argument.
            if expression.strip() != "path" and not expression.strip().startswith("encodeURIComponent("):
                line = source[:match.start()].count("\n") + 1
                offenders.append(f"line {line}: ${{{expression.strip()}}} in `{template[:60]}`")
    assert not offenders, "unencoded URL segments:\n  " + "\n  ".join(offenders)


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


# The renderers that replace a container's innerHTML and must re-bind after.
# Names are asserted to exist rather than skipped: an earlier version of this
# list said "refreshTrainedModels", which is not what the function is called, so
# the check quietly passed over the models table entirely.
LIST_RENDERERS = ("refreshCustomVoices", "refreshModels", "refreshTrainingJobs")


@pytest.mark.parametrize("renderer", LIST_RENDERERS)
def test_bindactions_is_used_by_every_rebuilt_list(source: str, renderer: str):
    """Rebuilding a list with innerHTML detaches its listeners; each renderer
    that does so must re-bind, or its buttons silently stop working."""
    assert "function bindActions(" in source, "bindActions() helper is missing"

    start = source.find(f"async function {renderer}()")
    assert start != -1, (
        f"{renderer}() not found in app.js — if it was renamed, rename it here "
        f"too rather than letting this check skip it"
    )
    # To the start of the next top-level function, so the body is not truncated
    # at the first dedented brace inside it.
    nxt = source.find("\nasync function ", start + 1)
    other = source.find("\nfunction ", start + 1)
    end = min(x for x in (nxt, other, len(source)) if x != -1)
    body = source[start:end]

    assert "innerHTML" in body, f"{renderer}() no longer rebuilds its container?"
    assert "bindActions(" in body, (
        f"{renderer}() rebuilds its container with innerHTML but never calls "
        f"bindActions(), so its action buttons do nothing"
    )
    # Against the assignment that writes the built list, not merely the last
    # innerHTML in the function — that one is the catch block's error message,
    # which comes after bindActions in source order and needs no handlers.
    built = body.find(".innerHTML = html")
    assert built != -1, (
        f"{renderer}() no longer assigns its built markup via `.innerHTML = html`; "
        f"update this check to match how it renders now"
    )
    assert body.index("bindActions(") > built, (
        f"{renderer}() calls bindActions() before writing the list, so the "
        f"handlers attach to elements that are then replaced"
    )


def _template_actions() -> set[str]:
    return set(re.findall(r'data-action="([a-z-]+)"', INDEX_HTML.read_text(encoding="utf-8")))


def test_every_data_action_has_a_handler(source: str):
    """A data-action with no matching bindActions key is a dead button — it
    looks live, does nothing, and reports no error. Actions are emitted by the
    list renderers in app.js and by the static controls in the template."""
    emitted = set(re.findall(r'data-action="([a-z-]+)"', source)) | _template_actions()
    bound: set[str] = set()
    for block in re.finditer(r"bindActions\([^,]+,\s*\{(.*?)\}\s*\)", source, re.S):
        bound.update(re.findall(r"^\s*'?([a-zA-Z-]+)'?\s*:", block.group(1), re.M))

    assert emitted, "no data-action attributes found — did the renderers change?"
    assert not (emitted - bound), (
        f"these data-action values are emitted but have no handler: "
        f"{sorted(emitted - bound)}"
    )
    assert not (bound - emitted), (
        f"these handlers are registered but nothing emits them: "
        f"{sorted(bound - emitted)}"
    )


def test_data_action_events_name_a_real_event():
    """`data-action-event` picks the DOM event a control listens for; a typo
    there binds a listener that never fires."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    events = set(re.findall(r'data-action-event="([^"]*)"', html))
    assert events, "no data-action-event found — did the change-driven controls move?"
    assert events <= {"change", "input", "click", "submit"}, events


class _InlineHandlerFinder(HTMLParser):
    """Collects every event-handler attribute (on*) the way a browser parses them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[str] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name.lower().startswith("on"):
                line = self.getpos()[0]
                self.found.append(f"line {line}: <{tag} {name}=\"{(value or '')[:50]}\">")

    handle_startendtag = handle_starttag


def test_template_has_no_inline_event_handlers():
    """No `onclick=`/`onchange=`/... anywhere in index.html.

    Inline handlers are inline script as far as a Content-Security-Policy is
    concerned, so any one left keeps `script-src 'unsafe-inline'` mandatory.
    Controls carry `data-action` and are wired by bindStaticActions().
    """
    finder = _InlineHandlerFinder()
    finder.feed(INDEX_HTML.read_text(encoding="utf-8"))
    assert not finder.found, "inline event handlers in index.html:\n  " + "\n  ".join(finder.found)


def test_markup_built_in_app_js_has_no_inline_handlers(source: str):
    """The same rule for the HTML app.js writes into the page, interpolated or not."""
    code = re.sub(r"(?m)^\s*//[^\n]*", "", _strip_block_comments(source))
    finder = _InlineHandlerFinder()
    for match in re.finditer(r"`([^`]*)`", code, re.S):
        finder.feed(match.group(1).replace("${", "").replace("}", ""))
    assert not finder.found, "inline event handlers in markup built by app.js:\n  " + "\n  ".join(finder.found)


def test_app_js_parses(tmp_path):
    """`node --check`: catches a syntax slip that the string-level tests above would miss."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    result = subprocess.run([node, "--check", str(APP_JS)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_static_ids_app_js_looks_up_exist_in_the_template(source: str):
    """`getElementById('x')` on a missing id is a silent no-op for most helpers
    (setElementText, showStatus, ...), so a renamed element just stops updating."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    template_ids = set(re.findall(r'\bid="([^"{}]+)"', html))
    created_by_script = set(re.findall(r'id="([^"$]+)"', source)) | set(re.findall(r"\.id = '([^']+)'", source))
    lookups = set()
    for pattern in (
        r"getElementById\(\s*'([^']+)'",
        r"(?:setElementText|setElementFormattedText|setInputPlaceholder|setInputValue|showStatus|populateSelectOptions|applyInputNumberConfig|getSelectedTrainingDeploymentTarget|setGroupVisible)\(\s*'([^']+)'",
    ):
        lookups.update(re.findall(pattern, source))
    missing = sorted(lookups - template_ids - created_by_script)
    assert not missing, f"app.js looks up ids that index.html does not define: {missing}"
    # The TTS panel hides what an engine does not have by these wrappers' ids.
    assert {"tts-quality-group", "tts-gender-group", "tts-speed-group", "tts-voice-group",
            "custom-voices-panel"} <= lookups, "the engine-dependent groups are no longer looked up"


# --- script/template DOM contracts -------------------------------------------


def test_every_tab_has_the_panel_and_button_id_showtab_looks_up():
    """`showTab(x)` reveals `#x` and highlights `#x-button`.

    Both are looked up by id and both fail silently when absent — a mistyped tab
    id shows nothing and highlights nothing, with no console error. Cheap to
    assert, invisible to catch by hand.
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    tab_ids = sorted(set(re.findall(r'data-action="show-tab"\s+data-tab="([^"]+)"', html)))
    assert tab_ids, "no show-tab controls found in index.html — did the tabs move?"

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
