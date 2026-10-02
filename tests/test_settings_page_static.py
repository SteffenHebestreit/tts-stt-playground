"""Static invariants of the Settings page (`settings.html` + `settings.js`).

The page runs under its own Content-Security-Policy (`script-src 'self'; style-src
'self'`, `_SETTINGS_PAGE_CSP` in app.py): anything that needs inline script, an inline
style or markup built from strings would not work there, and markup built from strings
is also how a key name or a server message would turn into script. These read the
source; what the script does is tested in test_frontend_ui_logic.py (under Node, in the
real template) and test_frontend_ui_browser.py (in Chromium, under the real policy).

The numbers in the test names are the page tests of the phase-1 plan (54-58).
"""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from frontend_loader import SERVICE_DIR, load_frontend_app

SETTINGS_JS = SERVICE_DIR / "static" / "js" / "settings.js"
SETTINGS_HTML = SERVICE_DIR / "templates" / "settings.html"
APP_JS = SERVICE_DIR / "static" / "js" / "app.js"
INDEX_HTML = SERVICE_DIR / "templates" / "index.html"


@pytest.fixture(scope="module")
def source() -> str:
    return SETTINGS_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def template() -> str:
    return SETTINGS_HTML.read_text(encoding="utf-8")


def _code(text: str) -> str:
    """The script without comments (line numbers kept), so that explaining a rule is not a finding."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    text = re.sub(r"(?m)^[ \t]*//.*$", "", text)
    return re.sub(r"(?m)[ \t]{2,}//[ \t].*$", "", text)


class _Page(HTMLParser):
    """Every start tag with its attributes, and any text inside a <script>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict]] = []
        self.inline_script: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, {name: value or "" for name, value in attrs}))
        self._in_script = tag == "script"

    handle_startendtag = handle_starttag

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script and data.strip():
            self.inline_script.append(data.strip()[:80])


def _parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    return page


# --- 54: it parses -------------------------------------------------------------------------------


def test_54_settings_js_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    result = subprocess.run([node, "--check", str(SETTINGS_JS)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- 55: no markup from strings, no eval, nothing inline --------------------------------------------

FORBIDDEN = {
    "innerHTML": r"\binnerHTML\b",
    "outerHTML": r"\bouterHTML\b",
    "insertAdjacentHTML": r"\binsertAdjacentHTML\b",
    "document.write": r"\bdocument\.write(?:ln)?\s*\(",
    "eval": r"\beval\s*\(",
    "new Function": r"\bnew\s+Function\b",
    "Function()": r"(?<![\w.])Function\s*\(",
    "a string run by setTimeout/setInterval": r"\bset(?:Timeout|Interval)\s*\(\s*['\"`]",
    "a style attribute or style property (style-src 'self' refuses them)": r"\.style\b|\.cssText\b|setAttribute\(\s*['\"]style",
    "an on* handler property": r"\.on[a-z]+\s*=(?!=)",
    "an on* attribute": r"setAttribute\(\s*['\"]on",
    "a javascript: URL": r"(?i)javascript:",
    "localStorage (a key must not outlive the tab)": r"\blocalStorage\b",
    "document.cookie": r"\bdocument\.cookie\b",
}


@pytest.mark.parametrize("what", sorted(FORBIDDEN))
def test_55_the_script_builds_no_markup_from_strings_and_runs_no_strings(source, what):
    code = _code(source)
    found = [f"line {code[:m.start()].count(chr(10)) + 1}: {code[m.start():].splitlines()[0][:80]}"
             for m in re.finditer(FORBIDDEN[what], code)]
    assert not found, f"settings.js uses {what}:\n  " + "\n  ".join(found)


def test_55_text_reaches_the_page_through_text_nodes(source):
    """The positive side of the rule: one way in for text, and it is textContent."""
    code = _code(source)
    assert "textContent" in code and "createTextNode" in code
    assert "createElement" in code


# --- 56: every action has a handler, every id it looks up exists ---------------------------------


def _table(source: str, name: str) -> set[str]:
    block = re.search(rf"const {name} = \{{(.*?)\n\}};", source, re.S)
    assert block, f"{name} not found in settings.js"
    return set(re.findall(r"^\s*'([a-z-]+)'\s*:", block.group(1), re.M))


def test_56_every_action_has_a_handler_and_every_handler_an_action(source, template):
    emitted = set(re.findall(r'data-action="([a-z-]+)"', template)) | set(re.findall(r"actionButton\(\s*'([a-z-]+)'", source))
    handled = _table(source, "ACTIONS")
    assert emitted and handled
    assert not emitted - handled, f"buttons that do nothing: {sorted(emitted - handled)}"
    assert not handled - emitted, f"handlers nothing triggers: {sorted(handled - emitted)}"
    assert not re.search(r"dataset\.action\s*=(?!=)|setAttribute\(\s*'data-action'", source), \
        "an action is only ever emitted through actionButton()"
    enter = set(re.findall(r'data-enter="([a-z-]+)"', template))
    assert enter and enter <= handled, f"Enter runs an action that does not exist: {sorted(enter - handled)}"


def test_56_every_watched_input_has_a_watcher(source, template):
    watched = set(re.findall(r'data-watch="([a-z-]+)"', template)) | set(re.findall(r"\bwatch: '([a-z-]+)'", source))
    assert watched and watched == _table(source, "WATCHERS")


def test_56_every_id_the_script_looks_up_exists(source, template):
    """A lookup of a missing id is a silent no-op for setText/setShown, so a renamed element
    would just stop updating."""
    code = _code(source)
    in_template = set(re.findall(r'\bid="([^"{}]+)"', template))
    created = set(re.findall(r"\bid: '([^']+)'", code))
    created_prefixes = set(re.findall(r"\bid(?::| =) `([^`$]+)\$\{", code))
    looked_up = set(re.findall(r"\b(?:byId|setText|setShown|hideDialog|dialogOpen|renderFields)\(\s*'([^']+)'", code))
    for call in re.finditer(r"\b(?:showDialog|copyFrom)\(([^;]*?)\);", code):
        looked_up |= set(re.findall(r"'([a-z][\w-]*)'", call.group(1)))
    prefixes = set(re.findall(r"\b(?:byId|setText|setShown|renderFields)\(\s*`([^`$]+)\$\{", code))
    assert len(looked_up) > 20 and prefixes, "the lookups were not found: did the helpers change?"
    missing = sorted(looked_up - in_template - created)
    assert not missing, f"settings.js looks up ids that neither the template nor the script defines: {missing}"
    unknown = sorted(prefix for prefix in prefixes
                     if prefix not in created_prefixes and not any(i.startswith(prefix) for i in in_template))
    assert not unknown, f"settings.js looks up ids by prefixes nothing creates: {unknown}"


def test_56_every_group_of_the_schema_has_its_card(template):
    sys.path.insert(0, str(SERVICE_DIR))
    try:
        schema = importlib.import_module("settings_schema")
    finally:
        sys.path.remove(str(SERVICE_DIR))
    ids = set(re.findall(r'\bid="([^"{}]+)"', template))
    missing = [group for group, _label in schema.GROUPS if f"fields-{group}" not in ids or f"card-{group}" not in ids]
    assert not missing, f"settings the page would have nowhere to show: {missing}"


# --- 57: nothing inline in the template ------------------------------------------------------------


def test_57_the_template_has_no_inline_script_style_or_handler(template):
    page = _parse(template)
    scripts = [attrs for tag, attrs in page.tags if tag == "script"]
    assert scripts, "the page loads no script?"
    assert all(attrs.get("src", "").startswith("/static/") for attrs in scripts), scripts
    assert page.inline_script == [], "inline script: the page's CSP would refuse to run it"
    assert not [tag for tag, _ in page.tags if tag in ("style", "form", "iframe", "object", "embed", "base")]
    handlers = [(tag, name) for tag, attrs in page.tags for name in attrs if name == "style" or name.startswith("on")]
    assert not handlers, f"inline style or handler attributes: {handlers}"


def test_57_everything_the_template_loads_is_this_servers_own(template):
    page = _parse(template)
    stylesheets = [attrs for tag, attrs in page.tags if tag == "link" and attrs.get("rel") == "stylesheet"]
    assert stylesheets and all(attrs["href"].startswith("/static/") for attrs in stylesheets)
    urls = [value for _tag, attrs in page.tags for name, value in attrs.items()
            if name in ("src", "href", "action", "formaction", "poster", "data") and value]
    foreign = [url for url in urls if re.match(r"(?i)\s*(?:[a-z][a-z0-9+.-]*:|//)", url)]
    assert not foreign, f"the CSP allows 'self' only, and links stay on this server: {foreign}"
    assert any(tag == "meta" and attrs.get("name") == "viewport" for tag, attrs in page.tags), "phone width"
    assert any(tag == "html" and attrs.get("lang") for tag, attrs in page.tags)


# --- 58: served under its policy ------------------------------------------------------------------


def test_58_the_page_is_served_under_its_own_policy_and_everything_it_loads_is_allowed(tmp_path):
    module = load_frontend_app({"TTS_STT_SETTINGS_DIR": str(tmp_path)})
    client = TestClient(module.app)
    r = client.get("/settings", headers={"Accept": "text/html"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    policy = r.headers["content-security-policy"]
    assert policy == module._SETTINGS_PAGE_CSP
    directives = dict(part.strip().split(" ", 1) for part in policy.split(";"))
    assert (directives["default-src"], directives["script-src"], directives["style-src"]) == ("'self'",) * 3
    assert (directives["frame-ancestors"], directives["base-uri"], directives["form-action"]) == ("'none'",) * 3
    assert "unsafe" not in policy
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
    assert "{{" not in r.text and "}}" not in r.text, "the template variables were filled in"

    page = _parse(r.text)
    assets = [attrs["src"] for tag, attrs in page.tags if tag == "script"]
    assets += [attrs["href"] for tag, attrs in page.tags if tag == "link" and attrs.get("rel") == "stylesheet"]
    for url in assets:
        asset = client.get(url)
        assert asset.status_code == 200, url
        if url.split("?")[0].endswith(".css"):
            assert not re.search(r"@import|url\(\s*['\"]?\s*(?:[a-z]+:)?//", asset.text), \
                "the stylesheet loads nothing from elsewhere"


# --- the page and the main UI ---------------------------------------------------------------------


def test_the_page_and_the_main_ui_share_one_key_slot(source):
    """A key entered on either page works on both: the same per-tab sessionStorage slot."""
    slot = re.search(r"const API_KEY_STORAGE_KEY = '([^']+)'", APP_JS.read_text(encoding="utf-8"))
    assert slot, "app.js no longer names its key slot API_KEY_STORAGE_KEY"
    assert f"const API_KEY_STORAGE_KEY = '{slot.group(1)}';" in source
    assert "window.sessionStorage" in _code(source)


def test_the_main_page_links_to_the_settings():
    index = INDEX_HTML.read_text(encoding="utf-8")
    link = re.search(r'<a href="/settings"[^>]*>(.*?)</a>', index, re.S)
    assert link and "Settings" in link.group(1)


def test_an_empty_field_shows_an_example_that_reads_as_one(source, template):
    """A placeholder that looks like a host name reads as a value that is set, in an empty field
    that is exactly the list a reader is checking. And no real host name ships as an example."""
    placeholders = re.search(r"const PLACEHOLDERS = \{(.*?)\};", source, re.S)
    assert placeholders, "settings.js no longer names its placeholders PLACEHOLDERS"
    examples = re.findall(r"^\s*([A-Z_]+): '([^']*)',$", placeholders.group(1), re.M)
    assert {key for key, _ in examples} == {"TRUSTED_HOSTS", "TRUSTED_ORIGINS", "ALLOWED_ORIGINS"}
    for key, text in examples:
        assert text.startswith("e.g. ") and "example." in text, key
    schema = (SERVICE_DIR / "settings_schema.py").read_text(encoding="utf-8")
    for name, text in (("settings.js", source), ("settings.html", template), ("settings_schema.py", schema)):
        assert "speach" not in text, f"{name} ships the owner's host name as an example"
