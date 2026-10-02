"""Page-level tests of the web UI, driven by a real Chromium through Playwright.

The gateway (`frontend-service/app.py`) is served for real, so the template,
static files and registry the page receives are the ones that ship. Everything
behind `/api/**` is answered by the test (no backend is running) and the live
transcription WebSocket is played by the test too, frame by frame, so a scenario
such as "the server sends a shorter `confirmed`" or "the socket closes with 1011"
can be produced exactly.

Skips when Playwright or its Chromium is not installed (CI only installs the
test requirements; the browser is preinstalled in the dev image). Like the other
frontend suites it honours `FRONTEND_SERVICE_DIR`, which is how these were run
against the previous UI code to show that they fail there.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import threading
import time
import wave
from urllib.parse import urlparse

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

# tests/requirements.txt does not list uvicorn (the other suites use TestClient);
# a real socket is what a browser needs.
uvicorn = pytest.importorskip("uvicorn", reason="uvicorn not installed")

from frontend_loader import load_frontend_app  # noqa: E402

# Every wait in this file. Lowered when the suite is pointed at older code
# (FRONTEND_SERVICE_DIR) so the tests that are meant to fail there do so quickly.
TIMEOUT_MS = int(os.getenv("FRONTEND_UI_TIMEOUT_MS", "10000"))

CHROMIUM_ARGS = [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
]


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def gateway_url():
    module = load_frontend_app()
    server = uvicorn.Server(uvicorn.Config(module.app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:
        pytest.fail("the gateway did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            instance = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        except Exception as exc:  # browser binary missing or not launchable here
            pytest.skip(f"chromium is not available: {exc}")
        yield instance
        instance.close()


def _wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class Api:
    """Answers `/api/**` for the page and remembers what it was asked."""

    def __init__(self):
        self.calls: list[dict] = []
        self.unmocked: list[str] = []
        self.deferred: list = []
        self.table: dict[tuple[str, str], object] = {}
        self._defaults()

    def _defaults(self):
        health = {"healthy": True, "latency_ms": 3, "status_code": 200}
        self.on("GET", "/api/health", {"providers": {
            "piper": {**health, "default_language": "de"},
            "whisper": health,
            "qwen3": health,
        }})
        self.on("GET", "/api/providers/piper/voices", {"voices": [
            {"id": "de_DE-thorsten-medium", "name": "thorsten", "language": "de_DE", "kind": "default",
             "description": "medium", "raw": {"quality": "medium"}},
        ], "default_language": "de"})
        self.on("GET", "/api/providers/piper/custom-voices", {"voices": []})
        self.on("GET", "/api/providers/qwen3/voices", {"voices": [
            {"id": "Vivian", "name": "Vivian", "kind": "builtin"}]})
        self.on("GET", "/api/providers/qwen3/models", {
            "models": [
                {"id": "base-1", "name": "Base", "description": "clone", "is_current": True, "capabilities_text": "clone"},
                {"id": "custom-CustomVoice", "name": "Custom", "description": "", "is_current": False},
            ],
            "current_model": {"id": "base-1", "name": "Base", "capabilities_text": "clone"},
        })
        self.on("GET", "/api/providers/qwen3/status", {"model_loaded": True, "model_name": "Base", "device_type": "gpu",
                                                      "device_name": "CUDA", "speakers": ["Vivian"]})
        self.on("GET", "/api/providers/qwen3/saved-voices", {"voices": [
            {"id": "v1", "name": "Anna", "language": "German", "reference_preview": "Hallo",
             "created_at_display": "2026-01-01 10:00 UTC"}]})
        self.on("POST", "/api/providers/qwen3/models/select", {"model": {"id": "x", "name": "X"}})
        self.on("GET", "/api/training/deployment-targets", {"default_target": "piper-volume", "targets": {
            "piper-volume": {"display_name": "Piper shared volume", "deployment_contract": "c", "capabilities": []}}})
        self.on("GET", "/api/training/jobs", {"jobs": []})
        self.on("POST", "/api/tts", ("audio/wav", _wav()))
        self.on("POST", "/api/stt", {"text": "hallo welt", "language": "de"})
        # asked before the live socket is opened, to collect an API key if one is needed
        self.on("POST", "/api/auth/check", ("text/plain", b""), status=204)

    def on(self, method, path, body, status=200):
        """`body` is JSON-able, a (content_type, bytes) pair, or a callable(route, request)."""
        self.table[(method, path)] = (status, body)

    def calls_to(self, method, path_prefix):
        return [c for c in self.calls if c["method"] == method and c["path"].startswith(path_prefix)]

    def handle(self, route):
        request = route.request
        parsed = urlparse(request.url)
        try:
            body = request.post_data
        except UnicodeDecodeError:       # multipart upload with binary parts
            body = None
        self.calls.append({"method": request.method, "path": parsed.path, "url": request.url, "body": body})
        entry = self.table.get((request.method, parsed.path))
        if entry is None:
            self.unmocked.append(f"{request.method} {parsed.path}")
            route.fulfill(status=404, content_type="application/json", body=json.dumps({"detail": "unmocked"}))
            return
        status, body = entry
        if callable(body):
            body(route, request)
        elif isinstance(body, tuple):
            route.fulfill(status=status, content_type=body[0], body=body[1])
        else:
            route.fulfill(status=status, content_type="application/json", body=json.dumps(body))


class Live:
    """Plays the /ws/stt server for one or more sockets."""

    def __init__(self):
        self.sockets = []
        self.received = []

    def handler(self, ws):
        index = len(self.sockets)
        self.sockets.append(ws)
        ws.on_message(lambda message, index=index: self.received.append((index, message)))

    def send(self, frame, socket=-1):
        self.sockets[socket].send(json.dumps(frame))

    def close(self, code, reason="", socket=-1):
        self.sockets[socket].close(code=code, reason=reason)

    def sent_stop(self, socket=0):
        return any(i == socket and isinstance(m, str) and '"stop"' in m for i, m in self.received)


class Ui:
    def __init__(self, page, api, live, base):
        self.page, self.api, self.live, self.base = page, api, live, base
        self.console: list[str] = []

    def open(self):
        self.page.goto(self.base, wait_until="load")
        self.page.wait_for_selector("#stt-tab-button")
        self.page.wait_for_function("document.querySelectorAll('#tts-language-select option').length > 1")
        return self

    def text(self, selector):
        return self.page.inner_text(selector)

    def wait_text(self, selector, fragment, timeout=TIMEOUT_MS):
        self.page.wait_for_function(
            "([sel, text]) => { const el = document.querySelector(sel); return !!el && el.innerText.includes(text); }",
            arg=[selector, fragment], timeout=timeout,
        )

    def wait_calls(self, method, prefix, count=1, timeout=TIMEOUT_MS):
        deadline = time.time() + timeout / 1000
        while time.time() < deadline:
            if len(self.api.calls_to(method, prefix)) >= count:
                return self.api.calls_to(method, prefix)
            self.page.wait_for_timeout(25)
        pytest.fail(f"no {method} {prefix} (wanted {count}); saw {[c['method'] + ' ' + c['path'] for c in self.api.calls]}")

    def errors(self):
        return [m for m in self.console if "favicon" not in m]

    def use_qwen3(self):
        self.page.click("label.radio-label:has(input[name=tts-engine][value=qwen3])")
        self.page.wait_for_selector("#qwen3-tts-tab-button", state="visible")

    def use_engine(self, provider_id):
        self.page.click(f"label.radio-label:has(input[name=tts-engine][value={provider_id}])")
        self.page.wait_for_function(
            "(id) => document.querySelector(`input[name=tts-engine][value=${id}]`).checked", arg=provider_id)


@contextlib.contextmanager
def _ui_session(browser, base):
    context = browser.new_context(accept_downloads=False)
    page = context.new_page()
    page.set_default_timeout(TIMEOUT_MS)
    api, live = Api(), Live()
    harness = Ui(page, api, live, base)
    page.on("pageerror", lambda err: harness.console.append(f"pageerror: {err}"))
    page.on("console", lambda m: harness.console.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    page.on("dialog", lambda dialog: dialog.accept())
    page.route("**/favicon.ico", lambda route: route.fulfill(status=204))
    page.route("**/api/**", api.handle)
    page.route_web_socket("**/ws/stt**", live.handler)
    try:
        yield harness
    finally:
        context.close()


@pytest.fixture
def ui(browser, gateway_url):
    with _ui_session(browser, gateway_url) as harness:
        yield harness


# --- loading and wiring ------------------------------------------------------


def test_page_loads_without_errors_and_without_inline_handlers(ui):
    ui.open()
    inline = ui.page.evaluate("""() => Array.from(document.querySelectorAll('*'))
        .flatMap((el) => Array.from(el.attributes).filter((a) => a.name.startsWith('on'))
        .map((a) => el.tagName + '[' + a.name + ']'))""")
    assert inline == []
    assert ui.errors() == []
    assert ui.api.unmocked == [], "the page called an API this test does not know about"
    # The worklet cache-buster is now available without the inline bootstrap script.
    assert ui.page.evaluate("document.documentElement.dataset.appVersion")


def test_ui_works_under_a_script_src_self_policy_without_inline_scripts(ui):
    """The point of removing the inline handlers: a CSP that forbids inline script
    must not break the page. Inline `<script>` blocks are stripped from the served
    HTML and the policy is enforced by the browser, so a handler or bootstrap that
    still needs inline execution shows up as a CSP violation or a dead control."""
    def serve(route):
        response = route.fetch()
        html = re.sub(r"<script>.*?</script>", "", response.text(), flags=re.S)
        headers = {**response.headers, "content-security-policy": "script-src 'self'"}
        route.fulfill(response=response, body=html, headers=headers)

    ui.page.route("**/", serve)
    ui.open()
    ui.page.click("#training-tab-button")
    assert ui.page.evaluate("document.querySelector('.tab-content.active').id") == "training-tab"
    # The registry-driven controls came from the embedded JSON, not from window.PROVIDER_REGISTRY.
    assert ui.page.evaluate("document.querySelectorAll('#tts-language-select option').length") > 1
    ui.page.click("#start-training-button")
    ui.wait_text("#training-progress-status", "voice model name")
    assert ui.errors() == [], "a CSP violation or script error was reported"


def test_every_static_control_is_wired(ui):
    """Each control the inline handlers used to drive still does its job."""
    ui.open()
    api, page = ui.api, ui.page

    # tabs
    page.click("#training-tab-button")
    assert page.get_attribute("#training-tab-button", "aria-current") == "page"
    assert page.get_attribute("#stt-tab-button", "aria-current") is None

    # validation paths of the request buttons
    page.click("#start-training-button")
    ui.wait_text("#training-progress-status", "voice model name")
    page.click("#resume-training-button")
    ui.wait_text("#continue-status", "voice model name")
    page.click("#train-from-dataset-button")
    ui.wait_text("#continue-status", "voice model name")
    page.click("#stt-tab-button")
    page.click("#process-stt-button")
    ui.wait_text("#stt-result-status", "select an audio file")

    # piper TTS tab
    page.click("#tts-tab-button")
    before = len(api.calls_to("GET", "/api/providers/piper/voices"))
    page.click("#tts-tab [data-action=refresh-tts-voices]")
    ui.wait_calls("GET", "/api/providers/piper/voices", before + 1)
    before = len(api.calls_to("GET", "/api/providers/piper/custom-voices"))
    page.click("#tts-tab [data-action=refresh-custom-voices]")
    ui.wait_calls("GET", "/api/providers/piper/custom-voices", before + 1)
    page.click("#generate-tts-button")
    ui.wait_text("#tts-result-status", "Speech generated")
    assert json.loads(api.calls_to("POST", "/api/tts")[0]["body"])["provider"] == "piper"

    # qwen3 tabs
    ui.use_qwen3()
    page.click("#qwen3-builtin-generate-button")
    ui.wait_text("#qwen3-builtin-status", "Speech generated")
    page.select_option("#qwen3-model-select", "custom-CustomVoice")
    ui.wait_calls("POST", "/api/providers/qwen3/models/select")

    page.click("#qwen3-cloning-tab-button")
    page.wait_for_function("document.querySelectorAll('#qwen3-saved-voice-select option').length > 0")
    before = len(api.calls_to("GET", "/api/providers/qwen3/saved-voices"))
    page.click("#saved-voices-refresh-button")
    ui.wait_calls("GET", "/api/providers/qwen3/saved-voices", before + 1)
    page.check("input[name=voice-source][value=upload]", force=True)
    assert page.is_visible("#clone-audio-mode") and not page.is_visible("#clone-saved-mode")
    page.check("input[name=voice-source][value=saved]", force=True)
    assert page.is_visible("#clone-saved-mode")
    assert not page.is_visible("#ref-text-group")
    page.check("input[name=voice-source][value=upload]", force=True)
    page.check("#enable-ref-text")
    assert page.is_visible("#ref-text-group")
    page.check("input[name=voice-source][value=saved]", force=True)
    page.click("#generate-qwen3-speech-btn")          # saved-voice mode, voice Anna selected
    ui.wait_calls("POST", "/api/providers/qwen3/saved-voices/v1/tts")


def test_engine_switch_leaves_the_panel_that_belongs_to_the_other_engine(ui):
    """The old check asked the *panel* for `piper-only`, but only the tab buttons
    carry it, so choosing Qwen3 while looking at the Piper panel left that panel
    showing under an engine it does not serve."""
    ui.open()
    ui.page.click("#tts-tab-button")
    assert ui.page.evaluate("document.querySelector('.tab-content.active').id") == "tts-tab"
    ui.use_qwen3()
    assert ui.page.evaluate("document.querySelector('.tab-content.active').id") == "qwen3-tts-tab"
    ui.page.click("#qwen3-cloning-tab-button")
    ui.page.click("label.radio-label:has(input[name=tts-engine][value=piper])")
    assert ui.page.evaluate("document.querySelector('.tab-content.active').id") == "stt-tab"


# --- catalog display ---------------------------------------------------------


def test_custom_voice_card_shows_the_quality_from_the_normalised_catalog(ui):
    ui.api.on("GET", "/api/providers/piper/custom-voices", {"voices": [
        {"id": "luna", "name": "luna", "language": "de_DE", "kind": "custom", "description": "high",
         "raw": {"quality": "high", "language": "de_DE"}}]})
    ui.open()
    ui.page.click("#tts-tab-button")
    ui.wait_text("#custom-voices-list", "luna")
    assert "Quality: high" in ui.text("#custom-voices-list")


def test_tts_auto_language_option_is_not_called_auto_detect(ui):
    """Piper's "auto" was labelled 'Auto-Detect' although it fell through to a
    fixed language; the server now reports which one, and the option names it."""
    ui.open()
    ui.page.wait_for_function(
        "Array.from(document.querySelectorAll('#tts-language-select option')).some(o => o.value === 'auto' && o.textContent.includes('German'))")
    label = ui.page.evaluate("Array.from(document.querySelectorAll('#tts-language-select option')).find(o => o.value === 'auto').textContent")
    assert "Auto-Detect" not in label
    assert "German" in label
    # The wire value is unchanged.
    ui.page.click("#tts-tab-button")
    ui.page.click("#generate-tts-button")
    ui.wait_calls("POST", "/api/tts")
    assert json.loads(ui.api.calls_to("POST", "/api/tts")[0]["body"])["language"] == "auto"


def test_the_stt_language_list_keeps_its_auto_detect_label(ui):
    """Whisper really does detect the language, so its 'Auto-Detect' stays."""
    ui.open()
    labels = ui.page.evaluate("Array.from(document.querySelectorAll('#stt-language option')).map(o => o.textContent)")
    assert any("Auto-Detect" in label for label in labels)


# --- one panel, three engines: Piper, Magpie and Chatterbox ----------------------
#
# A second gateway with both optional engines enabled. The panel used to show Piper's
# controls (and send their values) for all three, keep Piper's voice list after a
# switch, and ask Magpie and Chatterbox for Piper's custom voices.

SPEAKERS = ["Aria", "Jason", "John", "Leo", "Sofia"]
PANEL_GROUPS = ["#tts-quality-group", "#tts-gender-group", "#tts-speed-group", "#tts-voice-group",
                "#custom-voices-panel"]


@pytest.fixture(scope="module")
def three_engine_gateway_url():
    module = load_frontend_app({"ENABLE_MAGPIE_TTS": "true", "ENABLE_CHATTERBOX_TTS": "true"})
    server, thread, port = _serve(module)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def engines_ui(browser, three_engine_gateway_url):
    with _ui_session(browser, three_engine_gateway_url) as harness:
        # What the gateway answers for Magpie's /speakers (speaker-catalog-v1).
        harness.api.on("GET", "/api/providers/magpie/voices", {
            "provider": "magpie", "contract": "speaker-catalog-v1", "default_language": "de",
            "voices": [{"id": name, "name": name, "language": "multilingual", "kind": "builtin",
                        "description": "Built-in speaker. Languages: de, en, es, fr..."} for name in SPEAKERS],
        })
        yield harness


def _visible_groups(ui):
    return [group for group in PANEL_GROUPS if ui.page.is_visible(group)]


def test_the_panel_shows_each_engine_only_the_controls_it_has(engines_ui):
    ui = engines_ui
    ui.open()
    ui.page.click("#tts-tab-button")
    assert _visible_groups(ui) == PANEL_GROUPS
    ui.use_engine("magpie")
    assert _visible_groups(ui) == ["#tts-voice-group"]
    ui.page.wait_for_function(
        "Array.from(document.querySelectorAll('#tts-voice-select option')).some(o => o.value === 'Aria')")
    ui.use_engine("chatterbox")
    assert _visible_groups(ui) == []
    ui.use_engine("piper")
    assert _visible_groups(ui) == PANEL_GROUPS
    assert ui.api.unmocked == [], "an engine was asked for a list it does not have"
    assert ui.errors() == []


def test_a_piper_voice_is_not_sent_to_magpie(engines_ui):
    ui = engines_ui
    ui.open()
    ui.page.click("#tts-tab-button")
    ui.page.wait_for_function("document.querySelectorAll('#tts-voice-select option').length > 1")
    ui.page.select_option("#tts-voice-select", "de_DE-thorsten-medium")
    ui.use_engine("magpie")
    ui.page.click("#generate-tts-button")
    ui.wait_text("#tts-result-status", "Speech generated")
    body = json.loads(ui.api.calls_to("POST", "/api/tts")[-1]["body"])
    assert body["provider"] == "magpie"
    assert not {"voice", "speed", "quality", "gender"} & set(body), body

    ui.page.wait_for_function(
        "Array.from(document.querySelectorAll('#tts-voice-select option')).some(o => o.value === 'Aria')")
    labels = ui.page.evaluate("Array.from(document.querySelectorAll('#tts-voice-select option')).map(o => o.textContent)")
    assert labels == ["Service default voice", *SPEAKERS]


def test_switching_back_to_piper_on_the_open_tab_fills_its_custom_voices(engines_ui):
    ui = engines_ui
    ui.api.on("GET", "/api/providers/piper/custom-voices", {"voices": [
        {"id": "luna", "name": "luna", "language": "de_DE", "kind": "custom", "description": "high",
         "raw": {"quality": "high", "language": "de_DE"}}]})
    ui.open()
    ui.use_engine("magpie")
    ui.page.click("#tts-tab-button")
    ui.page.wait_for_function(
        "Array.from(document.querySelectorAll('#tts-voice-select option')).some(o => o.value === 'Aria')")
    assert not ui.page.is_visible("#custom-voices-panel")
    assert ui.api.calls_to("GET", "/api/providers/piper/custom-voices") == []
    ui.use_engine("piper")
    ui.wait_text("#custom-voices-list", "luna")
    assert ui.page.is_visible("#custom-voices-panel")


# --- request paths and text safety ------------------------------------------


def test_ids_from_the_job_table_are_encoded_in_request_urls(ui):
    hostile = "j/1?x#y"
    ui.api.on("GET", "/api/training/jobs", {"jobs": [
        {"job_id": hostile, "status": "completed", "voice_name": "luna", "progress": 100,
         "deployment_target_label": "Piper shared volume", "created_at_display": "2026-01-01 10:00 UTC"}]})
    ui.api.on("DELETE", "/api/training/model/j%2F1%3Fx%23y", {"ok": True})
    ui.open()
    ui.page.click("#training-tab-button")
    ui.wait_text("#models-list", "luna")
    ui.page.click("#models-list [data-action=delete]")
    deadline = time.time() + TIMEOUT_MS / 1000
    while time.time() < deadline and not [c for c in ui.api.calls if c["method"] == "DELETE"]:
        ui.page.wait_for_timeout(25)
    delete = [c for c in ui.api.calls if c["method"] == "DELETE"]
    assert delete, "the delete request was never sent"
    assert delete[0]["url"].endswith("/api/training/model/j%2F1%3Fx%23y")


def test_backend_error_text_is_never_interpreted_as_markup(ui):
    detail = "<img src=x onerror=\"window.__pwned=1\">"
    ui.api.on("POST", "/api/providers/qwen3/models/select", {"detail": detail}, status=500)
    ui.open()
    ui.use_qwen3()
    ui.page.click("#qwen3-tts-tab-button")
    ui.page.wait_for_function("document.querySelectorAll('#qwen3-model-select option').length > 1")
    ui.page.select_option("#qwen3-model-select", "custom-CustomVoice")
    ui.page.wait_for_selector("#notification-container .notification-error")
    assert ui.page.evaluate("document.querySelectorAll('#notification-container img').length") == 0
    assert ui.page.evaluate("window.__pwned") is None
    assert detail in ui.text("#notification-container")
    assert ui.page.get_attribute("#notification-container .notification-error", "role") == "alert"


# --- training progress -------------------------------------------------------


def _start_training(ui, statuses):
    """Open the page on a fake clock and start a job whose polls return `statuses`."""
    answers = list(statuses)

    def status(route, request):
        body = answers.pop(0) if len(answers) > 1 else answers[0]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

    ui.api.on("POST", "/api/training/train", {"job_id": "job-1"})
    ui.api.on("GET", "/api/training/status/job-1", status)
    ui.page.clock.install()
    ui.open()
    ui.page.click("#training-tab-button")
    ui.page.fill("#training-voice-name", "luna")
    ui.page.set_input_files("#training-files", {"name": "a.wav", "mimeType": "audio/wav", "buffer": _wav()})
    ui.page.click("#start-training-button")
    ui.wait_calls("GET", "/api/training/status/job-1")


def _polls_over(ui, seconds):
    for _ in range(seconds // 5):
        ui.page.clock.run_for(5000)
        ui.page.wait_for_timeout(60)
    return len(ui.api.calls_to("GET", "/api/training/status/job-1"))


@pytest.mark.parametrize("final", ["cancelled", "interrupted"])
def test_progress_polling_stops_when_the_job_is_cancelled_or_interrupted(ui, final):
    _start_training(ui, [
        {"status": "running", "progress": 10, "current_epoch": 1, "total_epochs": 10},
        {"status": final, "progress": 10},
    ])
    polls = _polls_over(ui, 90)
    assert polls == 2, f"a {final} job kept being polled ({polls} status requests in 90 s)"
    assert final in ui.text("#training-progress-status").lower()


def test_resuming_does_not_stack_a_second_polling_loop(ui):
    _start_training(ui, [{"status": "running", "progress": 10, "current_epoch": 1, "total_epochs": 10}])
    ui.api.on("POST", "/api/training/resume", {"job_id": "job-1"})
    ui.page.fill("#continue-voice-name", "luna")
    ui.page.click("#resume-training-button")
    ui.wait_calls("POST", "/api/training/resume")
    ui.page.wait_for_timeout(100)
    baseline = len(ui.api.calls_to("GET", "/api/training/status/job-1"))
    polls = _polls_over(ui, 30) - baseline
    assert polls == 6, f"expected one poll every 5 s for 30 s, saw {polls}"


# --- in-flight buttons -------------------------------------------------------


def test_a_button_is_disabled_while_its_request_is_in_flight(ui):
    held = []
    ui.api.on("POST", "/api/tts", lambda route, request: held.append(route))
    ui.open()
    ui.page.click("#tts-tab-button")
    ui.page.click("#generate-tts-button")
    ui.page.wait_for_function("document.querySelector('#generate-tts-button').disabled")
    assert ui.page.get_attribute("#generate-tts-button", "aria-busy") == "true"
    ui.page.evaluate("document.querySelector('#generate-tts-button').click()")   # a second click
    ui.page.wait_for_timeout(150)
    assert len(ui.api.calls_to("POST", "/api/tts")) == 1, "a second click started a second synthesis"

    held[0].fulfill(status=200, content_type="audio/wav", body=_wav())
    ui.page.wait_for_function("!document.querySelector('#generate-tts-button').disabled")
    assert ui.page.get_attribute("#generate-tts-button", "aria-busy") is None


# --- accessibility -----------------------------------------------------------


def test_status_areas_are_live_regions_and_errors_are_alerts(ui):
    ui.open()
    regions = ui.page.evaluate("""() => Object.fromEntries(
        Array.from(document.querySelectorAll('.status-display')).map((el) => [el.id, el.getAttribute('aria-live')]))""")
    assert regions["stt-result-status"] == "polite"
    assert regions["live-stt-status"] == "polite"
    assert regions["qwen3-tts-system-status"] == "off", "the 30 s health card must not be re-announced"
    assert ui.page.get_attribute("#live-stt-confirmed", "aria-live") == "polite"
    ui.page.click("#process-stt-button")
    ui.wait_text("#stt-result-status", "select an audio file")
    assert ui.page.get_attribute("#stt-result-status .error", "role") == "alert"


def test_the_engine_radios_can_be_reached_with_the_keyboard(ui):
    """They were display:none, which removes them from the tab order entirely."""
    ui.open()
    ui.page.focus("input[name=tts-engine]")
    assert ui.page.evaluate("document.activeElement && document.activeElement.name") == "tts-engine"
    ui.page.keyboard.press("ArrowRight")
    assert ui.page.evaluate("document.querySelector('input[name=tts-engine]:checked').value") == "qwen3"
    ui.page.wait_for_selector("#qwen3-tts-tab-button", state="visible")


def test_a_disabled_button_looks_disabled(ui):
    ui.open()
    enabled = ui.page.evaluate("""() => {
        const button = document.querySelector('#process-stt-button');
        const before = getComputedStyle(button).opacity;
        button.disabled = true;
        return before;
    }""")
    assert float(enabled) == 1.0
    # Buttons animate their properties, so the new opacity is reached a moment later.
    ui.page.wait_for_function(
        "parseFloat(getComputedStyle(document.querySelector('#process-stt-button')).opacity) < 1")


# --- live transcription ------------------------------------------------------


def _start_live(ui):
    ui.open()
    ui.page.click("#live-stt-button")
    ui.wait_text("#live-stt-status", "Listening")


def test_live_language_and_stop_use_the_documented_protocol(ui):
    _start_live(ui)
    deadline = time.time() + TIMEOUT_MS / 1000
    while time.time() < deadline and not any(isinstance(m, bytes) for _, m in ui.live.received):
        ui.page.wait_for_timeout(50)
    first = ui.live.received[0][1]
    assert json.loads(first) == {"language": "auto"}
    assert any(isinstance(m, bytes) and len(m) > 0 for _, m in ui.live.received), \
        "the capture pipeline delivered no PCM to the socket"
    ui.page.click("#live-stt-button")
    ui.wait_text("#live-stt-status", "Finishing")
    assert ui.live.sent_stop()
    ui.live.send({"type": "final", "text": "Guten Morgen zusammen", "language": "de", "duration": 3.2})
    ui.live.close(1000)
    ui.wait_text("#live-stt-status", "Final transcript")
    assert ui.text("#live-stt-confirmed").strip() == "Guten Morgen zusammen"
    assert "3.2s" in ui.text("#live-stt-status")


def test_a_shorter_confirmed_frame_does_not_erase_the_transcript(ui):
    _start_live(ui)
    ui.live.send({"type": "partial", "confirmed": "eins zwei drei", "pending": "vier", "decode_ms": 120})
    ui.wait_text("#live-stt-confirmed", "eins zwei drei")
    ui.live.send({"type": "partial", "confirmed": "drei", "pending": "vier fünf", "decode_ms": 130})
    ui.live.send({"type": "partial", "confirmed": "eins zwei drei vier", "pending": "fünf", "decode_ms": 130})
    ui.wait_text("#live-stt-confirmed", "eins zwei drei vier")
    assert ui.text("#live-stt-confirmed").strip() == "eins zwei drei vier"
    # And in between, while the short frame was the latest one:
    ui.live.send({"type": "partial", "confirmed": "eins", "pending": "zwei", "decode_ms": 90})
    ui.page.wait_for_function("document.querySelector('#live-stt-pending').innerText.trim() !== 'fünf'")
    assert ui.text("#live-stt-confirmed").strip().startswith("eins zwei drei vier")
    assert "decode 90 ms" in ui.text("#live-stt-latency")


def test_error_frames_from_the_server_are_shown(ui):
    _start_live(ui)
    ui.live.send({"type": "error", "code": "bad_language", "message": "Unknown language 'xx'", "error": "Unknown language 'xx'"})
    ui.wait_text("#live-stt-status", "Unknown language 'xx'")
    assert ui.page.get_attribute("#live-stt-status .error", "role") == "alert"
    # The server keeps the session open after such an error; the next partial restores the status.
    ui.live.send({"type": "partial", "confirmed": "hallo", "pending": "", "decode_ms": 10})
    ui.wait_text("#live-stt-status", "Listening")


def test_legacy_relay_error_frames_are_shown_too(ui):
    _start_live(ui)
    ui.live.send({"type": "error", "error": "Live transcription unavailable: whisper could not be reached."})
    ui.wait_text("#live-stt-status", "could not be reached")


@pytest.mark.parametrize("code, fragment", [
    (1011, "internal error"),
    (1001, "restarting"),
    (1008, "refused"),
    (1013, "busy"),
])
def test_unexpected_close_codes_give_a_readable_message(ui, code, fragment):
    _start_live(ui)
    ui.live.close(code)
    ui.wait_text("#live-stt-status", fragment)
    ui.page.wait_for_function("document.querySelector('#live-stt-button').innerText.includes('Start')")
    assert not ui.page.evaluate("document.querySelector('#live-stt-button').disabled")


def test_a_specific_error_frame_is_not_replaced_by_the_generic_close_text(ui):
    _start_live(ui)
    ui.live.send({"type": "error", "code": "too_many_sessions", "message": "Too many live sessions (limit 2)"})
    ui.live.close(1013)
    ui.wait_text("#live-stt-status", "Too many live sessions (limit 2)")
    ui.page.wait_for_timeout(200)
    assert "busy" not in ui.text("#live-stt-status")


def test_closing_while_waiting_for_the_final_transcript_is_reported(ui):
    _start_live(ui)
    ui.live.send({"type": "partial", "confirmed": "hallo welt", "pending": "", "decode_ms": 10})
    ui.wait_text("#live-stt-confirmed", "hallo welt")
    ui.page.click("#live-stt-button")
    ui.wait_text("#live-stt-status", "Finishing")
    ui.live.close(1011)
    ui.wait_text("#live-stt-status", "internal error")
    assert ui.text("#live-stt-confirmed").strip() == "hallo welt", "the text read so far stays on screen"


def test_an_empty_final_does_not_blank_the_transcript_already_shown(ui):
    _start_live(ui)
    ui.live.send({"type": "partial", "confirmed": "hallo welt", "pending": "", "decode_ms": 10})
    ui.wait_text("#live-stt-confirmed", "hallo welt")
    ui.page.click("#live-stt-button")
    ui.live.send({"type": "final", "text": "", "language": None, "duration": 1.0})
    ui.wait_text("#live-stt-status", "Final transcript")
    assert ui.text("#live-stt-confirmed").strip() == "hallo welt"


def test_frames_of_a_retired_session_do_not_touch_the_new_one(ui):
    _start_live(ui)
    ui.page.click("#live-stt-button")                      # stop; socket 0 stays open for its final
    ui.wait_text("#live-stt-status", "Finishing")
    ui.page.click("#live-stt-button")                      # start again -> socket 1
    ui.wait_text("#live-stt-status", "Listening")
    assert len(ui.live.sockets) == 2
    ui.live.send({"type": "partial", "confirmed": "neu", "pending": "", "decode_ms": 5}, socket=1)
    ui.wait_text("#live-stt-confirmed", "neu")
    ui.live.send({"type": "final", "text": "alt alt alt", "language": "de", "duration": 9}, socket=0)
    ui.live.close(1000, socket=0)
    ui.page.wait_for_timeout(300)
    assert ui.text("#live-stt-confirmed").strip() == "neu"
    assert "Listening" in ui.text("#live-stt-status")


def test_live_transcription_explains_a_missing_microphone_api(ui):
    """A plain-http page on a LAN/NAS address has no navigator.mediaDevices at all;
    the old code surfaced 'Cannot read properties of undefined'."""
    ui.page.add_init_script("Object.defineProperty(navigator, 'mediaDevices', { value: undefined, configurable: true })")
    ui.open()
    ui.page.click("#live-stt-button")
    ui.wait_text("#live-stt-status", "https")
    assert "undefined" not in ui.text("#live-stt-status")
    assert not ui.page.evaluate("document.querySelector('#live-stt-button').disabled")


# --- API documentation page --------------------------------------------------


def test_api_docs_status_comes_from_the_gateway_and_reflects_reality(ui):
    """The page used to start every card as 'Healthy' and probe each backend's own
    localhost port from the browser: wrong on any host but the developer's, and a
    CORS failure where those ports are not published."""
    ui.api.on("GET", "/api/health", {"providers": {
        "piper": {"healthy": True, "latency_ms": 2},
        "whisper": {"healthy": False, "error": "ConnectError"},
        "qwen3": {"healthy": True, "model_resident": False},
    }})
    requested = []
    ui.page.on("request", lambda request: requested.append(request.url))
    ui.page.goto(ui.base + "/api-docs", wait_until="load")

    ui.wait_text("#stt-status", "Offline")
    assert ui.text("#tts-status") == "Healthy"
    assert ui.text("#xtts-status") == "Idle", "a healthy but unloaded model is idle, not down"
    assert ui.text("#training-status") == "Offline", "a provider missing from the answer is not healthy"
    assert not [url for url in requested if re.search(r":(5000|5001|5002|5004|8080)/", url)], \
        "the docs page called a backend port directly"

    hostnames = ui.page.evaluate(
        "Array.from(document.querySelectorAll('a.api-link')).map(a => new URL(a.href).hostname)")
    assert "localhost" not in hostnames or urlparse(ui.base).hostname == "localhost"


# --- API key and DNS rebinding, against a real gateway ---------------------------------
#
# Nothing under /api is mocked here: the requests reach a gateway that runs with
# API_KEY (only its backends are stubbed), so what is checked is the whole loop -
# the 401 challenge, the prompt, the header on the retry, the WebSocket subprotocol -
# rather than the page's half alone.

KEY = "s3cret-key"


class _HangingUpstream:
    """The STT service's WebSocket: takes anything, says nothing, never hangs up."""

    def __init__(self, dialled, url):
        dialled.append(url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, message):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        import asyncio
        await asyncio.Event().wait()


class KeyedUi:
    def __init__(self, page, base, dialled):
        self.page, self.base, self.dialled = page, base, dialled
        self.dialogs: list[str] = []
        self.answers: list = []                     # what the next prompts get; None dismisses
        self.requests: list[tuple] = []             # (method, path, Authorization header)
        page.on("dialog", self._dialog)
        page.on("request", lambda r: self.requests.append(
            (r.method, urlparse(r.url).path, r.headers.get("authorization"))))
        page.route("**/favicon.ico", lambda route: route.fulfill(status=204))

    def _dialog(self, dialog):
        self.dialogs.append(dialog.message)
        answer = self.answers.pop(0) if self.answers else None
        dialog.dismiss() if answer is None else dialog.accept(answer)

    def open(self):
        self.page.goto(self.base, wait_until="load")
        self.page.wait_for_selector("#stt-tab-button")

    def open_tts(self):
        self.open()
        self.page.click("#tts-tab-button")
        self.page.wait_for_selector("#generate-tts-button", state="visible")

    def posts(self, path):
        return [auth for method, p, auth in self.requests if method == "POST" and p == path]

    def wait_posts(self, path, count):
        deadline = time.time() + TIMEOUT_MS / 1000
        while time.time() < deadline and len(self.posts(path)) < count:
            self.page.wait_for_timeout(25)
        assert len(self.posts(path)) >= count, f"wanted {count} POST {path}, saw {self.requests}"

    def wait_text(self, selector, fragment):
        self.page.wait_for_function(
            "([sel, text]) => { const el = document.querySelector(sel); return !!el && el.innerText.includes(text); }",
            arg=[selector, fragment], timeout=TIMEOUT_MS)


def _serve(module):
    server = uvicorn.Server(uvicorn.Config(module.app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:
        pytest.fail("the gateway did not start")
    return server, thread, server.servers[0].sockets[0].getsockname()[1]


@pytest.fixture
def keyed_ui(browser, monkeypatch):
    from frontend_loader import install_stub
    import websockets

    module = load_frontend_app({"API_KEY": KEY})

    def backend(method, url, kwargs):
        import httpx
        if url.endswith("/tts"):
            return httpx.Response(200, content=_wav(), headers={"content-type": "audio/wav"})
        return {"status": "healthy", "providers": {}, "voices": []}

    install_stub(monkeypatch, module, backend)
    dialled = []
    monkeypatch.setattr(websockets, "connect", lambda url, **kw: _HangingUpstream(dialled, url))
    server, thread, port = _serve(module)
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(TIMEOUT_MS)
    yield KeyedUi(page, f"http://127.0.0.1:{port}", dialled)
    context.close()
    server.should_exit = True
    thread.join(timeout=10)


def test_with_an_api_key_the_ui_asks_once_and_sends_it_from_then_on(keyed_ui):
    ui = keyed_ui
    ui.answers.append(KEY)
    ui.open_tts()
    ui.page.click("#generate-tts-button")
    ui.wait_text("#tts-result-status", "Speech generated")
    ui.page.click("#generate-tts-button")
    ui.wait_posts("/api/tts", 3)

    assert len(ui.dialogs) == 1 and "API key" in ui.dialogs[0]
    assert ui.posts("/api/tts") == [None, f"Bearer {KEY}", f"Bearer {KEY}"]
    assert ui.page.evaluate("sessionStorage.getItem('tts-stt.api-key')") == KEY
    assert ui.page.evaluate("localStorage.length") == 0, "the key must not outlive the tab"


def test_a_wrong_key_is_reported_and_asked_for_again(keyed_ui):
    ui = keyed_ui
    ui.answers.extend(["wrong", KEY])
    ui.open_tts()
    ui.page.click("#generate-tts-button")
    ui.wait_text("#tts-result-status", "API key")
    assert "Speech generated" not in ui.page.inner_text("#tts-result-status")
    assert ui.page.evaluate("sessionStorage.getItem('tts-stt.api-key')") is None

    ui.page.click("#generate-tts-button")
    ui.wait_text("#tts-result-status", "Speech generated")
    assert len(ui.dialogs) == 2


def test_declining_the_prompt_leaves_the_action_refused(keyed_ui):
    ui = keyed_ui
    ui.answers.append(None)
    ui.open_tts()
    ui.page.click("#generate-tts-button")
    ui.wait_text("#tts-result-status", "API key")
    assert ui.posts("/api/tts") == [None], "no retry without a key"


def test_live_transcription_needs_the_key_and_the_socket_carries_it(keyed_ui):
    ui = keyed_ui
    ui.answers.append(KEY)
    ui.open()
    ui.page.click("#live-stt-button")
    ui.wait_text("#live-stt-status", "Listening")

    assert ui.posts("/api/auth/check") == [None, f"Bearer {KEY}"]
    assert len(ui.dialogs) == 1
    assert len(ui.dialled) == 1, "the relay accepted the socket and dialled the STT service"


def test_live_transcription_without_a_key_is_refused_before_the_relay_dials_anything(keyed_ui):
    ui = keyed_ui
    ui.answers.append(None)
    ui.open()
    ui.page.click("#live-stt-button")
    ui.wait_text("#live-stt-status", "API key")
    assert "valid API key is required" in ui.page.inner_text("#live-stt-status")
    assert ui.dialled == [], "an unauthenticated socket reached the STT service"
    assert not ui.page.evaluate("document.querySelector('#live-stt-button').disabled")


def test_a_page_on_a_rebound_hostname_gets_nothing_from_the_gateway(browser, monkeypatch):
    """Chromium is told that evil.example is the gateway's own address, which is what a DNS
    rebinding attack achieves: the page's origin and the request's Host are the attacker's
    name, they match, and the browser calls it same-origin. The gateway refuses it by name."""
    module = load_frontend_app()
    from frontend_loader import install_stub
    stub = install_stub(monkeypatch, module, lambda m, u, k: {"status": "ok"})
    server, thread, port = _serve(module)
    rebinding = browser.browser_type.launch(
        headless=True, args=[*CHROMIUM_ARGS, "--host-resolver-rules=MAP evil.example 127.0.0.1"])
    try:
        context = rebinding.new_context()
        page = context.new_page()
        page.set_default_timeout(TIMEOUT_MS)

        control = page.goto(f"http://127.0.0.1:{port}/", wait_until="load")
        assert control.status == 200
        page.wait_for_selector("#stt-tab-button")
        page.wait_for_timeout(300)                 # the UI's own start-up calls have been made
        stub.calls.clear()

        rebound = page.goto(f"http://evil.example:{port}/", wait_until="load")
        assert rebound.status == 403
        assert "host_not_allowed" in page.content() or "not allowed" in page.content()
        # A browser gets the refusal as a page that can run and fetch nothing.
        assert rebound.headers["content-security-policy"].startswith("default-src 'none'")

        # The attack itself: a script in a document of the rebound origin calling the destructive
        # API. In a real attack that is the attacker's own page, served before the name was
        # re-pointed; the refusal page runs nothing, so the JSON refusal of an API path (a
        # document of that origin without a policy) stands in for it here.
        assert page.goto(f"http://evil.example:{port}/api/health", wait_until="load").status == 403
        outcome = page.evaluate("""async () => {
            const response = await fetch('/api/training/model/job-1', { method: 'DELETE' });
            return response.status;
        }""")
        assert outcome == 403
        assert stub.calls == [], "the request reached a backend"
    finally:
        rebinding.close()
        server.should_exit = True
        thread.join(timeout=10)
