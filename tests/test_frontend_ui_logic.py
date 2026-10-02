"""Behavioural tests of the browser controller's logic (`app.js`), run under Node.

There is no JS test runner in this repo and CI has no browser, but it does have
Node, and most of what went wrong in `app.js` was plain logic: a polling loop
that never ended, URL segments that were never encoded, a transcript merge that
let a shorter frame erase text. So these load the real `app.js` into a `vm`
context with a small fake DOM, fake timers and a scripted `fetch`, then call the
real functions and assert on what they did. Page-level behaviour (does the
template wire up, does the WebSocket handling hold under real frames) is in
`test_frontend_ui_browser.py`.

`FRONTEND_SERVICE_DIR` (see `frontend_loader`) points these at another checkout
of the service, which is how they are shown to fail against the previous code.

The Settings page's script (`settings.js`) has its own driver further down: it
runs in the real `settings.html`, parsed into a small DOM, and is fed the
answers a real gateway gave.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from frontend_loader import SERVICE_DIR, install_stub, load_frontend_app

APP_JS = SERVICE_DIR / "static" / "js" / "app.js"
NODE = shutil.which("node") or ("/opt/node22/bin/node" if Path("/opt/node22/bin/node").exists() else None)

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const [appPath, scenarioPath, registryJson] = process.argv.slice(2);

class FakeElement {
    constructor(tag = 'div', id = '') {
        this.tagName = String(tag).toUpperCase();
        this.id = id;
        this.children = [];
        this.attrs = {};
        this.dataset = {};
        this.style = {};
        this.listeners = {};
        this._text = '';
        this.innerHTML = '';
        this.value = '';
        this.options = [];
        this.disabled = false;
        this.className = '';
        const classes = new Set();
        this.classList = {
            add: (...c) => c.forEach((x) => classes.add(x)),
            remove: (...c) => c.forEach((x) => classes.delete(x)),
            contains: (x) => classes.has(x),
        };
    }
    set textContent(v) { this._text = String(v); this.children = []; }
    get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
    hasAttribute(k) { return k in this.attrs; }
    removeAttribute(k) { delete this.attrs[k]; }
    appendChild(c) { this.children.push(c); return c; }
    replaceChildren(...nodes) { this._text = ''; this.children = nodes; }
    removeChild(child) { this.children = this.children.filter((c) => c !== child); }
    remove() { this.removed = true; }
    select() {}
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
    querySelectorAll() { return []; }
    querySelector() { return null; }
    click() { this.clicks = (this.clicks || 0) + 1; }
}

const elements = new Map();
const harness = {
    requests: [],
    fetchQueue: [],
    alerts: [],
    prompts: [],
    promptAnswers: [],
    storageThrows: false,
    consoleErrors: [],
    notifications: () => {
        const container = elements.get('notification-container');
        return container ? container.children.map((c) => ({ text: c.textContent, className: c.className })) : [];
    },
    el: (id) => {
        if (!elements.has(id)) elements.set(id, new FakeElement('div', id));
        return elements.get(id);
    },
    // Opt-in: make an element behave like a single <select> where the code relies on
    // it. `options` are its children, `value` is the selected option's (the first one
    // once the selected option is gone, as in a browser), assigning `value` selects
    // the option that has it, and `innerHTML = ''` empties it.
    select: (id) => {
        const el = harness.el(id);
        let selected = null;
        const current = () => {
            if (!el.children.includes(selected)) selected = el.children[0] || null;
            return selected;
        };
        Object.defineProperty(el, 'options', { get: () => el.children, configurable: true });
        Object.defineProperty(el, 'value', {
            get: () => (current() ? current().value : ''),
            set: (v) => { selected = el.children.find((o) => o.value === String(v)) || null; },
            configurable: true,
        });
        Object.defineProperty(el, 'innerHTML', {
            get: () => '',
            set: (v) => { if (String(v) === '') el.children = []; },
            configurable: true,
        });
        return el;
    },
    status: (id) => {
        const box = harness.el(id).children[0];
        return box ? { className: box.className, text: box.textContent, attrs: box.attrs } : null;
    },
    json: (body, status = 200) => ({
        ok: status < 400,
        status,
        statusText: '',
        headers: { get: () => null },
        json: async () => body,
        blob: async () => ({ type: 'audio/wav' }),
    }),
    // What the gateway answers when API_KEY is set and the request had no valid key.
    unauthorized: (challenge = 'Bearer') => ({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        headers: { get: (name) => (String(name).toLowerCase() === 'www-authenticate' ? challenge : null) },
        json: async () => ({ detail: 'A valid API key is required (Authorization: Bearer <key>).' }),
        blob: async () => ({ type: 'application/json' }),
    }),
};

// sessionStorage, backed by a Map; `harness.storageThrows` makes it behave like a
// browser with site data blocked (every access throws).
const storageData = new Map();
const sessionStorage = {
    getItem: (k) => { if (harness.storageThrows) throw new Error('blocked'); return storageData.has(k) ? storageData.get(k) : null; },
    setItem: (k, v) => { if (harness.storageThrows) throw new Error('blocked'); storageData.set(k, String(v)); },
    removeItem: (k) => { if (harness.storageThrows) throw new Error('blocked'); storageData.delete(k); },
};
harness.storage = storageData;

// Fake timers: nothing runs until the scenario says so.
let timerSeq = 0;
const timers = new Map();
const addTimer = (fn, ms) => { const id = ++timerSeq; timers.set(id, { fn, ms }); return id; };
harness.pendingTimers = () => timers.size;
// Let every already-resolved promise (a mocked fetch, the code after its await) run.
// Scenarios use this instead of awaiting monitorTrainingProgress(), so they do
// not depend on whether it returns its first poll.
harness.flush = () => new Promise((resolve) => setImmediate(resolve));
harness.runTimers = async () => {
    const due = Array.from(timers.entries());
    timers.clear();
    for (const [, timer] of due) await timer.fn();
};

const document = {
    hidden: false,
    body: new FakeElement('body'),
    documentElement: { dataset: {} },
    getElementById: (id) => harness.el(id),
    createElement: (tag) => new FakeElement(tag),
    createTextNode: (text) => { const n = new FakeElement('#text'); n.textContent = text; return n; },
    addEventListener: () => {},
    querySelector: () => null,
    querySelectorAll: () => [],
};

const context = {
    document,
    console: { ...console, error: (...a) => harness.consoleErrors.push(a.map(String).join(' ')), warn: () => {} },
    setTimeout: addTimer,
    clearTimeout: (id) => timers.delete(id),
    setInterval: addTimer,
    clearInterval: (id) => timers.delete(id),
    fetch: async (url, options = {}) => {
        const sent = options && options.headers ? new Headers(options.headers) : null;
        harness.requests.push({
            url: String(url),
            method: (options && options.method) || 'GET',
            authorization: sent ? sent.get('authorization') : null,
            body: options ? options.body : undefined,
        });
        const next = harness.fetchQueue.shift();
        if (typeof next === 'function') return next(url, options);
        return next || harness.json({});
    },
    sessionStorage,
    prompt: (message) => {
        harness.prompts.push(String(message));
        const answer = harness.promptAnswers.shift();
        return answer === undefined ? null : answer;
    },
    Headers, TextEncoder, btoa,
    confirm: () => true,
    alert: (m) => harness.alerts.push(String(m)),
    AbortSignal, FormData, Blob, Event, Promise, JSON, Math, Number, String, Array, Object, Set, Map, WeakMap, Date,
    URL: Object.assign(function (...a) { return new URL(...a); }, {
        createObjectURL: () => 'blob:fake', revokeObjectURL: () => {},
    }),
    CSS: { escape: (s) => String(s) },
    navigator: { mediaDevices: undefined, clipboard: { writeText: async () => {} } },
    WebSocket: { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 },
    location: { protocol: 'http:', host: 'gateway.test' },
    harness,
};
context.window = context;
context.window.location = context.location;
context.window.PROVIDER_REGISTRY = JSON.parse(registryJson);
vm.createContext(context);

vm.runInContext(fs.readFileSync(appPath, 'utf8'), context, { filename: 'app.js' });

(async () => {
    const source = fs.readFileSync(scenarioPath, 'utf8');
    const result = await vm.runInContext(`(async () => {\n${source}\n})()`, context, { filename: 'scenario.js' });
    process.stdout.write(JSON.stringify({ ok: true, result }));
})().catch((error) => {
    process.stdout.write(JSON.stringify({ ok: false, error: String(error && error.stack || error) }));
    process.exitCode = 1;
});
"""

REGISTRY = {
    "ui": {"default_tts_provider": "piper", "default_stt_provider": "whisper", "training_provider": "piper-training"},
    "providers": {
        "piper": {
            "kind": "tts",
            "display_name": "PiperTTS",
            "ui": {"family": "piper"},
            "settings": {"languages": [
                {"value": "auto", "label": "Auto-Detect"},
                {"value": "de", "label": "German"},
                {"value": "en", "label": "English"},
            ]},
        },
        "whisper": {"kind": "stt", "capabilities": ["transcribe", "live_transcribe"]},
        "qwen3-asr": {"kind": "stt", "capabilities": ["transcribe"]},
        "qwen3": {"kind": "tts", "ui": {"family": "qwen3"}},
    },
}


def run_js(tmp_path: Path, scenario: str, registry: dict | None = None):
    """Load app.js, run `scenario` (an async function body) and return its result."""
    driver = tmp_path / "driver.js"
    driver.write_text(DRIVER, encoding="utf-8")
    body = tmp_path / "scenario.js"
    body.write_text(textwrap.dedent(scenario), encoding="utf-8")
    completed = subprocess.run(
        [NODE, str(driver), str(APP_JS), str(body), json.dumps(registry or REGISTRY)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        outcome = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"node produced no result.\nstdout: {completed.stdout}\nstderr: {completed.stderr}")
    if not outcome["ok"]:
        pytest.fail(f"scenario threw:\n{outcome['error']}")
    return outcome["result"]


# --- monitorTrainingProgress -------------------------------------------------


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled", "interrupted"])
def test_monitor_stops_polling_on_a_terminal_state(tmp_path, terminal):
    """The loop used to stop only on completed/failed. A cancelled or interrupted
    job never changes again, yet it was polled every five seconds for as long as
    the page stayed open."""
    result = run_js(tmp_path, f"""
        harness.fetchQueue.push(harness.json({{ status: 'running', progress: 10, current_epoch: 1, total_epochs: 10 }}));
        monitorTrainingProgress('job-1');
        await harness.flush();
        const afterRunning = harness.pendingTimers();

        harness.fetchQueue.push(harness.json({{ status: '{terminal}', progress: 10 }}));
        await harness.runTimers();
        await harness.flush();
        return {{
            afterRunning,
            afterTerminal: harness.pendingTimers(),
            requests: harness.requests.length,
            status: harness.status('training-progress-status'),
        }};
    """)
    assert result["afterRunning"] == 1, "a running job must schedule the next poll"
    assert result["afterTerminal"] == 0, f"'{terminal}' is terminal, but another poll was scheduled"
    # Two status polls, then the jobs list refresh the terminal branch triggers.
    assert result["status"] is not None and result["status"]["text"], "the user is told how the job ended"


def test_monitor_says_why_a_cancelled_or_interrupted_job_stopped(tmp_path):
    result = run_js(tmp_path, """
        const seen = {};
        for (const state of ['cancelled', 'interrupted']) {
            harness.fetchQueue.push(harness.json({ status: state }));
            monitorTrainingProgress('job-' + state);
            await harness.flush();
            seen[state] = harness.status('training-progress-status');
        }
        return seen;
    """)
    assert "cancelled" in result["cancelled"]["text"].lower()
    assert result["cancelled"]["className"] == "info"
    assert "interrupted" in result["interrupted"]["text"].lower()
    assert "Resume" in result["interrupted"]["text"]


def test_starting_a_monitor_ends_the_previous_loop_for_the_job(tmp_path):
    """Resume / Train-from-dataset call monitorTrainingProgress again for a job that
    is already being watched. Each call used to add another self-rescheduling loop
    (none was ever cleared), so the request rate multiplied with every resume."""
    result = run_js(tmp_path, """
        const running = () => harness.json({ status: 'running', progress: 5, current_epoch: 1, total_epochs: 10 });
        for (let i = 0; i < 3; i++) {
            harness.fetchQueue.push(running());
            monitorTrainingProgress('job-1');
            await harness.flush();
        }
        const pendingAfterThreeStarts = harness.pendingTimers();

        // Let time pass: every round of timers must cost exactly one request.
        const before = harness.requests.length;
        for (let i = 0; i < 3; i++) {
            harness.fetchQueue.push(running());
            await harness.runTimers();
            await harness.flush();
        }
        return { pendingAfterThreeStarts, requestsInThreeRounds: harness.requests.length - before };
    """)
    assert result["pendingAfterThreeStarts"] == 1
    assert result["requestsInThreeRounds"] == 3


def test_a_superseded_monitor_does_not_repaint_after_its_request_returns(tmp_path):
    """A poll already in flight when a new monitor starts must not schedule a
    follow-up or overwrite the newer monitor's panel."""
    result = run_js(tmp_path, """
        let release;
        const slow = new Promise((resolve) => { release = resolve; });
        harness.fetchQueue.push(() => slow);                      // monitor A, still in flight
        monitorTrainingProgress('job-a');
        await harness.flush();
        harness.fetchQueue.push(harness.json({ status: 'running', progress: 50, current_epoch: 5, total_epochs: 10 }));
        monitorTrainingProgress('job-b');
        await harness.flush();
        const panelAfterB = harness.el('training-progress').innerHTML;

        release(harness.json({ status: 'running', progress: 99, current_epoch: 9, total_epochs: 10 }));
        await harness.flush();
        return {
            timers: harness.pendingTimers(),
            panelUnchanged: harness.el('training-progress').innerHTML === panelAfterB,
        };
    """)
    assert result["timers"] == 1, "only monitor B may have a poll scheduled"
    assert result["panelUnchanged"]


def test_monitor_gives_up_after_repeated_failures_and_says_so(tmp_path):
    result = run_js(tmp_path, """
        for (let i = 0; i < 5; i++) harness.fetchQueue.push(harness.json({}, 503));
        monitorTrainingProgress('job-1');
        await harness.flush();
        while (harness.pendingTimers()) { await harness.runTimers(); await harness.flush(); }
        return { requests: harness.requests.length, status: harness.status('training-progress-status') };
    """)
    assert result["requests"] == 5
    assert result["status"]["className"] == "error"


def test_monitor_reports_a_vanished_job_immediately(tmp_path):
    result = run_js(tmp_path, """
        harness.fetchQueue.push(harness.json({}, 404));
        monitorTrainingProgress('gone');
        await harness.flush();
        return { timers: harness.pendingTimers(), status: harness.status('training-progress-status') };
    """)
    assert result["timers"] == 0
    assert result["status"]["className"] == "error"


def test_monitor_progress_panel_escapes_registry_text(tmp_path):
    """The progress line is built from a registry template plus job fields."""
    registry = json.loads(json.dumps(REGISTRY))
    registry["providers"]["piper-training"] = {"ui": {"messages": {"start_training": {
        "progress": "<img src=x onerror=alert(1)> {progress}%"}}}}
    result = run_js(tmp_path, """
        harness.fetchQueue.push(harness.json({ status: 'training', progress: 5, current_epoch: 1, total_epochs: 10 }));
        monitorTrainingProgress('job-1');
        await harness.flush();
        return harness.el('training-progress').innerHTML;
    """, registry)
    assert "<img" not in result
    assert "&lt;img" in result


# --- path segments -----------------------------------------------------------

HOSTILE = "a/b?c#d e"
ENCODED = "a%2Fb%3Fc%23d%20e"


def test_every_id_is_encoded_before_it_becomes_a_url_segment(tmp_path):
    """Job, voice and provider ids went into URLs raw. An id with a `/`, `?` or `#`
    then addressed another route (or lost the rest of the path)."""
    result = run_js(tmp_path, f"""
        const id = {json.dumps(HOSTILE)};
        harness.el('qwen3-saved-voice-select').value = id;
        harness.el('qwen3-saved-voice-select').selectedOptions = [{{ textContent: 'v' }}];

        await deleteModel(id);
        await cancelJob(id);
        await downloadModel(id);
        await deployExportedModel(id, 'name');
        await viewJobDetails(id);
        await deleteSavedVoice();
        await deleteCustomVoice(id);
        monitorTrainingProgress(id);
        await harness.flush();
        return {{
            urls: harness.requests.map((r) => r.method + ' ' + r.url),
            provider: getProviderApiPath('we/ird', '/voices'),
        }};
    """)
    urls = result["urls"]
    assert f"DELETE /api/training/model/{ENCODED}" in urls
    assert f"DELETE /api/training/job/{ENCODED}" in urls
    assert f"GET /api/training/download/{ENCODED}" in urls
    assert f"POST /api/training/export/{ENCODED}" in urls
    assert f"GET /api/training/status/{ENCODED}" in urls
    assert f"DELETE /api/providers/qwen3/saved-voices/{ENCODED}" in urls
    assert f"DELETE /api/providers/piper/custom-voices/{ENCODED}" in urls
    assert not [u for u in urls if HOSTILE in u], "a raw id reached a URL"
    assert result["provider"] == "/api/providers/we%2Fird/voices"


# --- live transcription ------------------------------------------------------


def _merge(tmp_path, cases):
    return run_js(tmp_path, f"""
        return {json.dumps(cases)}.map(([shown, confirmed, pending]) =>
            mergeLiveTranscript(shown, confirmed, pending));
    """)


def test_a_shorter_confirmed_never_erases_shown_text(tmp_path):
    """`confirmed` was assigned straight to the page, so any frame with less text
    than the last one (an older server that confirmed per decode window, a
    reordered frame) wiped what the user was reading."""
    out = _merge(tmp_path, [
        ["guten Morgen liebe Zuhörer", "liebe Zuhörer", "heute"],   # window slid: only the tail
        ["guten Morgen", "", "guten"],                               # empty confirmed
    ])
    assert out[0]["confirmed"] == "guten Morgen liebe Zuhörer"
    assert out[1]["confirmed"] == "guten Morgen"


def test_pending_does_not_repeat_words_the_page_already_shows(tmp_path):
    out = _merge(tmp_path, [
        # frame lags one word behind: its pending starts with the word already confirmed on screen
        ["eins zwei drei", "eins zwei", "drei vier"],
        # frame lags and its pending is only what is already shown
        ["eins zwei drei", "eins zwei", "drei"],
    ])
    assert out[0] == {"confirmed": "eins zwei drei", "pending": "vier"}
    assert out[1] == {"confirmed": "eins zwei drei", "pending": ""}


def test_growth_and_same_length_revisions_are_taken_from_the_server(tmp_path):
    out = _merge(tmp_path, [
        ["hallo", "hallo welt", "wie"],
        ["", "hallo", ""],
        ["hallo welt", "Hallo Welt", ""],                            # same length, revised casing
        [None, None, None],
    ])
    assert out[0] == {"confirmed": "hallo welt", "pending": "wie"}
    assert out[1] == {"confirmed": "hallo", "pending": ""}
    assert out[2]["confirmed"] == "Hallo Welt"
    assert out[3] == {"confirmed": "", "pending": ""}


@pytest.mark.parametrize("code, fragment", [
    (1001, "restarting"),
    (1008, "refused"),
    (1011, "internal error"),
    (1013, "busy"),
    (4408, "idle"),
    (1006, "lost"),
    (4999, "4999"),
])
def test_close_codes_get_a_readable_message(tmp_path, code, fragment):
    text = run_js(tmp_path, f"return describeLiveClose({code}, '');")
    assert fragment in text


def test_normal_closes_are_not_reported(tmp_path):
    assert run_js(tmp_path, "return [1000, 1005, undefined].map((c) => describeLiveClose(c, ''));") == [None, None, None]


def test_close_reason_from_the_server_is_included(tmp_path):
    assert "provider 'x' is not live" in run_js(
        tmp_path, "return describeLiveClose(1008, \"provider 'x' is not live\");")


def test_error_frames_are_read_from_message_or_the_legacy_error_key(tmp_path):
    out = run_js(tmp_path, """
        return [
            describeLiveError({ type: 'error', message: 'new style', error: 'new style' }),
            describeLiveError({ type: 'error', error: 'legacy relay frame' }),
            describeLiveError({ type: 'error' }),
        ];
    """)
    assert out == ["new style", "legacy relay frame", "Streaming error"]


def test_mic_errors_are_explained(tmp_path):
    out = run_js(tmp_path, """
        const insecure = describeMicError(new Error('x'));
        navigator.mediaDevices = { getUserMedia: () => {} };
        const denied = describeMicError({ name: 'NotAllowedError' });
        const missing = describeMicError({ name: 'NotFoundError' });
        const busy = describeMicError({ name: 'NotReadableError' });
        return { insecure, denied, missing, busy };
    """)
    assert "https" in out["insecure"].lower(), "plain-http deployments have no navigator.mediaDevices at all"
    assert "denied" in out["denied"]
    assert "No microphone" in out["missing"]
    assert "in use" in out["busy"]


def test_live_socket_targets_the_provider_that_can_do_live_transcription(tmp_path):
    assert run_js(tmp_path, "return getLiveSttProviderId();") == "whisper"


# --- catalog reads and the language option ----------------------------------


def test_voice_quality_is_read_where_the_gateway_puts_it(tmp_path):
    """The normaliser nests the backend record under `raw`; `voice.quality` is
    always undefined, so the custom-voice card never showed a quality."""
    out = run_js(tmp_path, """
        return [
            getVoiceQuality({ id: 'v', raw: { quality: 'high' }, description: 'high' }),
            getVoiceQuality({ id: 'v', quality: 'low' }),
            getVoiceQuality({ id: 'v' }),
        ];
    """)
    assert out == ["high", "low", ""]


def test_auto_language_is_not_labelled_auto_detect_for_piper(tmp_path):
    """A TTS backend cannot detect a language from nothing; 'auto' means the
    server decides, and the fallback is a deployment setting."""
    out = run_js(tmp_path, """
        const before = ttsLanguageOptions('piper').find((o) => o.value === 'auto').label;
        noteServerDefaultLanguage('piper', 'de_DE');
        const after = ttsLanguageOptions('piper').find((o) => o.value === 'auto').label;
        return { before, after, others: ttsLanguageOptions('piper').filter((o) => o.value !== 'auto').map((o) => o.label) };
    """)
    assert "Auto-Detect" not in out["before"] and "Auto-Detect" not in out["after"]
    assert "server decides" in out["before"]
    assert "German" in out["after"], "the default is named from what the server reports"
    assert out["others"] == ["German", "English"]


def test_a_deliberate_registry_label_for_auto_is_kept(tmp_path):
    registry = json.loads(json.dumps(REGISTRY))
    registry["providers"]["piper"]["settings"]["languages"][0]["label"] = "Ask the server"
    out = run_js(tmp_path, "return ttsLanguageOptions('piper')[0].label;", registry)
    assert out == "Ask the server"


# --- the generic TTS panel: Piper, Magpie and Chatterbox share it ---------------------
#
# The panel showed Piper's quality, gender and speed controls for all three engines and
# sent what they held, kept the voice list of whichever engine filled it last (so a
# Piper voice id went to Magpie, which answers it with a 400), and asked Chatterbox and
# Magpie for Piper's lists. What each engine has is in the gateway's registry, so these
# run against the registry the gateway really embeds in the page.

PANEL_GROUPS = ["tts-quality-group", "tts-gender-group", "tts-speed-group", "tts-voice-group", "custom-voices-panel"]
SPEAKERS = ["Aria", "Jason", "John", "Leo", "Sofia"]


def _gateway_registry(env=None):
    """The gateway's registry with Magpie and Chatterbox enabled, minus the status row.

    `show_status` is dropped so the health poll every engine switch starts is not one
    more request in the lists these tests compare.
    """
    module = load_frontend_app({"ENABLE_MAGPIE_TTS": "true", "ENABLE_CHATTERBOX_TTS": "true", **(env or {})})
    registry = json.loads(json.dumps(module.PROVIDER_REGISTRY))
    for provider in registry["providers"].values():
        provider.get("ui", {}).pop("show_status", None)
    return registry


def _magpie_voice_answer(monkeypatch):
    """What the gateway's /api/providers/magpie/voices answers for the service's /speakers."""
    module = load_frontend_app({"ENABLE_MAGPIE_TTS": "true"})
    install_stub(monkeypatch, module, lambda method, url, kwargs: {
        "speakers": SPEAKERS, "languages": ["de", "en", "es", "fr", "ja", "zh"], "default_language": "de"})
    return TestClient(module.app).get("/api/providers/magpie/voices").json()


def test_the_panel_shows_only_the_controls_the_engine_has(tmp_path):
    out = run_js(tmp_path, f"""
        const groups = {json.dumps(PANEL_GROUPS)};
        const seen = [];
        for (const engine of ['piper', 'magpie', 'chatterbox', 'piper']) {{
            switchTTSEngine(engine);
            seen.push(groups.filter((id) => harness.el(id).style.display !== 'none'));
        }}
        return seen;
    """, _gateway_registry())
    piper, magpie, chatterbox, piper_again = out
    assert piper == PANEL_GROUPS and piper_again == PANEL_GROUPS
    assert magpie == ["tts-voice-group"], "Magpie has its five speakers and nothing else of Piper's panel"
    assert chatterbox == []


def test_generate_sends_only_the_fields_the_engine_has(tmp_path):
    """The hidden controls still hold Piper's values; they must not reach another engine."""
    out = run_js(tmp_path, """
        const bodies = {};
        for (const engine of ['piper', 'magpie', 'chatterbox']) {
            switchTTSEngine(engine);
            harness.el('tts-text').value = 'Hallo Welt.';
            harness.el('tts-language-select').value = 'de';
            harness.el('tts-quality-select').value = 'high';
            harness.el('tts-gender-select').value = 'female';
            harness.el('tts-speed').value = '1.5';
            harness.el('tts-voice-select').value = 'auto';
            const mark = harness.requests.length;
            await generateTTS();
            bodies[engine] = JSON.parse(harness.requests.slice(mark).find((r) => r.url === '/api/tts').body);
        }
        return bodies;
    """, _gateway_registry())
    assert list(out["piper"].items()) == [
        ("provider", "piper"), ("text", "Hallo Welt."), ("speed", 1.5), ("output_format", "wav"),
        ("instructions", ""), ("language", "de"), ("quality", "high"), ("gender", "female"),
    ], "Piper's body (and its key order) is what it always was"
    for engine in ("magpie", "chatterbox"):
        assert out[engine] == {"provider": engine, "text": "Hallo Welt.", "output_format": "wav",
                               "instructions": "", "language": "de"}, engine


def test_switching_engines_never_offers_or_sends_the_previous_engines_voices(tmp_path, monkeypatch):
    magpie = _magpie_voice_answer(monkeypatch)
    out = run_js(tmp_path, f"""
        const voices = harness.select('tts-voice-select');
        const urls = (from) => harness.requests.slice(from).map((r) => r.url);
        harness.fetchQueue.push(harness.json({{ voices: [{{ id: 'de_DE-thorsten-medium', name: 'thorsten',
            language: 'de_DE', kind: 'default', raw: {{ quality: 'medium' }} }}] }}));
        await refreshTTSVoices();
        voices.value = 'de_DE-thorsten-medium';
        const picked = voices.value;

        // Magpie's catalog is still on its way while the user generates.
        let answer;
        harness.fetchQueue.push(() => new Promise((resolve) => {{ answer = resolve; }}));
        let mark = harness.requests.length;
        switchTTSEngine('magpie');
        const whileLoading = {{ options: voices.options.map((o) => o.value), value: voices.value, requests: urls(mark) }};
        harness.el('tts-text').value = 'Hallo.';
        mark = harness.requests.length;
        await generateTTS();
        const sent = JSON.parse(harness.requests.slice(mark).find((r) => r.url === '/api/tts').body);

        answer(harness.json({json.dumps(magpie)}));
        await harness.flush();
        const loaded = voices.options.map((o) => [o.value, o.textContent]);

        mark = harness.requests.length;
        switchTTSEngine(currentTTSEngine);       // what initializeApp does: not a switch
        const sameEngine = {{ requests: urls(mark), options: voices.options.length }};

        mark = harness.requests.length;
        switchTTSEngine('chatterbox');
        const chatterbox = {{ options: voices.options.map((o) => o.value), requests: urls(mark) }};
        return {{ picked, whileLoading, sent, loaded, sameEngine, chatterbox }};
    """, _gateway_registry())
    assert out["picked"] == "de_DE-thorsten-medium"
    assert out["whileLoading"] == {"options": ["auto"], "value": "auto", "requests": ["/api/providers/magpie/voices"]}
    assert "voice" not in out["sent"], "a Piper voice id was sent to Magpie, which refuses it with a 400"
    assert out["loaded"] == [["auto", "Service default voice"], *[[name, name] for name in SPEAKERS]], (
        "a built-in speaker is listed by its name, and 'auto' says it is one fixed voice")
    assert out["sameEngine"] == {"requests": [], "options": 1 + len(SPEAKERS)}
    assert out["chatterbox"] == {"options": ["auto"], "requests": []}, "Chatterbox has no voice catalog to ask for"


def test_the_custom_voices_panel_is_filled_for_piper_and_only_for_piper(tmp_path):
    """Under Magpie the panel stayed at 'Loading custom voices...' (or showed 'Is the
    PiperTTS service running?'), and a switch back to Piper on the open tab never filled it."""
    out = run_js(tmp_path, """
        const urls = (from) => harness.requests.slice(from).map((r) => r.url);
        let mark = harness.requests.length;
        switchTTSEngine('magpie');
        switchTTSEngine('piper');             // the tab is not open: showTab fills it later
        const tabClosed = urls(mark);
        await harness.flush();

        switchTTSEngine('magpie');
        await harness.flush();
        mark = harness.requests.length;
        showTab('tts-tab');
        await harness.flush();
        const tabUnderMagpie = urls(mark);

        harness.fetchQueue.push(harness.json({ voices: [] }), harness.json({ voices: [] }));
        mark = harness.requests.length;
        switchTTSEngine('piper');
        await harness.flush();
        return { tabClosed, tabUnderMagpie, toPiper: urls(mark), list: harness.el('custom-voices-list').textContent };
    """, _gateway_registry())
    assert out["tabClosed"] == ["/api/providers/magpie/voices", "/api/providers/piper/voices"]
    assert out["tabUnderMagpie"] == ["/api/providers/magpie/voices"]
    assert out["toPiper"] == ["/api/providers/piper/voices", "/api/providers/piper/custom-voices"]
    assert "No custom trained voices" in out["list"]


def _two_engine_registry():
    registry = json.loads(json.dumps(REGISTRY))
    registry["providers"]["piper"]["ui"]["sections"] = {"tts": {"text_sample": "Piper sample."}}
    registry["providers"]["magpie"] = {
        "kind": "tts",
        "ui": {"family": "piper", "sections": {"tts": {"text_sample": "Magpie Beispiel."}}},
        "settings": {"languages": [{"value": "auto", "label": "Automatic"}]},
    }
    return registry


def test_switching_engines_keeps_the_text_the_user_typed(tmp_path):
    """The TTS engines share one text box; comparing two engines on one text must
    not lose that text to the next engine's sample."""
    out = run_js(tmp_path, """
        const box = document.getElementById('tts-text');
        box.value = 'Mein eigener Text.';
        switchTTSEngine('magpie');
        const afterMagpie = box.value;
        switchTTSEngine('qwen3');
        switchTTSEngine('piper');
        return { afterMagpie, afterPiper: box.value };
    """, _two_engine_registry())
    assert out == {"afterMagpie": "Mein eigener Text.", "afterPiper": "Mein eigener Text."}


def test_an_untouched_or_empty_box_takes_the_new_engines_sample(tmp_path):
    out = run_js(tmp_path, """
        const box = document.getElementById('tts-text');
        box.value = '';
        switchTTSEngine('piper');
        const fromEmpty = box.value;
        switchTTSEngine('magpie');
        const fromSample = box.value;
        box.value = '  ';
        switchTTSEngine('piper');
        return { fromEmpty, fromSample, fromBlank: box.value };
    """, _two_engine_registry())
    assert out == {"fromEmpty": "Piper sample.", "fromSample": "Magpie Beispiel.", "fromBlank": "Piper sample."}


def test_magpies_automatic_language_names_the_default_the_service_reports(tmp_path, monkeypatch):
    """The registry labelled it "Automatic (service default)", a label app.js keeps as one
    chosen on purpose, so the German the service reports was never named."""
    magpie = _magpie_voice_answer(monkeypatch)
    assert magpie["default_language"] == "de"
    out = run_js(tmp_path, f"""
        const languages = harness.select('tts-language-select');
        const auto = () => languages.options.find((o) => o.value === 'auto').textContent;
        harness.fetchQueue.push(harness.json({json.dumps(magpie)}));
        switchTTSEngine('magpie');
        const beforeTheAnswer = auto();
        await harness.flush();
        return {{ beforeTheAnswer, after: auto(), selected: languages.value }};
    """, _gateway_registry())
    assert out["beforeTheAnswer"] == "Automatic - server decides"
    assert out["after"] == "Automatic - server decides (default: German)"
    assert out["selected"] == "auto"


# --- text safety -------------------------------------------------------------


def test_notifications_render_text_not_markup(tmp_path):
    out = run_js(tmp_path, """
        showNotification('<img src=x onerror=alert(1)>', 'error');
        const n = harness.el('notification-container').children[0];
        return { text: n.textContent, html: n.innerHTML, role: n.attrs.role };
    """)
    assert out["text"] == "<img src=x onerror=alert(1)>"
    assert out["html"] == ""
    assert out["role"] == "alert"


def test_bindactions_binds_each_element_action_pair_once(tmp_path):
    """Static wiring runs on `document`; list renderers run on their container.
    An element visited by both must not fire its handler twice per click."""
    out = run_js(tmp_path, """
        const button = document.createElement('button');
        button.dataset.action = 'go';
        const root = { querySelectorAll: () => [button] };
        let calls = 0;
        bindActions(root, { go: () => { calls++; } });
        bindActions(root, { go: () => { calls++; } });
        button.listeners.click.forEach((fn) => fn());
        return { listeners: button.listeners.click.length, calls };
    """)
    assert out == {"listeners": 1, "calls": 1}


def test_bindactions_uses_the_declared_event(tmp_path):
    out = run_js(tmp_path, """
        const select = document.createElement('select');
        select.dataset.action = 'pick';
        select.dataset.actionEvent = 'change';
        bindActions({ querySelectorAll: () => [select] }, { pick: () => {} });
        return Object.keys(select.listeners);
    """)
    assert out == ["change"]


def test_withbusy_disables_the_button_for_the_duration_and_restores_it(tmp_path):
    out = run_js(tmp_path, """
        const button = document.createElement('button');
        button.textContent = 'Go';
        let release;
        const running = withBusy(button, () => new Promise((r) => { release = r; }), 'Working...');
        const during = { disabled: button.disabled, busy: button.getAttribute('aria-busy'), label: button.textContent };
        const second = await withBusy(button, async () => 'ran');   // a second click while busy
        release();
        await running;
        const after = { disabled: button.disabled, busy: button.getAttribute('aria-busy'), label: button.textContent };

        const failing = document.createElement('button');
        try { await withBusy(failing, async () => { throw new Error('boom'); }); } catch (e) { /* expected */ }
        return { during, second: second === undefined, after, failedButtonRestored: failing.disabled === false };
    """)
    assert out["during"] == {"disabled": True, "busy": "true", "label": "Working..."}
    assert out["second"], "a click on a busy button must not start the task again"
    assert out["after"] == {"disabled": False, "busy": None, "label": "Go"}
    assert out["failedButtonRestored"]


def test_progress_ticks_are_hidden_from_screen_readers_but_results_are_not(tmp_path):
    out = run_js(tmp_path, """
        showStatus('s', 'info', 'tick 1.0s', { silent: true });
        const tick = harness.status('s');
        showStatus('s', 'error', 'failed');
        const error = harness.status('s');
        showStatus('s', 'success', 'done');
        const success = harness.status('s');
        return { tick: tick.attrs, error: error.attrs, success: success.attrs };
    """)
    assert out["tick"] == {"aria-hidden": "true"}
    assert out["error"] == {"role": "alert"}
    assert out["success"] == {}


# --- clipboard and form validation ------------------------------------------


def test_copy_works_without_the_async_clipboard_api(tmp_path):
    """`navigator.clipboard` does not exist on a plain-http LAN/NAS address, so the
    old code threw a TypeError before its own fallback could run."""
    out = run_js(tmp_path, """
        harness.el('full-transcription').textContent = 'hallo welt';
        navigator.clipboard = undefined;
        let commands = 0;
        document.execCommand = (name) => { commands += name === 'copy' ? 1 : 0; return true; };
        await copyTranscription();
        return { commands, notes: harness.notifications() };
    """)
    assert out["commands"] == 1
    assert [n["className"] for n in out["notes"]] == ["notification notification-success"]


def test_copy_reports_failure_instead_of_claiming_success(tmp_path):
    out = run_js(tmp_path, """
        harness.el('full-transcription').textContent = 'hallo welt';
        navigator.clipboard = { writeText: async () => { throw new Error('denied'); } };
        document.execCommand = () => false;
        await copyTranscription();
        return harness.notifications();
    """)
    assert [n["className"] for n in out] == ["notification notification-error"]


def test_training_form_rejects_unparseable_numbers_instead_of_sending_nan(tmp_path):
    out = run_js(tmp_path, """
        harness.el('training-voice-name').value = 'luna';
        harness.el('training-files').files = [{ name: 'a.wav' }];
        harness.el('training-epochs').value = '';
        harness.el('training-batch-size').value = '32';
        await startTraining();
        const emptyEpochs = harness.status('training-progress-status');

        harness.el('training-epochs').value = '100';
        harness.el('training-batch-size').value = '';        // the "Loading batch sizes..." placeholder
        await startTraining();
        return { emptyEpochs, emptyBatch: harness.status('training-progress-status'), requests: harness.requests.length };
    """)
    assert out["emptyEpochs"]["className"] == "error" and "epochs" in out["emptyEpochs"]["text"]
    assert out["emptyBatch"]["className"] == "error" and "batch" in out["emptyBatch"]["text"].lower()
    assert out["requests"] == 0, "nothing may be sent with NaN in the form"


# --- the API key ------------------------------------------------------------------
#
# With API_KEY set the gateway wants `Authorization: Bearer <key>` on every /v1 call and
# every state-changing /api call, the UI's included (a request that merely looks like the
# UI is exactly what a DNS-rebinding page sends). The page learns it needs a key from a
# 401 carrying `WWW-Authenticate: Bearer`, asks once, keeps the key for the tab, and
# sends it from then on. Without API_KEY none of this shows.


def test_without_a_key_nothing_extra_is_sent_and_nobody_is_asked(tmp_path):
    out = run_js(tmp_path, """
        const response = await fetch('/api/training/model/job-1', { method: 'DELETE' });
        return { ok: response.ok, requests: harness.requests, prompts: harness.prompts,
                 stored: Array.from(harness.storage.keys()) };
    """)
    assert out["ok"] is True
    assert [r["authorization"] for r in out["requests"]] == [None]
    assert out["prompts"] == [] and out["stored"] == []


def test_a_challenge_prompts_once_retries_with_the_key_and_remembers_it(tmp_path):
    out = run_js(tmp_path, """
        harness.promptAnswers.push('  s3cret  ');
        harness.fetchQueue.push(harness.unauthorized(), harness.json({ deleted: true }));
        const first = await fetch('/api/training/model/job-1', { method: 'DELETE' });
        const firstBody = await first.json();

        // later calls carry the key straight away, reads included
        harness.fetchQueue.push(harness.json({ ok: 1 }), harness.json({ ok: 2 }));
        await fetch('/api/providers/qwen3/unload', { method: 'POST' });
        await fetch('/api/health');
        return {
            firstBody,
            auth: harness.requests.map((r) => r.authorization),
            urls: harness.requests.map((r) => r.url),
            prompts: harness.prompts,
            stored: harness.storage.get('tts-stt.api-key'),
        };
    """)
    assert out["firstBody"] == {"deleted": True}
    assert out["auth"] == [None, "Bearer s3cret", "Bearer s3cret", "Bearer s3cret"]
    assert len(out["prompts"]) == 1 and "requires an API key" in out["prompts"][0]
    assert out["stored"] == "s3cret", "kept for the tab, whitespace trimmed"


def test_declining_the_prompt_shows_the_401_and_asks_again_next_time(tmp_path):
    out = run_js(tmp_path, """
        harness.promptAnswers.push(null, '   ');                   // cancel, then an empty entry
        harness.fetchQueue.push(harness.unauthorized(), harness.unauthorized());
        const a = await fetch('/api/tts', { method: 'POST', body: '{}' });
        const b = await fetch('/api/tts', { method: 'POST', body: '{}' });
        return { statuses: [a.status, b.status], requests: harness.requests.length,
                 prompts: harness.prompts.length, stored: harness.storage.size };
    """)
    assert out["statuses"] == [401, 401]
    assert out["requests"] == 2, "no retry without a key"
    assert out["prompts"] == 2 and out["stored"] == 0


def test_a_refused_key_is_dropped_and_asked_for_again(tmp_path):
    out = run_js(tmp_path, """
        setGatewayApiKey('rotated-away');
        harness.promptAnswers.push('fresh');
        harness.fetchQueue.push(harness.unauthorized(), harness.json({ ok: true }));
        const response = await fetch('/api/training/train', { method: 'POST', body: 'x' });
        return { ok: response.ok, auth: harness.requests.map((r) => r.authorization),
                 prompt: harness.prompts[0], stored: harness.storage.get('tts-stt.api-key') };
    """)
    assert out["ok"] is True
    assert out["auth"] == ["Bearer rotated-away", "Bearer fresh"]
    assert "not accepted" in out["prompt"]
    assert out["stored"] == "fresh"


def test_a_wrong_key_is_not_kept(tmp_path):
    out = run_js(tmp_path, """
        harness.promptAnswers.push('typo');
        harness.fetchQueue.push(harness.unauthorized(), harness.unauthorized());
        const response = await fetch('/api/tts', { method: 'POST', body: '{}' });
        return { status: response.status, requests: harness.requests.length, stored: harness.storage.size };
    """)
    assert out["status"] == 401
    assert out["requests"] == 2, "one retry only, never a loop"
    assert out["stored"] == 0


def test_simultaneous_challenges_share_one_prompt(tmp_path):
    out = run_js(tmp_path, """
        harness.promptAnswers.push('s3cret');
        let seen = 0;
        const handler = () => (seen++ < 2 ? harness.unauthorized() : harness.json({ ok: true }));
        harness.fetchQueue.push(handler, handler, handler, handler);
        const [a, b] = await Promise.all([
            fetch('/api/training/model/a', { method: 'DELETE' }),
            fetch('/api/training/model/b', { method: 'DELETE' }),
        ]);
        return { ok: [a.ok, b.ok], prompts: harness.prompts.length,
                 auth: harness.requests.map((r) => r.authorization) };
    """)
    assert out["ok"] == [True, True]
    assert out["prompts"] == 1, "two dialogs for one missing key"
    assert out["auth"].count("Bearer s3cret") == 2


def test_the_key_is_only_ever_sent_to_the_gateway(tmp_path):
    out = run_js(tmp_path, """
        setGatewayApiKey('s3cret');
        for (const url of ['/static/js/mic-worklet.js', 'https://cdn.example/x.js', '//evil.example/api/x',
                           'api/relative', '/apix', '/v10/models']) {
            await fetch(url);
        }
        await fetch('/api/health');
        await fetch('/v1/models');
        return harness.requests.map((r) => [r.url, r.authorization]);
    """)
    assert [entry for entry in out if entry[1]] == [["/api/health", "Bearer s3cret"], ["/v1/models", "Bearer s3cret"]]


def test_a_401_that_is_not_a_key_challenge_does_not_prompt(tmp_path):
    out = run_js(tmp_path, """
        harness.fetchQueue.push(harness.json({ detail: 'nope' }, 401), harness.unauthorized('Basic realm="x"'));
        const a = await fetch('/api/tts', { method: 'POST', body: '{}' });
        const b = await fetch('/api/tts', { method: 'POST', body: '{}' });
        return { statuses: [a.status, b.status], prompts: harness.prompts.length };
    """)
    assert out["statuses"] == [401, 401] and out["prompts"] == 0


def test_the_retry_sends_the_same_body_and_options(tmp_path):
    out = run_js(tmp_path, """
        harness.promptAnswers.push('k');
        harness.fetchQueue.push(harness.unauthorized(), harness.json({ ok: true }));
        const form = new FormData();
        form.append('provider', 'whisper');
        await fetch('/api/stt', { method: 'POST', body: form, headers: { 'X-Trace': '1' } });
        return { sameBody: harness.requests[0].body === form && harness.requests[1].body === form,
                 methods: harness.requests.map((r) => r.method) };
    """)
    assert out["sameBody"] is True
    assert out["methods"] == ["POST", "POST"]


def test_a_browser_that_blocks_storage_still_works_from_memory(tmp_path):
    out = run_js(tmp_path, """
        harness.storageThrows = true;
        harness.promptAnswers.push('s3cret');
        harness.fetchQueue.push(harness.unauthorized(), harness.json({ ok: 1 }), harness.json({ ok: 2 }));
        await fetch('/api/tts', { method: 'POST', body: '{}' });
        await fetch('/api/tts', { method: 'POST', body: '{}' });
        return { auth: harness.requests.map((r) => r.authorization), prompts: harness.prompts.length,
                 reread: readStoredApiKey() };
    """)
    assert out["auth"] == [None, "Bearer s3cret", "Bearer s3cret"]
    assert out["prompts"] == 1
    assert out["reread"] == ""


def test_the_key_is_read_back_from_session_storage(tmp_path):
    out = run_js(tmp_path, """
        sessionStorage.setItem('tts-stt.api-key', 'from-an-earlier-load');
        return readStoredApiKey();
    """)
    assert out == "from-an-earlier-load"


def test_the_key_is_never_written_to_local_storage_or_the_page(tmp_path):
    out = run_js(tmp_path, """
        harness.promptAnswers.push('s3cret');
        harness.fetchQueue.push(harness.unauthorized(), harness.json({ ok: 1 }));
        await fetch('/api/tts', { method: 'POST', body: '{}' });
        return { local: typeof localStorage, storedKeys: Array.from(harness.storage.keys()) };
    """)
    assert out == {"local": "undefined", "storedKeys": ["tts-stt.api-key"]}


# --- the API key on the live-transcription WebSocket ---------------------------------------


def _b64url(text: str) -> str:
    import base64
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode().rstrip("=")


def test_socket_protocols_carry_the_key_the_way_the_gateway_reads_it(tmp_path):
    key = "pässwörd, with spaces/and=signs+more"
    out = run_js(tmp_path, f"""
        const without = liveSocketProtocols();
        setGatewayApiKey({json.dumps(key)});
        return {{ without: without === undefined, with: liveSocketProtocols() }};
    """)
    assert out["without"] is True, "no key: the socket is opened exactly as before, with no protocol argument"
    assert out["with"] == ["tts-stt.v1", "bearer." + _b64url(key)]
    assert all(c.isalnum() or c in "-_." for c in out["with"][1]), "must stay inside the subprotocol token alphabet"


def test_live_transcription_asks_for_the_key_before_the_microphone_and_offers_it_to_the_socket(tmp_path):
    out = run_js(tmp_path, """
        const order = [];
        harness.promptAnswers.push('s3cret');
        harness.fetchQueue.push(
            () => { order.push('auth-check'); return harness.unauthorized(); },
            () => { order.push('auth-check-retry'); return harness.json({}, 204); });
        navigator.mediaDevices = { getUserMedia: async () => { order.push('microphone'); return { getTracks: () => [] }; } };
        const sockets = [];
        const RealConstants = WebSocket;
        globalThis.WebSocket = Object.assign(function FakeSocket(url, protocols) {
            order.push('socket');
            sockets.push({ url, protocols, argCount: arguments.length });
            this.readyState = 0;
        }, RealConstants);

        await toggleLiveTranscription();
        return { order, sockets, requests: harness.requests.map((r) => [r.method, r.url, r.authorization]) };
    """)
    assert out["order"] == ["auth-check", "auth-check-retry", "microphone", "socket"]
    assert out["requests"] == [["POST", "/api/auth/check", None], ["POST", "/api/auth/check", "Bearer s3cret"]]
    assert out["sockets"][0]["protocols"] == ["tts-stt.v1", "bearer." + _b64url("s3cret")]
    assert out["sockets"][0]["url"].startswith("ws://gateway.test/ws/stt?provider=")


def test_live_transcription_without_a_key_opens_the_socket_with_one_argument(tmp_path):
    out = run_js(tmp_path, """
        navigator.mediaDevices = { getUserMedia: async () => ({ getTracks: () => [] }) };
        const sockets = [];
        const RealConstants = WebSocket;
        globalThis.WebSocket = Object.assign(function FakeSocket(url, protocols) {
            sockets.push({ argCount: arguments.length });
            this.readyState = 0;
        }, RealConstants);
        await toggleLiveTranscription();
        return { sockets, prompts: harness.prompts.length };
    """)
    assert out["sockets"] == [{"argCount": 1}]
    assert out["prompts"] == 0


def test_a_socket_refused_for_the_key_says_so_and_forgets_the_key(tmp_path):
    out = run_js(tmp_path, """
        setGatewayApiKey('stale');
        navigator.mediaDevices = { getUserMedia: async () => ({ getTracks: () => [] }) };
        let socket;
        const RealConstants = WebSocket;
        globalThis.WebSocket = Object.assign(function FakeSocket() { socket = this; this.readyState = 0; }, RealConstants);
        await toggleLiveTranscription();
        socket.onclose({ code: 1008, reason: 'A valid API key is required' });
        return { stored: harness.storage.size, status: harness.status('live-stt-status') };
    """)
    assert out["stored"] == 0
    assert out["status"]["className"] == "error"
    assert "API key" in out["status"]["text"]


def test_an_unreachable_gateway_does_not_stop_the_socket_attempt(tmp_path):
    """The check is only there to collect a key; the socket reports a dead server itself."""
    out = run_js(tmp_path, """
        harness.fetchQueue.push(() => { throw new TypeError('Failed to fetch'); });
        navigator.mediaDevices = { getUserMedia: async () => ({ getTracks: () => [] }) };
        const sockets = [];
        const RealConstants = WebSocket;
        globalThis.WebSocket = Object.assign(function FakeSocket(url) { sockets.push(url); this.readyState = 0; }, RealConstants);
        await toggleLiveTranscription();
        return sockets.length;
    """)
    assert out == 1


# --- STT language --------------------------------------------------------------------------


@pytest.mark.parametrize("selected", ["auto", "de", "en"])
def test_the_stt_upload_always_says_which_language_it_wants(tmp_path, selected):
    """"auto" used to be left out of the form, which a service reads as "use the server default"
    (German in the TrueNAS profile) instead of "detect"."""
    out = run_js(tmp_path, f"""
        harness.el('stt-file').files = [new Blob(['RIFF....'])];
        harness.el('stt-language').value = {json.dumps(selected)};
        harness.el('enable-segmentation').checked = false;
        harness.el('stt-engine-select').value = 'whisper';
        harness.fetchQueue.push(harness.json({{ text: 'hallo', segments: [] }}));
        await processSTT();
        const form = harness.requests[0].body;
        return {{ url: harness.requests[0].url, fields: Array.from(form.keys()), language: form.get('language'),
                 provider: form.get('provider') }};
    """)
    assert out["url"] == "/api/stt"
    assert out["language"] == selected
    assert out["provider"] == "whisper"


def test_the_auth_check_is_made_once_per_accepted_key_not_on_every_start(tmp_path):
    out = run_js(tmp_path, """
        await ensureApiKeyForSocket();
        await ensureApiKeyForSocket();
        await ensureApiKeyForSocket();
        const afterAccepted = harness.requests.length;

        // the key changes (or is dropped after a refused socket): ask the gateway again
        setGatewayApiKey('another');
        await ensureApiKeyForSocket();
        return { afterAccepted, total: harness.requests.length };
    """)
    assert out == {"afterAccepted": 1, "total": 2}


def test_a_failed_auth_check_is_not_remembered_as_accepted(tmp_path):
    out = run_js(tmp_path, """
        harness.fetchQueue.push(() => { throw new TypeError('Failed to fetch'); }, harness.json({}, 500));
        await ensureApiKeyForSocket();            // network error
        await ensureApiKeyForSocket();            // server error
        await ensureApiKeyForSocket();            // finally fine
        await ensureApiKeyForSocket();            // remembered now
        return harness.requests.length;
    """)
    assert out == 3


# --- voice training offered or not -----------------------------------------------------------
#
# ENABLE_TRAINING=false (the Settings page or the app YAML) takes the piper-training entry out
# of the registry, and the gateway then answers every training route with a 404.


def _with_training() -> dict:
    registry = json.loads(json.dumps(REGISTRY))
    registry["providers"]["piper-training"] = {"kind": "training"}
    return registry


def test_without_a_training_provider_the_tab_goes_and_nothing_asks_the_training_api(tmp_path):
    """The tab stayed, with every call in it a 404, and every page load asked the training
    service for its deployment targets."""
    out = run_js(tmp_path, """
        initializeApp();
        await harness.flush();
        showTab('training-tab');
        await harness.flush();
        return {
            urls: harness.requests.map((r) => r.method + ' ' + r.url),
            removed: [Boolean(harness.el('training-tab-button').removed), Boolean(harness.el('training-tab').removed)],
            offered: isTrainingOffered(),
        };
    """)
    assert out["offered"] is False
    assert out["removed"] == [True, True]
    assert not [url for url in out["urls"] if "/api/training/" in url], out["urls"]


def test_with_a_training_provider_the_tab_stays_and_its_targets_load(tmp_path):
    out = run_js(tmp_path, """
        initializeApp();
        await harness.flush();
        return {
            urls: harness.requests.map((r) => r.method + ' ' + r.url),
            removed: [Boolean(harness.el('training-tab-button').removed), Boolean(harness.el('training-tab').removed)],
            offered: isTrainingOffered(),
        };
    """, _with_training())
    assert out["offered"] is True
    assert out["removed"] == [False, False]
    assert "GET /api/training/deployment-targets" in out["urls"]


# --- the Settings page (settings.js) -----------------------------------------------------------
#
# settings.js runs in the real settings.html, parsed into a small DOM that refuses what the
# page's Content-Security-Policy refuses (innerHTML, on* and style attributes), and it is fed
# the answers a real gateway gave to the same requests: if the API changes shape, these see
# it. The numbers in the test names are the page tests of the phase-1 plan.

SETTINGS_JS = SERVICE_DIR / "static" / "js" / "settings.js"
SETTINGS_HTML = SERVICE_DIR / "templates" / "settings.html"
NAS = "192.168.1.20:3000"
YAML_KEY = "the-key-from-the-app-yaml"
PAGE_KEY = re.compile(r"tts_[A-Za-z0-9_-]{43}")
CLAIM_CODE = re.compile(r"\b[0-9A-Z]{5}(?:-[0-9A-Z]{5}){3}\b")

SETTINGS_DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const nodeCrypto = require('crypto');
const [scriptPath, htmlPath, scenarioPath, fixturesPath] = process.argv.slice(2);
const fixtures = JSON.parse(fs.readFileSync(fixturesPath, 'utf8'));

// --- a small DOM: the real template, parsed, and what settings.js does with it ---------
//
// Strict where the page's CSP is strict: innerHTML/outerHTML/insertAdjacentHTML throw,
// and so does setting an on* or style attribute.

const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr']);
const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: String.fromCharCode(160) };
const kebab = (key) => String(key).replace(/[A-Z]/g, (c) => '-' + c.toLowerCase());

function decode(text) {
    return text.replace(/&(#x[0-9a-f]+|#[0-9]+|[a-z]+);/gi, (whole, name) => {
        if (name[0] === '#') {
            const hex = name[1] === 'x' || name[1] === 'X';
            return String.fromCodePoint(parseInt(name.slice(hex ? 2 : 1), hex ? 16 : 10));
        }
        return ENTITIES[name.toLowerCase()] ?? whole;
    });
}

let doc = null;

class FakeEvent {
    constructor(type, init = {}) {
        this.type = type;
        this.bubbles = init.bubbles !== false;
        this.key = init.key;
        this.target = null;
        this.currentTarget = null;
        this.defaultPrevented = false;
        this.stopped = false;
    }
    preventDefault() { this.defaultPrevented = true; }
    stopPropagation() { this.stopped = true; }
}

function dispatch(target, event) {
    event.target = target;
    const path = [];
    for (let node = target; node; node = node.parentNode) path.push(node);
    for (const node of event.bubbles ? path : [target]) {
        event.currentTarget = node;
        for (const listener of (node.listeners[event.type] || []).slice()) listener.call(node, event);
        if (event.stopped) break;
    }
    return !event.defaultPrevented;
}

class TextNode {
    constructor(data) {
        this.nodeType = 3;
        this.data = String(data);
        this.parentNode = null;
    }
    get textContent() { return this.data; }
    set textContent(value) { this.data = String(value); }
}

function walk(root, visit) {
    for (const child of root.childNodes) {
        if (child.nodeType !== 1) continue;
        visit(child);
        walk(child, visit);
    }
}

// Selectors: tag, #id, .class, [attr], [attr="value"], compounds of these, and the
// descendant and child combinators. Anything else throws, so a test never passes
// because a selector silently matched nothing.
const selectorCache = new Map();

function parseCompound(text) {
    const compound = { tag: null, id: null, classes: [], attrs: [] };
    const token = /([a-zA-Z][a-zA-Z0-9-]*|\*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:=(?:"([^"]*)"|'([^']*)'|([\w-]+)))?\]/y;
    let index = 0;
    while (index < text.length) {
        token.lastIndex = index;
        const m = token.exec(text);
        if (!m) throw new Error(`unsupported selector: ${text}`);
        if (m[1]) compound.tag = m[1].toLowerCase();
        else if (m[2]) compound.id = m[2];
        else if (m[3]) compound.classes.push(m[3]);
        else compound.attrs.push({ name: m[4].toLowerCase(), value: m[5] ?? m[6] ?? m[7] ?? null });
        index = token.lastIndex;
    }
    return compound;
}

function parseComplex(text) {
    const parts = [];
    const piece = /(\s*>\s*|\s+)?([^\s>]+)/g;
    const source = text.trim();
    let m;
    while ((m = piece.exec(source)) !== null) {
        const combinator = parts.length === 0 ? null : (m[1] && m[1].includes('>') ? '>' : ' ');
        parts.push({ combinator, compound: parseCompound(m[2]) });
    }
    if (!parts.length) throw new Error(`empty selector: ${text}`);
    return parts;
}

function selectorList(text) {
    if (!selectorCache.has(text)) selectorCache.set(text, String(text).split(',').map(parseComplex));
    return selectorCache.get(text);
}

function matchCompound(node, compound) {
    if (!node || node.nodeType !== 1) return false;
    if (compound.tag && compound.tag !== '*' && node.localName !== compound.tag) return false;
    if (compound.id && node.getAttribute('id') !== compound.id) return false;
    if (compound.classes.length) {
        const classes = node._classes();
        if (!compound.classes.every((name) => classes.has(name))) return false;
    }
    for (const attr of compound.attrs) {
        const value = node.getAttribute(attr.name);
        if (value === null || (attr.value !== null && value !== attr.value)) return false;
    }
    return true;
}

function matchesComplex(node, parts, index) {
    if (!matchCompound(node, parts[index].compound)) return false;
    if (index === 0) return true;
    if (parts[index].combinator === '>') return matchesComplex(node.parentNode, parts, index - 1);
    for (let up = node.parentNode; up && up.nodeType === 1; up = up.parentNode) {
        if (matchesComplex(up, parts, index - 1)) return true;
    }
    return false;
}

function matchesAny(node, selector) {
    return selectorList(selector).some((parts) => matchesComplex(node, parts, parts.length - 1));
}

/** Is the element out of sight: hidden itself or by an ancestor, or inside a closed dialog? */
function hiddenByAncestor(node) {
    for (let up = node; up && up.nodeType === 1; up = up.parentNode) {
        if (up.hasAttribute('hidden')) return true;
        if (up.localName === 'dialog' && !up.hasAttribute('open')) return true;
    }
    return false;
}

class Element {
    constructor(tag) {
        this.nodeType = 1;
        this.localName = String(tag).toLowerCase();
        this.tagName = this.localName.toUpperCase();
        this.childNodes = [];
        this.parentNode = null;
        this.attrs = new Map();
        this.listeners = {};
        this._value = null;
        this._checked = null;
        this._selected = null;
        this._noneSelected = false;
        const self = this;
        this.dataset = new Proxy({}, {
            get: (target, key) => {
                if (typeof key !== 'string') return undefined;
                const name = 'data-' + kebab(key);
                return self.attrs.has(name) ? self.attrs.get(name) : undefined;
            },
            set: (target, key, value) => { self.setAttribute('data-' + kebab(key), value); return true; },
            has: (target, key) => self.attrs.has('data-' + kebab(key)),
            deleteProperty: (target, key) => { self.removeAttribute('data-' + kebab(key)); return true; },
        });
        this.classList = {
            contains: (name) => self._classes().has(name),
            add: (...names) => { const set = self._classes(); names.forEach((n) => set.add(n)); self._setClasses(set); },
            remove: (...names) => { const set = self._classes(); names.forEach((n) => set.delete(n)); self._setClasses(set); },
            toggle: (name, force) => {
                const set = self._classes();
                const on = force === undefined ? !set.has(name) : Boolean(force);
                if (on) set.add(name); else set.delete(name);
                self._setClasses(set);
                return on;
            },
        };
    }
    _classes() { return new Set((this.attrs.get('class') || '').split(/\s+/).filter(Boolean)); }
    _setClasses(set) { this.attrs.set('class', Array.from(set).join(' ')); }
    getAttribute(name) {
        const key = String(name).toLowerCase();
        return this.attrs.has(key) ? this.attrs.get(key) : null;
    }
    setAttribute(name, value) {
        const key = String(name).toLowerCase();
        if (key.startsWith('on')) throw new Error(`an inline handler attribute was set: ${key}`);
        if (key === 'style') throw new Error('a style attribute was set (the page CSP refuses it)');
        this.attrs.set(key, String(value));
    }
    hasAttribute(name) { return this.attrs.has(String(name).toLowerCase()); }
    removeAttribute(name) { this.attrs.delete(String(name).toLowerCase()); }
    get children() { return this.childNodes.filter((node) => node.nodeType === 1); }
    get textContent() { return this.childNodes.map((node) => node.textContent).join(''); }
    set textContent(value) {
        this.replaceChildren();
        const text = value == null ? '' : String(value);
        if (text) this.appendChild(new TextNode(text));
    }
    get innerHTML() { throw new Error('innerHTML was read'); }
    set innerHTML(value) { throw new Error('innerHTML was written'); }
    get outerHTML() { throw new Error('outerHTML was read'); }
    set outerHTML(value) { throw new Error('outerHTML was written'); }
    insertAdjacentHTML() { throw new Error('insertAdjacentHTML was used'); }
    appendChild(node) {
        if (!node || typeof node !== 'object' || !('nodeType' in node)) throw new TypeError('appendChild: not a node');
        if (node.parentNode) node.parentNode.removeChild(node);
        node.parentNode = this;
        this.childNodes.push(node);
        return node;
    }
    append(...nodes) {
        for (const node of nodes) this.appendChild(typeof node === 'string' ? new TextNode(node) : node);
    }
    removeChild(node) {
        const index = this.childNodes.indexOf(node);
        if (index >= 0) {
            this.childNodes.splice(index, 1);
            node.parentNode = null;
        }
        return node;
    }
    replaceChildren(...nodes) {
        for (const node of this.childNodes) node.parentNode = null;
        this.childNodes = [];
        if (this.localName === 'select') this._noneSelected = false;
        this.append(...nodes);
    }
    remove() { if (this.parentNode) this.parentNode.removeChild(this); }
    contains(node) {
        for (let up = node; up; up = up.parentNode) if (up === this) return true;
        return false;
    }
    get isConnected() {
        for (let up = this; up; up = up.parentNode) if (up === doc) return true;
        return false;
    }
    matches(selector) { return matchesAny(this, selector); }
    closest(selector) {
        for (let up = this; up && up.nodeType === 1; up = up.parentNode) if (matchesAny(up, selector)) return up;
        return null;
    }
    querySelectorAll(selector) {
        const found = [];
        walk(this, (node) => { if (matchesAny(node, selector)) found.push(node); });
        return found;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    addEventListener(type, listener) { (this.listeners[type] = this.listeners[type] || []).push(listener); }
    removeEventListener(type, listener) { this.listeners[type] = (this.listeners[type] || []).filter((l) => l !== listener); }
    dispatchEvent(event) { return dispatch(this, event); }
    focus() {
        if (this.disabled || !this.isConnected || hiddenByAncestor(this)) return;
        doc.activeElement = this;
    }
    blur() { if (doc.activeElement === this) doc.activeElement = doc.body; }
    select() { harness.selected = this.value; }
    scrollIntoView() {}
    click() {
        if (this.disabled) return;
        if (this.localName === 'input' && (this.type === 'checkbox' || this.type === 'radio')) {
            const before = this.checked;
            this.checked = this.type === 'radio' ? true : !before;
            dispatch(this, new FakeEvent('click'));
            if (this.checked !== before) {
                dispatch(this, new FakeEvent('input'));
                dispatch(this, new FakeEvent('change'));
            }
            return;
        }
        dispatch(this, new FakeEvent('click'));
    }
    showModal() {
        if (this.hasAttribute('open')) throw new Error(`InvalidStateError: #${this.id} is already open`);
        if (!this.isConnected) throw new Error('InvalidStateError: the dialog is not in the document');
        this.setAttribute('open', '');
    }
    close() {
        if (!this.hasAttribute('open')) return;
        this.removeAttribute('open');
        // As in a browser: the close event comes a moment later, and does not bubble.
        Promise.resolve().then(() => dispatch(this, new FakeEvent('close', { bubbles: false })));
    }
    get options() { return this.querySelectorAll('option'); }
    _selectedOption() {
        const options = this.options;
        const chosen = options.find((option) => option._selected === true);
        if (chosen) return chosen;
        if (this._noneSelected) return null;
        return options.find((option) => !option.disabled) || null;
    }
}

const STRING_PROPS = {
    id: 'id', className: 'class', name: 'name', title: 'title', htmlFor: 'for', placeholder: 'placeholder',
    min: 'min', max: 'max', step: 'step', rows: 'rows', maxLength: 'maxlength', role: 'role', href: 'href',
};
for (const [prop, attr] of Object.entries(STRING_PROPS)) {
    Object.defineProperty(Element.prototype, prop, {
        get() { const value = this.getAttribute(attr); return value === null ? '' : value; },
        set(value) { this.setAttribute(attr, value); },
        configurable: true,
    });
}
const BOOLEAN_PROPS = { hidden: 'hidden', disabled: 'disabled', readOnly: 'readonly', open: 'open', required: 'required' };
for (const [prop, attr] of Object.entries(BOOLEAN_PROPS)) {
    Object.defineProperty(Element.prototype, prop, {
        get() { return this.hasAttribute(attr); },
        set(value) { if (value) this.setAttribute(attr, ''); else this.removeAttribute(attr); },
        configurable: true,
    });
}
Object.defineProperty(Element.prototype, 'type', {
    get() {
        const value = this.getAttribute('type');
        if (value !== null) return value.toLowerCase();
        return this.localName === 'input' ? 'text' : this.localName === 'button' ? 'submit' : '';
    },
    set(value) { this.setAttribute('type', value); },
    configurable: true,
});
Object.defineProperty(Element.prototype, 'value', {
    get() {
        if (this.localName === 'select') {
            const option = this._selectedOption();
            return option ? option.value : '';
        }
        if (this.localName === 'option') return this.hasAttribute('value') ? this.getAttribute('value') : this.textContent;
        if (this._value !== null) return this._value;
        if (this.localName === 'textarea') return this.textContent;
        if (this.hasAttribute('value')) return this.getAttribute('value');
        return this.type === 'checkbox' || this.type === 'radio' ? 'on' : '';
    },
    set(value) {
        const text = value == null ? '' : String(value);
        if (this.localName === 'select') {
            let matched = false;
            for (const option of this.options) {
                option._selected = !matched && option.value === text;
                if (option._selected) matched = true;
            }
            this._noneSelected = !matched;
            return;
        }
        if (this.localName === 'option') {
            this.setAttribute('value', text);
            return;
        }
        this._value = text;
    },
    configurable: true,
});
Object.defineProperty(Element.prototype, 'checked', {
    get() { return this._checked === null ? this.hasAttribute('checked') : this._checked; },
    set(value) {
        this._checked = Boolean(value);
        if (this._checked && this.type === 'radio' && this.name && doc) {
            for (const other of doc.querySelectorAll('input')) {
                if (other !== this && other.type === 'radio' && other.name === this.name) other._checked = false;
            }
        }
    },
    configurable: true,
});

class FakeDocument {
    constructor() {
        this.nodeType = 9;
        this.parentNode = null;
        this.listeners = {};
        this.readyState = 'loading';
        this.documentElement = null;
        this.body = null;
        this.activeElement = null;
    }
    get childNodes() { return this.documentElement ? [this.documentElement] : []; }
    createElement(tag) { return new Element(tag); }
    createTextNode(text) { return new TextNode(text); }
    getElementById(id) {
        let found = null;
        walk(this, (node) => { if (!found && node.getAttribute('id') === String(id)) found = node; });
        return found;
    }
    querySelectorAll(selector) { return Element.prototype.querySelectorAll.call(this, selector); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    addEventListener(type, listener) { (this.listeners[type] = this.listeners[type] || []).push(listener); }
    removeEventListener(type, listener) { this.listeners[type] = (this.listeners[type] || []).filter((l) => l !== listener); }
    dispatchEvent(event) { return dispatch(this, event); }
    execCommand(name) { harness.execCommands.push(String(name)); return false; }
}

/** The template, parsed the simple way: it is our own, well-formed HTML. */
function parseDocument(html) {
    const document = new FakeDocument();
    const top = new Element('#top');
    const stack = [top];
    const token = /<!--[\s\S]*?-->|<!DOCTYPE[^>]*>|<\/([a-zA-Z][a-zA-Z0-9-]*)\s*>|<([a-zA-Z][a-zA-Z0-9-]*)((?:\s+[^\s"'>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*)\s*(\/?)>|([^<]+)/gi;
    let m;
    while ((m = token.exec(html)) !== null) {
        if (m[1]) {
            const tag = m[1].toLowerCase();
            for (let i = stack.length - 1; i > 0; i -= 1) {
                if (stack[i].localName === tag) {
                    stack.length = i;
                    break;
                }
            }
        } else if (m[2]) {
            const element = new Element(m[2]);
            const attrs = /([^\s"'>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
            let a;
            while ((a = attrs.exec(m[3])) !== null) element.attrs.set(a[1].toLowerCase(), decode(a[2] ?? a[3] ?? a[4] ?? ''));
            stack[stack.length - 1].appendChild(element);
            if (!VOID.has(element.localName) && !m[4]) stack.push(element);
        } else if (m[5]) {
            stack[stack.length - 1].appendChild(new TextNode(decode(m[5])));
        }
    }
    const html_ = top.children.find((node) => node.localName === 'html');
    if (!html_) throw new Error('the template has no <html> element');
    html_.parentNode = document;
    document.documentElement = html_;
    document.body = html_.children.find((node) => node.localName === 'body');
    document.activeElement = document.body;
    return document;
}

doc = parseDocument(fs.readFileSync(htmlPath, 'utf8'));

// --- the browser around it ------------------------------------------------------------------

const storageData = new Map();
const harness = {
    requests: [],
    routes: new Map(),
    unmocked: [],
    confirms: [],
    confirmAnswer: true,
    consoleErrors: [],
    execCommands: [],
    clipboard: null,
    selected: null,
    storage: storageData,
    storageThrows: false,
    FakeEvent,
    /** Answer `method path` with these, in turn; the last one keeps answering. */
    on(method, path, ...answers) {
        this.routes.set(`${method.toUpperCase()} ${path}`, answers);
    },
    calls(method, path) {
        return this.requests.filter((r) => r.method === method && r.path === path);
    },
    flush: async () => {
        for (let i = 0; i < 3; i += 1) await new Promise((resolve) => setImmediate(resolve));
    },
    el: (id) => doc.getElementById(id),
    find(target) {
        if (typeof target !== 'string') return target;
        const element = /^[\w-]+$/.test(target) ? doc.getElementById(target) : doc.querySelector(target);
        if (!element) throw new Error(`nothing matches ${target}`);
        return element;
    },
    visible(target) {
        const element = typeof target === 'string' && /^[\w-]+$/.test(target) ? doc.getElementById(target) : doc.querySelector(target);
        return Boolean(element) && !hiddenByAncestor(element);
    },
    text(target) {
        const element = typeof target === 'string' && /^[\w-]+$/.test(target) ? doc.getElementById(target) : doc.querySelector(target);
        return element ? element.textContent : null;
    },
    /** A user's click: refused on what nobody can see (a hidden element, a closed dialog). */
    async click(target) {
        const element = this.find(target);
        if (hiddenByAncestor(element)) throw new Error(`clicked something that is not on screen: ${target}`);
        element.click();
        await this.flush();
    },
    async type(target, text) {
        const element = this.find(target);
        if (hiddenByAncestor(element)) throw new Error(`typed into something that is not on screen: ${target}`);
        if (element.disabled) throw new Error(`typed into a disabled field: ${target}`);
        element.value = text;
        dispatch(element, new FakeEvent('input'));
        dispatch(element, new FakeEvent('change'));
        await this.flush();
    },
    async check(target, on = true) {
        const element = this.find(target);
        if (element.checked !== on) await this.click(element);
    },
    async choose(target, value) {
        const element = this.find(target);
        element.value = value;
        dispatch(element, new FakeEvent('input'));
        dispatch(element, new FakeEvent('change'));
        await this.flush();
    },
    async press(target, key) {
        dispatch(this.find(target), new FakeEvent('keydown', { key }));
        await this.flush();
    },
    async boot() {
        doc.readyState = 'interactive';
        dispatch(doc, new FakeEvent('DOMContentLoaded', { bubbles: false }));
        await this.flush();
    },
};

function answer(spec) {
    const status = spec.status ?? 200;
    const headers = {};
    for (const [name, value] of Object.entries(spec.headers || {})) headers[name.toLowerCase()] = String(value);
    if (status === 401 && !('www-authenticate' in headers)) headers['www-authenticate'] = 'Bearer';
    const text = spec.body === undefined ? '' : JSON.stringify(spec.body);
    return {
        status,
        ok: status >= 200 && status < 300,
        headers: { get: (name) => headers[String(name).toLowerCase()] ?? null },
        json: async () => JSON.parse(text),
    };
}

async function fakeFetch(url, init = {}) {
    const method = String(init.method || 'GET').toUpperCase();
    const path = String(url).split('?')[0];
    const headers = {};
    for (const [name, value] of Object.entries(init.headers || {})) headers[name.toLowerCase()] = String(value);
    let body;
    if (init.body !== undefined) {
        try {
            body = JSON.parse(init.body);
        } catch {
            body = init.body;
        }
    }
    const request = { method, path, url: String(url), headers, body, cache: init.cache };
    harness.requests.push(request);
    const answers = harness.routes.get(`${method} ${path}`);
    if (!answers || !answers.length) {
        harness.unmocked.push(`${method} ${path}`);
        return answer({ status: 404, body: { detail: 'unmocked' } });
    }
    let spec = answers.length > 1 ? answers.shift() : answers[0];
    if (typeof spec === 'function') spec = spec(request);
    if (spec && spec.networkError) throw new TypeError('Failed to fetch');
    return answer(spec);
}

const sessionStorage = {
    getItem: (key) => { if (harness.storageThrows) throw new Error('blocked'); return storageData.has(key) ? storageData.get(key) : null; },
    setItem: (key, value) => { if (harness.storageThrows) throw new Error('blocked'); storageData.set(key, String(value)); },
    removeItem: (key) => { if (harness.storageThrows) throw new Error('blocked'); storageData.delete(key); },
};

let timerSeq = 0;
const timers = new Map();
harness.pendingTimers = () => timers.size;
harness.runTimers = async () => {
    const due = Array.from(timers.values());
    timers.clear();
    for (const timer of due) await timer.fn();
    await harness.flush();
};

const windowListeners = {};
const context = {
    document: doc,
    console: { ...console, error: (...a) => harness.consoleErrors.push(a.map(String).join(' ')), warn: () => {} },
    setTimeout: (fn, ms) => { const id = ++timerSeq; timers.set(id, { fn, ms }); return id; },
    clearTimeout: (id) => timers.delete(id),
    setInterval: (fn, ms) => { const id = ++timerSeq; timers.set(id, { fn, ms }); return id; },
    clearInterval: (id) => timers.delete(id),
    fetch: fakeFetch,
    sessionStorage,
    confirm: (message) => { harness.confirms.push(String(message)); return harness.confirmAnswer; },
    navigator: { clipboard: { writeText: async (text) => { harness.clipboard = String(text); } } },
    isSecureContext: false,
    crypto: {
        getRandomValues: (array) => {
            const bytes = nodeCrypto.randomBytes(array.length);
            for (let i = 0; i < array.length; i += 1) array[i] = bytes[i];
            return array;
        },
    },
    btoa,
    atob,
    location: { protocol: 'http:', host: '192.168.1.20:3000', hostname: '192.168.1.20' },
    addEventListener: (type, listener) => { (windowListeners[type] = windowListeners[type] || []).push(listener); },
    removeEventListener: () => {},
    harness,
    fixtures,
};
context.window = context;
harness.windowListeners = windowListeners;
vm.createContext(context);

// Every route the page asks at start answers like the gateway did, unless a scenario says otherwise.
if (fixtures.view) harness.on('GET', '/api/settings', { status: 200, body: fixtures.view });
if (fixtures.engines) harness.on('GET', '/api/settings/engines', { status: 200, body: fixtures.engines });

vm.runInContext(fs.readFileSync(scriptPath, 'utf8'), context, { filename: 'settings.js' });

(async () => {
    const source = fs.readFileSync(scenarioPath, 'utf8');
    const result = await vm.runInContext(`(async () => {\n${source}\n})()`, context, { filename: 'scenario.js' });
    process.stdout.write(JSON.stringify({ ok: true, result: result === undefined ? null : result }));
})().catch((error) => {
    process.stdout.write(JSON.stringify({ ok: false, error: String((error && error.stack) || error) }));
    process.exitCode = 1;
});
"""


def _new_page_key() -> str:
    return "tts_" + base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def _answer(response) -> dict:
    """A gateway response, as the page's fetch replays it."""
    headers = {name: value for name, value in response.headers.items()
               if name.lower() in ("retry-after", "www-authenticate")}
    return {"status": response.status_code, "body": response.json(), "headers": headers}


class _LogLines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture(scope="module")
def gateway_answers(tmp_path_factory):
    """What real gateways answered the requests the page makes, by name.

    One gateway's app YAML sets API_KEY, TRUSTED_HOSTS=truenas.k2o and offers Magpie as the
    default TTS engine (installed: its name resolves; no other optional engine's does). The
    other has no key and is claimed with the one-time code from its log.
    """
    async def magpie_installed(host):
        return "magpie" in host

    keyed = load_frontend_app({
        "TTS_STT_SETTINGS_DIR": str(tmp_path_factory.mktemp("keyed")), "API_KEY": YAML_KEY,
        "TRUSTED_HOSTS": "truenas.k2o", "ENABLE_MAGPIE_TTS": "true", "DEFAULT_TTS_PROVIDER": "magpie"})
    keyless = load_frontend_app({"TTS_STT_SETTINGS_DIR": str(tmp_path_factory.mktemp("keyless"))})
    keyed._engine_resolver = keyless._engine_resolver = magpie_installed
    page = {"Host": NAS, "Origin": f"http://{NAS}"}

    def call(client, method, path, body=None, headers=None):
        return _answer(client.request(method, path, json=body, headers=headers or page))

    answers: dict = {}
    log = _LogLines()
    claim_log = logging.getLogger("tts_stt.settings.claim")
    claim_log.addHandler(log)
    try:
        with pytest.MonkeyPatch.context() as patch:
            install_stub(patch, keyed, lambda method, url, kwargs: {"status": "ok"})
            client = TestClient(keyed.app)
            admin = {**page, "Authorization": f"Bearer {YAML_KEY}"}
            answers["view"] = call(client, "GET", "/api/settings", headers=admin)
            answers["engines"] = call(client, "GET", "/api/settings/engines", headers=admin)
            dry_run = lambda changes, **extra: call(client, "PUT", "/api/settings", {  # noqa: E731
                "base_revision": 0, "set": changes, "dry_run": True}, headers={**admin, **extra})
            answers["dry_run_hosts"] = dry_run({"TRUSTED_HOSTS": ["truenas.k2o", "speach.k2o"]})
            answers["dry_run_invalid_host"] = dry_run({"TRUSTED_HOSTS": ["truenas.k2o", "*.de"]})
            answers["dry_run_unoffer"] = dry_run({"ENABLE_MAGPIE_TTS": False, "DEFAULT_TTS_PROVIDER": "piper"})
            answers["would_lock_out"] = dry_run({"TRUSTED_HOSTS": []}, Host="truenas.k2o:3000",
                                                Origin="http://truenas.k2o:3000")
            answers["dry_run_proxy"] = dry_run({"TRUST_PROXY_HEADERS": True})
            answers["save_proxy"] = call(client, "PUT", "/api/settings", {
                "base_revision": 0, "set": {"TRUST_PROXY_HEADERS": True},
                "acknowledge": ["TRUST_PROXY_HEADERS:proxy_headers_on"]}, headers=admin)
            answers["confirm"] = call(client, "POST", "/api/settings/confirm",
                                      {"revision": answers["save_proxy"]["body"]["confirm"]["revision"]}, headers=admin)
            answers["view_saved"] = call(client, "GET", "/api/settings", headers=admin)
            answers["conflict"] = dry_run({"MAX_TTS_CHARS": 100})
            answers["access_refused"] = call(client, "PUT", "/api/settings/access", {
                "base_revision": answers["view"]["body"]["keys_revision"], "deployment_key_role": "client"}, headers=admin)
            answers["key_created"] = call(client, "POST", "/api/settings/keys", {
                "name": "Home Assistant", "role": "client", "key": _new_page_key()}, headers=admin)

            client = TestClient(keyless.app)
            answers["view_unclaimed"] = call(client, "GET", "/api/settings")
            answers["claim_code"] = call(client, "POST", "/api/settings/claim-code", {})
            laptop = _new_page_key()
            answers["claim"] = call(client, "POST", "/api/settings/claim", {
                "code": CLAIM_CODE.findall(" ".join(log.lines))[-1], "name": "Laptop", "key": laptop})
            owner = {**page, "Authorization": f"Bearer {laptop}"}
            answers["view_claimed"] = call(client, "GET", "/api/settings", headers=owner)
            answers["access_require"] = call(client, "PUT", "/api/settings/access", {
                "base_revision": answers["view_claimed"]["body"]["keys_revision"], "require_key": True}, headers=owner)
            answers["second_admin"] = call(client, "POST", "/api/settings/keys", {
                "name": "Phone", "role": "admin", "key": _new_page_key()}, headers=owner)
            answers["view_two_admins"] = call(client, "GET", "/api/settings", headers=owner)
            answers["revoke_in_use"] = call(client, "DELETE", f"/api/settings/keys/{answers['claim']['body']['id']}",
                                            headers=owner)
    finally:
        claim_log.removeHandler(log)
    assert {name: answer["status"] for name, answer in answers.items()} == {
        "view": 200, "engines": 200, "dry_run_hosts": 200, "dry_run_invalid_host": 400, "dry_run_unoffer": 200,
        "would_lock_out": 409, "dry_run_proxy": 200, "save_proxy": 200, "confirm": 200, "view_saved": 200,
        "conflict": 409, "access_refused": 403, "key_created": 201, "view_unclaimed": 200, "claim_code": 202,
        "claim": 201, "view_claimed": 200, "access_require": 200, "second_admin": 201, "view_two_admins": 200,
        "revoke_in_use": 200,
    }
    return answers


def run_settings_js(tmp_path: Path, scenario: str, answers: dict, start: str | None = "view"):
    """Open the Settings page (GET /api/settings answers `answers[start]`), then run `scenario`.

    `harness` drives the page like a user (click, type, check), and `fixtures.answers` holds
    the gateway's answers for `harness.on(method, path, ...answers)`.
    """
    driver = tmp_path / "settings_driver.js"
    driver.write_text(SETTINGS_DRIVER, encoding="utf-8")
    body = tmp_path / "scenario.js"
    body.write_text(textwrap.dedent(scenario), encoding="utf-8")
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps({
        "view": answers[start]["body"] if start else None,
        "engines": answers["engines"]["body"],
        "answers": answers,
    }), encoding="utf-8")
    completed = subprocess.run(
        [NODE, str(driver), str(SETTINGS_JS), str(SETTINGS_HTML), str(body), str(fixtures)],
        capture_output=True, text=True, encoding="utf-8", timeout=60,    # node writes UTF-8 on every platform
    )
    try:
        outcome = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"node produced no result.\nstdout: {completed.stdout}\nstderr: {completed.stderr}")
    if not outcome["ok"]:
        pytest.fail(f"scenario threw:\n{outcome['error']}\nstderr: {completed.stderr}")
    return outcome["result"]


# The key the page's tab holds when a scenario opens it as the admin.
OPEN_AS_ADMIN = f"harness.storage.set('tts-stt.api-key', {json.dumps(YAML_KEY)});\nawait harness.boot();\n"


def test_59_host_names_are_read_one_per_line_and_only_changes_are_sent(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        const parsed = [
            parseList('speach.k2o\\n\\n  TTS.Example.com \\r\\n*.K2O,truenas.k2o\\tspeach.k2o'),
            parseList(' \\n , \\t'),
            parseList('https://TTS.example.com, https://tts.example.com'),
        ];
        await harness.type('input-TRUSTED_HOSTS', 'truenas.k2o\\nSpeach.k2o\\n\\nspeach.k2o, ');
        const bar = [harness.visible('save-bar'), harness.text('save-bar-count')];
        harness.on('PUT', '/api/settings', fixtures.answers.dry_run_hosts);
        await harness.click('review-button');
        const review = harness.text('review-rows');
        await harness.click('review-back');
        await harness.type('input-TRUSTED_HOSTS', '  TRUENAS.k2o \\n');
        return {
            parsed, bar, review,
            sent: harness.calls('PUT', '/api/settings').map((r) => r.body),
            draftAfter: S.draft.size, barAfter: harness.visible('save-bar'), unmocked: harness.unmocked,
        };
    """, gateway_answers)
    assert out["parsed"] == [["speach.k2o", "tts.example.com", "*.k2o", "truenas.k2o"], [], ["https://tts.example.com"]]
    assert out["bar"] == [True, "1 unsaved change"]
    assert out["sent"] == [{"base_revision": 0, "set": {"TRUSTED_HOSTS": ["truenas.k2o", "speach.k2o"]},
                            "reset": [], "dry_run": True}]
    assert "added speach.k2o" in out["review"] and "truenas.k2o" not in out["review"], "only what changes is listed"
    assert (out["draftAfter"], out["barAfter"]) == (0, False), "typing the value in force back leaves nothing to save"
    assert out["unmocked"] == []


def test_59_a_removed_host_name_is_highlighted_in_the_review(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.type('input-TRUSTED_HOSTS', 'speach.k2o');
        harness.on('PUT', '/api/settings', fixtures.answers.dry_run_hosts);
        await harness.click('review-button');
        const removed = harness.el('review-rows').querySelectorAll('li.review-removed').map((li) => li.textContent);
        return { removed, warning: harness.text('review-rows') };
    """, gateway_answers)
    assert out["removed"] == ["removed truenas.k2o"]
    assert "Anyone who opens this server as truenas.k2o is refused after this change." in out["warning"]


def test_60_un_offering_the_default_engine_moves_the_default_to_a_built_in_one(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        const select = () => harness.el('input-DEFAULT_TTS_PROVIDER');
        const before = { value: select().value, options: select().options.map((o) => o.value) };
        await harness.click('input-ENABLE_MAGPIE_TTS');            // un-tick "Offer Magpie-TTS"
        const after = {
            value: select().value, options: select().options.map((o) => o.value),
            hint: harness.text('hint-DEFAULT_TTS_PROVIDER'), stt: harness.el('input-DEFAULT_STT_PROVIDER').value,
        };
        harness.on('PUT', '/api/settings', fixtures.answers.dry_run_unoffer);
        await harness.click('review-button');
        const review = { sent: harness.calls('PUT', '/api/settings')[0].body.set,
                         open: harness.el('review-dialog').open, rows: harness.text('review-rows') };
        await harness.click('review-back');
        await harness.click('input-ENABLE_MAGPIE_TTS');            // offered again: the default goes back
        return {
            before, after, review, again: { value: select().value, hint: harness.visible('hint-DEFAULT_TTS_PROVIDER'),
                                            draft: S.draft.size, bar: harness.visible('save-bar') },
        };
    """, gateway_answers)
    assert out["before"] == {"value": "magpie", "options": ["piper", "qwen3", "magpie"]}
    assert out["after"]["value"] == "piper" and "magpie" not in out["after"]["options"]
    assert out["after"]["hint"] == "Magpie-TTS is no longer offered, so the default moves to Piper."
    assert out["after"]["stt"] == "whisper", "the speech-to-text default is not touched"
    assert out["review"]["sent"] == {"ENABLE_MAGPIE_TTS": False, "DEFAULT_TTS_PROVIDER": "piper"}
    assert out["review"]["open"] is True and "Magpie-TTS" in out["review"]["rows"] and "Piper" in out["review"]["rows"]
    assert out["again"] == {"value": "magpie", "hint": False, "draft": 0, "bar": False},         "ticking Offer twice changes nothing"


def test_59_an_edit_the_server_reads_as_the_value_in_force_is_not_saved(tmp_path, gateway_answers):
    """http://Speach.k2o:3000/ is speach.k2o to the server: nothing to review, nothing kept as unsaved."""
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.type('input-TRUSTED_HOSTS', 'http://TRUENAS.k2o:3000/');
        const before = S.draft.size;
        harness.on('PUT', '/api/settings', { status: 200, body: { revision: 0, changed: [], dry_run: true, written: false,
            reload_main_ui: false, confirm: null, warnings: [], notes: {}, dropped: {} } });
        await harness.click('review-button');
        return { before, after: S.draft.size, open: harness.el('review-dialog').open, bar: harness.visible('save-bar'),
                 status: harness.text('settings-status'), field: harness.el('input-TRUSTED_HOSTS').value };
    """, gateway_answers)
    assert out["before"] == 1
    assert (out["after"], out["open"], out["bar"]) == (0, False, False)
    assert "Nothing to save" in out["status"] and out["field"] == "truenas.k2o"


def test_after_an_engine_change_the_page_offers_the_way_back_to_the_app(tmp_path, gateway_answers):
    """The main page renders its engines on load: after a change it has to be loaded again."""
    saved = {**gateway_answers["dry_run_unoffer"]["body"], "dry_run": False, "written": True}
    assert saved["reload_main_ui"] is True
    answers = {**gateway_answers, "save_unoffer": {"status": 200, "body": saved}}
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.click('input-ENABLE_MAGPIE_TTS');
        harness.on('PUT', '/api/settings', fixtures.answers.dry_run_unoffer, fixtures.answers.save_unoffer);
        await harness.click('review-button');
        await harness.click('review-save');
        const link = harness.el('status-app-link');
        return { href: link && link.getAttribute('href'), text: link && link.textContent,
                 status: harness.text('settings-status'), draft: S.draft.size };
    """, answers)
    assert (out["href"], out["text"], out["draft"]) == ("/", "Back to the app", 0)
    assert "engine changes" in out["status"]


def test_61_a_new_key_is_saved_only_after_i_have_stored_it(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.click('create-key-button');
        const shown = harness.el('new-key-value').value;
        const save = harness.el('key-dialog-save');
        const states = [save.disabled];
        await harness.type('new-key-name', 'Home Assistant');
        states.push(save.disabled);
        await harness.check('new-key-stored');
        states.push(save.disabled);
        await harness.check('new-key-stored', false);
        states.push(save.disabled);
        save.click();                                        // a disabled button does nothing
        await harness.flush();
        const early = harness.calls('POST', '/api/settings/keys').length;
        await harness.check('new-key-stored');
        harness.on('POST', '/api/settings/keys', fixtures.answers.key_created);
        await harness.click('key-dialog-save');
        return {
            shown, states, early, posted: harness.calls('POST', '/api/settings/keys').map((r) => r.body),
            done: harness.visible('key-dialog-done'), form: harness.visible('key-dialog-form'),
            useHere: harness.visible('use-new-key'), text: harness.text('key-dialog-done-text'),
            tabKey: harness.storage.get('tts-stt.api-key'),
            elsewhere: harness.requests.filter((r) => JSON.stringify(r).includes(shown)).map((r) => r.method + ' ' + r.path),
        };
    """, gateway_answers)
    assert PAGE_KEY.fullmatch(out["shown"]), out["shown"]
    assert out["states"] == [True, True, False, True], "Save opens with the name and 'I have stored it' only"
    assert out["early"] == 0
    assert out["posted"] == [{"name": "Home Assistant", "role": "client", "key": out["shown"]}]
    assert (out["done"], out["form"], out["useHere"]) == (True, False, False), "a client key is not offered for this tab"
    assert out["shown"] not in out["text"]
    assert out["tabKey"] == YAML_KEY
    assert out["elsewhere"] == ["POST /api/settings/keys"], "the key went to the server once, in the create request"


def test_61_the_claim_creates_the_admin_key_only_after_i_have_stored_it(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, """
        await harness.boot();
        const card = harness.visible('claim-card');
        const requireOffered = harness.visible('claim-require-row');
        const shown = harness.el('claim-key').value;
        const button = harness.el('claim-button');
        const states = [button.disabled];
        harness.on('POST', '/api/settings/claim-code', fixtures.answers.claim_code);
        await harness.click('[data-action="print-code"]');
        const printed = harness.text('claim-code-status');
        await harness.type('claim-code', 'ABCDE-FGHJK-MNPQR-STVWX');
        states.push(button.disabled);
        await harness.check('claim-stored');
        states.push(button.disabled);
        harness.on('POST', '/api/settings/claim', fixtures.answers.claim);
        harness.on('GET', '/api/settings', { status: 200, body: fixtures.answers.view_claimed.body });
        await harness.click('claim-button');
        return {
            card, shown, states, printed, requireOffered,
            posts: harness.requests.filter((r) => r.method === 'POST').map((r) => [r.path, r.body]),
            tabKey: harness.storage.get('tts-stt.api-key'),
            lastGet: harness.calls('GET', '/api/settings').slice(-1)[0].headers.authorization,
            cardAfter: harness.visible('claim-card'), status: harness.text('settings-status'),
            fieldsWritable: !harness.el('input-TRUSTED_HOSTS').disabled,
        };
    """, gateway_answers, start="view_unclaimed")
    assert out["card"] is True and out["requireOffered"] is True
    assert PAGE_KEY.fullmatch(out["shown"])
    assert out["states"] == [True, True, False]
    assert "log" in out["printed"]
    assert out["posts"] == [["/api/settings/claim-code", {}],
                            ["/api/settings/claim", {"code": "ABCDE-FGHJK-MNPQR-STVWX", "name": "Admin", "key": out["shown"]}]]
    assert out["tabKey"] == out["shown"], "the new admin key is used in this tab (and on the main page)"
    assert out["lastGet"] == f"Bearer {out['shown']}"
    assert out["cardAfter"] is False and "claimed" in out["status"] and out["fieldsWritable"] is True


def test_62_refused_values_are_shown_next_to_their_field_as_text(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        const hostile = '<img src=x onerror="window.pwned=1"> is not a host name.';
        await harness.type('input-TRUSTED_HOSTS', 'truenas.k2o\\n<img src=x onerror="window.pwned=1">');
        await harness.type('input-MAX_TTS_CHARS', '0');
        harness.on('PUT', '/api/settings', { status: 400, body: {
            detail: 'Some values are not valid.', code: 'invalid_input', errors: {
                TRUSTED_HOSTS: hostile, MAX_TTS_CHARS: 'Expected a whole number from 1 to 100000.',
                base_revision: 'Expected the revision the change is based on.' } } });
        await harness.click('review-button');
        const box = harness.el('error-TRUSTED_HOSTS');
        return {
            text: box.textContent, nodes: box.childNodes.map((n) => n.nodeType), shown: harness.visible('error-TRUSTED_HOSTS'),
            chars: harness.text('error-MAX_TTS_CHARS'),
            invalid: harness.el('input-TRUSTED_HOSTS').getAttribute('aria-invalid'),
            marked: harness.el('setting-TRUSTED_HOSTS').classList.contains('has-error'),
            status: harness.text('settings-status'),
            alert: Boolean(harness.el('settings-status').querySelector('[role="alert"]')),
            open: harness.el('review-dialog').open, focus: document.activeElement.id,
            pwned: window.pwned === 1, draft: S.draft.size, hostile,
        };
    """, gateway_answers)
    assert out["text"] == out["hostile"] and out["nodes"] == [3], "one text node, no markup"
    assert out["shown"] is True and out["pwned"] is False
    assert out["chars"] == "Expected a whole number from 1 to 100000."
    assert (out["invalid"], out["marked"]) == ("true", True)
    assert "base_revision: Expected the revision the change is based on." in out["status"] and out["alert"] is True
    assert out["open"] is False and out["focus"] == "input-TRUSTED_HOSTS"
    assert out["draft"] == 2, "the edits are kept to be corrected"


def test_62_the_gateways_own_refusal_of_a_public_suffix_wildcard_lands_on_the_field(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.type('input-TRUSTED_HOSTS', 'truenas.k2o\\n*.de');
        harness.on('PUT', '/api/settings', fixtures.answers.dry_run_invalid_host);
        await harness.click('review-button');
        return harness.text('error-TRUSTED_HOSTS');
    """, gateway_answers)
    assert out == gateway_answers["dry_run_invalid_host"]["body"]["errors"]["TRUSTED_HOSTS"]
    assert "'*.de'" in out


def test_63_a_proxy_change_is_confirmed_right_after_it_is_saved(tmp_path, gateway_answers):
    revision = gateway_answers["save_proxy"]["body"]["confirm"]["revision"]
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.click('input-TRUST_PROXY_HEADERS');
        harness.on('PUT', '/api/settings', fixtures.answers.dry_run_proxy, fixtures.answers.save_proxy);
        harness.on('POST', '/api/settings/confirm', fixtures.answers.confirm);
        await harness.click('review-button');
        const notice = harness.text('review-confirm');
        const locked = harness.el('review-save').disabled;
        await harness.check('#review-warnings input');
        const unlocked = !harness.el('review-save').disabled;
        harness.on('GET', '/api/settings', { status: 200, body: fixtures.answers.view_saved.body });
        const before = harness.requests.length;
        await harness.click('review-save');
        return {
            notice, locked, unlocked,
            order: harness.requests.slice(before).map((r) => r.method + ' ' + r.path),
            saved: harness.calls('PUT', '/api/settings')[1].body,
            confirmed: harness.calls('POST', '/api/settings/confirm').map((r) => r.body),
            status: harness.text('settings-status'), open: harness.el('review-dialog').open,
        };
    """, gateway_answers)
    assert "confirmed within 60 seconds" in out["notice"]
    assert out["locked"] is True and out["unlocked"] is True, "the warning has to be ticked first"
    assert out["order"] == ["PUT /api/settings", "POST /api/settings/confirm", "GET /api/settings"]
    assert out["saved"]["acknowledge"] == ["TRUST_PROXY_HEADERS:proxy_headers_on"] and "dry_run" not in out["saved"]
    assert out["confirmed"] == [{"revision": revision}]
    assert "is confirmed and stays" in out["status"] and out["open"] is False


def test_63_a_confirmation_the_new_rules_refuse_says_when_the_change_is_undone(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.click('input-TRUST_PROXY_HEADERS');
        harness.on('PUT', '/api/settings', fixtures.answers.dry_run_proxy, fixtures.answers.save_proxy);
        harness.on('POST', '/api/settings/confirm', { status: 403, body: {
            detail: "Host 'proxy.example' is not allowed.", code: 'host_not_allowed' } });
        await harness.click('review-button');
        await harness.check('#review-warnings input');
        harness.on('GET', '/api/settings', { status: 200, body: fixtures.answers.view_saved.body });
        await harness.click('review-save');
        return { status: harness.text('settings-status'),
                 alert: Boolean(harness.el('settings-status').querySelector('[role="alert"]')) };
    """, gateway_answers)
    assert "could not be confirmed from this page" in out["status"] and "It is undone at" in out["status"]
    assert out["alert"] is True


# --- more of the page ----------------------------------------------------------------------------


def test_the_key_is_shared_with_the_main_page_and_a_refused_one_is_forgotten(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, """
        harness.storage.set('tts-stt.api-key', 'an-old-key');
        harness.on('GET', '/api/settings', { status: 401, body: { detail: 'An admin key is required.', code: 'invalid_api_key' } },
                   { status: 200, body: fixtures.answers.view.body });
        await harness.boot();
        const refused = { panel: harness.visible('key-panel'), main: harness.visible('settings-main'),
                          text: harness.text('key-panel-text'), stored: harness.storage.get('tts-stt.api-key') ?? null,
                          focus: document.activeElement.id };
        await harness.type('key-input', '  the-key-from-the-app-yaml  ');
        await harness.press('key-input', 'Enter');
        return {
            refused, stored: harness.storage.get('tts-stt.api-key'),
            sent: harness.calls('GET', '/api/settings').map((r) => r.headers.authorization || null),
            panel: harness.visible('key-panel'), main: harness.visible('settings-main'), field: harness.el('key-input').value,
        };
    """, gateway_answers, start=None)
    assert out["refused"] == {"panel": True, "main": False, "text": "This key was not accepted. Enter an admin key of this server.",
                              "stored": None, "focus": "key-input"}
    assert out["stored"] == YAML_KEY, "kept in the slot app.js reads, sessionStorage 'tts-stt.api-key'"
    assert out["sent"] == ["Bearer an-old-key", f"Bearer {YAML_KEY}"]
    assert (out["panel"], out["main"], out["field"]) == (False, True, "")


def test_a_mistyped_key_does_not_replace_the_one_that_works(tmp_path, gateway_answers):
    """Trying another key keeps the tab's key until the server takes the new one: a typo used to
    end in a 401 that cleared the slot, here and on the main page."""
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.click('[data-action="change-key"]');
        await harness.type('key-input', 'tts_typo');
        harness.on('GET', '/api/settings', { status: 401, body: { detail: 'An admin key is required.', code: 'invalid_api_key' } });
        await harness.press('key-input', 'Enter');
        return { stored: harness.storage.get('tts-stt.api-key'), error: harness.text('key-panel-error'),
                 errorShown: harness.visible('key-panel-error'), main: harness.visible('settings-main'),
                 tried: harness.calls('GET', '/api/settings').slice(-1)[0].headers.authorization };
    """, gateway_answers)
    assert out["tried"] == "Bearer tts_typo"
    assert out["stored"] == YAML_KEY, "the working key stays"
    assert (out["error"], out["errorShown"], out["main"]) == ("This key was not accepted.", True, True)


def test_a_client_key_is_asked_for_an_admin_key_and_kept_for_the_api(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, """
        harness.storage.set('tts-stt.api-key', 'home-assistant');
        harness.on('GET', '/api/settings', { status: 403, body: {
            detail: 'This key may use the API but not the Settings; an admin key is required.', code: 'admin_key_required' } });
        await harness.boot();
        return { panel: harness.visible('key-panel'), text: harness.text('key-panel-text'),
                 stored: harness.storage.get('tts-stt.api-key') };
    """, gateway_answers, start=None)
    assert out["panel"] is True and "may use the API but not change settings" in out["text"]
    assert out["stored"] == "home-assistant"


def test_reset_to_the_app_yaml_drops_the_saved_value(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        const badge = harness.text('#setting-TRUST_PROXY_HEADERS .badge-saved');
        const yaml = harness.text('#setting-TRUST_PROXY_HEADERS .setting-note');
        await harness.click('reset-TRUST_PROXY_HEADERS');
        const shown = { checked: harness.el('input-TRUST_PROXY_HEADERS').checked, undo: harness.visible('undo-TRUST_PROXY_HEADERS'),
                        focus: document.activeElement.id, changed: harness.text('changed-TRUST_PROXY_HEADERS') };
        harness.on('PUT', '/api/settings', { status: 200, body: { revision: 3, changed: ['TRUST_PROXY_HEADERS'], dry_run: true,
            written: false, reload_main_ui: false, confirm: null, warnings: [], notes: {}, dropped: {} } });
        await harness.click('review-button');
        return { badge, yaml, shown, sent: harness.calls('PUT', '/api/settings')[0].body, rows: harness.text('review-rows') };
    """, gateway_answers, start="view_saved")
    assert out["badge"] == "Set here"
    assert out["yaml"] == "App YAML: Off. Not used while the value set here applies."
    assert out["shown"] == {"checked": False, "undo": True, "focus": "undo-TRUST_PROXY_HEADERS",
                            "changed": "Back to the app YAML on save"}
    assert out["sent"] == {"base_revision": gateway_answers["view_saved"]["body"]["revision"], "set": {},
                           "reset": ["TRUST_PROXY_HEADERS"], "dry_run": True}
    assert "Back to the app YAML value." in out["rows"]


def test_api_access_is_changed_through_its_own_endpoint_and_revision(tmp_path, gateway_answers):
    view = gateway_answers["view_claimed"]["body"]
    out = run_settings_js(tmp_path, """
        harness.storage.set('tts-stt.api-key', 'tts_laptop');
        await harness.boot();
        await harness.click('input-require_key-true');
        await harness.click('review-button');
        const rows = harness.text('review-rows');
        harness.on('PUT', '/api/settings/access', fixtures.answers.access_require);
        await harness.click('review-save');
        return { rows, puts: harness.requests.filter((r) => r.method === 'PUT').map((r) => [r.path, r.body]),
                 status: harness.text('settings-status') };
    """, gateway_answers, start="view_claimed")
    assert "Open on the network" in out["rows"] and "Require a key" in out["rows"]
    assert "Every client then has to send a key" in out["rows"]
    assert out["puts"] == [["/api/settings/access", {"base_revision": view["keys_revision"], "require_key": True}]], \
        "no dry run and no gateway.json write for keys.json settings"
    assert "API access is changed" in out["status"]


def test_the_yaml_key_cannot_offer_to_make_itself_a_client_and_the_refusal_is_shown(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        const client = harness.el('input-deployment_key_role').options.find((o) => o.value === 'client');
        const disabled = client.disabled;
        const note = harness.text('setting-deployment_key_role');
        // what the server says when it is tried anyway
        client.disabled = false;
        await harness.choose('input-deployment_key_role', 'client');
        harness.on('PUT', '/api/settings/access', fixtures.answers.access_refused);
        await harness.click('review-button');
        await harness.click('review-save');
        return { disabled, note, open: harness.el('review-dialog').open, error: harness.text('review-error') };
    """, gateway_answers)
    assert out["disabled"] is True and "cannot make itself a client" in out["note"]
    assert out["open"] is True, "the refusal is shown where the change was asked for"
    assert out["error"] == gateway_answers["access_refused"]["body"]["detail"]


def test_a_change_that_would_lock_me_out_cannot_be_saved(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.type('input-TRUSTED_HOSTS', '');
        harness.on('PUT', '/api/settings', fixtures.answers.would_lock_out);
        await harness.click('review-button');
        return { open: harness.el('review-dialog').open, error: harness.text('review-error'),
                 save: harness.el('review-save').disabled, puts: harness.calls('PUT', '/api/settings').length };
    """, gateway_answers)
    assert out["open"] is True and out["save"] is True and out["puts"] == 1
    assert out["error"] == gateway_answers["would_lock_out"]["body"]["detail"]
    assert "speach" not in out["error"] and "truenas.k2o" in out["error"]


def test_a_lockout_found_only_by_the_save_keeps_the_review_open_and_blocked(tmp_path, gateway_answers):
    """The dry run passed, then something changed before Save (another worker, a proxy): the save's own
    would_lock_out is shown in the review, and Save stays off."""
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.type('input-TRUSTED_HOSTS', 'speach.k2o');
        harness.on('PUT', '/api/settings', fixtures.answers.dry_run_hosts, fixtures.answers.would_lock_out);
        await harness.click('review-button');
        await harness.click('review-save');
        return { open: harness.el('review-dialog').open, error: harness.text('review-error'),
                 save: harness.el('review-save').disabled, puts: harness.calls('PUT', '/api/settings').length };
    """, gateway_answers)
    assert out["open"] is True and out["save"] is True and out["puts"] == 2
    assert out["error"] == gateway_answers["would_lock_out"]["body"]["detail"]


def test_a_conflicting_save_reloads_the_settings_and_keeps_the_edits(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        await harness.type('input-MAX_TTS_CHARS', '100');
        harness.on('PUT', '/api/settings', fixtures.answers.conflict);
        harness.on('GET', '/api/settings', { status: 200, body: fixtures.answers.view_saved.body });
        await harness.click('review-button');
        return { gets: harness.calls('GET', '/api/settings').length, status: harness.text('settings-status'),
                 draft: Array.from(S.draft.keys()), field: harness.el('input-MAX_TTS_CHARS').value,
                 open: harness.el('review-dialog').open };
    """, gateway_answers)
    assert out["gets"] == 2 and "changed in the meantime" in out["status"]
    assert out["draft"] == ["MAX_TTS_CHARS"] and out["field"] == "100" and out["open"] is False


def test_revoking_the_key_this_tab_uses_asks_first_and_then_forgets_it(tmp_path, gateway_answers):
    view = gateway_answers["view_two_admins"]["body"]
    laptop = next(record for record in view["keys"] if record["name"] == "Laptop")
    out = run_settings_js(tmp_path, f"""
        harness.storage.set('tts-stt.api-key', 'tts_laptop');
        await harness.boot();
        harness.confirmAnswer = false;
        await harness.click('[data-key-id="{laptop['id']}"]');
        const declined = harness.calls('DELETE', '/api/settings/keys/{laptop['id']}').length;
        harness.confirmAnswer = true;
        harness.on('DELETE', '/api/settings/keys/{laptop['id']}', fixtures.answers.revoke_in_use);
        harness.on('GET', '/api/settings', {{ status: 401, body: {{ detail: 'An admin key is required.', code: 'invalid_api_key' }} }});
        await harness.click('[data-key-id="{laptop['id']}"]');
        return {{ asked: harness.confirms, declined, stored: harness.storage.get('tts-stt.api-key') ?? null,
                  panel: harness.visible('key-panel'), status: harness.text('settings-status'),
                  reloadedWith: harness.calls('GET', '/api/settings').slice(-1)[0].headers.authorization ?? null }};
    """, gateway_answers, start="view_two_admins")
    assert out["declined"] == 0
    assert len(out["asked"]) == 2 and "This tab uses it" in out["asked"][0]
    assert out["stored"] is None and out["panel"] is True
    assert out["reloadedWith"] is None, "a revoked key sent again would count as a wrong guess toward the 429"
    assert 'The key "Laptop" is revoked.' in out["status"]


def test_the_page_never_asks_for_more_than_it_needs_on_start(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        return { requests: harness.requests.map((r) => [r.method, r.path, r.cache, r.headers['content-type'] || null]),
                 chips: ['ENABLE_MAGPIE_TTS', 'ENABLE_CANARY_ASR'].map((key) => harness.text('engine-state-' + key)),
                 install: harness.visible('install-ENABLE_CANARY_ASR'), status: harness.text('settings-status'),
                 timers: harness.pendingTimers() };
    """, gateway_answers)
    assert out["requests"] == [["GET", "/api/settings", "no-store", None], ["GET", "/api/settings/engines", "no-store", None]]
    assert out["chips"] == ["Running", "Not installed"] and out["install"] is True
    assert out["status"] == "", "nothing left saying the page is still loading"
    assert out["timers"] == 0, "no polling"


def test_the_limits_say_what_they_cost_as_they_are_typed(tmp_path, gateway_answers):
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        const memory = [harness.text('hint-MAX_UPLOAD_MB')];
        await harness.type('input-MAX_CONCURRENT_UPLOADS', '8');
        memory.push(harness.text('hint-MAX_UPLOAD_MB'));
        const chars = [harness.visible('hint-MAX_TTS_CHARS')];
        await harness.type('input-MAX_TTS_CHARS', '6000');
        chars.push(harness.text('hint-MAX_TTS_CHARS'));
        await harness.type('input-MAX_TTS_CHARS', '5000');
        chars.push(harness.visible('hint-MAX_TTS_CHARS'));
        return { memory, chars, workers: fixtures.answers.view.body.deployment.workers };
    """, gateway_answers)
    times = chr(0xd7)
    assert out["workers"] == 2
    assert out["memory"] == [f"Worst case: 2 workers {times} 4 uploads at once {times} 512 MB = 4.0 GB of memory.",
                             f"Worst case: 2 workers {times} 8 uploads at once {times} 512 MB = 8.0 GB of memory."]
    assert out["chars"][0] is False and out["chars"][2] is False
    assert "refuse more than 5000 characters" in out["chars"][1]


def test_every_server_banner_is_shown_and_the_pending_one_offers_to_keep_the_change(tmp_path, gateway_answers):
    view = json.loads(json.dumps(gateway_answers["view_saved"]["body"]))
    view["pending"] = {"revision": view["revision"], "deadline": 4102444800, "seconds_left": 30, "keys": ["TRUST_PROXY_HEADERS"]}
    view["banners"] = [{"id": "pending", "level": "warning", "text": "The change of TRUST_PROXY_HEADERS is undone soon."},
                       {"id": "not_mounted", "level": "error", "text": "Settings cannot be saved here."}]
    answers = {**gateway_answers, "custom": {"status": 200, "body": view}}
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        const banners = harness.el('settings-banners').children.map((b) => [b.dataset.banner, b.className]);
        const lines = harness.text('settings-banners');
        harness.on('POST', '/api/settings/confirm', fixtures.answers.confirm);
        await harness.click('[data-action="confirm-pending"]');
        return { banners, lines, confirmed: harness.calls('POST', '/api/settings/confirm').map((r) => r.body),
                 live: harness.el('settings-banners').getAttribute('aria-live'), timers: harness.pendingTimers() };
    """, answers, start="custom")
    assert out["banners"] == [["pending", "banner warning"], ["not_mounted", "banner error"]]
    assert '/app/settings"' in out["lines"], "the YAML lines to add are shown"
    assert out["confirmed"] == [{"revision": view["revision"]}]
    assert out["live"] == "polite"
    assert out["timers"] == 1, "one reload after the deadline shows whether the change stayed"


def _damaged_keys_answer(tmp_path) -> dict:
    """What a gateway with API_KEY answers its own YAML key while keys.json is damaged."""
    folder = tmp_path / "damaged-keys"
    folder.mkdir()
    module = load_frontend_app({"TTS_STT_SETTINGS_DIR": str(folder), "API_KEY": YAML_KEY})
    (folder / "keys.json").write_text("{damaged", encoding="utf-8")
    response = TestClient(module.app).get("/api/settings", headers={
        "Host": NAS, "Origin": f"http://{NAS}", "Authorization": f"Bearer {YAML_KEY}"})
    return _answer(response)


def test_a_damaged_key_file_opens_the_repair_card_and_keeps_the_tabs_key(tmp_path, gateway_answers):
    """keys.json damaged: no key can pass, so the page offers the repair (a claim with a one-time
    code), not the key prompt, and keeps the tab's key, which the API may still take."""
    damaged = _damaged_keys_answer(tmp_path)
    assert damaged["status"] == 403 and damaged["body"]["code"] == "keys_damaged" and damaged["body"]["can_claim"]
    answers = {**gateway_answers, "damaged": damaged}
    out = run_settings_js(tmp_path, f"""
        harness.storage.set('tts-stt.api-key', {json.dumps(YAML_KEY)});
        harness.on('GET', '/api/settings', fixtures.answers.damaged);
        await harness.boot();
        const card = {{
            shown: harness.visible('claim-card'), title: harness.text('claim-title'), intro: harness.text('claim-intro'),
            keyPanel: harness.visible('key-panel'), main: harness.visible('settings-main'),
            cancel: harness.visible('claim-cancel'), require: harness.visible('claim-require-row'),
            stored: harness.storage.get('tts-stt.api-key'), status: harness.text('settings-status'),
        }};
        harness.on('POST', '/api/settings/claim-code', fixtures.answers.claim_code);
        await harness.click('[data-action="print-code"]');
        await harness.type('claim-code', 'ABCDE-FGHJK-MNPQR-STVWX');
        await harness.check('claim-stored');
        harness.on('POST', '/api/settings/claim', fixtures.answers.claim);
        harness.on('GET', '/api/settings', {{ status: 200, body: fixtures.answers.view_claimed.body }});
        await harness.click('claim-button');
        return {{ card, status: harness.text('settings-status'), cardAfter: harness.visible('claim-card'),
                  main: harness.visible('settings-main'), tabKey: harness.storage.get('tts-stt.api-key') }};
    """, answers, start=None)
    card = out["card"]
    assert (card["shown"], card["keyPanel"], card["main"], card["cancel"], card["require"]) == (True, False, False, False, False)
    assert card["title"] == "Repair the API keys" and "keys.json" in card["intro"] and "damaged" in card["intro"]
    assert card["stored"] == YAML_KEY, "a 403 is no Bearer challenge: the tab's key is kept"
    assert card["status"] == ""
    assert "written again" in out["status"] and "make any that are missing again" in out["status"]
    assert (out["cardAfter"], out["main"]) == (False, True)
    assert PAGE_KEY.fullmatch(out["tabKey"]), "the new admin key is used in this tab"


def test_a_damaged_key_file_that_the_page_cannot_repair_says_what_to_do(tmp_path, gateway_answers):
    damaged = _damaged_keys_answer(tmp_path)
    answers = {**gateway_answers, "damaged": {**damaged, "body": {**damaged["body"], "can_claim": False}}}
    out = run_settings_js(tmp_path, """
        harness.on('GET', '/api/settings', fixtures.answers.damaged);
        await harness.boot();
        return { card: harness.visible('claim-card'), keyPanel: harness.visible('key-panel'),
                 status: harness.text('settings-status'),
                 alert: Boolean(harness.el('settings-status').querySelector('[role="alert"]')) };
    """, answers, start=None)
    assert (out["card"], out["keyPanel"], out["alert"]) == (False, False, True)
    assert "repair or delete settings/keys.json" in out["status"]


def test_the_history_names_the_yaml_key_by_its_id_and_never_by_a_name_alone(tmp_path, gateway_answers):
    """A page key named "deployment key" (made before such names were refused) is still a page key."""
    saved_by = gateway_answers["view_saved"]["body"]["history"][0]["saved_by"]
    assert saved_by["credential_id"] == "yaml:API_KEY", "the gateway records the YAML key by its id"
    view = json.loads(json.dumps(gateway_answers["view_saved"]["body"]))
    base = view["history"][0]
    where = {"ip": "192.168.1.20", "host": NAS}
    view["history"] = [
        {**base, "id": "20261002T091203Z-r3", "revision": 3,
         "saved_by": {**where, "credential": "deployment key", "credential_id": "0a1b2c3d"}},
        {**base, "id": "20261002T091202Z-r2", "revision": 2,
         "saved_by": {**where, "credential": "deployment key", "credential_id": "yaml:API_KEY"}},
        {**base, "id": "20261002T091201Z-r1", "revision": 1, "saved_by": {**where, "credential": "deployment key"}},
    ]
    answers = {**gateway_answers, "custom": {"status": 200, "body": view}}
    out = run_settings_js(tmp_path, OPEN_AS_ADMIN + """
        return harness.el('history-list').children.map((item) => item.querySelector('.setting-note').textContent);
    """, answers, start="custom")
    assert out[0].endswith('By "deployment key" (192.168.1.20, 192.168.1.20:3000).')
    assert "By the app YAML key (" in out[1]
    assert "By the app YAML key (" in out[2], "a version saved before ids were recorded"
