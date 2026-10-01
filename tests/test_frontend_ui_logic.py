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
"""

from __future__ import annotations

import json
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
