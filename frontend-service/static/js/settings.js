// The Settings page (/settings): shows and changes the gateway's live settings.
//
// The page runs under a strict Content-Security-Policy (script-src 'self' and
// style-src 'self', see _SETTINGS_PAGE_CSP in app.py), so everything here is built
// with createElement and textContent: no innerHTML, no eval, no inline handler or
// style attribute. Controls say what they do, and one delegated listener per event
// type dispatches:
//   data-action="name"   a click                      -> ACTIONS
//   data-field="KEY"     the input of a setting       -> onFieldInput
//   data-watch="name"    another input that matters   -> WATCHERS
//   data-enter="name"    Enter in that input          -> ACTIONS
//
// Everything comes from the gateway's internal Settings API (/api/settings*, see
// app.py). The key travels the way app.js sends it, `Authorization: Bearer`, from
// the same per-tab slot (sessionStorage 'tts-stt.api-key'), so a key entered on
// either page works on both. It never goes into a URL, an attribute or a log.
//
// Saving: edits collect in a draft; "Review and save" asks the server for a dry
// run (warnings to tick, notes, a commit-confirm notice, a lockout refusal) and
// shows old -> new; Save sends the same change with the ticked warnings, and a
// change of TRUSTED_ORIGINS / TRUST_PROXY_HEADERS is confirmed right away, through
// the new rules.
'use strict';

const API_KEY_STORAGE_KEY = 'tts-stt.api-key';
const PAGE_KEY_PREFIX = 'tts_';
const PREFERENCES_FILE = 'gateway.json';
const ACCESS_FILE = 'keys.json';
const DEFAULT_KEYS = ['DEFAULT_TTS_PROVIDER', 'DEFAULT_STT_PROVIDER'];
const LIST_KINDS = new Set(['hosts', 'origins']);

const ENGINE_NAMES = {
    piper: 'Piper', qwen3: 'Qwen3-TTS', whisper: 'Whisper', 'qwen3-asr': 'Qwen3-ASR',
    canary: 'Canary-ASR', parakeet: 'Parakeet-ASR', chatterbox: 'Chatterbox-TTS',
    magpie: 'Magpie-TTS', 'whisper-cpp': 'whisper.cpp', 'piper-training': 'Voice training',
};
const SOURCE_LABELS = {
    default: 'Default', yaml: 'App YAML', saved: 'Set here', locked: 'Locked by the app YAML',
    fail_closed: 'keys.json damaged',
};
const STATE_LABELS = { running: 'Running', not_reachable: 'Not answering', not_installed: 'Not installed' };
const EVENT_LABELS = {
    save: 'Saved', restore: 'Restored', discard: 'Back to the app YAML',
    'auto-revert': 'Undone: not confirmed in time',
};
const ROLE_LABELS = { admin: 'admin: use the API and change settings', client: 'client: use the API only' };
// What an empty list field shows. Examples only, marked as such: a placeholder that reads like
// a name (a real host name, say) is taken for a value that is set.
const PLACEHOLDERS = {
    TRUSTED_HOSTS: 'e.g. tts.example.com (one name per line)',
    TRUSTED_ORIGINS: 'e.g. https://tts.example.com',
    ALLOWED_ORIGINS: 'e.g. https://dashboard.example.com',
};
// How the history and the audit trail name the app YAML's API_KEY (settings_store.DEPLOYMENT_KEY_ID):
// by this id, not by the name, which a page key could otherwise share.
const DEPLOYMENT_KEY_ID = 'yaml:API_KEY';
// The claim card: claim (no admin key yet), recovery (a lost admin key) or repair (keys.json damaged).
const CLAIM_TEXTS = {
    claim: {
        title: 'Claim this server',
        intro: 'Nobody can change settings yet: this server has no admin key. A one-time code from the container log '
            + 'proves that you run it and creates your admin key.',
    },
    recovery: {
        title: 'Recover admin access',
        intro: 'A one-time code from the container log creates a new admin key. The other keys stay as they are; '
            + 'revoke the lost one afterwards.',
    },
    repair: {
        title: 'Repair the API keys',
        intro: 'keys.json in the settings folder is damaged or cannot be read: no key made here works, for these '
            + 'settings or for the API (only an API_KEY in the app YAML does). A one-time code from the container '
            + 'log creates a new admin key and writes the file again; a damaged one is kept as keys.json.damaged. '
            + 'Keys made here that are gone with it have to be made again.',
    },
};
const CONFLICT_TEXT = 'The settings were changed in the meantime, maybe in another tab. The page shows them now; '
    + 'your unsaved changes are kept where they still differ. Review them again.';
// Typographic characters, made from their code points so that this file stays ASCII.
const TIMES = String.fromCharCode(0xd7);
const ARROW = String.fromCharCode(0x2192);
const ELLIPSIS = String.fromCharCode(0x2026);
const EN_DASH = String.fromCharCode(0x2013);

// Everything the page knows. The draft maps a setting's key to { value } (save this
// value) or { reset: true } (drop the value saved here: the app YAML applies again).
const S = {
    view: null,             // the last answer of GET /api/settings
    engines: null,          // the last answer of GET /api/settings/engines
    draft: new Map(),
    moved: new Map(),       // DEFAULT_* -> why the page moved that default engine
    review: null,           // { save, blocked } while the review dialog is open
    newKey: null,           // { key, saved } while the create-key dialog is open
    claim: null,            // { mode: 'claim' | 'recovery', key } while the claim card is open
    pendingTimer: null,
    returnFocus: null,
};

// --- the API key, shared with app.js -------------------------------------------------

let memoryKey = '';

function readStoredKey() {
    try {
        return window.sessionStorage.getItem(API_KEY_STORAGE_KEY) || '';
    } catch {
        return memoryKey;   // storage blocked (private window, site data off): this page's memory only
    }
}

function storeKey(key) {
    memoryKey = key || '';
    try {
        if (memoryKey) window.sessionStorage.setItem(API_KEY_STORAGE_KEY, memoryKey);
        else window.sessionStorage.removeItem(API_KEY_STORAGE_KEY);
    } catch {
        // the memory copy above serves this page
    }
}

/** A key as the server expects one: tts_ and 32 random bytes in base64url (43 characters). */
function generateKey() {
    const bytes = new Uint8Array(32);
    window.crypto.getRandomValues(bytes);
    let binary = '';
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return PAGE_KEY_PREFIX + window.btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

// --- talking to the Settings API ------------------------------------------------------

function isBearerChallenge(response) {
    const header = response.headers && typeof response.headers.get === 'function'
        ? response.headers.get('WWW-Authenticate') : '';
    return response.status === 401 && /bearer/i.test(header || '');
}

/**
 * One call of the Settings API; never throws (a network failure is status 0).
 *
 * A Bearer challenge (401) to the key this tab keeps means the server knows no
 * such key at all (a client key is answered 403), so the shared slot forgets it,
 * as app.js does: offering it again would only count as another wrong guess.
 * `candidate` is a key to try instead, which is kept only if it works (useKey).
 */
async function api(method, path, body, candidate) {
    const trying = candidate !== undefined;
    const key = trying ? candidate : readStoredKey();
    const headers = { Accept: 'application/json' };
    if (key) headers.Authorization = `Bearer ${key}`;
    const init = { method, headers, cache: 'no-store', credentials: 'same-origin' };
    if (body !== undefined) {
        headers['Content-Type'] = 'application/json';
        init.body = JSON.stringify(body);
    }
    let response;
    try {
        response = await fetch(path, init);
    } catch {
        return {
            status: 0, ok: false, retryAfter: null, sentKey: key,
            data: { detail: 'The server did not answer. Check that it runs, then try again.', code: 'network_error' },
        };
    }
    let data = null;
    try {
        data = await response.json();
    } catch {
        data = null;
    }
    if (!trying && key && isBearerChallenge(response) && readStoredKey() === key) storeKey('');
    const retryAfter = response.headers && typeof response.headers.get === 'function'
        ? response.headers.get('Retry-After') : null;
    return {
        status: response.status,
        ok: response.ok,
        data: data && typeof data === 'object' && !Array.isArray(data) ? data : {},
        retryAfter,
        sentKey: key,
    };
}

/** The server's own message for a refusal (always plain text), or a generic one. */
function errorText(result, fallback) {
    const detail = result && result.data ? result.data.detail : null;
    let text = typeof detail === 'string' && detail ? detail : '';
    if (!text) text = fallback || `The request failed (HTTP ${result ? result.status : '?'}).`;
    if (result && result.retryAfter && /^\d+$/.test(String(result.retryAfter))) {
        text += ` Try again in ${result.retryAfter} seconds.`;
    }
    return text;
}

// --- building the page ------------------------------------------------------------------

// Properties set as properties; anything else named in el() becomes an attribute.
const PROPERTIES = new Set([
    'id', 'className', 'type', 'value', 'checked', 'disabled', 'hidden', 'htmlFor', 'name', 'title',
    'readOnly', 'rows', 'min', 'max', 'step', 'placeholder', 'maxLength',
]);

/** createElement with properties, attributes, data-* values and children (strings become text). */
function el(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(props || {})) {
        if (value === undefined || value === null) continue;
        if (/^on|^style$|html$|^srcdoc$/i.test(name)) throw new Error(`el(): ${name} is not set this way`);
        if (name === 'text') node.textContent = String(value);
        else if (name === 'data') Object.entries(value).forEach(([key, item]) => { node.dataset[key] = String(item); });
        else if (PROPERTIES.has(name)) node[name] = value;
        else node.setAttribute(name, String(value));
    }
    appendChildren(node, children);
    return node;
}

function appendChildren(node, children) {
    for (const child of children.flat(Infinity)) {
        if (child === null || child === undefined || child === false) continue;
        node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    }
    return node;
}

/** Every button that does something is made here, so its action is easy to find. */
function actionButton(action, label, props) {
    const { className = 'btn-secondary', data = {}, ...rest } = props || {};
    return el('button', { type: 'button', className, data: { ...data, action }, text: label, ...rest });
}

function byId(id) {
    return document.getElementById(id);
}

function setText(id, text) {
    const node = byId(id);
    if (node) node.textContent = text == null ? '' : String(text);
}

function setShown(id, shown) {
    const node = byId(id);
    if (node) node.hidden = !shown;
}

/**
 * The page's status line: success and progress are read out politely, an error interrupts.
 * `withAppLink` adds "Back to the app", for a change the main page shows only once reloaded.
 */
function setStatus(type, text, withAppLink) {
    const box = byId('settings-status');
    if (!box) return;
    if (!text) {
        box.replaceChildren();
        return;
    }
    const line = el('div', { className: type || 'info', text });
    if (type === 'error') line.setAttribute('role', 'alert');
    if (withAppLink) appendChildren(line, [' ', el('a', { href: '/', id: 'status-app-link', text: 'Back to the app' })]);
    box.replaceChildren(line);
}

/** Run `task` with `button` disabled and marked busy, so a second click cannot start it twice. */
async function withBusy(button, task) {
    if (button && button.disabled) return undefined;
    if (button) {
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
    }
    try {
        return await task(button);
    } finally {
        if (button) {
            button.disabled = false;
            button.removeAttribute('aria-busy');
        }
        refreshGatedButtons();
    }
}

// --- values -------------------------------------------------------------------------------

/** One entry per name: split on lines, commas and spaces, trimmed, lower-cased, de-duplicated. */
function parseList(text) {
    const seen = new Set();
    const entries = [];
    for (const raw of String(text == null ? '' : text).split(/[\s,]+/)) {
        const entry = raw.trim().toLowerCase();
        if (!entry || seen.has(entry)) continue;
        seen.add(entry);
        entries.push(entry);
    }
    return entries;
}

/** A number when the text is one; otherwise the text, which the server refuses with its own message. */
function parseNumber(text) {
    const trimmed = String(text == null ? '' : text).trim();
    if (trimmed === '') return trimmed;
    const number = Number(trimmed);
    return Number.isFinite(number) ? number : trimmed;
}

function sameValue(a, b) {
    if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((item, i) => item === b[i]);
    return a === b;
}

function settingFor(key) {
    return S.view ? S.view.settings.find((setting) => setting.key === key) || null : null;
}

/** What a field shows: the draft's value, the app YAML value for a pending reset, or the value in force. */
function displayValue(setting) {
    const change = S.draft.get(setting.key);
    if (!change) return setting.value;
    return change.reset ? setting.yaml_value : change.value;
}

function draftValues() {
    const values = {};
    for (const setting of S.view ? S.view.settings : []) values[setting.key] = displayValue(setting);
    return values;
}

function isEditable(setting) {
    const view = S.view;
    if (!view || !setting || setting.locked || setting.read_only) return false;
    return setting.file === ACCESS_FILE ? Boolean(view.can_manage_keys) : Boolean(view.can_write);
}

function engineName(id) {
    return ENGINE_NAMES[id] || String(id);
}

function formatValue(setting, value) {
    if (Array.isArray(value)) return value.length ? value.join(', ') : '(none)';
    if (typeof value === 'boolean') {
        if (setting && setting.key === 'require_key') return value ? 'Require a key' : 'Open on the network';
        return value ? 'On' : 'Off';
    }
    if (value === null || value === undefined || value === '') return '(empty)';
    if (setting && setting.kind === 'provider') return engineName(value);
    return String(value);
}

function formatTime(iso) {
    if (!iso) return 'an unknown time';
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? String(iso) : date.toLocaleString();
}

function formatClock(epochSeconds) {
    const date = new Date(Number(epochSeconds) * 1000);
    return Number.isNaN(date.getTime()) ? 'soon' : date.toLocaleTimeString();
}

function formatMegabytes(total) {
    return total >= 1024 ? `${(total / 1024).toFixed(1)} GB` : `${Math.round(total)} MB`;
}

/** {provider id: kind} on offer with these values: what the server's offered_engines() computes. */
function offeredEngines(values) {
    const view = S.view;
    const flags = new Map();
    for (const setting of view.settings) if (setting.engine) flags.set(setting.engine, setting);
    const fixed = new Set((view.engines && view.engines.registry_json) || []);
    const offered = {};
    for (const [provider, kind] of Object.entries((view.engines && view.engines.offered) || {})) {
        if (flags.has(provider) && !fixed.has(provider)) continue;   // its flag decides, below
        offered[provider] = kind;
    }
    for (const [provider, setting] of flags) {
        if (!fixed.has(provider) && values[setting.key] === true) offered[provider] = setting.engine_kind;
    }
    return offered;
}

// --- the draft ------------------------------------------------------------------------------

function setDraft(key, value) {
    const setting = settingFor(key);
    if (!setting) return;
    if (sameValue(value, setting.value)) S.draft.delete(key);
    else S.draft.set(key, { value });
}

/**
 * A default engine that is no longer offered moves to one that is: the schema's
 * default (a built-in) when it is offered, otherwise the first engine of its kind.
 * The server refuses a default that is not offered, so leaving it would only fail.
 */
function moveUnofferedDefaults() {
    const values = draftValues();
    const offered = offeredEngines(values);
    for (const key of DEFAULT_KEYS) {
        const setting = settingFor(key);
        if (!setting || !isEditable(setting)) continue;
        const current = values[key];
        if (offered[current] === setting.engine_kind) continue;
        const fallback = [setting.default, ...Object.keys(offered)].find((id) => offered[id] === setting.engine_kind);
        if (!fallback || fallback === current) continue;
        setDraft(key, fallback);
        S.moved.set(key, `${engineName(current)} is no longer offered, so the default moves to ${engineName(fallback)}.`);
    }
}

/** A default the page moved goes back once its engine is offered again (ticking Offer twice changes nothing). */
function restoreMovedDefaults() {
    const offered = offeredEngines(draftValues());
    for (const key of Array.from(S.moved.keys())) {
        const setting = settingFor(key);
        if (setting && offered[setting.value] === setting.engine_kind) {
            S.draft.delete(key);
            S.moved.delete(key);
        }
    }
}

/** After an engine is offered or not: the default engines follow. */
function followEngineChange() {
    restoreMovedDefaults();
    moveUnofferedDefaults();
}

/** Save took these: they leave the draft, whatever spelling the server stored them in. */
function forgetSaved(keys) {
    for (const key of keys) {
        S.draft.delete(key);
        S.moved.delete(key);
    }
}

/** The changes Save would make: gateway.json values to set or reset, and the access settings. */
function pendingChanges() {
    const set = {};
    const reset = [];
    const access = {};
    for (const [key, change] of S.draft) {
        const setting = settingFor(key);
        if (!setting) continue;
        if (setting.file === ACCESS_FILE) access[key] = change.value;
        else if (change.reset) reset.push(key);
        else set[key] = change.value;
    }
    return {
        set, reset, access,
        preferences: Object.keys(set).length + reset.length > 0,
        accessChanged: Object.keys(access).length > 0,
    };
}

/** After the view changed: drop draft entries that are now in force, or that can no longer be made. */
function pruneDraft() {
    for (const [key, change] of Array.from(S.draft)) {
        const setting = settingFor(key);
        const done = !setting || !isEditable(setting)
            || (change.reset ? !setting.saved : sameValue(change.value, setting.value));
        if (done) S.draft.delete(key);
    }
    for (const key of Array.from(S.moved.keys())) if (!S.draft.has(key)) S.moved.delete(key);
}

function readField(element) {
    const setting = settingFor(element.dataset.field);
    if (!setting) return undefined;
    switch (setting.kind) {
        case 'hosts':
        case 'origins':
            return parseList(element.value);
        case 'bool':
            return setting.key === 'require_key' ? element.value === 'true' : Boolean(element.checked);
        case 'int':
        case 'number':
            return parseNumber(element.value);
        default:
            return element.value;
    }
}

function onFieldInput(element) {
    const key = element.dataset.field;
    const setting = settingFor(key);
    if (!isEditable(setting)) return;
    if (element.type === 'radio' && !element.checked) return;
    setDraft(key, readField(element));
    clearFieldError(key);
    if (DEFAULT_KEYS.includes(key)) S.moved.delete(key);
    if (setting.engine) {
        followEngineChange();
        refreshProviderSelects();
    }
    updateDependents();
    updateSaveBar();
}

function resetField(key) {
    const setting = settingFor(key);
    if (!isEditable(setting) || !setting.saved) return;
    S.draft.set(key, { reset: true });
    clearFieldError(key);
    if (setting.engine) followEngineChange();
    renderFields(`undo-${key}`);
}

function undoReset(key) {
    const setting = settingFor(key);
    S.draft.delete(key);
    if (setting && setting.engine) followEngineChange();
    renderFields(`reset-${key}`);
}

function undoDraft() {
    S.draft.clear();
    S.moved.clear();
    renderFields(null);
    setStatus(null, '');
}

// --- rendering ------------------------------------------------------------------------------

function render() {
    renderBanners();
    renderWho();
    const view = S.view;
    setShown('settings-main', Boolean(view));
    if (view) {
        renderReadonlyNote();
        renderFields(null);
        renderKeys();
        renderHistory();
        renderFacts();
        renderClaimCard();
    } else {
        updateSaveBar();
    }
}

function levelClass(level) {
    return level === 'error' ? 'error' : level === 'warning' ? 'warning' : 'info';
}

function renderBanners() {
    const region = byId('settings-banners');
    if (!region) return;
    const view = S.view;
    const banners = [];
    for (const banner of (view && view.banners) || []) {
        const box = el('div', { className: `banner ${levelClass(banner.level)}`, data: { banner: banner.id } },
            el('p', { text: banner.text }));
        if (banner.id === 'not_mounted') box.appendChild(mountLines(view.mount));
        if (banner.id === 'pending' && view.can_write) {
            box.appendChild(actionButton('confirm-pending', 'Keep the change', { className: 'btn-secondary btn-sm' }));
        }
        banners.push(box);
    }
    region.replaceChildren(...banners);
}

function mountLines(mount) {
    const lines = (mount && mount.lines) || {};
    return el('div', { className: 'mount-lines' },
        el('p', { text: 'TrueNAS app YAML, under frontend-service, volumes:' }),
        el('pre', { text: (lines.truenas || []).join('\n') }),
        el('p', { text: 'docker-compose.yml, under frontend-service, volumes:' }),
        el('pre', { text: (lines.compose || []).join('\n') }));
}

function renderWho() {
    const box = byId('settings-who');
    if (!box) return;
    const view = S.view;
    const stored = readStoredKey();
    const parts = [];
    if (view && view.credential) {
        const credential = view.credential;
        const whose = credential.deployment_key ? 'the key from the app YAML' : `the key "${credential.name}"`;
        parts.push(el('span', { text: `This tab uses ${whose} (${credential.role}).` }));
    } else if (view && stored) {
        parts.push(el('span', { text: 'The key kept in this tab is not one this server knows.' }));
    } else if (view && !view.claimed) {
        parts.push(el('span', { text: 'Anyone who reaches this page can read these settings until the server is claimed.' }));
    }
    if (view && (view.claimed || stored)) {
        parts.push(actionButton('change-key', 'Use another key', { className: 'btn-secondary btn-sm' }));
    }
    if (view && stored) {
        parts.push(actionButton('forget-key', 'Forget the key in this tab', { className: 'btn-secondary btn-sm' }));
    }
    box.replaceChildren(...parts);
    box.hidden = parts.length === 0;
}

function readonlyReason(view) {
    if (!view.enabled) return 'ENABLE_SETTINGS_UI=false in the app YAML.';
    if (!view.mount.writable) return 'the settings folder is not mounted (see above).';
    if (!view.claimed) return 'this server has no admin key yet. Claim it with a one-time code.';
    if (!view.credential || view.credential.role !== 'admin') return 'this tab has no admin key.';
    if (view.safe_mode) return 'a SAFE-MODE file in the settings folder. API keys can still be managed.';
    return 'the server does not take changes right now.';
}

function renderReadonlyNote() {
    const view = S.view;
    const note = byId('readonly-note');
    if (!note) return;
    note.hidden = Boolean(view.can_write);
    note.textContent = view.can_write ? '' : `Read-only: ${readonlyReason(view)}`;
}

function sourceBadge(setting) {
    return el('span', {
        className: `badge badge-${setting.source}`,
        text: SOURCE_LABELS[setting.source] || setting.source,
        title: 'Where the value in force comes from',
    });
}

function applyBadge(setting) {
    const text = setting.apply === 'live' ? 'Applies immediately' : String(setting.apply || '');
    return text ? el('span', { className: 'badge badge-apply', text }) : null;
}

function fieldLabel(setting, forId) {
    return el('label', { className: 'setting-label', htmlFor: forId, id: `label-${setting.key}`, text: setting.label });
}

function textareaRows(values) {
    return Math.min(10, Math.max(3, (Array.isArray(values) ? values.length : 0) + 1));
}

/** The input of one setting, and the label to put in its head (null when the control carries it). */
function fieldControl(setting, shown, editable, describedBy) {
    const key = setting.key;
    const id = `input-${key}`;
    const common = { id, disabled: !editable, 'aria-describedby': describedBy, data: { field: key } };
    switch (setting.kind) {
        case 'hosts':
        case 'origins': {
            const area = el('textarea', {
                ...common, rows: textareaRows(shown), spellcheck: 'false', autocomplete: 'off', autocapitalize: 'none',
                placeholder: PLACEHOLDERS[key] || null,
            });
            area.value = (Array.isArray(shown) ? shown : []).join('\n');
            return { control: area, label: fieldLabel(setting, id) };
        }
        case 'bool': {
            if (key === 'require_key') {
                const option = (value, text) => {
                    const radio = el('input', {
                        type: 'radio', name: 'require_key', value: String(value), id: `${id}-${value}`,
                        disabled: !editable, data: { field: key },
                    });
                    radio.checked = shown === value;
                    return el('label', { className: 'check', htmlFor: radio.id }, radio, el('span', { text }));
                };
                const group = el('div', {
                    className: 'setting-choice', id, role: 'radiogroup', 'aria-labelledby': `label-${key}`,
                    'aria-describedby': describedBy,
                }, option(false, 'Open on the network: no key needed'), option(true, 'Require a key'));
                return { control: group, label: el('span', { className: 'setting-label', id: `label-${key}`, text: setting.label }) };
            }
            const box = el('input', { ...common, type: 'checkbox' });
            box.checked = shown === true;
            return { control: null, label: el('label', { className: 'check setting-label', htmlFor: id, id: `label-${key}` }, box, el('span', { text: setting.label })) };
        }
        case 'int':
        case 'number': {
            const input = el('input', {
                ...common, type: 'number', min: setting.min, max: setting.max,
                step: setting.kind === 'int' ? '1' : 'any', inputmode: setting.kind === 'int' ? 'numeric' : 'decimal',
            });
            input.value = shown === null || shown === undefined ? '' : String(shown);
            return { control: input, label: fieldLabel(setting, id) };
        }
        case 'provider': {
            const select = el('select', common);
            fillProviderOptions(select, setting, shown);
            return { control: select, label: fieldLabel(setting, id) };
        }
        case 'choice': {
            const select = el('select', common);
            const deploymentKeyInUse = Boolean(S.view.credential && S.view.credential.deployment_key);
            for (const choice of setting.choices || []) {
                const option = el('option', { value: choice, text: ROLE_LABELS[choice] || choice });
                // The YAML key cannot make itself a client (the server refuses it too).
                if (key === 'deployment_key_role' && choice === 'client' && deploymentKeyInUse && shown !== 'client') {
                    option.disabled = true;
                }
                select.appendChild(option);
            }
            select.value = shown;
            return { control: select, label: fieldLabel(setting, id) };
        }
        default:
            return { control: el('p', { text: formatValue(setting, shown) }), label: el('span', { className: 'setting-label', text: setting.label }) };
    }
}

function fillProviderOptions(select, setting, selected) {
    const offered = offeredEngines(draftValues());
    const ids = Object.keys(offered).filter((id) => offered[id] === setting.engine_kind);
    if (selected && !ids.includes(selected)) ids.push(selected);
    select.replaceChildren(...ids.map((id) => el('option', {
        value: id,
        text: offered[id] === setting.engine_kind ? engineName(id) : `${engineName(id)} (not offered)`,
    })));
    select.value = selected;
}

function refreshProviderSelects() {
    for (const key of DEFAULT_KEYS) {
        const setting = settingFor(key);
        const select = byId(`input-${key}`);
        if (setting && select) fillProviderOptions(select, setting, displayValue(setting));
    }
}

/** The lines under a field: the YAML value it overrides, why it is ignored or read-only. */
function fieldNotes(setting) {
    const notes = [];
    const add = (text, className = 'setting-note') => notes.push(el('p', { className, text }));
    if (setting.file === PREFERENCES_FILE) {
        if (setting.source === 'saved' && !sameValue(setting.yaml_value, setting.value)) {
            add(`App YAML: ${formatValue(setting, setting.yaml_value)}. Not used while the value set here applies.`);
        }
        if (setting.ignored && setting.saved) {
            add(`Saved here: ${formatValue(setting, setting.saved_value)}. Ignored: ${setting.locked
                ? 'SETTINGS_LOCKED_KEYS keeps the app YAML value.' : 'saved values are switched off right now.'}`);
        }
        if (setting.dropped) add(`A value in ${PREFERENCES_FILE} was ignored: ${setting.dropped}`, 'setting-note note-warning');
    }
    if (setting.read_only) add(setting.read_only, 'setting-note note-readonly');
    if (setting.key === 'deployment_key_role' && isEditable(setting)) {
        if (S.view.credential && S.view.credential.deployment_key) {
            add('This tab uses the key from the app YAML, which cannot make itself a client: do that with an admin key created here.');
        } else if (!S.view.keys.some((record) => record.role === 'admin')) {
            add('Create an admin key first: without one, nobody could change settings any more.');
        }
    }
    return notes;
}

function fieldActions(setting) {
    if (setting.file !== PREFERENCES_FILE || !isEditable(setting)) return null;
    const change = S.draft.get(setting.key);
    if (change && change.reset) {
        return el('div', { className: 'setting-actions' },
            actionButton('undo-reset', 'Keep the value set here', {
                className: 'btn-secondary btn-sm', id: `undo-${setting.key}`, data: { key: setting.key },
            }));
    }
    if (!setting.saved) return null;
    return el('div', { className: 'setting-actions' },
        actionButton('reset-field', 'Reset to app YAML', {
            className: 'btn-secondary btn-sm', id: `reset-${setting.key}`, data: { key: setting.key },
            'aria-label': `Reset "${setting.label}" to the app YAML value`,
        }));
}

function renderSetting(setting) {
    const key = setting.key;
    const editable = isEditable(setting);
    const describedBy = `help-${key} error-${key}`;
    const { control, label } = fieldControl(setting, displayValue(setting), editable, describedBy);
    const head = el('div', { className: 'setting-head' }, label, sourceBadge(setting), applyBadge(setting),
        el('span', { className: 'badge badge-changed', id: `changed-${key}`, hidden: true }));
    if (setting.engine) head.appendChild(el('span', { className: 'chip chip-checking', id: `engine-state-${key}`, text: 'Checking...' }));
    return el('div', { className: 'setting', id: `setting-${key}`, data: { key } },
        head,
        control,
        el('p', { className: 'setting-help', id: `help-${key}`, text: setting.help }),
        fieldNotes(setting),
        setting.engine ? el('details', { className: 'install-hint', id: `install-${key}`, hidden: true }) : null,
        el('p', { className: 'setting-hint', id: `hint-${key}`, hidden: true }),
        el('p', { className: 'setting-error', id: `error-${key}`, hidden: true }),
        fieldActions(setting));
}

/** (Re)build every field from the view and the draft, then put the focus back where it was. */
function renderFields(focusId) {
    if (!S.view) return;
    const active = document.activeElement;
    const keepFocus = focusId || (active && active.id) || null;
    const groups = new Map();
    for (const setting of S.view.settings) {
        if (!groups.has(setting.group)) groups.set(setting.group, []);
        groups.get(setting.group).push(renderSetting(setting));
    }
    for (const [group, rows] of groups) {
        const container = byId(`fields-${group}`);
        if (container) container.replaceChildren(...rows);
    }
    renderEngineStates();
    updateDependents();
    updateSaveBar();
    const target = keepFocus ? byId(keepFocus) : null;
    if (target && typeof target.focus === 'function') target.focus();
}

function memoryHint(values) {
    const workers = Number(S.view.deployment && S.view.deployment.workers) || 1;
    const uploads = Number(values.MAX_CONCURRENT_UPLOADS);
    const size = Number(values.MAX_UPLOAD_MB);
    if (!(uploads > 0) || !(size > 0)) return '';
    const plural = (count, word) => `${count} ${word}${count === 1 ? '' : 's'}`;
    return `Worst case: ${plural(workers, 'worker')} ${TIMES} ${plural(uploads, 'upload')} at once ${TIMES} ${size} MB `
        + `= ${formatMegabytes(workers * uploads * size)} of memory.`;
}

function hintFor(setting, values) {
    if (setting.key === 'MAX_UPLOAD_MB') return memoryHint(values);
    if (setting.warn_above !== null && setting.warn_above !== undefined && Number(values[setting.key]) > setting.warn_above) {
        return `Qwen3-TTS, Chatterbox and Magpie refuse more than ${setting.warn_above} characters themselves; `
            + 'saving asks you to confirm this.';
    }
    if (DEFAULT_KEYS.includes(setting.key)) return S.moved.get(setting.key) || '';
    return '';
}

/** What depends on the draft without rebuilding a field: change markers and live hints. */
function updateDependents() {
    if (!S.view) return;
    const values = draftValues();
    for (const setting of S.view.settings) {
        const row = byId(`setting-${setting.key}`);
        if (!row) continue;
        const change = S.draft.get(setting.key);
        row.classList.toggle('is-changed', Boolean(change));
        const badge = byId(`changed-${setting.key}`);
        if (badge) {
            badge.hidden = !change;
            badge.textContent = change && change.reset ? 'Back to the app YAML on save' : 'Changed';
        }
        const hint = byId(`hint-${setting.key}`);
        if (hint) {
            const text = hintFor(setting, values);
            hint.textContent = text;
            hint.hidden = !text;
        }
    }
}

function updateSaveBar() {
    const count = S.view ? S.draft.size : 0;
    setShown('save-bar', count > 0);
    setText('save-bar-count', count === 1 ? '1 unsaved change' : `${count} unsaved changes`);
}

function clearFieldError(key) {
    const box = byId(`error-${key}`);
    if (box) {
        box.textContent = '';
        box.hidden = true;
    }
    const row = byId(`setting-${key}`);
    if (row) row.classList.remove('has-error');
    const input = byId(`input-${key}`);
    if (input) input.removeAttribute('aria-invalid');
}

function clearFieldErrors() {
    for (const setting of S.view ? S.view.settings : []) clearFieldError(setting.key);
}

/** Show each message next to its field, as text; returns what has no field on the page. */
function showFieldErrors(errors) {
    const unplaced = [];
    let first = null;
    for (const [key, message] of Object.entries(errors || {})) {
        const box = byId(`error-${key}`);
        if (!box) {
            unplaced.push(`${key}: ${message}`);
            continue;
        }
        box.textContent = String(message);
        box.hidden = false;
        const row = byId(`setting-${key}`);
        if (row) row.classList.add('has-error');
        const input = byId(`input-${key}`);
        if (input) input.setAttribute('aria-invalid', 'true');
        first = first || input;
    }
    if (first && typeof first.focus === 'function') first.focus();
    return unplaced;
}

function renderEngineStates() {
    if (!S.view) return;
    const engines = new Map(((S.engines && S.engines.engines) || []).map((engine) => [engine.key, engine]));
    for (const setting of S.view.settings) {
        if (!setting.engine) continue;
        const engine = engines.get(setting.key);
        const state = engine ? engine.state : (S.engines ? 'unknown' : 'checking');
        const chip = byId(`engine-state-${setting.key}`);
        if (chip) {
            chip.className = `chip chip-${state}`;
            chip.textContent = STATE_LABELS[state] || (state === 'checking' ? 'Checking...' : 'Not checked');
        }
        const install = byId(`install-${setting.key}`);
        if (install) {
            const show = Boolean(engine && engine.state === 'not_installed' && engine.install);
            install.hidden = !show;
            install.replaceChildren(...(show ? [
                el('summary', { text: 'How to install it' }),
                el('p', { text: `TrueNAS: ${engine.install.truenas}` }),
                el('p', {}, 'Compose: ', el('code', { text: engine.install.compose })),
            ] : []));
        }
    }
}

function renderKeys() {
    const view = S.view;
    const list = byId('keys-list');
    if (!list) return;
    const items = [];
    const usedHere = () => el('span', { className: 'badge badge-saved', text: 'used by this tab' });
    if (view.access.deployment_key_present) {
        items.push(el('li', { className: 'item' },
            el('div', { className: 'item-main' },
                el('strong', { text: 'Key from the app YAML' }), ' ',
                el('span', { className: 'badge', text: view.access.deployment_key_role }), ' ',
                view.credential && view.credential.deployment_key ? usedHere() : null),
            el('p', { className: 'setting-note', text: 'API_KEY in the app YAML. It cannot be changed or removed here.' })));
    }
    for (const record of view.keys) {
        const inUse = Boolean(view.credential && view.credential.id === record.id);
        items.push(el('li', { className: 'item' },
            el('div', { className: 'item-main' },
                el('strong', { text: record.name }), ' ',
                el('span', { className: 'badge', text: record.role }), ' ',
                el('code', { text: `${record.hint}${ELLIPSIS}` }), ' ',
                inUse ? usedHere() : null),
            el('p', { className: 'setting-note', text: `Created ${formatTime(record.created_at)}.` }),
            actionButton('revoke-key', 'Revoke', {
                className: 'btn-secondary btn-sm btn-danger', data: { keyId: record.id },
                disabled: !view.can_manage_keys, 'aria-label': `Revoke the key "${record.name}"`,
            })));
    }
    if (!view.keys.length) items.push(el('li', { className: 'item item-empty', text: 'No keys created here yet.' }));
    list.replaceChildren(...items);
    setText('keys-count', `${view.keys.length} of ${view.max_keys} keys created here.`);
    const create = byId('create-key-button');
    if (create) create.disabled = !view.can_manage_keys || view.keys.length >= view.max_keys;
}

/** Who saved a version: the app YAML key by its id; a page key by its name. */
function savedByWhom(by) {
    if (by.credential_id === DEPLOYMENT_KEY_ID) return 'the app YAML key';
    // A version saved before credentials were recorded by id names the YAML key only by name.
    if (!by.credential_id && by.credential === 'deployment key') return 'the app YAML key';
    return by.credential ? `"${by.credential}"` : 'someone';
}

function describeEntry(entry) {
    const changed = entry.changed && entry.changed.length ? entry.changed.join(', ') : 'nothing';
    const by = entry.saved_by || {};
    const where = [by.ip, by.host].filter(Boolean).join(', ');
    return `Changed: ${changed}. By ${savedByWhom(by)}${where ? ` (${where})` : ''}.`;
}

function renderHistory() {
    const view = S.view;
    const list = byId('history-list');
    if (!list) return;
    const items = view.history.map((entry) => {
        const current = entry.revision === view.revision;
        return el('li', { className: 'item' },
            el('div', { className: 'item-main' },
                el('strong', { text: EVENT_LABELS[entry.event] || entry.event }),
                ` ${EN_DASH} revision ${entry.revision}, ${formatTime(entry.saved_at)} `,
                current ? el('span', { className: 'badge badge-saved', text: 'current' }) : null),
            el('p', { className: 'setting-note', text: describeEntry(entry) }),
            current ? null : actionButton('restore', 'Restore', {
                className: 'btn-secondary btn-sm', data: { historyId: entry.id },
                disabled: !view.can_write, 'aria-label': `Restore revision ${entry.revision}`,
            }));
    });
    if (!items.length) items.push(el('li', { className: 'item item-empty', text: 'Nothing saved here yet.' }));
    list.replaceChildren(...items);
    const discard = byId('discard-button');
    if (discard) {
        discard.disabled = !view.can_write
            || !view.settings.some((setting) => setting.file === PREFERENCES_FILE && setting.saved);
    }
}

function renderFacts() {
    const facts = byId('yaml-facts');
    if (!facts) return;
    const deployment = S.view.deployment || {};
    const rows = [
        ['API_KEY', deployment.api_key_set ? 'set (the deployment key)' : 'not set'],
        ['ALLOWED_HOSTS', deployment.allow_any_host ? "'*': the host check is off, except on this page" : 'the host check is on'],
        ['ALLOWED_ORIGINS', deployment.allowed_origins_wildcard ? "'*': any web page may call the API" : 'no wildcard'],
        ['ALLOW_CREDENTIALS', deployment.allow_credentials ? 'on' : 'off'],
        ['ENABLE_SETTINGS_UI', deployment.settings_ui_enabled ? 'true' : 'false: this page is read-only'],
        ['SETTINGS_LOCKED_KEYS', (deployment.locked_keys || []).length ? deployment.locked_keys.join(', ') : 'none'],
        ['FRONTEND_WORKERS', String(deployment.workers == null ? '' : deployment.workers)],
    ];
    facts.replaceChildren(...rows.flatMap(([name, value]) => [el('dt', {}, el('code', { text: name })), el('dd', { text: value })]));
}

// --- loading --------------------------------------------------------------------------------

/**
 * GET /api/settings and show it; callers that have something to report set the status
 * afterwards. With `candidate`, that key is tried and kept only if the server takes it:
 * a mistyped key never replaces the one that works.
 */
async function loadView(candidate) {
    const result = await api('GET', '/api/settings', undefined, candidate);
    const loaded = result.ok && Array.isArray(result.data.settings);
    if (candidate !== undefined && !loaded) {
        setText('key-panel-error', result.status === 401 ? 'This key was not accepted.'
            : result.data.code === 'admin_key_required'
                ? 'This key may use the API but not change settings: an admin key is needed here.'
                : errorText(result, 'The key could not be checked.'));
        setShown('key-panel-error', true);
        return false;
    }
    if (loaded) {
        if (candidate !== undefined) storeKey(candidate);
        S.view = result.data;
        setStatus(null, '');
        setShown('key-panel', false);
        pruneDraft();
        render();
        schedulePendingReload();
        if (!S.engines) loadEngines();
        return true;
    }
    S.view = null;
    if (result.status === 401) {
        showKeyPanel(result.sentKey
            ? 'This key was not accepted. Enter an admin key of this server.'
            : 'This server has an admin key. Enter it to see and change the settings.');
        setStatus(null, '');
    } else if (result.data.code === 'keys_damaged') {
        // No key can pass until keys.json is written again: the repair card, not the key prompt
        // (and the key in this tab is kept: it may be the app YAML key the API still takes).
        setShown('key-panel', false);
        if (result.data.can_claim) {
            setStatus(null, '');
            openClaim('repair');
        } else {
            setStatus('error', `${errorText(result)} This page cannot write the settings folder here: repair or `
                + "delete settings/keys.json in the app's dataset, then load this page again.");
        }
    } else if (result.data.code === 'admin_key_required') {
        showKeyPanel('The key in this tab may use the API but not change settings. Enter an admin key.');
        setStatus(null, '');
    } else {
        setStatus('error', errorText(result, 'The settings could not be loaded.'));
    }
    render();
    return false;
}

/** After an unconfirmed change's deadline, show what the server did with it (kept or undone). */
function schedulePendingReload() {
    if (S.pendingTimer) clearTimeout(S.pendingTimer);
    S.pendingTimer = null;
    const pending = S.view && S.view.pending;
    if (!pending) return;
    const reload = () => {
        // Not under an open dialog: its contents were made from the view that is shown now.
        if (dialogOpen('review-dialog') || dialogOpen('key-dialog')) S.pendingTimer = setTimeout(reload, 5000);
        else {
            S.pendingTimer = null;
            loadView();
        }
    };
    S.pendingTimer = setTimeout(reload, (Math.max(0, Number(pending.seconds_left) || 0) + 2) * 1000);
}

async function loadEngines() {
    setText('engines-line', 'Checking which engines run...');
    const result = await api('GET', '/api/settings/engines');
    if (!result.ok || !Array.isArray(result.data.engines)) {
        S.engines = { engines: [] };
        setText('engines-line', `The engines could not be checked: ${errorText(result)}`);
        renderEngineStates();
        return;
    }
    S.engines = result.data;
    setText('engines-line', `Checked ${formatTime(result.data.checked_at)}. "Not installed" means its container `
        + 'does not exist yet: it is installed once in the app YAML.');
    renderEngineStates();
}

function showKeyPanel(text) {
    setText('key-panel-text', text);
    setShown('key-panel-error', false);
    setShown('key-panel', true);
    const input = byId('key-input');
    if (input && typeof input.focus === 'function') input.focus();
}

async function useKey() {
    const input = byId('key-input');
    const key = input ? input.value.trim() : '';
    setShown('key-panel-error', false);
    if (!key) {
        setText('key-panel-error', 'Enter the key first.');
        setShown('key-panel-error', true);
        return;
    }
    input.value = '';
    if (await loadView(key)) setStatus('success', 'This tab now uses the key you entered.');
}

async function forgetKey() {
    storeKey('');
    await loadView();
    setStatus('info', 'This tab no longer keeps a key, here or on the main page.');
}

// --- the claim card: a one-time code from the container log -----------------------------------

function renderClaimCard() {
    const view = S.view;
    if (!view) return;
    if (!view.claimed && view.can_claim && !S.claim) openClaim('claim');
    else if (view.claimed && S.claim && S.claim.mode === 'claim') closeClaim();
}

function openClaim(mode) {
    if (S.claim && S.claim.mode === mode) {
        setShown('claim-card', true);
        return;
    }
    S.claim = { mode, key: generateKey() };
    const recovery = mode !== 'claim';
    const texts = CLAIM_TEXTS[mode] || CLAIM_TEXTS.claim;
    setText('claim-title', texts.title);
    setText('claim-intro', texts.intro);
    const view = S.view;
    const requireLocked = !view || view.access.deployment_key_present
        || (view.deployment.locked_keys || []).includes('require_key') || view.access.require_key;
    setShown('claim-require-row', !recovery && !requireLocked);
    setShown('claim-cancel', mode === 'recovery' || Boolean(view && view.claimed));
    const code = byId('claim-code');
    if (code) code.value = '';
    const key = byId('claim-key');
    if (key) key.value = S.claim.key;
    for (const box of [byId('claim-stored'), byId('claim-require')]) {
        if (box) box.checked = false;
    }
    setText('claim-code-status', '');
    setText('claim-copy-status', '');
    setShown('claim-error', false);
    setShown('claim-card', true);
    updateClaimButton();
    if (recovery) {
        const card = byId('claim-card');
        if (card && typeof card.scrollIntoView === 'function') card.scrollIntoView({ block: 'start' });
        const title = byId('claim-title');
        if (title) {
            title.setAttribute('tabindex', '-1');
            if (typeof title.focus === 'function') title.focus();
        }
    }
}

function closeClaim() {
    S.claim = null;
    const key = byId('claim-key');
    if (key) key.value = '';
    setShown('claim-card', false);
}

function updateClaimButton() {
    const button = byId('claim-button');
    if (!button) return;
    const value = (id) => {
        const input = byId(id);
        return input ? input.value.trim() : '';
    };
    const stored = byId('claim-stored');
    button.disabled = !(S.claim && value('claim-code') && value('claim-name') && stored && stored.checked);
}

function showClaimError(text) {
    setText('claim-error', text);
    setShown('claim-error', Boolean(text));
}

async function printCode() {
    showClaimError('');
    const result = await api('POST', '/api/settings/claim-code', {});
    if (result.status === 202) {
        setText('claim-code-status', `${errorText(result, 'A one-time code was printed to the container log.')} `
            + 'Enter it below.');
        return;
    }
    showClaimError(errorText(result, 'No code could be printed.'));
}

async function submitClaim() {
    if (!S.claim) return;
    const name = byId('claim-name').value.trim();
    const body = { code: byId('claim-code').value.trim(), name, key: S.claim.key };
    const row = byId('claim-require-row');
    const requireBox = byId('claim-require');
    if (row && !row.hidden && requireBox && requireBox.checked) body.require_key = true;
    showClaimError('');
    const result = await api('POST', '/api/settings/claim', body);
    if (!result.ok) {
        const errors = result.data.errors ? Object.values(result.data.errors).join(' ') : '';
        showClaimError(errors || errorText(result, 'The claim did not work.'));
        return;
    }
    const mode = S.claim.mode;
    storeKey(S.claim.key);
    closeClaim();
    await loadView();
    const done = {
        claim: `This server is claimed. This tab now uses the new admin key "${name}".`,
        recovery: `This tab now uses the new admin key "${name}". Revoke the lost key under API keys.`,
        repair: `The API keys are written again. This tab now uses the new admin key "${name}". Check the keys `
            + 'under API keys and make any that are missing again.',
    };
    setStatus('success', done[mode] || done.claim);
}

// --- dialogs --------------------------------------------------------------------------------

function dialogOpen(id) {
    const dialog = byId(id);
    return Boolean(dialog && dialog.open);
}

/**
 * Open a dialog and focus `focusId` in it. Closing it focuses `opener` again (or whatever
 * had the focus): pass the button that opened it, because withBusy() has just disabled
 * that button, which moved the focus to the page itself.
 */
function showDialog(id, focusId, opener) {
    const dialog = byId(id);
    if (!dialog) return;
    if (!dialog.open) {
        S.returnFocus = opener || document.activeElement || null;
        if (typeof dialog.showModal === 'function') dialog.showModal();
        else dialog.setAttribute('open', '');
    }
    const target = focusId ? byId(focusId) : null;
    if (target && typeof target.focus === 'function') target.focus();
}

function hideDialog(id) {
    const dialog = byId(id);
    if (!dialog || !dialog.open) return;
    if (typeof dialog.close === 'function') dialog.close();
    else {
        dialog.removeAttribute('open');
        onDialogClosed(id);
    }
}

/** Also runs when Escape closes a dialog. */
function onDialogClosed(id) {
    if (id === 'review-dialog') S.review = null;
    if (id === 'key-dialog') {
        S.newKey = null;
        const value = byId('new-key-value');
        if (value) value.value = '';
    }
    const back = S.returnFocus;
    S.returnFocus = null;
    if (back && back.isConnected && typeof back.focus === 'function') back.focus();
}

// --- review and save --------------------------------------------------------------------------

function changeRow(setting, before, after, note) {
    const row = el('div', { className: 'review-row', data: { key: setting.key } },
        el('p', { className: 'review-label' }, el('strong', { text: setting.label }), ' ', el('code', { text: setting.key })));
    if (LIST_KINDS.has(setting.kind) && Array.isArray(before) && Array.isArray(after)) {
        const removed = before.filter((item) => !after.includes(item));
        const added = after.filter((item) => !before.includes(item));
        const items = [
            ...removed.map((item) => el('li', { className: 'review-removed' }, el('span', { className: 'review-mark', text: 'removed' }), ' ', item)),
            ...added.map((item) => el('li', { className: 'review-added' }, el('span', { className: 'review-mark', text: 'added' }), ' ', item)),
        ];
        if (!items.length) items.push(el('li', { text: 'The same entries in another order.' }));
        row.appendChild(el('ul', { className: 'review-list' }, items));
        if (setting.key === 'TRUSTED_HOSTS' && removed.length) {
            row.appendChild(el('p', {
                className: 'setting-note note-warning',
                text: `Anyone who opens this server as ${removed.join(', ')} is refused after this change.`,
            }));
        }
    } else {
        row.appendChild(el('p', { className: 'review-change' },
            el('span', { className: 'review-old', text: formatValue(setting, before) }),
            ` ${ARROW} `,
            el('span', { className: 'review-new', text: formatValue(setting, after) })));
    }
    if (note) row.appendChild(el('p', { className: 'setting-note', text: note }));
    return row;
}

/** What a change of API access means for the clients that use this server. */
function accessNote(key, value) {
    if (key === 'require_key' && value === true) {
        return 'Every client then has to send a key: scripts, Home Assistant and OpenAI-style clients without one are refused.';
    }
    if (key === 'deployment_key_role' && value === 'client') {
        return 'The key from the app YAML may then only use the API; changing settings needs an admin key made here.';
    }
    return null;
}

function draftRows() {
    const rows = [];
    for (const setting of S.view.settings) {
        const change = S.draft.get(setting.key);
        if (!change) continue;
        rows.push(change.reset
            ? changeRow(setting, setting.value, setting.yaml_value, 'Back to the app YAML value.')
            : changeRow(setting, setting.value, change.value, accessNote(setting.key, change.value)));
    }
    return rows;
}

function warningCheck(warning) {
    const setting = settingFor(warning.key);
    const id = `ack-${String(warning.token).replace(/[^A-Za-z0-9_-]/g, '-')}`;
    const box = el('input', { type: 'checkbox', id, data: { watch: 'review', token: warning.token } });
    return el('label', { className: 'check warning-check', htmlFor: id }, box,
        el('span', { text: `${setting ? setting.label : warning.key}: ${warning.message}` }));
}

function reviewTokens() {
    return Array.from(byId('review-warnings').querySelectorAll('input[type="checkbox"]'))
        .filter((box) => box.checked).map((box) => box.dataset.token);
}

function updateReviewSave() {
    const save = byId('review-save');
    if (!save || save.getAttribute('aria-busy') === 'true') return;
    const boxes = Array.from(byId('review-warnings').querySelectorAll('input[type="checkbox"]'));
    save.disabled = !S.review || S.review.blocked || !boxes.every((box) => box.checked);
}

function setReviewError(text) {
    setText('review-error', text);
    setShown('review-error', Boolean(text));
}

function addReviewWarnings(warnings) {
    const container = byId('review-warnings');
    const known = new Set(Array.from(container.querySelectorAll('input[type="checkbox"]')).map((box) => box.dataset.token));
    if (!container.querySelector('p')) {
        container.appendChild(el('p', { className: 'review-warnings-title', text: 'Tick each of these to confirm you mean it:' }));
    }
    for (const warning of warnings || []) {
        if (!known.has(warning.token)) container.appendChild(warningCheck(warning));
    }
    updateReviewSave();
}

/** What a dry run says: warnings to tick, notes, the confirmation it would start, or why it is refused. */
function previewParts(preview) {
    if (!preview) return { warnings: [], notes: {}, confirm: null, blocked: '' };
    if (!preview.ok) return { warnings: [], notes: {}, confirm: null, blocked: errorText(preview) };
    return {
        warnings: preview.data.warnings || [], notes: preview.data.notes || {},
        confirm: preview.data.confirm || null, blocked: '',
    };
}

function confirmNotice(confirm) {
    const keys = (confirm.keys || []).join(' and ');
    return `A change of ${keys} has to be confirmed within ${Math.round(Number(confirm.window) || 60)} seconds, `
        + 'or it is undone. This page confirms it right after saving, through the new rules, so it stays only if '
        + 'this page still reaches the server under them.';
}

function openReview(spec) {
    const parts = previewParts(spec.preview);
    S.review = { save: spec.save, blocked: Boolean(parts.blocked) };
    setText('review-title', spec.title);
    setText('review-intro', spec.intro || '');
    byId('review-rows').replaceChildren(...spec.rows);
    const notes = [];
    for (const [key, texts] of Object.entries(parts.notes)) {
        const setting = settingFor(key);
        for (const text of texts) notes.push(el('p', { className: 'setting-note', text: `${setting ? setting.label : key}: ${text}` }));
    }
    byId('review-notes').replaceChildren(...notes);
    byId('review-warnings').replaceChildren();
    if (parts.warnings.length) addReviewWarnings(parts.warnings);
    setText('review-confirm', parts.confirm ? confirmNotice(parts.confirm) : '');
    setShown('review-confirm', Boolean(parts.confirm));
    setReviewError(parts.blocked);
    setText('review-save', spec.saveLabel || 'Save');
    updateReviewSave();
    const firstWarning = byId('review-warnings').querySelector('input[type="checkbox"]');
    showDialog('review-dialog', firstWarning ? firstWarning.id : parts.blocked ? 'review-back' : 'review-save', spec.opener);
}

function closeReview() {
    hideDialog('review-dialog');
    S.review = null;
}

/** POST /api/settings/confirm: keeps a TRUSTED_ORIGINS / TRUST_PROXY_HEADERS change. */
async function confirmChange(confirm) {
    const result = await api('POST', '/api/settings/confirm', { revision: confirm.revision });
    const keys = (confirm.keys || []).join(' and ') || 'the proxy settings';
    if (result.ok) return { ok: true, message: `The change of ${keys} is confirmed and stays.` };
    if (result.data.code === 'confirm_expired') {
        return { ok: false, message: `The change of ${keys} came too late to be confirmed and was undone.` };
    }
    return {
        ok: false,
        message: `The change of ${keys} could not be confirmed from this page (${errorText(result)}). It is undone at `
            + `${formatClock(confirm.deadline)} unless it is confirmed from a page that still reaches the server.`,
    };
}

/** After a write the server took: confirm a proxy change at once, and say what happened. */
async function finishWrite(answer, what) {
    if (!answer.written) return { type: 'info', message: 'Nothing changed: these values were already in force.' };
    let type = 'success';
    let message = `${what}. In force now, in every worker.`;
    if (answer.confirm) {
        const confirmed = await confirmChange(answer.confirm);
        message += ` ${confirmed.message}`;
        if (!confirmed.ok) type = 'error';
    }
    if (answer.reload_main_ui) message += ' The main page shows the engine changes when it is loaded again.';
    return { type, message, appLink: Boolean(answer.reload_main_ui) };
}

/** A write refused: what the review dialog does next ({ done: false } keeps it open). */
function writeRefused(result) {
    const code = result.data.code;
    if (code === 'needs_confirmation') {
        addReviewWarnings(result.data.warnings);
        setReviewError('Something changed since the check: confirm the new warning first.');
        return { done: false };
    }
    if (code === 'would_lock_out') {
        if (S.review) S.review.blocked = true;
        setReviewError(errorText(result));
        return { done: false };
    }
    if (code === 'invalid_input' || code === 'locked_by_deployment') {
        return {
            done: true, reload: false, fieldErrors: result.data.errors || {}, type: 'error',
            message: `${errorText(result)} See the marked fields.`,
        };
    }
    if (code === 'revision_conflict') return { done: true, type: 'error', message: CONFLICT_TEXT };
    if (result.status === 401 || code === 'admin_key_required' || code === 'too_many_attempts') {
        return { done: true, type: 'error', message: errorText(result) };
    }
    setReviewError(errorText(result));
    return { done: false };
}

async function saveDraft(changes, acknowledge) {
    const messages = [];
    let type = 'success';
    let appLink = false;
    if (changes.preferences) {
        const result = await api('PUT', '/api/settings', {
            base_revision: S.view.revision, set: changes.set, reset: changes.reset, acknowledge,
        });
        if (!result.ok) return writeRefused(result);
        forgetSaved([...Object.keys(changes.set), ...changes.reset]);
        const written = await finishWrite(result.data, 'Saved');
        messages.push(written.message);
        appLink = written.appLink;
        if (written.type === 'error') type = 'error';
    }
    if (changes.accessChanged) {
        const result = await api('PUT', '/api/settings/access', { base_revision: S.view.keys_revision, ...changes.access });
        if (!result.ok) {
            if (!messages.length) return writeRefused(result);
            type = 'error';
            messages.push(`API access was not changed: ${errorText(result)}`);
        } else {
            forgetSaved(Object.keys(changes.access));
            messages.push('API access is changed, in force now.');
        }
    }
    return { done: true, type, message: messages.join(' '), appLink };
}

async function review(button) {
    if (!S.view || !S.draft.size) return;
    clearFieldErrors();
    const changes = pendingChanges();
    let preview = null;
    if (changes.preferences) {
        setStatus('info', 'Checking the changes...');
        preview = await api('PUT', '/api/settings', {
            base_revision: S.view.revision, set: changes.set, reset: changes.reset, dry_run: true,
        });
        const lockout = preview.status === 409 && preview.data.code === 'would_lock_out';
        if (!preview.ok && !lockout) {
            await showRefusal(preview);
            return;
        }
        setStatus(null, '');
        if (preview.ok && !(preview.data.changed || []).length && !changes.accessChanged) {
            // The server reads every edit as what is saved already (say http://Name:3000 for name).
            forgetSaved([...Object.keys(changes.set), ...changes.reset]);
            renderFields(null);
            setStatus('info', 'Nothing to save: the server reads these values as the ones in force.');
            return;
        }
    }
    openReview({
        title: 'Review the changes',
        intro: 'Saving puts these in force at once, in every worker.',
        rows: draftRows(),
        preview,
        saveLabel: 'Save',
        save: (acknowledge) => saveDraft(changes, acknowledge),
        opener: button,
    });
}

/** A refusal outside the review dialog: field errors, a reload after a conflict, or the message. */
async function showRefusal(result) {
    const outcome = writeRefused(result);
    if (outcome.done === false) {
        setStatus('error', errorText(result));
        return;
    }
    if (outcome.reload !== false) await loadView();
    const unplaced = outcome.fieldErrors ? showFieldErrors(outcome.fieldErrors) : [];
    setStatus('error', [outcome.message, ...unplaced].join(' '));
}

async function saveReview() {
    const spec = S.review;
    if (!spec || spec.blocked) return;
    setReviewError('');
    const outcome = await spec.save(reviewTokens());
    if (!outcome || outcome.done === false) return;
    closeReview();
    if (outcome.reload !== false) await loadView();
    const unplaced = outcome.fieldErrors ? showFieldErrors(outcome.fieldErrors) : [];
    setStatus(outcome.type, [outcome.message, ...unplaced].join(' '), outcome.appLink);
}

async function startRestore(button) {
    const historyId = button.dataset.historyId;
    const entry = S.view.history.find((item) => item.id === historyId);
    if (!entry) return;
    const preview = await api('POST', '/api/settings/restore', {
        history_id: historyId, base_revision: S.view.revision, dry_run: true,
    });
    const lockout = preview.status === 409 && preview.data.code === 'would_lock_out';
    if (!preview.ok && !lockout) {
        await showRefusal(preview);
        return;
    }
    const changed = preview.ok ? preview.data.changed || [] : entry.changed || [];
    openReview({
        title: `Restore revision ${entry.revision}`,
        intro: changed.length
            ? `These go back to the values saved ${formatTime(entry.saved_at)}:`
            : 'That version holds the values in force now: restoring it changes nothing.',
        rows: changed.map((key) => {
            const setting = settingFor(key);
            return setting
                ? el('div', { className: 'review-row' },
                    el('p', { className: 'review-label' }, el('strong', { text: setting.label }), ' ', el('code', { text: key })),
                    el('p', { className: 'review-change', text: `Now ${formatValue(setting, setting.value)}; back to the value of revision ${entry.revision}.` }))
                : el('p', { text: key });
        }),
        preview,
        saveLabel: 'Restore',
        save: async (acknowledge) => {
            const result = await api('POST', '/api/settings/restore', {
                history_id: historyId, base_revision: S.view.revision, acknowledge,
            });
            if (!result.ok) return writeRefused(result);
            return { done: true, ...(await finishWrite(result.data, `Revision ${entry.revision} is restored`)) };
        },
        opener: button,
    });
}

async function startDiscard(button) {
    const preview = await api('POST', '/api/settings/discard', { base_revision: S.view.revision, dry_run: true });
    const lockout = preview.status === 409 && preview.data.code === 'would_lock_out';
    if (!preview.ok && !lockout) {
        await showRefusal(preview);
        return;
    }
    const changed = preview.ok ? preview.data.changed || [] : [];
    openReview({
        title: 'Back to the app YAML',
        intro: 'Every value saved here is dropped and the app YAML values apply again. The dropped version stays '
            + 'in the history and can be restored. API keys are not affected.',
        rows: changed.map((key) => {
            const setting = settingFor(key);
            return setting ? changeRow(setting, setting.value, setting.yaml_value, null) : el('p', { text: key });
        }),
        preview,
        saveLabel: 'Back to the app YAML',
        save: async () => {
            const result = await api('POST', '/api/settings/discard', { base_revision: S.view.revision });
            if (!result.ok) return writeRefused(result);
            return { done: true, ...(await finishWrite(result.data, 'The app YAML values apply again')) };
        },
        opener: button,
    });
}

async function confirmPending() {
    const pending = S.view && S.view.pending;
    if (!pending) return;
    const outcome = await confirmChange(pending);
    await loadView();
    setStatus(outcome.ok ? 'success' : 'error', outcome.message);
}

// --- keys -------------------------------------------------------------------------------------

function openKeyDialog(opener) {
    S.newKey = { key: generateKey(), saved: null };
    const name = byId('new-key-name');
    if (name) name.value = '';
    const client = byId('new-key-role-client');
    if (client) client.checked = true;
    const stored = byId('new-key-stored');
    if (stored) stored.checked = false;
    const value = byId('new-key-value');
    if (value) value.value = S.newKey.key;
    setText('new-key-copy-status', '');
    setText('key-dialog-error', '');
    setShown('key-dialog-error', false);
    setShown('key-dialog-form', true);
    setShown('key-dialog-done', false);
    updateKeyDialog();
    showDialog('key-dialog', 'new-key-name', opener);
}

function selectedRole() {
    const admin = byId('new-key-role-admin');
    return admin && admin.checked ? 'admin' : 'client';
}

/** Save stays off until the key has a name and its owner has ticked "I have stored it". */
function updateKeyDialog() {
    const save = byId('key-dialog-save');
    if (!save || save.getAttribute('aria-busy') === 'true') return;
    const name = byId('new-key-name');
    const stored = byId('new-key-stored');
    save.disabled = !(S.newKey && !S.newKey.saved && name && name.value.trim() && stored && stored.checked);
}

async function saveNewKey() {
    if (!S.newKey || S.newKey.saved) return;
    const name = byId('new-key-name').value.trim();
    const role = selectedRole();
    const result = await api('POST', '/api/settings/keys', { name, role, key: S.newKey.key });
    if (!result.ok) {
        const errors = result.data.errors ? Object.values(result.data.errors).join(' ') : '';
        setText('key-dialog-error', errors || errorText(result, 'The key was not saved.'));
        setShown('key-dialog-error', true);
        return;
    }
    S.newKey.saved = { name, role, hint: result.data.hint };
    setText('key-dialog-done-text', `The key "${name}" (${role}) is saved and works from now on. The server keeps `
        + `only its fingerprint; it shows the key as ${result.data.hint}${ELLIPSIS} from now on.`);
    setShown('use-new-key', role === 'admin');
    setShown('key-dialog-form', false);
    setShown('key-dialog-done', true);
    const close = byId('key-dialog-done').querySelector('button[data-action="key-dialog-close"]');
    if (close && typeof close.focus === 'function') close.focus();
    await loadView();
}

async function useNewKey() {
    if (!S.newKey || !S.newKey.saved) return;
    storeKey(S.newKey.key);
    hideDialog('key-dialog');
    if (await loadView()) setStatus('success', 'This tab now uses the new key.');
}

async function revokeKey(button) {
    const id = button.dataset.keyId;
    const record = S.view.keys.find((item) => item.id === id);
    if (!record) return;
    const inUse = Boolean(S.view.credential && S.view.credential.id === id);
    const question = inUse
        ? `Revoke "${record.name}"? This tab uses it, so this page asks for another admin key afterwards.`
        : `Revoke "${record.name}"? Whatever uses it is refused from the next request on.`;
    if (!window.confirm(question)) return;
    const result = await api('DELETE', `/api/settings/keys/${encodeURIComponent(id)}`);
    if (!result.ok) {
        setStatus('error', errorText(result, 'The key was not revoked.'));
        return;
    }
    if (inUse) storeKey('');
    await loadView();
    setStatus('success', `The key "${record.name}" is revoked.`);
}

async function copyFrom(inputId, statusId) {
    const input = byId(inputId);
    if (!input || !input.value) return;
    let copied = false;
    try {
        if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(input.value);
            copied = true;
        }
    } catch {
        copied = false;
    }
    if (!copied) {
        // Plain http (a NAS on the LAN) has no clipboard API; the selection still copies.
        try {
            input.focus();
            input.select();
            copied = Boolean(document.execCommand && document.execCommand('copy'));
        } catch {
            copied = false;
        }
    }
    setText(statusId, copied ? 'Copied.' : 'The key is selected: copy it with Ctrl+C (Cmd+C on a Mac).');
}

// --- wiring -------------------------------------------------------------------------------------

function refreshGatedButtons() {
    updateReviewSave();
    updateKeyDialog();
    updateClaimButton();
}

const ACTIONS = {
    'use-key': () => useKey(),
    'change-key': () => showKeyPanel('Enter the admin key to use in this tab.'),
    'forget-key': () => forgetKey(),
    // Unclaimed, the same code claims the server.
    'start-recovery': () => openClaim(S.view && !S.view.claimed ? 'claim' : 'recovery'),
    'print-code': (button) => withBusy(button, printCode),
    'claim': (button) => withBusy(button, submitClaim),
    'cancel-claim': () => closeClaim(),
    'copy-claim-key': () => copyFrom('claim-key', 'claim-copy-status'),
    'reset-field': (button) => resetField(button.dataset.key),
    'undo-reset': (button) => undoReset(button.dataset.key),
    'undo-draft': () => undoDraft(),
    'review': (button) => withBusy(button, review),
    'review-save': (button) => withBusy(button, saveReview),
    'review-cancel': () => closeReview(),
    'confirm-pending': (button) => withBusy(button, confirmPending),
    'recheck-engines': (button) => withBusy(button, loadEngines),
    'restore': (button) => withBusy(button, () => startRestore(button)),
    'discard': (button) => withBusy(button, startDiscard),
    'create-key': (button) => openKeyDialog(button),
    'copy-key': () => copyFrom('new-key-value', 'new-key-copy-status'),
    'key-dialog-save': (button) => withBusy(button, saveNewKey),
    'key-dialog-close': () => hideDialog('key-dialog'),
    'use-new-key': (button) => withBusy(button, useNewKey),
    'revoke-key': (button) => withBusy(button, () => revokeKey(button)),
};

const WATCHERS = {
    'claim': () => updateClaimButton(),
    'key-dialog': () => updateKeyDialog(),
    'review': () => updateReviewSave(),
};

function onClick(event) {
    const target = event.target && typeof event.target.closest === 'function'
        ? event.target.closest('[data-action]') : null;
    if (!target || target.disabled) return;
    const handler = ACTIONS[target.dataset.action];
    if (!handler) return;
    event.preventDefault();
    handler(target);
}

function onInput(event) {
    const target = event.target;
    if (!target || typeof target.closest !== 'function') return;
    const field = target.closest('[data-field]');
    if (field) {
        onFieldInput(field);
        return;
    }
    const watched = target.closest('[data-watch]');
    const watcher = watched ? WATCHERS[watched.dataset.watch] : null;
    if (watcher) watcher(watched);
}

function onKeydown(event) {
    if (event.key !== 'Enter') return;
    const target = event.target;
    const action = target && target.dataset ? target.dataset.enter : null;
    if (!action || !ACTIONS[action]) return;
    event.preventDefault();
    ACTIONS[action](target);
}

function onBeforeUnload(event) {
    if (!S.draft.size) return;
    event.preventDefault();
    event.returnValue = '';   // older browsers want a value to show their "leave the page?" question
}

function start() {
    document.addEventListener('click', onClick);
    document.addEventListener('input', onInput);
    document.addEventListener('change', onInput);
    document.addEventListener('keydown', onKeydown);
    // Escape closes a dialog without any of the page's buttons: tidy up then as well.
    const reviewDialog = byId('review-dialog');
    if (reviewDialog) reviewDialog.addEventListener('close', () => onDialogClosed('review-dialog'));
    const keyDialog = byId('key-dialog');
    if (keyDialog) keyDialog.addEventListener('close', () => onDialogClosed('key-dialog'));
    window.addEventListener('beforeunload', onBeforeUnload);
    return loadView();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();
