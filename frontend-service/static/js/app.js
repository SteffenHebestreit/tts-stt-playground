// Global variables
let currentTTSEngine = 'piper'; // 'piper' or 'qwen3'

// TTS Engine Switching
function switchTTSEngine(engine) {
    currentTTSEngine = engine;

    const piperElements = document.querySelectorAll('.piper-only');
    const qwen3Elements = document.querySelectorAll('.qwen3-only');

    piperElements.forEach(el => el.style.display = 'none');
    qwen3Elements.forEach(el => el.style.display = 'none');

    if (engine === 'piper') {
        piperElements.forEach(el => el.style.display = '');
        const currentTab = document.querySelector('.tab-content.active');
        if (currentTab && currentTab.classList.contains('qwen3-only')) {
            showTab('stt-tab');
        }
    } else if (engine === 'qwen3') {
        qwen3Elements.forEach(el => el.style.display = '');
        const currentTab = document.querySelector('.tab-content.active');
        if (currentTab && currentTab.classList.contains('piper-only')) {
            showTab('qwen3-tts-tab');
        }
    }

    updateServiceStatus();
}

// Notification system
function showNotification(message, type = 'info') {
    let notificationContainer = document.getElementById('notification-container');
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notification-container';
        notificationContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 350px;
        `;
        document.body.appendChild(notificationContainer);
    }

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 14px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        animation: slideInRight 0.3s ease-out;
        background-color: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
    `;
    notification.textContent = message;

    notification.addEventListener('click', () => notification.remove());
    notificationContainer.appendChild(notification);

    setTimeout(() => {
        if (notification.parentNode) notification.remove();
    }, 5000);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkServiceHealth();
});

function initializeApp() {
    setupFileDragDrop();
    setupRangeSliders();
    refreshTTSVoices();
    switchTTSEngine(currentTTSEngine);

    setTimeout(() => {
        checkServiceHealth();
        if (currentTTSEngine === 'qwen3') {
            updateQwen3TTSStatus();
        }
    }, 1000);

    setInterval(() => {
        checkServiceHealth();
        if (currentTTSEngine === 'qwen3') {
            updateQwen3TTSStatus();
        }
    }, 30000);
}

function setupEventListeners() {
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const match = this.getAttribute('onclick').match(/'([^']+)'/);
            if (match) showTab(match[1]);
        });
    });
}

function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));

    const tabEl = document.getElementById(tabId);
    if (tabEl) tabEl.classList.add('active');

    const btn = document.querySelector(`.tab-button[onclick*="${tabId}"]`);
    if (btn) btn.classList.add('active');

    // Tab-specific initialization
    if (tabId === 'training-tab') {
        refreshModels();
        refreshTrainingJobs();
    } else if (tabId === 'tts-tab') {
        refreshTTSVoices();
        refreshCustomVoices();
    } else if (tabId === 'qwen3-tts-tab') {
        updateQwen3TTSStatus();
        loadQwen3Models();
    } else if (tabId === 'qwen3-cloning-tab') {
        loadQwen3Models();
        loadSavedVoices();
    }
}

// File drag and drop
function setupFileDragDrop() {
    const dropZones = [
        { zone: 'stt-drop-zone', input: 'stt-file', callback: () => handleSTTFile(document.getElementById('stt-file')) },
        { zone: 'training-drop-zone', input: 'training-files', callback: () => handleTrainingFiles(document.getElementById('training-files')) },
        { zone: 'qwen3-voice-drop-zone', input: 'qwen3-voice-file', callback: handleQwen3VoiceFile }
    ];

    dropZones.forEach(({ zone, input, callback }) => {
        const dropZone = document.getElementById(zone);
        const fileInput = document.getElementById(input);
        if (!dropZone || !fileInput) return;

        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('drag-over');
        });

        dropZone.addEventListener('dragenter', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            if (!this.contains(e.relatedTarget)) this.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                callback(files[0]);
            }
        });

        dropZone.addEventListener('click', function(e) {
            if (e.target !== fileInput) fileInput.click();
        });

        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) callback(this.files[0]);
        });
    });
}

function setupRangeSliders() {
    const speedSlider = document.getElementById('tts-speed');
    const speedValue = document.getElementById('tts-speed-value');
    if (speedSlider && speedValue) {
        speedSlider.addEventListener('input', function() {
            speedValue.textContent = this.value + 'x';
        });
    }
}

// ============================================================
// Service Health Checks
// ============================================================

async function checkServiceHealth() {
    const services = [
        { name: 'tts', url: window.TTS_SERVICE_URL, element: 'tts-status' },
        { name: 'stt', url: window.STT_SERVICE_URL, element: 'stt-status-indicator' },
        { name: 'training', url: window.VOICE_TRAINING_URL, element: 'training-status' },
        { name: 'qwen3-tts', url: window.QWEN3_TTS_SERVICE_URL, element: 'qwen3-tts-status' },
        { name: 'qwen3-asr', url: window.QWEN3_ASR_SERVICE_URL, element: 'qwen3-asr-status' }
    ];

    for (const service of services) {
        const element = document.getElementById(service.element);
        if (!element) continue;

        element.classList.remove('healthy', 'error');
        element.classList.add('loading');

        try {
            const timeout = service.name === 'training' ? 15000 : 5000;
            const response = await fetch(`${service.url}/health`, {
                signal: AbortSignal.timeout(timeout)
            });
            element.classList.remove('loading');
            element.classList.add(response.ok ? 'healthy' : 'error');
        } catch {
            element.classList.remove('loading');
            element.classList.add('error');
        }
    }
}

function updateServiceStatus() {
    checkServiceHealth();
}

// Helper functions
function showStatus(elementId, type, message) {
    const element = document.getElementById(elementId);
    if (element) element.innerHTML = `<div class="${type}">${message}</div>`;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function toggleRefText() {
    const checked = document.getElementById('enable-ref-text').checked;
    document.getElementById('ref-text-group').style.display = checked ? '' : 'none';
}

// ============================================================
// PiperTTS Functions
// ============================================================

async function generateTTS() {
    const text = document.getElementById('tts-text').value.trim();
    const language = document.getElementById('tts-language-select').value;
    const quality = document.getElementById('tts-quality-select').value;
    const gender = document.getElementById('tts-gender-select').value;
    const voice = document.getElementById('tts-voice-select').value;
    const speed = parseFloat(document.getElementById('tts-speed').value);
    const audioPlayer = document.getElementById('tts-audio-player');

    if (!text) {
        showStatus('tts-result-status', 'error', 'Please enter some text to synthesize');
        return;
    }

    try {
        showStatus('tts-result-status', 'info', 'Generating speech with PiperTTS...');
        audioPlayer.innerHTML = '';

        const requestData = { text, speed, output_format: 'wav' };
        if (voice !== 'auto') requestData.voice = voice;
        if (language !== 'auto') requestData.language = language;
        if (quality !== 'medium') requestData.quality = quality;
        if (gender !== 'any') requestData.gender = gender;

        const response = await fetch(`${window.TTS_SERVICE_URL}/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `TTS generation failed: ${response.statusText}`);
        }

        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);

        audioPlayer.innerHTML = `
            <audio controls style="width: 100%;">
                <source src="${audioUrl}" type="audio/wav">
            </audio>
        `;

        showStatus('tts-result-status', 'success', 'Speech generated successfully!');
    } catch (error) {
        console.error('TTS generation error:', error);
        showStatus('tts-result-status', 'error', `Generation failed: ${error.message}`);
    }
}

async function refreshTTSVoices() {
    try {
        const response = await fetch(`${window.TTS_SERVICE_URL}/voices`);
        const data = await response.json();

        const voiceSelect = document.getElementById('tts-voice-select');
        voiceSelect.innerHTML = '<option value="auto">Auto-Select Best Voice</option>';

        if (data.voices) {
            Object.keys(data.voices).forEach(voiceId => {
                const voice = data.voices[voiceId];
                const option = document.createElement('option');
                option.value = voiceId;
                const quality = voice.quality ? ` (${voice.quality})` : '';
                option.textContent = voice.language
                    ? `${voice.language} - ${voice.speaker || voice.name}${quality}`
                    : voice.name;
                voiceSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error refreshing voices:', error);
    }
}

// Custom voice management for PiperTTS
async function refreshCustomVoices() {
    const container = document.getElementById('custom-voices-list');
    if (!container) return;

    try {
        container.innerHTML = '<p>Loading voices...</p>';
        const response = await fetch(`${window.TTS_SERVICE_URL}/voices`);
        const data = await response.json();

        if (!data.voices) {
            container.innerHTML = '<p>No voices available.</p>';
            return;
        }

        // Filter custom voices (those that are user-trained, not default)
        const customVoices = [];
        const defaultVoices = [];

        Object.keys(data.voices).forEach(voiceId => {
            const voice = data.voices[voiceId];
            if (voice.model_type === 'custom') {
                customVoices.push({ id: voiceId, ...voice });
            } else {
                defaultVoices.push({ id: voiceId, ...voice });
            }
        });

        if (customVoices.length === 0) {
            container.innerHTML = '<p>No custom trained voices found. Train a voice model and it will appear here.</p>';
            return;
        }

        let html = '<div class="voices-grid">';
        customVoices.forEach(voice => {
            html += `
                <div class="voice-card">
                    <div class="voice-header">
                        <h4>${voice.name || voice.id}</h4>
                        <span class="voice-id">${voice.id}</span>
                    </div>
                    <div class="voice-info">
                        ${voice.language ? `<p>Language: ${voice.language}</p>` : ''}
                        ${voice.quality ? `<p>Quality: ${voice.quality}</p>` : ''}
                    </div>
                    <div class="voice-actions">
                        <button class="btn-secondary" onclick="testVoice('${voice.id}', '${(voice.language || 'en').split('_')[0]}')">Test</button>
                        <button class="btn-secondary" onclick="deleteCustomVoice('${voice.id}')" style="color: var(--error);">Delete</button>
                    </div>
                    <div id="voice-test-${voice.id}" class="audio-player"></div>
                </div>
            `;
        });
        html += '</div>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error refreshing custom voices:', error);
        container.innerHTML = '<p style="color: var(--error);">Failed to load voices. Is the PiperTTS service running?</p>';
    }
}

async function testVoice(voiceId, lang = 'en') {
    const playerDiv = document.getElementById(`voice-test-${voiceId}`);
    if (!playerDiv) return;

    try {
        playerDiv.innerHTML = '<div class="info" style="padding: 8px; font-size: 0.9rem;">Generating test audio...</div>';

        const testTexts = {
            'de': 'Hallo, das ist ein Test dieser Stimme.',
            'en': 'Hello, this is a test of this voice.',
            'fr': 'Bonjour, ceci est un test de cette voix.',
            'es': 'Hola, esta es una prueba de esta voz.',
        };
        const baseLang = lang.split('-')[0];
        const testText = testTexts[baseLang] || testTexts['en'];
        const response = await fetch(`${window.TTS_SERVICE_URL}/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: testText, voice: voiceId, output_format: 'wav' })
        });

        if (!response.ok) throw new Error('Test generation failed');

        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        playerDiv.innerHTML = `<audio controls autoplay style="width: 100%; margin-top: 8px;"><source src="${audioUrl}" type="audio/wav"></audio>`;
    } catch (error) {
        playerDiv.innerHTML = `<div class="error" style="padding: 8px; font-size: 0.9rem;">Test failed: ${error.message}</div>`;
    }
}

async function deleteCustomVoice(voiceId) {
    if (!confirm(`Delete custom voice "${voiceId}"? This cannot be undone.`)) return;

    try {
        const response = await fetch(`${window.TTS_SERVICE_URL}/voice/${voiceId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Delete failed');

        showNotification(`Voice "${voiceId}" deleted`, 'success');
        refreshCustomVoices();
        refreshTTSVoices();
    } catch (error) {
        console.error('Delete voice error:', error);
        showNotification(`Failed to delete voice: ${error.message}`, 'error');
    }
}

// ============================================================
// Qwen3-TTS Functions
// ============================================================

// Model management
async function loadQwen3Models() {
    const selects = [
        { select: document.getElementById('qwen3-model-select'), desc: document.getElementById('qwen3-model-description') },
        { select: document.getElementById('qwen3-clone-model-select'), desc: document.getElementById('qwen3-clone-model-description') }
    ].filter(s => s.select);

    if (selects.length === 0) return;

    try {
        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/models`, {
            signal: AbortSignal.timeout(5000)
        });
        if (!response.ok) throw new Error('Failed to fetch models');

        const data = await response.json();

        selects.forEach(({ select, desc }) => {
            select.innerHTML = '';
            Object.entries(data.models).forEach(([modelId, info]) => {
                const option = document.createElement('option');
                option.value = modelId;
                option.textContent = `${info.name} - ${info.description}`;
                if (modelId === data.current_model) option.selected = true;
                select.appendChild(option);
            });

            const currentInfo = data.models[data.current_model];
            if (desc && currentInfo) {
                desc.textContent = `Current: ${currentInfo.name} | Capabilities: ${currentInfo.capabilities.join(', ')}`;
            }
        });
    } catch {
        selects.forEach(({ select, desc }) => {
            select.innerHTML = '<option value="">Service unavailable</option>';
            if (desc) desc.textContent = '';
        });
    }
}

async function switchQwen3Model(modelId) {
    if (!modelId) return;

    // Both tabs have model selectors — disable all during switch
    const elements = [
        { select: document.getElementById('qwen3-model-select'), loading: document.getElementById('qwen3-model-loading') },
        { select: document.getElementById('qwen3-clone-model-select'), loading: document.getElementById('qwen3-clone-model-loading') }
    ];

    try {
        elements.forEach(({ select, loading }) => {
            if (loading) loading.style.display = '';
            if (select) select.disabled = true;
        });

        showNotification('Switching model... This may take a while if the model needs to download.', 'info');

        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/load_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelId }),
            signal: AbortSignal.timeout(300000) // 5 min timeout for download
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: 'Switch failed' }));
            throw new Error(err.detail || 'Model switch failed');
        }

        const result = await response.json();
        showNotification(`Model switched to ${result.model_info?.name || modelId}`, 'success');

        // Refresh both dropdowns and status
        loadQwen3Models();
        updateQwen3TTSStatus();
    } catch (error) {
        console.error('Model switch error:', error);
        showNotification(`Failed to switch model: ${error.message}`, 'error');
        loadQwen3Models();
    } finally {
        elements.forEach(({ select, loading }) => {
            if (loading) loading.style.display = 'none';
            if (select) select.disabled = false;
        });
    }
}

// Track current cloning mode and voice source
let currentCloneMode = 'saved'; // 'saved', 'audio', 'design', or 'unsupported'
let currentVoiceSource = 'saved'; // 'saved' or 'upload'

function switchVoiceSource(source) {
    currentVoiceSource = source;
    const savedMode = document.getElementById('clone-saved-mode');
    const audioMode = document.getElementById('clone-audio-mode');

    if (source === 'saved') {
        currentCloneMode = 'saved';
        if (savedMode) savedMode.style.display = '';
        if (audioMode) audioMode.style.display = 'none';
    } else {
        currentCloneMode = 'audio';
        if (savedMode) savedMode.style.display = 'none';
        if (audioMode) audioMode.style.display = '';
    }
}

function onCloneModelChange(modelId) {
    const isVoiceDesign = modelId.includes('VoiceDesign');
    const isCustomVoice = modelId.includes('CustomVoice');

    const savedMode = document.getElementById('clone-saved-mode');
    const audioMode = document.getElementById('clone-audio-mode');
    const designMode = document.getElementById('clone-design-mode');
    const voiceSource = document.getElementById('clone-voice-source');
    const title = document.getElementById('qwen3-cloning-title');
    const desc = document.getElementById('qwen3-cloning-description');
    const btn = document.getElementById('generate-qwen3-speech-btn');

    if (isVoiceDesign) {
        currentCloneMode = 'design';
        if (voiceSource) voiceSource.style.display = 'none';
        if (savedMode) savedMode.style.display = 'none';
        if (audioMode) audioMode.style.display = 'none';
        if (designMode) designMode.style.display = '';
        if (title) title.textContent = 'Voice Design (Qwen3-TTS)';
        if (desc) desc.textContent = 'Describe the voice you want using text and generate speech with that designed voice.';
        if (btn) { btn.textContent = 'Design & Generate'; btn.disabled = false; }
    } else if (isCustomVoice) {
        currentCloneMode = 'unsupported';
        if (voiceSource) voiceSource.style.display = 'none';
        if (savedMode) savedMode.style.display = 'none';
        if (audioMode) audioMode.style.display = 'none';
        if (designMode) designMode.style.display = 'none';
        if (title) title.textContent = 'Voice Cloning (Qwen3-TTS)';
        if (desc) desc.textContent = 'The CustomVoice model uses built-in speakers only and does not support voice cloning or voice design. Switch to a Base model (1.7B or 0.6B) for cloning, or use VoiceDesign for text-described voices.';
        if (btn) { btn.textContent = 'Generate Speech'; btn.disabled = true; }
    } else {
        // Base model — show voice source toggle
        if (voiceSource) voiceSource.style.display = '';
        if (designMode) designMode.style.display = 'none';
        if (title) title.textContent = 'Voice Cloning (Qwen3-TTS)';
        if (desc) desc.textContent = 'Use a saved voice for fast TTS, or upload a new sample to clone.';
        if (btn) { btn.textContent = 'Generate Speech'; btn.disabled = false; }
        // Restore saved/upload mode
        switchVoiceSource(currentVoiceSource);
        loadSavedVoices();
    }

    // Also trigger the model switch
    switchQwen3Model(modelId);
}

// --- Saved Voices ---

async function loadSavedVoices() {
    const select = document.getElementById('qwen3-saved-voice-select');
    const info = document.getElementById('saved-voice-info');
    if (!select) return;

    try {
        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/voices`);
        if (!response.ok) throw new Error('Failed to load voices');
        const data = await response.json();
        const voices = data.voices || [];

        select.innerHTML = '';
        if (voices.length === 0) {
            select.innerHTML = '<option value="">No saved voices — upload a sample first</option>';
            if (info) info.textContent = '';
            return;
        }

        voices.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.id;
            opt.textContent = v.name || v.id;
            opt.dataset.refText = v.ref_text || '';
            opt.dataset.lang = v.lang || '';
            opt.dataset.createdAt = v.created_at || '';
            select.appendChild(opt);
        });

        // Show info for first voice
        updateSavedVoiceInfo();
        select.onchange = updateSavedVoiceInfo;
    } catch (err) {
        console.error('Load saved voices error:', err);
        select.innerHTML = '<option value="">Error loading voices</option>';
    }
}

function updateSavedVoiceInfo() {
    const select = document.getElementById('qwen3-saved-voice-select');
    const info = document.getElementById('saved-voice-info');
    if (!select || !info) return;
    const opt = select.selectedOptions[0];
    if (opt && opt.value) {
        const parts = [];
        if (opt.dataset.refText) parts.push(`Ref: "${opt.dataset.refText.substring(0, 80)}..."`);
        if (opt.dataset.createdAt) parts.push(`Saved: ${opt.dataset.createdAt}`);
        info.textContent = parts.join(' | ');
    } else {
        info.textContent = '';
    }
}

async function deleteSavedVoice() {
    const select = document.getElementById('qwen3-saved-voice-select');
    if (!select || !select.value) {
        showNotification('No voice selected to delete.', 'error');
        return;
    }
    const voiceId = select.value;
    const voiceName = select.selectedOptions[0]?.textContent || voiceId;
    if (!confirm(`Delete saved voice "${voiceName}"?`)) return;

    try {
        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/voices/${voiceId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Delete failed');
        showNotification(`Voice "${voiceName}" deleted.`, 'success');
        loadSavedVoices();
    } catch (err) {
        showNotification(`Failed to delete voice: ${err.message}`, 'error');
    }
}

async function saveVoiceFromUpload(voiceFile, name) {
    const lang = document.getElementById('qwen3-tts-language-select')?.value || 'auto';
    const formData = new FormData();
    formData.append('name', name);
    formData.append('lang', lang);
    formData.append('file', voiceFile);

    showStatus('qwen3-generation-status', 'info', `Saving voice "${name}" (transcribing + extracting embedding)...`);

    const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/voices/save`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(120000),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Save failed' }));
        throw new Error(err.detail || 'Failed to save voice');
    }

    const result = await response.json();
    showNotification(`Voice "${name}" saved! Use it from "Saved Voices" for fast TTS.`, 'success');
    loadSavedVoices();
    return result;
}

// Built-in speaker TTS
async function generateQwen3BuiltinTTS() {
    const text = document.getElementById('qwen3-builtin-text').value.trim();
    const lang = document.getElementById('qwen3-builtin-language').value;
    const speaker = document.getElementById('qwen3-builtin-speaker').value;
    const instruct = document.getElementById('qwen3-builtin-instruct').value.trim();
    const audioPlayer = document.getElementById('qwen3-builtin-audio-player');

    if (!text) {
        showStatus('qwen3-builtin-status', 'error', 'Please enter some text to synthesize');
        return;
    }

    try {
        showStatus('qwen3-builtin-status', 'info', `Generating speech with ${speaker}...`);
        audioPlayer.innerHTML = '';

        const startTime = Date.now();

        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, lang, speaker, instruct }),
            signal: AbortSignal.timeout(60000)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Generation failed: ${response.statusText}`);
        }

        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        audioPlayer.innerHTML = `
            <audio controls autoplay style="width: 100%;">
                <source src="${audioUrl}" type="audio/wav">
            </audio>
        `;

        showStatus('qwen3-builtin-status', 'success', `Speech generated in ${duration}s (Speaker: ${speaker})`);
    } catch (error) {
        console.error('Qwen3-TTS generation error:', error);
        showStatus('qwen3-builtin-status', 'error', `Generation failed: ${error.message}`);
    }
}

// Voice cloning
function handleQwen3VoiceFile(file) {
    if (!file) return;
    const infoDiv = document.getElementById('qwen3-voice-file-info');
    infoDiv.innerHTML = `
        <strong>Selected:</strong> ${file.name}<br>
        <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
        <strong>Type:</strong> ${file.type}
    `;
    infoDiv.style.display = 'block';
}

async function generateQwen3TTS() {
    // Route to the correct handler based on current mode
    if (currentCloneMode === 'design') {
        return generateQwen3VoiceDesign();
    }
    if (currentCloneMode === 'saved') {
        return generateWithSavedVoice();
    }
    return generateQwen3VoiceClone();
}

async function generateWithSavedVoice() {
    const text = document.getElementById('qwen3-tts-text').value.trim();
    const lang = document.getElementById('qwen3-tts-language-select').value;
    const voiceId = document.getElementById('qwen3-saved-voice-select')?.value;
    const audioPlayer = document.getElementById('qwen3-audio-player');
    const generateBtn = document.getElementById('generate-qwen3-speech-btn');

    if (!text) {
        showStatus('qwen3-generation-status', 'error', 'Please enter some text.');
        return;
    }
    if (!voiceId) {
        showStatus('qwen3-generation-status', 'error', 'No saved voice selected. Upload a sample first.');
        return;
    }

    const formData = new FormData();
    formData.append('text', text);
    formData.append('lang', lang);

    try {
        showStatus('qwen3-generation-status', 'info', 'Generating speech with saved voice...');
        audioPlayer.innerHTML = '';
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';

        const startTime = Date.now();

        let progressInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            showStatus('qwen3-generation-status', 'info', `Generating speech... ${elapsed}s`);
        }, 500);

        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/voices/${voiceId}/tts`, {
            method: 'POST',
            body: formData,
            signal: AbortSignal.timeout(600000),
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Generation failed: ${response.statusText}`);
        }

        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        const genTime = response.headers.get('X-Generation-Time');
        const audioDur = response.headers.get('X-Audio-Duration');

        audioPlayer.innerHTML = `
            <audio controls autoplay style="width: 100%;">
                <source src="${audioUrl}" type="audio/wav">
            </audio>
        `;

        let statusMsg = `Speech generated in ${duration}s`;
        if (audioDur) statusMsg += ` (${parseFloat(audioDur).toFixed(1)}s audio)`;
        showStatus('qwen3-generation-status', 'success', statusMsg);
    } catch (error) {
        console.error('Saved voice TTS error:', error);
        showStatus('qwen3-generation-status', 'error', `Generation failed: ${error.message}`);
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Speech';
    }
}

async function generateQwen3VoiceClone() {
    const text = document.getElementById('qwen3-tts-text').value.trim();
    const lang = document.getElementById('qwen3-tts-language-select').value;
    const voiceFile = document.getElementById('qwen3-voice-file').files[0];
    const audioPlayer = document.getElementById('qwen3-audio-player');
    const generateBtn = document.getElementById('generate-qwen3-speech-btn');
    const useRefText = document.getElementById('enable-ref-text').checked;
    const refText = document.getElementById('qwen3-ref-text')?.value.trim();
    const saveName = document.getElementById('save-voice-name')?.value.trim();

    if (!text) {
        showStatus('qwen3-generation-status', 'error', 'Please enter some text.');
        return;
    }
    if (!voiceFile) {
        showStatus('qwen3-generation-status', 'error', 'Please select a voice sample file.');
        return;
    }
    if (useRefText && !refText) {
        showStatus('qwen3-generation-status', 'error', 'Please enter the reference audio transcript or uncheck the option.');
        return;
    }

    try {
        generateBtn.disabled = true;
        generateBtn.textContent = 'Processing...';
        audioPlayer.innerHTML = '';
        const startTime = Date.now();

        // Step 1: If save name provided, save the voice first (extracts embedding)
        if (saveName) {
            await saveVoiceFromUpload(voiceFile, saveName);
        }

        // Step 2: Generate the speech
        const formData = new FormData();
        formData.append('text', text);
        formData.append('lang', lang);
        formData.append('file', voiceFile);
        if (useRefText && refText) {
            formData.append('ref_text', refText);
        }

        const endpoint = useRefText && refText ? '/clone-with-ref-text' : '/clone';
        const autoTranscribing = !useRefText || !refText;
        const statusMsg = autoTranscribing
            ? 'Auto-transcribing reference audio via Qwen3-ASR, then cloning...'
            : 'Cloning and generating speech...';
        showStatus('qwen3-generation-status', 'info', statusMsg);

        let progressInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            const phase = autoTranscribing && elapsed < 10
                ? `Auto-transcribing + cloning... ${elapsed}s`
                : `Generating voice clone... ${elapsed}s`;
            showStatus('qwen3-generation-status', 'info', phase);
        }, 500);

        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}${endpoint}`, {
            method: 'POST',
            body: formData,
            signal: AbortSignal.timeout(600000)
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Voice cloning failed: ${response.statusText}`);
        }

        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        audioPlayer.innerHTML = `
            <audio controls autoplay style="width: 100%;">
                <source src="${audioUrl}" type="audio/wav">
            </audio>
        `;

        let msg = `Voice cloning completed in ${duration}s`;
        if (saveName) msg += ` (voice "${saveName}" saved for fast reuse)`;
        showStatus('qwen3-generation-status', 'success', msg);
    } catch (error) {
        console.error('Qwen3-TTS cloning error:', error);
        showStatus('qwen3-generation-status', 'error', `Generation failed: ${error.message}`);
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Speech';
    }
}

async function generateQwen3VoiceDesign() {
    const text = document.getElementById('qwen3-tts-text').value.trim();
    const lang = document.getElementById('qwen3-tts-language-select').value;
    const voiceDescription = document.getElementById('qwen3-voice-description').value.trim();
    const audioPlayer = document.getElementById('qwen3-audio-player');
    const generateBtn = document.getElementById('generate-qwen3-speech-btn');

    if (!text) {
        showStatus('qwen3-generation-status', 'error', 'Please enter some text to synthesize.');
        return;
    }
    if (!voiceDescription) {
        showStatus('qwen3-generation-status', 'error', 'Please describe the voice you want.');
        return;
    }

    try {
        showStatus('qwen3-generation-status', 'info', 'Designing voice and generating speech...');
        audioPlayer.innerHTML = '';
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';

        const startTime = Date.now();

        let progressInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            showStatus('qwen3-generation-status', 'info', `Designing voice... ${elapsed}s`);
        }, 500);

        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/voice_design`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, lang, voice_description: voiceDescription }),
            signal: AbortSignal.timeout(600000)
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Voice design failed: ${response.statusText}`);
        }

        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        audioPlayer.innerHTML = `
            <audio controls autoplay style="width: 100%;">
                <source src="${audioUrl}" type="audio/wav">
            </audio>
        `;

        showStatus('qwen3-generation-status', 'success', `Voice design completed in ${duration}s`);
    } catch (error) {
        console.error('Voice design error:', error);
        showStatus('qwen3-generation-status', 'error', `Generation failed: ${error.message}`);
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Design & Generate';
    }
}

async function updateQwen3TTSStatus() {
    try {
        const response = await fetch(`${window.QWEN3_TTS_SERVICE_URL}/status`, {
            signal: AbortSignal.timeout(5000)
        });

        if (response.ok) {
            const status = await response.json();
            updateQwen3TTSStatusDisplay(status);
        } else {
            updateQwen3TTSStatusDisplay(null);
        }
    } catch {
        updateQwen3TTSStatusDisplay(null);
    }
}

function updateQwen3TTSStatusDisplay(status) {
    const statusElement = document.getElementById('qwen3-tts-system-status');
    if (!statusElement) return;

    if (!status) {
        statusElement.innerHTML = '<div class="status-error">Qwen3-TTS Service Unavailable</div>';
        return;
    }

    const memoryMB = status.gpu_memory_allocated ? Math.round(status.gpu_memory_allocated / (1024 * 1024)) : 0;
    const memoryGB = (memoryMB / 1024).toFixed(1);
    const speakers = status.builtin_speakers ? status.builtin_speakers.join(', ') : '';

    const modelName = status.current_model_info?.name || status.current_model || 'Unknown';

    statusElement.innerHTML = `
        <div class="status-success">
            <h4>Qwen3-TTS Service Online</h4>
            <div class="status-grid">
                <div class="status-item">
                    <strong>Device:</strong> ${status.device?.toUpperCase() || 'Unknown'}
                    ${status.cuda_available ? '(GPU)' : '(CPU)'}
                </div>
                <div class="status-item">
                    <strong>Model:</strong> ${status.model_loaded ? modelName : 'Not Loaded'}
                </div>
                ${status.cuda_available && status.gpu_memory_allocated ? `
                <div class="status-item">
                    <strong>GPU Memory:</strong> ${memoryGB}GB
                </div>` : ''}
                ${speakers ? `
                <div class="status-item" style="grid-column: 1 / -1;">
                    <strong>Speakers:</strong> ${speakers}
                </div>` : ''}
            </div>
        </div>
    `;
}

// ============================================================
// STT Functions
// ============================================================

async function processSTT() {
    const fileInput = document.getElementById('stt-file');
    const language = document.getElementById('stt-language').value;
    const enableSegmentation = document.getElementById('enable-segmentation').checked;
    const sttEngine = document.getElementById('stt-engine-select').value;
    const resultsDiv = document.getElementById('stt-results');

    if (!fileInput.files.length) {
        showStatus('stt-result-status', 'error', 'Please select an audio file');
        return;
    }

    const serviceUrl = sttEngine === 'qwen3-asr'
        ? window.QWEN3_ASR_SERVICE_URL
        : window.STT_SERVICE_URL;

    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);
    if (language !== 'auto') formData.append('language', language);

    try {
        const engineLabel = sttEngine === 'qwen3-asr' ? 'Qwen3-ASR' : 'Whisper';
        showStatus('stt-result-status', 'info', `Processing audio with ${engineLabel}...`);
        resultsDiv.innerHTML = '';

        const response = await fetch(`${serviceUrl}/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `STT processing failed: ${response.statusText}`);
        }

        const result = await response.json();

        if (enableSegmentation && result.segments && result.segments.length > 0) {
            resultsDiv.innerHTML = `
                <div class="result-header">
                    <h3>Transcription with Segmentation</h3>
                    <button class="btn-secondary btn-sm" onclick="copyTranscription()">Copy Text</button>
                </div>
                <div class="segment-stats">
                    <strong>Language:</strong> ${result.language || 'Unknown'} |
                    <strong>Duration:</strong> ${result.duration ? result.duration.toFixed(2) + 's' : 'N/A'} |
                    <strong>Segments:</strong> ${result.segments.length}
                </div>
                <div class="segments-container">
                    ${result.segments.map(seg => `
                        <div class="segment-item">
                            <div class="segment-time">${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s</div>
                            <div class="segment-text">${seg.text}</div>
                        </div>
                    `).join('')}
                </div>
                <div id="full-transcription" style="display:none;">${result.text || result.segments.map(s => s.text).join(' ')}</div>
            `;
        } else {
            resultsDiv.innerHTML = `
                <div class="result-header">
                    <h3>Transcription Result</h3>
                    <button class="btn-secondary btn-sm" onclick="copyTranscription()">Copy Text</button>
                </div>
                <div class="transcription-text" id="full-transcription">${result.text}</div>
                ${result.language ? `<div class="result-meta"><strong>Language:</strong> ${result.language}</div>` : ''}
            `;
        }

        showStatus('stt-result-status', 'success', 'Audio processed successfully!');
    } catch (error) {
        console.error('STT processing error:', error);
        showStatus('stt-result-status', 'error', `Processing failed: ${error.message}`);
    }
}

function copyTranscription() {
    const el = document.getElementById('full-transcription');
    if (!el) return;

    const text = el.textContent || el.innerText;
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Transcription copied to clipboard', 'success');
    }).catch(() => {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showNotification('Transcription copied to clipboard', 'success');
    });
}

function handleSTTFile(input) {
    const fileInfo = document.getElementById('stt-file-info');
    if (input.files.length > 0) {
        const file = input.files[0];
        fileInfo.innerHTML = `
            <div class="file-details">
                <strong>${file.name}</strong><br>
                <span>Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</span> |
                <span>Type: ${file.type}</span>
            </div>
        `;
        fileInfo.style.display = 'block';
    } else {
        fileInfo.innerHTML = '';
        fileInfo.style.display = 'none';
    }
}

// ============================================================
// Training Functions
// ============================================================

async function startTraining() {
    const voiceName = document.getElementById('training-voice-name').value.trim();
    const language = document.getElementById('training-language').value;
    const gender = document.getElementById('training-gender').value;
    const epochs = parseInt(document.getElementById('training-epochs').value);
    const batchSize = parseInt(document.getElementById('training-batch-size').value);
    const fileInput = document.getElementById('training-files');
    const progressDiv = document.getElementById('training-progress');

    if (!voiceName) {
        showStatus('training-progress-status', 'error', 'Please enter a voice model name');
        return;
    }
    if (!fileInput.files.length) {
        showStatus('training-progress-status', 'error', 'Please select training audio files');
        return;
    }

    const formData = new FormData();
    formData.append('model_name', voiceName);
    formData.append('language', language);
    formData.append('gender', gender);
    formData.append('epochs', epochs);
    formData.append('batch_size', batchSize);

    for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('audio_files', fileInput.files[i]);
    }

    try {
        showStatus('training-progress-status', 'info', 'Starting VITS training pipeline...');
        progressDiv.innerHTML = '';

        const response = await fetch(`${window.VOICE_TRAINING_URL}/train`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Training failed: ${response.statusText}`);
        }

        const result = await response.json();
        showStatus('training-progress-status', 'success', 'Training started successfully!');

        progressDiv.innerHTML = `
            <div class="training-info">
                <p><strong>Job ID:</strong> ${result.job_id}</p>
                <p><strong>Voice Name:</strong> ${voiceName}</p>
                <p><strong>Language:</strong> ${language}</p>
                <p><strong>Epochs:</strong> ${epochs}</p>
            </div>
        `;

        monitorTrainingProgress(result.job_id);
    } catch (error) {
        console.error('Training error:', error);
        showStatus('training-progress-status', 'error', `Training failed: ${error.message}`);
    }
}

function handleTrainingFiles(input) {
    const fileInfo = document.getElementById('training-files-info');
    if (input.files.length > 0) {
        const totalSize = Array.from(input.files).reduce((sum, f) => sum + f.size, 0);
        fileInfo.innerHTML = `
            <div class="file-details">
                <strong>${input.files.length} file(s) selected</strong><br>
                <span>Total Size: ${(totalSize / 1024 / 1024).toFixed(2)} MB</span><br>
                ${Array.from(input.files).map(f => `<span>- ${f.name}</span>`).join('<br>')}
            </div>
        `;
        fileInfo.style.display = 'block';
    } else {
        fileInfo.innerHTML = '';
        fileInfo.style.display = 'none';
    }
}

async function monitorTrainingProgress(sessionId) {
    const progressDiv = document.getElementById('training-progress');

    const checkProgress = async () => {
        try {
            const response = await fetch(`${window.VOICE_TRAINING_URL}/status/${sessionId}`);
            const status = await response.json();

            if (status.status === 'completed') {
                showStatus('training-progress-status', 'success', 'Training completed! You can now export the model to TTS.');
                refreshTrainingJobs();
                refreshModels();
                return;
            } else if (status.status === 'failed') {
                showStatus('training-progress-status', 'error', 'Training failed. Check training jobs for details.');
                refreshTrainingJobs();
                return;
            } else if (status.status === 'running' || status.status === 'training') {
                const progress = status.progress || 0;
                progressDiv.innerHTML = `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <p>Progress: ${progress.toFixed(1)}% (Epoch ${status.current_epoch || 0}/${status.total_epochs || 1000})</p>
                `;
                setTimeout(checkProgress, 5000);
            } else {
                // Unknown status — keep polling
                setTimeout(checkProgress, 5000);
            }
        } catch (error) {
            console.error('Progress monitoring error:', error);
        }
    };

    checkProgress();
}

// ============================================================
// Model Management Functions
// ============================================================

async function refreshModels() {
    const modelsList = document.getElementById('models-list');
    if (!modelsList) return;

    try {
        modelsList.innerHTML = '<p>Loading trained models...</p>';

        const response = await fetch(`${window.VOICE_TRAINING_URL}/jobs`);
        if (!response.ok) throw new Error('Failed to fetch models');

        const data = await response.json();
        const jobs = Array.isArray(data) ? data : (data.jobs || []);
        const completedModels = jobs.filter(job => job.status === 'completed');

        if (completedModels.length === 0) {
            modelsList.innerHTML = '<p>No trained models found. Start training to create your first model!</p>';
            return;
        }

        let html = '<table class="data-table"><thead><tr><th>Voice Name</th><th>Job ID</th><th>Created</th><th>Actions</th></tr></thead><tbody>';

        completedModels.forEach(job => {
            html += `
                <tr>
                    <td>${job.voice_name || job.job_id}</td>
                    <td><code>${job.job_id}</code></td>
                    <td>${new Date(job.created_at).toLocaleDateString()}</td>
                    <td class="action-buttons">
                        <button class="btn-secondary btn-sm" onclick="exportModelToTTS('${job.job_id}', '${job.voice_name || job.job_id}')">Export to TTS</button>
                        <button class="btn-secondary btn-sm" onclick="downloadModel('${job.job_id}')">Download</button>
                        <button class="btn-secondary btn-sm btn-danger" onclick="deleteModel('${job.job_id}')">Delete</button>
                    </td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        modelsList.innerHTML = html;
    } catch (error) {
        console.error('Failed to refresh models:', error);
        modelsList.innerHTML = '<p style="color: var(--error);">Failed to load models. Is the training service running?</p>';
    }
}

async function refreshTrainingJobs() {
    const jobsList = document.getElementById('training-jobs-list');
    if (!jobsList) return;

    try {
        jobsList.innerHTML = '<p>Loading training jobs...</p>';

        const response = await fetch(`${window.VOICE_TRAINING_URL}/jobs`);
        if (!response.ok) throw new Error('Failed to fetch training jobs');

        const data = await response.json();
        const jobs = Array.isArray(data) ? data : (data.jobs || []);

        if (jobs.length === 0) {
            jobsList.innerHTML = '<p>No training jobs found.</p>';
            return;
        }

        let html = '<table class="data-table"><thead><tr><th>Voice Name</th><th>Status</th><th>Progress</th><th>Created</th><th>Actions</th></tr></thead><tbody>';

        jobs.forEach(job => {
            const statusClass = job.status === 'completed' ? 'status-badge-success' :
                              job.status === 'failed' ? 'status-badge-error' :
                              job.status === 'training' || job.status === 'running' ? 'status-badge-active' :
                              job.status === 'interrupted' ? 'status-badge-warning' : 'status-badge-default';

            const voiceName = job.voice_name || job.model_name || job.job_id;
            let actionButtons = `<button class="btn-secondary btn-sm" onclick="viewJobDetails('${job.job_id}')">Details</button>`;
            if (job.status === 'interrupted') {
                actionButtons += ` <button class="btn-secondary btn-sm" onclick="resumeTraining('${voiceName}')">Resume</button>`;
            } else if (job.status !== 'completed' && job.status !== 'failed') {
                actionButtons += ` <button class="btn-secondary btn-sm btn-danger" onclick="cancelJob('${job.job_id}')">Cancel</button>`;
            }

            html += `
                <tr>
                    <td>${voiceName}</td>
                    <td><span class="status-badge ${statusClass}">${job.status}</span></td>
                    <td>${(job.progress || 0).toFixed(1)}%</td>
                    <td>${new Date(job.created_at).toLocaleDateString()}</td>
                    <td class="action-buttons">${actionButtons}</td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        jobsList.innerHTML = html;
    } catch (error) {
        console.error('Failed to refresh training jobs:', error);
        jobsList.innerHTML = '<p style="color: var(--error);">Failed to load training jobs.</p>';
    }
}

async function exportModelToTTS(jobId, modelName) {
    try {
        showNotification(`Exporting "${modelName}" to TTS service...`, 'info');

        const formData = new FormData();
        formData.append('model_name', modelName);

        const response = await fetch(`${window.VOICE_TRAINING_URL}/export/${jobId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Export failed' }));
            throw new Error(errorData.detail || 'Export failed');
        }

        showNotification(`Model "${modelName}" exported to TTS! Refresh voices to use it.`, 'success');

        // Refresh the PiperTTS voices
        await fetch(`${window.TTS_SERVICE_URL}/refresh_voices`, { method: 'POST' }).catch(() => {});
        refreshTTSVoices();
    } catch (error) {
        console.error('Export error:', error);
        showNotification(`Export failed: ${error.message}`, 'error');
    }
}

async function downloadModel(jobId) {
    try {
        const response = await fetch(`${window.VOICE_TRAINING_URL}/download/${jobId}`);
        if (!response.ok) throw new Error('Download failed');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${jobId}_model.onnx`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showNotification('Model download started', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showNotification('Failed to download model', 'error');
    }
}

async function deleteModel(jobId) {
    if (!confirm(`Delete model "${jobId}" and all training data? This cannot be undone.`)) return;

    try {
        const response = await fetch(`${window.VOICE_TRAINING_URL}/model/${jobId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Delete failed');

        showNotification('Model deleted successfully', 'success');
        refreshModels();
        refreshTTSVoices();
    } catch (error) {
        console.error('Delete error:', error);
        showNotification('Failed to delete model', 'error');
    }
}

async function cancelJob(jobId) {
    if (!confirm('Cancel this training job?')) return;

    try {
        const response = await fetch(`${window.VOICE_TRAINING_URL}/job/${jobId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Cancel failed');

        showNotification('Training job cancelled', 'success');
        refreshTrainingJobs();
    } catch (error) {
        console.error('Cancel error:', error);
        showNotification('Failed to cancel job', 'error');
    }
}

async function resumeTrainingManual() {
    const voiceName = document.getElementById('continue-voice-name').value.trim();
    if (!voiceName) {
        showStatus('continue-status', 'error', 'Please enter a voice model name');
        return;
    }
    resumeTraining(voiceName, 'continue-status');
}

async function trainFromDataset() {
    const voiceName = document.getElementById('continue-voice-name').value.trim();
    const epochs = parseInt(document.getElementById('continue-epochs').value) || 10000;
    if (!voiceName) {
        showStatus('continue-status', 'error', 'Please enter a voice model name');
        return;
    }

    if (!confirm(`Start training "${voiceName}" from the existing prepared dataset (train.json / val.json)?`)) return;

    try {
        showStatus('continue-status', 'info', `Starting training for "${voiceName}" from existing dataset...`);

        const formData = new FormData();
        formData.append('model_name', voiceName);
        formData.append('epochs', epochs);

        const response = await fetch(`${window.VOICE_TRAINING_URL}/train-from-dataset`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to start training' }));
            throw new Error(errorData.detail || 'Failed to start training');
        }

        const result = await response.json();
        showStatus('continue-status', 'success', `Training started! Job ID: ${result.job_id}`);
        showNotification(`Training started for "${voiceName}"`, 'success');
        refreshTrainingJobs();
        monitorTrainingProgress(result.job_id);
    } catch (error) {
        console.error('Train from dataset error:', error);
        showStatus('continue-status', 'error', `Failed: ${error.message}`);
    }
}

async function resumeTraining(voiceName, statusElementId = 'training-progress-status') {
    if (!confirm(`Resume training for voice "${voiceName}" from the last checkpoint?`)) return;

    try {
        showNotification(`Resuming training for "${voiceName}"...`, 'info');

        const formData = new FormData();
        formData.append('model_name', voiceName);

        const response = await fetch(`${window.VOICE_TRAINING_URL}/resume-training`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Resume failed' }));
            throw new Error(errorData.detail || 'Resume failed');
        }

        const result = await response.json();
        showNotification(`Training resumed for "${voiceName}"`, 'success');
        showStatus(statusElementId, 'info', `Resumed training for "${voiceName}" — monitoring progress...`);
        refreshTrainingJobs();
        monitorTrainingProgress(result.job_id);
    } catch (error) {
        console.error('Resume error:', error);
        showStatus(statusElementId, 'error', `Resume failed: ${error.message}`);
        showNotification(`Resume failed: ${error.message}`, 'error');
    }
}

async function viewJobDetails(jobId) {
    try {
        const response = await fetch(`${window.VOICE_TRAINING_URL}/status/${jobId}`);
        if (!response.ok) throw new Error('Failed to fetch job details');

        const job = await response.json();

        let details = `Job: ${job.config?.voice_name || jobId}\n`;
        details += `Status: ${job.status}\n`;
        details += `Progress: ${(job.progress || 0).toFixed(1)}%\n`;
        details += `Current Epoch: ${job.current_epoch || 0}\n`;

        if (job.config) {
            details += `\nConfiguration:\n`;
            details += `  Epochs: ${job.config.epochs || 'N/A'}\n`;
            details += `  Batch Size: ${job.config.batch_size || 'N/A'}\n`;
            details += `  Learning Rate: ${job.config.learning_rate || 'N/A'}\n`;
        }

        if (job.best_loss) details += `\nBest Loss: ${job.best_loss.toFixed(4)}\n`;

        if (job.logs && job.logs.length > 0) {
            details += '\nRecent Logs:\n';
            job.logs.slice(-5).forEach(log => {
                details += `  ${log.timestamp}: ${log.message}\n`;
            });
        }

        alert(details);
    } catch (error) {
        console.error('Job details error:', error);
        showNotification('Failed to fetch job details', 'error');
    }
}
