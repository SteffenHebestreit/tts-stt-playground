// Frontend controller for the TTS/STT demo UI.
// Manages tab navigation, service health polling, TTS/STT actions,
// training workflows, and Qwen3 voice-cloning interactions.

// Global variables
const providerRegistry = window.PROVIDER_REGISTRY || { providers: {}, ui: {} };
let currentTTSEngine = providerRegistry.ui?.default_tts_provider || 'piper';
let trainingDeploymentRegistry = null;
const qwen3ProviderId = 'qwen3';

function removeOptionalProvider(providerId) {
    if (providerRegistry.providers?.[providerId]) {
        delete providerRegistry.providers[providerId];
    }

    const statusElement = document.getElementById(`service-status-${providerId}`);
    if (statusElement) {
        statusElement.remove();
    }

    const sttSelect = document.getElementById('stt-engine-select');
    if (sttSelect) {
        const option = sttSelect.querySelector(`option[value="${providerId}"]`);
        if (option) {
            option.remove();
        }
    }
}

function pruneOptionalProviders() {
    if (!providerRegistry.ui?.enable_whisper_cpp) {
        removeOptionalProvider('whisper-cpp');
        if (providerRegistry.ui?.default_stt_provider === 'whisper-cpp') {
            providerRegistry.ui.default_stt_provider = 'whisper';
        }
    }
}

function getProvider(providerId) {
    return providerRegistry.providers?.[providerId] || null;
}

function getProviderUrl(providerId) {
    return getProvider(providerId)?.browser_url || null;
}

function getProviderDisplayName(providerId) {
    return getProvider(providerId)?.display_name || providerId;
}

function getProviderFamily(providerId) {
    return getProvider(providerId)?.ui?.family || providerId;
}

function getProviderContract(providerId, contractName) {
    return getProvider(providerId)?.contracts?.[contractName] || null;
}

function getProviderApiPath(providerId, path) {
    return `/api/providers/${providerId}${path}`;
}

function getProviderSettings(providerId) {
    return getProvider(providerId)?.settings || {};
}

function getProviderUI(providerId) {
    return getProvider(providerId)?.ui || {};
}

function getTrainingProviderId() {
    return providerRegistry.ui?.training_provider || 'piper-training';
}

function getTrainingBrowserUrl() {
    return getProviderUrl(getTrainingProviderId());
}

async function fetchTrainingRequest(primaryPath, fallbackPath, options = {}) {
    const fallbackUrl = getTrainingBrowserUrl();
    let primaryError = null;

    try {
        const response = await fetch(primaryPath, options);
        if (response.ok || response.status < 500 || !fallbackUrl) {
            return response;
        }
    } catch (error) {
        primaryError = error;
        if (!fallbackUrl) {
            throw error;
        }
    }

    try {
        return await fetch(`${fallbackUrl}${fallbackPath}`, options);
    } catch (fallbackError) {
        if (primaryError) {
            throw primaryError;
        }
        throw fallbackError;
    }
}

function getProviderMessages(providerId) {
    return getProviderUI(providerId)?.messages || {};
}

function formatMessage(template, values = {}) {
    if (!template) {
        return '';
    }

    return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (_match, key) => {
        const value = values[key];
        return value === undefined || value === null ? '' : String(value);
    });
}

function setSingleSelectOption(select, label, value = '') {
    if (!select) return;
    select.innerHTML = `<option value="${value}">${label}</option>`;
}

function setElementText(elementId, text) {
    const element = document.getElementById(elementId);
    if (element && text) {
        element.textContent = text;
    }
}

function setElementHTML(elementId, html) {
    const element = document.getElementById(elementId);
    if (element && html) {
        element.innerHTML = html;
    }
}

function setInputPlaceholder(elementId, placeholder) {
    const element = document.getElementById(elementId);
    if (element && placeholder) {
        element.placeholder = placeholder;
    }
}

function setInputValue(elementId, value) {
    const element = document.getElementById(elementId);
    if (element && value) {
        element.value = value;
    }
}

function applyGlobalCopy() {
    const copy = providerRegistry.ui?.copy || {};
    setElementText('app-subtitle', copy.app_subtitle);
    setElementText('stt-tab-button', copy.stt_tab_label);
    setElementText('stt-title', copy.stt_title);
    setElementText('stt-description', copy.stt_description);
}

function applyTrainingProviderCopy() {
    const providerId = getTrainingProviderId();
    const ui = getProviderUI(providerId);
    const copy = ui?.sections?.training || {};
    const forms = ui?.forms || {};
    const startForm = forms.start_training || {};
    const continueForm = forms.continue_training || {};
    const modelManagementForm = forms.model_management || {};

    setElementText('training-tab-button', ui?.tab_label);
    setElementText('training-title', copy.title);
    setElementText('training-description', copy.description);
    setElementText('training-models-description', copy.models_description);

    setElementText('training-voice-name-label', startForm.fields?.voice_name?.label);
    setInputPlaceholder('training-voice-name', startForm.fields?.voice_name?.placeholder || copy.voice_name_placeholder);
    setElementText('training-language-label', startForm.fields?.language?.label);
    setElementText('training-gender-label', startForm.fields?.gender?.label);
    setElementText('training-files-label', startForm.fields?.files?.label);
    setElementText('training-files-drop-text', startForm.fields?.files?.drop_text);
    setElementText('training-files-hint', startForm.fields?.files?.hint);
    setElementText('training-epochs-label', startForm.fields?.epochs?.label);
    setElementText('training-batch-size-label', startForm.fields?.batch_size?.label);
    setElementText('training-deployment-target-label', startForm.fields?.deployment_target?.label);
    setElementText('start-training-button', startForm.actions?.submit);

    setElementText('continue-training-title', continueForm.title);
    setElementHTML('continue-training-description', continueForm.description);
    setElementText('continue-voice-name-label', continueForm.fields?.voice_name?.label);
    setInputPlaceholder('continue-voice-name', continueForm.fields?.voice_name?.placeholder || copy.continue_voice_name_placeholder);
    setElementText('continue-epochs-label', continueForm.fields?.epochs?.label);
    setElementText('continue-deployment-target-label', continueForm.fields?.deployment_target?.label);
    setElementText('resume-training-button', continueForm.actions?.resume);
    setElementText('train-from-dataset-button', continueForm.actions?.train_from_dataset);

    setElementText('model-deployment-target-label', modelManagementForm.fields?.deployment_target?.label);
}

function applyTTSProviderCopy(providerId) {
    const ui = getProviderUI(providerId);
    const copy = ui?.sections?.tts || {};
    setElementText('tts-tab-button', ui?.tab_label);
    setElementText('tts-title', copy.title);
    setElementText('tts-description', copy.description);
    setInputPlaceholder('tts-text', copy.text_placeholder);
    setInputValue('tts-text', copy.text_sample);
    setElementText('custom-voices-title', copy.custom_voices_title);
    setElementText('custom-voices-description', copy.custom_voices_description);
}

function applyQwen3ProviderCopy() {
    const ui = getProviderUI(qwen3ProviderId);
    const builtinCopy = ui?.sections?.builtin_tts || {};
    const cloningCopy = ui?.sections?.cloning || {};
    const builtinForm = ui?.forms?.builtin_tts || {};
    const cloningForm = ui?.forms?.voice_clone || {};

    setElementText('qwen3-tts-tab-button', ui?.tab_label);
    setElementText('qwen3-cloning-tab-button', ui?.clone_tab_label);
    setElementText('qwen3-builtin-title', builtinCopy.title);
    setElementText('qwen3-builtin-description', builtinCopy.description);
    setElementText('qwen3-builtin-text-label', builtinForm.fields?.text?.label);
    setInputPlaceholder('qwen3-builtin-text', builtinForm.fields?.text?.placeholder || builtinCopy.text_placeholder);
    setInputValue('qwen3-builtin-text', builtinForm.fields?.text?.sample || builtinCopy.text_sample);
    setElementText('qwen3-builtin-language-label', builtinForm.fields?.language?.label);
    setElementText('qwen3-builtin-speaker-label', builtinForm.fields?.speaker?.label);
    setElementText('qwen3-builtin-instruct-label', builtinForm.fields?.instruction?.label);
    setInputPlaceholder('qwen3-builtin-instruct', builtinForm.fields?.instruction?.placeholder || builtinCopy.instruction_placeholder);
    setElementText('qwen3-builtin-instruct-hint', builtinForm.fields?.instruction?.hint || builtinCopy.instruction_hint);
    setElementText('qwen3-builtin-generate-button', builtinForm.actions?.generate);

    setElementText('qwen3-cloning-title', cloningCopy.title);
    setElementText('qwen3-cloning-description', cloningCopy.description);
    setElementText('qwen3-clone-model-label', cloningForm.fields?.model?.label);
    setElementText('qwen3-tts-text-label', cloningForm.fields?.text?.label);
    setInputPlaceholder('qwen3-tts-text', cloningForm.fields?.text?.placeholder || cloningCopy.text_placeholder);
    setInputValue('qwen3-tts-text', cloningForm.fields?.text?.sample || cloningCopy.text_sample);
    setElementText('qwen3-tts-language-label', cloningForm.fields?.language?.label);
    setElementText('clone-voice-source-label', cloningForm.fields?.voice_source?.label);
    setElementText('clone-saved-source-label', cloningCopy.saved_source_label);
    setElementText('clone-upload-source-label', cloningCopy.upload_source_label);
    setElementText('qwen3-saved-voice-select-label', cloningForm.fields?.saved_voice?.label);
    setElementText('saved-voices-refresh-button', cloningForm.actions?.refresh_saved);
    setElementText('saved-voices-delete-button', cloningForm.actions?.delete_saved);
    setElementText('qwen3-voice-file-label', cloningForm.fields?.voice_file?.label);
    setElementText('qwen3-voice-file-drop-text', cloningForm.fields?.voice_file?.drop_text);
    setElementText('qwen3-voice-file-hint', cloningForm.fields?.voice_file?.hint);
    setElementText('save-voice-name-label', cloningForm.fields?.save_voice_name?.label);
    setInputPlaceholder('save-voice-name', cloningForm.fields?.save_voice_name?.placeholder || cloningCopy.save_voice_placeholder);
    setElementText('save-voice-name-hint', cloningForm.fields?.save_voice_name?.hint);
    setElementText('enable-ref-text-label', cloningForm.fields?.ref_text_toggle?.label);
    setElementText('qwen3-ref-text-label', cloningForm.fields?.ref_text?.label);
    setInputPlaceholder('qwen3-ref-text', cloningForm.fields?.ref_text?.placeholder || cloningCopy.ref_text_placeholder);
    setElementText('qwen3-ref-text-hint', cloningForm.fields?.ref_text?.hint || cloningCopy.ref_text_hint);
    setElementText('qwen3-voice-description-label', cloningForm.fields?.voice_description?.label);
    setInputPlaceholder('qwen3-voice-description', cloningForm.fields?.voice_description?.placeholder || cloningCopy.voice_description_placeholder);
    setElementText('qwen3-voice-description-hint', cloningForm.fields?.voice_description?.hint || cloningCopy.voice_description_hint);
    setElementText('generate-qwen3-speech-btn', cloningForm.actions?.generate);
}

function getQwen3CloneModeCopy(mode) {
    return getProviderUI(qwen3ProviderId)?.sections?.cloning?.modes?.[mode] || {};
}

function getQwen3GenerateButtonLabel(mode = currentCloneMode) {
    const actions = getProviderUI(qwen3ProviderId)?.forms?.voice_clone?.actions || {};

    if (mode === 'design') {
        return getQwen3CloneModeCopy('design').button_label || actions.generate_design || actions.generate || 'Generate Speech';
    }

    if (mode === 'unsupported') {
        return getQwen3CloneModeCopy('unsupported').button_label || actions.generate || 'Generate Speech';
    }

    return getQwen3CloneModeCopy('saved').button_label || actions.generate || 'Generate Speech';
}

function populateSelectOptions(selectId, options, fallbackValue = '') {
    const select = document.getElementById(selectId);
    if (!select) return;

    const previousValue = select.value;
    select.innerHTML = '';

    (options || []).forEach((item) => {
        const option = document.createElement('option');
        if (typeof item === 'string') {
            option.value = item;
            option.textContent = item;
        } else {
            option.value = item.value;
            option.textContent = item.label || item.value;
        }
        select.appendChild(option);
    });

    const allowedValues = Array.from(select.options).map(option => option.value);
    if (previousValue && allowedValues.includes(previousValue)) {
        select.value = previousValue;
    } else if (fallbackValue && allowedValues.includes(String(fallbackValue))) {
        select.value = String(fallbackValue);
    }
}

function applyInputNumberConfig(inputId, config, fallbackValue) {
    const input = document.getElementById(inputId);
    if (!input || !config) return;

    if (config.min !== undefined) input.min = config.min;
    if (config.max !== undefined) input.max = config.max;
    if (config.step !== undefined) input.step = config.step;
    if (fallbackValue !== undefined) input.value = String(fallbackValue);
}

function applyTTSProviderSettings(providerId) {
    const settings = getProviderSettings(providerId);
    const defaults = settings.defaults || {};
    populateSelectOptions('tts-language-select', settings.languages, defaults.language || 'auto');
    populateSelectOptions('tts-quality-select', settings.qualities, defaults.quality || 'medium');
    populateSelectOptions('tts-gender-select', settings.genders, defaults.gender || 'any');

    const speedConfig = settings.speed || {};
    const speedSlider = document.getElementById('tts-speed');
    const speedValue = document.getElementById('tts-speed-value');
    if (speedSlider) {
        if (speedConfig.min !== undefined) speedSlider.min = speedConfig.min;
        if (speedConfig.max !== undefined) speedSlider.max = speedConfig.max;
        if (speedConfig.step !== undefined) speedSlider.step = speedConfig.step;
        speedSlider.value = String(defaults.speed ?? speedConfig.default ?? 1.0);
    }
    if (speedValue && speedSlider) {
        speedValue.textContent = `${speedSlider.value}x`;
    }
}

function applySTTProviderSettings() {
    const providerId = document.getElementById('stt-engine-select')?.value || providerRegistry.ui?.default_stt_provider;
    const provider = getProvider(providerId);
    const settings = getProviderSettings(providerId);
    const defaults = settings.defaults || {};
    populateSelectOptions('stt-language', settings.languages, defaults.language || 'auto');

    const segmentationCheckbox = document.getElementById('enable-segmentation');
    if (segmentationCheckbox) {
        const supportsSegments = Boolean(provider?.capabilities?.includes('segments'));
        segmentationCheckbox.disabled = !supportsSegments;
        segmentationCheckbox.checked = supportsSegments ? Boolean(defaults.enable_segmentation) : false;
        segmentationCheckbox.title = supportsSegments
            ? 'Enable audio segmentation for training workflows'
            : 'This provider does not expose segmentation metadata';
    }
}

function applyTrainingProviderSettings() {
    const providerId = getTrainingProviderId();
    const settings = getProviderSettings(providerId);
    const defaults = settings.defaults || {};
    populateSelectOptions('training-language', settings.languages, defaults.language || 'en');
    populateSelectOptions('training-gender', settings.genders, defaults.gender || 'female');
    populateSelectOptions('training-batch-size', settings.batch_sizes, defaults.batch_size || '32');
    applyInputNumberConfig('training-epochs', settings.epochs, defaults.epochs || 1000);
    applyInputNumberConfig('continue-epochs', settings.epochs, defaults.epochs || 1000);
}

function applyQwen3ProviderSettings() {
    const settings = getProviderSettings(qwen3ProviderId);
    const defaults = settings.defaults || {};
    populateSelectOptions('qwen3-builtin-language', settings.languages, defaults.language || 'English');
    populateSelectOptions('qwen3-tts-language-select', settings.languages, defaults.language || 'English');
}

async function loadQwen3BuiltinSpeakers() {
    const select = document.getElementById('qwen3-builtin-speaker');
    if (!select) return;

    try {
        const response = await fetch(getProviderApiPath(qwen3ProviderId, '/voices'));
        if (!response.ok) {
            throw new Error('Failed to load speaker catalog');
        }

        const data = await response.json();
        const voices = Array.isArray(data.voices) ? data.voices : [];
        const speakerOptions = voices
            .filter(voice => voice.kind === 'builtin')
            .map(voice => ({ value: voice.id, label: voice.name }));
        const defaultSpeaker = getProviderSettings(qwen3ProviderId)?.defaults?.speaker || 'Vivian';

        if (speakerOptions.length > 0) {
            populateSelectOptions('qwen3-builtin-speaker', speakerOptions, defaultSpeaker);
        }
    } catch (error) {
        console.error('Failed to load Qwen3 speakers:', error);
    }
}

function getTrainingTargetSelectIds() {
    return [
        'training-deployment-target',
        'continue-deployment-target',
        'model-deployment-target',
    ];
}

function getTrainingDeploymentLabel(targetId) {
    if (!targetId || !trainingDeploymentRegistry?.targets?.[targetId]) {
        return targetId || 'default';
    }
    return trainingDeploymentRegistry.targets[targetId].display_name || targetId;
}

function populateTrainingDeploymentTargetSelects() {
    const registry = trainingDeploymentRegistry;
    const selectIds = getTrainingTargetSelectIds();
    const targets = registry?.targets || {};
    const defaultTarget = registry?.default_target || '';

    selectIds.forEach((selectId) => {
        const select = document.getElementById(selectId);
        const description = document.getElementById(`${selectId}-description`);
        if (!select) return;

        select.innerHTML = '';
        Object.entries(targets).forEach(([targetId, target]) => {
            const option = document.createElement('option');
            option.value = targetId;
            option.textContent = target.display_name || targetId;
            option.selected = targetId === defaultTarget;
            select.appendChild(option);
        });

        select.onchange = () => {
            const selectedTarget = targets[select.value];
            if (description) {
                description.textContent = selectedTarget
                    ? `${selectedTarget.deployment_contract} | Capabilities: ${(selectedTarget.capabilities || []).join(', ')}`
                    : '';
            }
        };

        select.dispatchEvent(new Event('change'));
    });
}

async function loadTrainingDeploymentTargets(force = false) {
    if (trainingDeploymentRegistry && !force) {
        populateTrainingDeploymentTargetSelects();
        return trainingDeploymentRegistry;
    }

    const selectIds = getTrainingTargetSelectIds();
    try {
        const response = await fetch('/api/training/deployment-targets', {
            signal: AbortSignal.timeout(5000)
        });
        if (!response.ok) {
            throw new Error('Failed to fetch deployment targets');
        }

        trainingDeploymentRegistry = await response.json();
        populateTrainingDeploymentTargetSelects();
        return trainingDeploymentRegistry;
    } catch (error) {
        console.error('Failed to load deployment targets:', error);
        trainingDeploymentRegistry = null;
        selectIds.forEach((selectId) => {
            const select = document.getElementById(selectId);
            const description = document.getElementById(`${selectId}-description`);
            if (select) {
                select.innerHTML = '<option value="">Deployment targets unavailable</option>';
            }
            if (description) {
                description.textContent = 'The training service did not return deployment target metadata.';
            }
        });
        return null;
    }
}

function getSelectedTrainingDeploymentTarget(selectId) {
    const select = document.getElementById(selectId);
    return select?.value || trainingDeploymentRegistry?.default_target || '';
}

// TTS Engine Switching
/** Switch the visible UI panels to the selected TTS engine family. */
function switchTTSEngine(engine) {
    currentTTSEngine = engine;
    const family = getProviderFamily(engine);
    applyTTSProviderSettings(engine);
    applyTTSProviderCopy(engine);

    const piperElements = document.querySelectorAll('.piper-only');
    const qwen3Elements = document.querySelectorAll('.qwen3-only');

    piperElements.forEach(el => el.style.display = 'none');
    qwen3Elements.forEach(el => el.style.display = 'none');

    if (family === 'piper') {
        piperElements.forEach(el => el.style.display = '');
        const currentTab = document.querySelector('.tab-content.active');
        if (currentTab && currentTab.classList.contains('qwen3-only')) {
            showTab('stt-tab');
        }
    } else if (family === 'qwen3') {
        qwen3Elements.forEach(el => el.style.display = '');
        const currentTab = document.querySelector('.tab-content.active');
        if (currentTab && currentTab.classList.contains('piper-only')) {
            showTab('qwen3-tts-tab');
        }
    }

    updateServiceStatus();
}

// Notification system
/** Render a temporary toast-style notification in the top-right corner. */
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

/** Initialize UI widgets, voice lists, and background health polling. */
function initializeApp() {
    pruneOptionalProviders();
    applyGlobalCopy();
    applyTrainingProviderCopy();
    applyQwen3ProviderCopy();
    applySTTProviderSettings();
    applyTrainingProviderSettings();
    applyQwen3ProviderSettings();
    applyTTSProviderSettings(currentTTSEngine);
    applyTTSProviderCopy(currentTTSEngine);
    setupFileDragDrop();
    setupRangeSliders();
    refreshTTSVoices();
    loadQwen3BuiltinSpeakers();
    loadTrainingDeploymentTargets();
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

/** Bind dynamic control listeners (tab buttons use their inline onclick handlers). */
function setupEventListeners() {
    const sttEngineSelect = document.getElementById('stt-engine-select');
    if (sttEngineSelect) {
        sttEngineSelect.addEventListener('change', applySTTProviderSettings);
    }
}

/** Activate a single tab and trigger tab-specific refresh actions. */
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));

    const tabEl = document.getElementById(tabId);
    if (tabEl) tabEl.classList.add('active');

    const btn = document.querySelector(`.tab-button[onclick*="${tabId}"]`);
    if (btn) btn.classList.add('active');

    // Tab-specific initialization
    if (tabId === 'training-tab') {
        loadTrainingDeploymentTargets();
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
/** Attach drag-and-drop behavior to the file upload zones. */
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

/** Synchronize slider labels with their current numeric values. */
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

/**
 * Refresh every backend status indicator from one same-origin call.
 *
 * The gateway probes the providers concurrently over the internal Docker
 * network. Doing it here in the browser meant one cross-origin request per
 * provider, issued sequentially — a single unreachable backend stalled the whole
 * row for its full timeout — and it forced every backend port to be published
 * just so the browser could reach it.
 */
async function checkServiceHealth() {
    const ids = Object.entries(providerRegistry.providers || {})
        .filter(([, provider]) => provider.ui?.show_status)
        .map(([providerId]) => providerId);

    const elements = new Map();
    for (const providerId of ids) {
        const element = document.getElementById(`service-status-${providerId}`);
        if (!element) continue;
        elements.set(providerId, element);
        element.classList.remove('healthy', 'error');
        element.classList.add('loading');
    }
    if (!elements.size) return;

    let health = {};
    try {
        const response = await fetch('/api/health', { signal: AbortSignal.timeout(15000) });
        if (response.ok) health = (await response.json()).providers || {};
    } catch {
        // Leave `health` empty: every indicator falls through to 'error' below.
    }

    for (const [providerId, element] of elements) {
        const entry = health[providerId];
        element.classList.remove('loading');
        element.classList.add(entry && entry.healthy ? 'healthy' : 'error');
        if (entry) {
            const detail = [entry.model_size, entry.device].filter(Boolean).join(' · ');
            if (!entry.healthy) {
                element.title = `unavailable${entry.error ? ` (${entry.error})` : ''}`;
            } else if (entry.model_resident === false) {
                // Healthy but idle: the model was unloaded to free memory and
                // the next request will reload it. Not an error state.
                element.classList.add('idle');
                element.title = `idle — model unloaded to free memory${detail ? ` (${detail})` : ''}`;
            } else {
                element.title = `ready${detail ? ` (${detail})` : ''} — ${entry.latency_ms} ms`;
            }
        }
    }
}

/** Trigger a health refresh after UI state changes. */
function updateServiceStatus() {
    checkServiceHealth();
}

// Helper functions
/** Render a status message into the given result/status container. */
function showStatus(elementId, type, message) {
    const element = document.getElementById(elementId);
    if (element) element.innerHTML = `<div class="${type}">${message}</div>`;
}

/** Escape a value for safe interpolation into innerHTML. */
function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/** Convert a byte count into a human-readable size string. */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTimestamp(value) {
    if (!value) return 'N/A';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'N/A' : date.toLocaleDateString();
}

/** Show or hide the manual reference-text input for voice cloning. */
function toggleRefText() {
    const checked = document.getElementById('enable-ref-text').checked;
    document.getElementById('ref-text-group').style.display = checked ? '' : 'none';
}

// Object URLs live until explicitly revoked. Keyed on the container element so
// that replacing a player — including when refreshCustomVoices() rebuilds the
// containers wholesale — releases the blob the old player was holding.
const _playerUrls = new WeakMap();

/**
 * Render an <audio> player for `blob` inside `container`, releasing whatever
 * blob the previous player in that container was holding.
 *
 * `autoplay` matters for perceived latency: without it every synthesis result
 * waits on a human noticing the player and clicking it, which dwarfs anything
 * the backend can save.
 */
function setAudioPlayer(container, blob, extraStyle = '') {
    if (!container) return null;
    const previous = _playerUrls.get(container);
    if (previous) URL.revokeObjectURL(previous);

    const url = URL.createObjectURL(blob);
    _playerUrls.set(container, url);
    container.innerHTML =
        `<audio controls autoplay preload="auto" style="width: 100%;${extraStyle}">` +
        `<source src="${url}" type="${blob.type || 'audio/wav'}"></audio>`;
    return url;
}

// ============================================================
// PiperTTS Functions
// ============================================================

/** Submit a PiperTTS generation request and render the returned audio. */
async function generateTTS() {
    const text = document.getElementById('tts-text').value.trim();
    const language = document.getElementById('tts-language-select').value;
    const quality = document.getElementById('tts-quality-select').value;
    const gender = document.getElementById('tts-gender-select').value;
    const voice = document.getElementById('tts-voice-select').value;
    const speed = parseFloat(document.getElementById('tts-speed').value);
    const audioPlayer = document.getElementById('tts-audio-player');
    const providerId = currentTTSEngine;
    const messages = getProviderMessages(providerId)?.tts_generation || {};

    if (!text) {
        showStatus('tts-result-status', 'error', messages.validation_text || 'Please enter some text to synthesize');
        return;
    }

    try {
        const providerLabel = getProviderDisplayName(providerId);
        showStatus(
            'tts-result-status',
            'info',
            formatMessage(messages.start, { provider: providerLabel }) || `Generating speech with ${providerLabel}...`
        );
        audioPlayer.innerHTML = '';

        const requestData = {
            provider: providerId,
            text,
            speed,
            output_format: 'wav',
            instructions: '',
        };
        if (voice !== 'auto') requestData.voice = voice;
        requestData.language = language;
        if (quality !== 'medium') requestData.quality = quality;
        if (gender !== 'any') requestData.gender = gender;

        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `TTS generation failed: ${response.statusText}`);
        }

        setAudioPlayer(audioPlayer, await response.blob());

        showStatus('tts-result-status', 'success', messages.success || 'Speech generated successfully!');
    } catch (error) {
        console.error('TTS generation error:', error);
        showStatus(
            'tts-result-status',
            'error',
            formatMessage(messages.error, { error: error.message }) || `Generation failed: ${error.message}`
        );
    }
}

/** Reload the list of Piper voices into the voice selector dropdown. */
async function refreshTTSVoices() {
    try {
        const providerId = currentTTSEngine;
        const messages = getProviderMessages(providerId)?.tts_generation || {};
        const response = await fetch(`/api/providers/${providerId}/voices`);
        const data = await response.json();

        const voiceSelect = document.getElementById('tts-voice-select');
        voiceSelect.innerHTML = `<option value="auto">${messages.voice_auto_option || 'Auto-Select Best Voice'}</option>`;

        if (Array.isArray(data.voices)) {
            data.voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.id;
                const quality = voice.raw?.quality ? ` (${voice.raw.quality})` : '';
                const description = voice.description ? ` - ${voice.description}` : '';
                option.textContent = voice.language
                    ? `${voice.language} - ${voice.name}${quality}`
                    : `${voice.name}${description}`;
                voiceSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error refreshing voices:', error);
    }
}

// Custom voice management for PiperTTS
/** Refresh the card view of custom Piper voices exported from training jobs. */
async function refreshCustomVoices() {
    const container = document.getElementById('custom-voices-list');
    const messages = getProviderMessages('piper')?.custom_voice_library || {};
    if (!container) return;

    try {
        container.innerHTML = `<p>${messages.loading || 'Loading voices...'}</p>`;
        const response = await fetch(getProviderApiPath('piper', '/custom-voices'));
        const data = await response.json();

        if (!Array.isArray(data.voices)) {
            container.innerHTML = `<p>${messages.empty_invalid || 'No voices available.'}</p>`;
            return;
        }

        const customVoices = data.voices;

        if (customVoices.length === 0) {
            container.innerHTML = `<p>${messages.empty || 'No custom trained voices found. Train a voice model and it will appear here.'}</p>`;
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
                        <button class="btn-secondary" onclick="testVoice('${voice.id}', '${(voice.language || 'en').split('_')[0]}')">${messages.action_test || 'Test'}</button>
                        <button class="btn-secondary" onclick="deleteCustomVoice('${voice.id}')" style="color: var(--error);">${messages.action_delete || 'Delete'}</button>
                    </div>
                    <div id="voice-test-${voice.id}" class="audio-player"></div>
                </div>
            `;
        });
        html += '</div>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error refreshing custom voices:', error);
        container.innerHTML = `<p style="color: var(--error);">${messages.unavailable || 'Failed to load voices. Is the PiperTTS service running?'}</p>`;
    }
}

/** Generate a short preview clip for a specific custom Piper voice. */
async function testVoice(voiceId, lang = 'en') {
    const playerDiv = document.getElementById(`voice-test-${voiceId}`);
    const messages = getProviderMessages('piper')?.custom_voice_library || {};
    if (!playerDiv) return;

    try {
        playerDiv.innerHTML = `<div class="info" style="padding: 8px; font-size: 0.9rem;">${messages.test_start || 'Generating test audio...'}</div>`;

        const testTexts = {
            'de': 'Hallo, das ist ein Test dieser Stimme.',
            'en': 'Hello, this is a test of this voice.',
            'fr': 'Bonjour, ceci est un test de cette voix.',
            'es': 'Hola, esta es una prueba de esta voz.',
        };
        const baseLang = lang.split('-')[0];
        const testText = testTexts[baseLang] || testTexts['en'];
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider: 'piper', text: testText, voice: voiceId, output_format: 'wav' })
        });

        if (!response.ok) throw new Error('Test generation failed');

        setAudioPlayer(playerDiv, await response.blob(), ' margin-top: 8px;');
    } catch (error) {
        playerDiv.innerHTML = `<div class="error" style="padding: 8px; font-size: 0.9rem;">${formatMessage(messages.test_error, { error: error.message }) || `Test failed: ${error.message}`}</div>`;
    }
}

/** Delete a custom Piper voice and refresh the dependent UI state. */
async function deleteCustomVoice(voiceId) {
    const messages = getProviderMessages('piper')?.custom_voice_library || {};
    if (!confirm(formatMessage(messages.delete_confirm, { voice_id: voiceId }) || `Delete custom voice "${voiceId}"? This cannot be undone.`)) return;

    try {
        const response = await fetch(getProviderApiPath('piper', `/custom-voices/${voiceId}`), { method: 'DELETE' });
        if (!response.ok) throw new Error('Delete failed');

        showNotification(formatMessage(messages.delete_success, { voice_id: voiceId }) || `Voice "${voiceId}" deleted`, 'success');
        refreshCustomVoices();
        refreshTTSVoices();
    } catch (error) {
        console.error('Delete voice error:', error);
        showNotification(formatMessage(messages.delete_error, { error: error.message }) || `Failed to delete voice: ${error.message}`, 'error');
    }
}

// ============================================================
// Qwen3-TTS Functions
// ============================================================

// Model management
/** Load Qwen3-TTS model options into every visible model selector. */
async function loadQwen3Models() {
    const messages = getProviderMessages(qwen3ProviderId)?.model_catalog || {};
    const selects = [
        { select: document.getElementById('qwen3-model-select'), desc: document.getElementById('qwen3-model-description') },
        { select: document.getElementById('qwen3-clone-model-select'), desc: document.getElementById('qwen3-clone-model-description') }
    ].filter(s => s.select);

    if (selects.length === 0) return;

    try {
        const response = await fetch(getProviderApiPath(qwen3ProviderId, '/models'), {
            signal: AbortSignal.timeout(5000)
        });
        if (!response.ok) throw new Error('Failed to fetch models');

        const data = await response.json();
        const models = Array.isArray(data.models) ? data.models : [];
        const currentModel = data.current_model || models[0] || null;

        selects.forEach(({ select, desc }) => {
            select.innerHTML = '';
            models.forEach((model) => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = `${model.name} - ${model.description}`;
                if (model.is_current) option.selected = true;
                select.appendChild(option);
            });

            if (desc && currentModel) {
                desc.textContent = formatMessage(messages.current_description, {
                    model: currentModel.name,
                    capabilities: currentModel.capabilities_text || '',
                }) || `Current: ${currentModel.name} | Capabilities: ${currentModel.capabilities_text || ''}`;
            }
        });
    } catch {
        selects.forEach(({ select, desc }) => {
            setSingleSelectOption(select, messages.unavailable_option || 'Service unavailable');
            if (desc) desc.textContent = '';
        });
    }
}

/** Switch the active Qwen3-TTS model and refresh related status displays. */
async function switchQwen3Model(modelId) {
    if (!modelId) return;

    const messages = getProviderMessages(qwen3ProviderId)?.model_switching || {};

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

        showNotification(
            messages.start || 'Switching model... This may take a while if the model needs to download.',
            'info'
        );

        const response = await fetch(getProviderApiPath(qwen3ProviderId, '/models/select'), {
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
        showNotification(
            formatMessage(messages.success, { model: result.model?.name || modelId }) || `Model switched to ${result.model?.name || modelId}`,
            'success'
        );

        // Refresh both dropdowns and status
        loadQwen3Models();
        updateQwen3TTSStatus();
        loadQwen3BuiltinSpeakers();
    } catch (error) {
        console.error('Model switch error:', error);
        showNotification(
            formatMessage(messages.error, { error: error.message }) || `Failed to switch model: ${error.message}`,
            'error'
        );
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

/** Toggle the cloning UI between saved-voice and fresh-upload modes. */
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

/** Reconfigure the cloning UI when the selected Qwen3 model changes. */
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
    const designCopy = getQwen3CloneModeCopy('design');
    const unsupportedCopy = getQwen3CloneModeCopy('unsupported');
    const savedCopy = getQwen3CloneModeCopy('saved');

    if (isVoiceDesign) {
        currentCloneMode = 'design';
        if (voiceSource) voiceSource.style.display = 'none';
        if (savedMode) savedMode.style.display = 'none';
        if (audioMode) audioMode.style.display = 'none';
        if (designMode) designMode.style.display = '';
        if (title) title.textContent = designCopy.title || title.textContent;
        if (desc) desc.textContent = designCopy.description || desc.textContent;
        if (btn) { btn.textContent = designCopy.button_label || 'Design & Generate'; btn.disabled = false; }
    } else if (isCustomVoice) {
        currentCloneMode = 'unsupported';
        if (voiceSource) voiceSource.style.display = 'none';
        if (savedMode) savedMode.style.display = 'none';
        if (audioMode) audioMode.style.display = 'none';
        if (designMode) designMode.style.display = 'none';
        if (title) title.textContent = unsupportedCopy.title || title.textContent;
        if (desc) desc.textContent = unsupportedCopy.description || desc.textContent;
        if (btn) { btn.textContent = unsupportedCopy.button_label || 'Generate Speech'; btn.disabled = true; }
    } else {
        // Base model — show voice source toggle
        if (voiceSource) voiceSource.style.display = '';
        if (designMode) designMode.style.display = 'none';
        if (title) title.textContent = savedCopy.title || title.textContent;
        if (desc) desc.textContent = savedCopy.description || desc.textContent;
        if (btn) { btn.textContent = savedCopy.button_label || 'Generate Speech'; btn.disabled = false; }
        // Restore saved/upload mode
        switchVoiceSource(currentVoiceSource);
        loadSavedVoices();
    }

    // Also trigger the model switch
    switchQwen3Model(modelId);
}

// --- Saved Voices ---

/** Load saved Qwen3 voice profiles into the voice selector. */
async function loadSavedVoices() {
    const select = document.getElementById('qwen3-saved-voice-select');
    const info = document.getElementById('saved-voice-info');
    const messages = getProviderMessages(qwen3ProviderId)?.saved_voice_library || {};
    if (!select) return;

    try {
        const response = await fetch(getProviderApiPath(qwen3ProviderId, '/saved-voices'));
        if (!response.ok) throw new Error('Failed to load voices');
        const data = await response.json();
        const voices = data.voices || [];

        select.innerHTML = '';
        if (voices.length === 0) {
            setSingleSelectOption(select, messages.empty_option || 'No saved voices - upload a sample first');
            if (info) info.textContent = '';
            return;
        }

        voices.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.id;
            opt.textContent = v.name || v.id;
            opt.dataset.referencePreview = v.reference_preview || '';
            opt.dataset.language = v.language || '';
            opt.dataset.createdAtDisplay = v.created_at_display || '';
            select.appendChild(opt);
        });

        // Show info for first voice
        updateSavedVoiceInfo();
        select.onchange = updateSavedVoiceInfo;
    } catch (err) {
        console.error('Load saved voices error:', err);
        setSingleSelectOption(select, messages.error_option || 'Error loading voices');
    }
}

/** Display metadata for the currently selected saved Qwen3 voice. */
function updateSavedVoiceInfo() {
    const select = document.getElementById('qwen3-saved-voice-select');
    const info = document.getElementById('saved-voice-info');
    const messages = getProviderMessages(qwen3ProviderId)?.saved_voice_library || {};
    if (!select || !info) return;
    const opt = select.selectedOptions[0];
    if (opt && opt.value) {
        const parts = [];
        if (opt.dataset.referencePreview) {
            parts.push(
                formatMessage(messages.info_ref, { ref_text: opt.dataset.referencePreview }) || `Ref: "${opt.dataset.referencePreview}"`
            );
        }
        if (opt.dataset.createdAtDisplay) {
            parts.push(
                formatMessage(messages.info_saved, { created_at: opt.dataset.createdAtDisplay }) || `Saved: ${opt.dataset.createdAtDisplay}`
            );
        }
        info.textContent = parts.join(' | ');
    } else {
        info.textContent = '';
    }
}

/** Delete the selected saved Qwen3 voice profile. */
async function deleteSavedVoice() {
    const select = document.getElementById('qwen3-saved-voice-select');
    const messages = getProviderMessages(qwen3ProviderId)?.saved_voice_library || {};
    if (!select || !select.value) {
        showNotification(messages.no_selection_delete || 'No voice selected to delete.', 'error');
        return;
    }
    const voiceId = select.value;
    const voiceName = select.selectedOptions[0]?.textContent || voiceId;
    if (!confirm(formatMessage(messages.delete_confirm, { voice_name: voiceName }) || `Delete saved voice "${voiceName}"?`)) return;

    try {
        const response = await fetch(getProviderApiPath(qwen3ProviderId, `/saved-voices/${voiceId}`), { method: 'DELETE' });
        if (!response.ok) throw new Error('Delete failed');
        showNotification(formatMessage(messages.delete_success, { voice_name: voiceName }) || `Voice "${voiceName}" deleted.`, 'success');
        loadSavedVoices();
    } catch (err) {
        showNotification(formatMessage(messages.delete_error, { error: err.message }) || `Failed to delete voice: ${err.message}`, 'error');
    }
}

/** Save an uploaded reference recording as a reusable Qwen3 voice profile. */
async function saveVoiceFromUpload(voiceFile, name) {
    const lang = document.getElementById('qwen3-tts-language-select')?.value || 'auto';
    const formData = new FormData();
    const messages = getProviderMessages(qwen3ProviderId)?.voice_library || {};
    formData.append('name', name);
    formData.append('lang', lang);
    formData.append('file', voiceFile);

    showStatus(
        'qwen3-generation-status',
        'info',
        formatMessage(messages.save_start, { name }) || `Saving voice "${name}" (transcribing + extracting embedding)...`
    );

    const response = await fetch(getProviderApiPath(qwen3ProviderId, '/saved-voices'), {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(120000),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Save failed' }));
        throw new Error(err.detail || 'Failed to save voice');
    }

    const result = await response.json();
    showNotification(
        formatMessage(messages.save_success, { name }) || `Voice "${name}" saved! Use it from "Saved Voices" for fast TTS.`,
        'success'
    );
    loadSavedVoices();
    return result;
}

// Built-in speaker TTS
/** Generate audio with a built-in Qwen3 speaker voice. */
async function generateQwen3BuiltinTTS() {
    const text = document.getElementById('qwen3-builtin-text').value.trim();
    const lang = document.getElementById('qwen3-builtin-language').value;
    const speaker = document.getElementById('qwen3-builtin-speaker').value;
    const instruct = document.getElementById('qwen3-builtin-instruct').value.trim();
    const audioPlayer = document.getElementById('qwen3-builtin-audio-player');
    const messages = getProviderMessages(qwen3ProviderId)?.builtin_tts || {};

    if (!text) {
        showStatus('qwen3-builtin-status', 'error', messages.validation_text || 'Please enter some text to synthesize');
        return;
    }

    try {
        showStatus(
            'qwen3-builtin-status',
            'info',
            formatMessage(messages.start, { speaker }) || `Generating speech with ${speaker}...`
        );
        audioPlayer.innerHTML = '';

        const startTime = Date.now();

        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: qwen3ProviderId,
                text,
                language: lang,
                voice: speaker,
                instructions: instruct,
            }),
            signal: AbortSignal.timeout(60000)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Generation failed: ${response.statusText}`);
        }

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        setAudioPlayer(audioPlayer, await response.blob());

        showStatus(
            'qwen3-builtin-status',
            'success',
            formatMessage(messages.success, { duration, speaker }) || `Speech generated in ${duration}s (Speaker: ${speaker})`
        );
    } catch (error) {
        console.error('Qwen3-TTS generation error:', error);
        showStatus(
            'qwen3-builtin-status',
            'error',
            formatMessage(messages.error, { error: error.message }) || `Generation failed: ${error.message}`
        );
    }
}

// Voice cloning
/** Show metadata for the currently selected Qwen3 reference voice file. */
function handleQwen3VoiceFile(file) {
    if (!file) return;
    const infoDiv = document.getElementById('qwen3-voice-file-info');
    infoDiv.innerHTML = `
        <strong>Selected:</strong> ${escapeHtml(file.name)}<br>
        <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
        <strong>Type:</strong> ${escapeHtml(file.type)}
    `;
    infoDiv.style.display = 'block';
}

/** Dispatch Qwen3 generation to saved-voice, clone, or voice-design mode. */
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

/** Generate audio using a previously saved Qwen3 voice profile. */
async function generateWithSavedVoice() {
    const text = document.getElementById('qwen3-tts-text').value.trim();
    const lang = document.getElementById('qwen3-tts-language-select').value;
    const voiceId = document.getElementById('qwen3-saved-voice-select')?.value;
    const audioPlayer = document.getElementById('qwen3-audio-player');
    const generateBtn = document.getElementById('generate-qwen3-speech-btn');
    const messages = getProviderMessages(qwen3ProviderId)?.saved_voice_tts || {};
    let progressInterval = null;

    if (!text) {
        showStatus('qwen3-generation-status', 'error', messages.validation_text || 'Please enter some text.');
        return;
    }
    if (!voiceId) {
        showStatus('qwen3-generation-status', 'error', messages.validation_voice || 'No saved voice selected. Upload a sample first.');
        return;
    }

    const formData = new FormData();
    formData.append('text', text);
    formData.append('lang', lang);

    try {
        showStatus('qwen3-generation-status', 'info', messages.start || 'Generating speech with saved voice...');
        audioPlayer.innerHTML = '';
        generateBtn.disabled = true;
        generateBtn.textContent = messages.action_busy || 'Generating...';

        const startTime = Date.now();

        progressInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            showStatus(
                'qwen3-generation-status',
                'info',
                formatMessage(messages.progress, { elapsed }) || `Generating speech... ${elapsed}s`
            );
        }, 500);

        const response = await fetch(getProviderApiPath(qwen3ProviderId, `/saved-voices/${voiceId}/tts`), {
            method: 'POST',
            body: formData,
            signal: AbortSignal.timeout(600000),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Generation failed: ${response.statusText}`);
        }

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        const genTime = response.headers.get('X-Generation-Time');
        const audioDur = response.headers.get('X-Audio-Duration');
        setAudioPlayer(audioPlayer, await response.blob());

        const audioDuration = audioDur ? parseFloat(audioDur).toFixed(1) : null;
        const statusMsg = audioDuration
            ? (formatMessage(messages.success_with_audio, { duration, audio_duration: audioDuration }) || `Speech generated in ${duration}s (${audioDuration}s audio)`)
            : (formatMessage(messages.success, { duration }) || `Speech generated in ${duration}s`);
        showStatus('qwen3-generation-status', 'success', statusMsg);
    } catch (error) {
        console.error('Saved voice TTS error:', error);
        showStatus(
            'qwen3-generation-status',
            'error',
            formatMessage(messages.error, { error: error.message }) || `Generation failed: ${error.message}`
        );
    } finally {
        if (progressInterval) clearInterval(progressInterval);
        generateBtn.disabled = false;
        generateBtn.textContent = getQwen3GenerateButtonLabel();
    }
}

/** Clone a voice from an uploaded reference clip and optionally save it. */
async function generateQwen3VoiceClone() {
    const text = document.getElementById('qwen3-tts-text').value.trim();
    const lang = document.getElementById('qwen3-tts-language-select').value;
    const voiceFile = document.getElementById('qwen3-voice-file').files[0];
    const audioPlayer = document.getElementById('qwen3-audio-player');
    const generateBtn = document.getElementById('generate-qwen3-speech-btn');
    const useRefText = document.getElementById('enable-ref-text').checked;
    const refText = document.getElementById('qwen3-ref-text')?.value.trim();
    const saveName = document.getElementById('save-voice-name')?.value.trim();
    const messages = getProviderMessages(qwen3ProviderId)?.voice_clone || {};
    let progressInterval = null;

    if (!text) {
        showStatus('qwen3-generation-status', 'error', messages.validation_text || 'Please enter some text.');
        return;
    }
    if (!voiceFile) {
        showStatus('qwen3-generation-status', 'error', messages.validation_voice_file || 'Please select a voice sample file.');
        return;
    }
    if (useRefText && !refText) {
        showStatus('qwen3-generation-status', 'error', messages.validation_ref_text || 'Please enter the reference audio transcript or uncheck the option.');
        return;
    }

    try {
        generateBtn.disabled = true;
        generateBtn.textContent = messages.action_busy || 'Processing...';
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

        const autoTranscribing = !useRefText || !refText;
        const statusMsg = autoTranscribing
            ? (messages.start_auto_transcribe || 'Auto-transcribing reference audio via Qwen3-ASR, then cloning...')
            : (messages.start_manual_ref || 'Cloning and generating speech...');
        showStatus('qwen3-generation-status', 'info', statusMsg);

        progressInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            const phase = autoTranscribing && elapsed < 10
                ? (formatMessage(messages.progress_auto_transcribe, { elapsed }) || `Auto-transcribing + cloning... ${elapsed}s`)
                : (formatMessage(messages.progress_generate, { elapsed }) || `Generating voice clone... ${elapsed}s`);
            showStatus('qwen3-generation-status', 'info', phase);
        }, 500);

        const response = await fetch(getProviderApiPath(qwen3ProviderId, '/voice-clone'), {
            method: 'POST',
            body: formData,
            signal: AbortSignal.timeout(600000)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Voice cloning failed: ${response.statusText}`);
        }

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        setAudioPlayer(audioPlayer, await response.blob());

        const msg = saveName
            ? (formatMessage(messages.success_with_save, { duration, name: saveName }) || `Voice cloning completed in ${duration}s (voice "${saveName}" saved for fast reuse)`)
            : (formatMessage(messages.success, { duration }) || `Voice cloning completed in ${duration}s`);
        showStatus('qwen3-generation-status', 'success', msg);
    } catch (error) {
        console.error('Qwen3-TTS cloning error:', error);
        showStatus(
            'qwen3-generation-status',
            'error',
            formatMessage(messages.error, { error: error.message }) || `Generation failed: ${error.message}`
        );
    } finally {
        if (progressInterval) clearInterval(progressInterval);
        generateBtn.disabled = false;
        generateBtn.textContent = getQwen3GenerateButtonLabel();
    }
}

/** Generate speech from a text-described target voice. */
async function generateQwen3VoiceDesign() {
    const text = document.getElementById('qwen3-tts-text').value.trim();
    const lang = document.getElementById('qwen3-tts-language-select').value;
    const voiceDescription = document.getElementById('qwen3-voice-description').value.trim();
    const audioPlayer = document.getElementById('qwen3-audio-player');
    const generateBtn = document.getElementById('generate-qwen3-speech-btn');
    const messages = getProviderMessages(qwen3ProviderId)?.voice_design || {};
    let progressInterval = null;

    if (!text) {
        showStatus('qwen3-generation-status', 'error', messages.validation_text || 'Please enter some text to synthesize.');
        return;
    }
    if (!voiceDescription) {
        showStatus('qwen3-generation-status', 'error', messages.validation_description || 'Please describe the voice you want.');
        return;
    }

    try {
        showStatus('qwen3-generation-status', 'info', messages.start || 'Designing voice and generating speech...');
        audioPlayer.innerHTML = '';
        generateBtn.disabled = true;
        generateBtn.textContent = messages.action_busy || 'Generating...';

        const startTime = Date.now();

        progressInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            showStatus(
                'qwen3-generation-status',
                'info',
                formatMessage(messages.progress, { elapsed }) || `Designing voice... ${elapsed}s`
            );
        }, 500);

        const response = await fetch(getProviderApiPath(qwen3ProviderId, '/voice-design'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, lang, voice_description: voiceDescription }),
            signal: AbortSignal.timeout(600000)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Voice design failed: ${response.statusText}`);
        }

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        setAudioPlayer(audioPlayer, await response.blob());

        showStatus(
            'qwen3-generation-status',
            'success',
            formatMessage(messages.success, { duration }) || `Voice design completed in ${duration}s`
        );
    } catch (error) {
        console.error('Voice design error:', error);
        showStatus(
            'qwen3-generation-status',
            'error',
            formatMessage(messages.error, { error: error.message }) || `Generation failed: ${error.message}`
        );
    } finally {
        if (progressInterval) clearInterval(progressInterval);
        generateBtn.disabled = false;
        generateBtn.textContent = getQwen3GenerateButtonLabel();
    }
}

/** Fetch the current Qwen3-TTS status payload from the backend. */
async function updateQwen3TTSStatus() {
    try {
        const response = await fetch(getProviderApiPath(qwen3ProviderId, '/status'), {
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

/** Render the Qwen3-TTS status card from the latest backend status data. */
function updateQwen3TTSStatusDisplay(status) {
    const statusElement = document.getElementById('qwen3-tts-system-status');
    const messages = getProviderMessages(qwen3ProviderId)?.runtime_status || {};
    if (!statusElement) return;

    if (!status) {
        statusElement.innerHTML = `<div class="status-error">${messages.unavailable || 'Qwen3-TTS Service Unavailable'}</div>`;
        return;
    }

    const memoryGB = status.gpu_memory_gb != null ? Number(status.gpu_memory_gb).toFixed(1) : null;
    const speakers = Array.isArray(status.speakers) ? status.speakers.join(', ') : '';

    const modelName = status.model_name || messages.unknown || 'Unknown';
    const deviceName = status.device_name || messages.unknown || 'Unknown';
    const deviceSuffix = status.device_type === 'gpu' ? (messages.gpu_suffix || 'GPU') : (messages.cpu_suffix || 'CPU');
    const modelFallback = messages.not_loaded || 'Not Loaded';

    statusElement.innerHTML = `
        <div class="status-success">
            <h4>${messages.online || 'Qwen3-TTS Service Online'}</h4>
            <div class="status-grid">
                <div class="status-item">
                    <strong>${messages.device_label || 'Device'}:</strong> ${deviceName}
                    (${deviceSuffix})
                </div>
                <div class="status-item">
                    <strong>${messages.model_label || 'Model'}:</strong> ${status.model_loaded ? modelName : modelFallback}
                </div>
                ${memoryGB ? `
                <div class="status-item">
                    <strong>${messages.gpu_memory_label || 'GPU Memory'}:</strong> ${memoryGB}GB
                </div>` : ''}
                ${speakers ? `
                <div class="status-item" style="grid-column: 1 / -1;">
                    <strong>${messages.speakers_label || 'Speakers'}:</strong> ${speakers}
                </div>` : ''}
            </div>
        </div>
    `;
}

// ============================================================
// STT Functions
// ============================================================

/** Submit an audio transcription request to the selected STT backend. */
async function processSTT() {
    const fileInput = document.getElementById('stt-file');
    const language = document.getElementById('stt-language').value;
    const enableSegmentation = document.getElementById('enable-segmentation').checked;
    const sttEngine = document.getElementById('stt-engine-select').value;
    const resultsDiv = document.getElementById('stt-results');
    const messages = getProviderMessages(sttEngine)?.transcription || {};

    if (!fileInput.files.length) {
        showStatus('stt-result-status', 'error', messages.validation_file || 'Please select an audio file');
        return;
    }

    const transcribeContract = getProviderContract(sttEngine, 'transcribe') || 'stt-form-v1';

    try {
        const formData = new FormData();
        formData.append('provider', sttEngine);
        formData.append('audio', fileInput.files[0]);
        if (language !== 'auto') formData.append('language', language);

        const engineLabel = getProviderDisplayName(sttEngine);
        showStatus(
            'stt-result-status',
            'info',
            formatMessage(messages.start, { provider: engineLabel }) || `Processing audio with ${engineLabel}...`
        );
        resultsDiv.innerHTML = '';

        const response = await fetch('/api/stt', {
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
                    <h3>${messages.segmented_heading || 'Transcription with Segmentation'}</h3>
                    <button class="btn-secondary btn-sm" onclick="copyTranscription()">${messages.copy_action || 'Copy Text'}</button>
                </div>
                <div class="segment-stats">
                    <strong>${messages.language_label || 'Language'}:</strong> ${escapeHtml(result.language || messages.unknown || 'Unknown')} |
                    <strong>${messages.duration_label || 'Duration'}:</strong> ${result.duration ? result.duration.toFixed(2) + 's' : (messages.not_available || 'N/A')} |
                    <strong>${messages.segments_label || 'Segments'}:</strong> ${result.segments.length}
                </div>
                <div class="segments-container">
                    ${result.segments.map(seg => `
                        <div class="segment-item">
                            <div class="segment-time">${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s</div>
                            <div class="segment-text">${escapeHtml(seg.text)}</div>
                        </div>
                    `).join('')}
                </div>
                <div id="full-transcription" style="display:none;">${escapeHtml(result.text || result.segments.map(s => s.text).join(' '))}</div>
            `;
        } else {
            resultsDiv.innerHTML = `
                <div class="result-header">
                    <h3>${messages.result_heading || 'Transcription Result'}</h3>
                    <button class="btn-secondary btn-sm" onclick="copyTranscription()">${messages.copy_action || 'Copy Text'}</button>
                </div>
                <div class="transcription-text" id="full-transcription">${escapeHtml(result.text)}</div>
                ${result.language ? `<div class="result-meta"><strong>${messages.language_label || 'Language'}:</strong> ${escapeHtml(result.language)}</div>` : ''}
            `;
        }

        showStatus('stt-result-status', 'success', messages.success || 'Audio processed successfully!');
    } catch (error) {
        console.error('STT processing error:', error);
        showStatus(
            'stt-result-status',
            'error',
            formatMessage(messages.error, { error: error.message }) || `Processing failed: ${error.message}`
        );
    }
}

// ============================================================
// Live microphone transcription (WebSocket to the Whisper service)
// ============================================================

const LIVE_STT_SAMPLE_RATE = 16000;
// ~4 s of PCM16 @ 16 kHz. Past this the network is not keeping up, and queueing
// more only adds latency the user cannot see until they press Stop.
const LIVE_STT_MAX_BUFFERED_BYTES = 131072;

const liveSTT = {
    socket: null,
    audioContext: null,
    processor: null,
    source: null,
    filter: null,
    sink: null,
    mediaStream: null,
    active: false,
    starting: false,
    droppedFrames: 0,
    worletWatchdog: null,
    // Incremented on every start. Callbacks belonging to a retired socket
    // compare against their captured value and bail out, so a late close from
    // session N can no longer reset state that now belongs to session N+1.
    generation: 0,
};

/** Encode a Float32 buffer already at 16 kHz as PCM16 little-endian. */
function encodePCM16(float32) {
    const pcm = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return pcm.buffer;
}

/**
 * Create an AudioContext locked to 16 kHz so the browser's own (properly
 * filtered) resampler does the rate conversion.
 *
 * The previous code resampled by picking every n-th sample with no anti-alias
 * filter, which folded everything above 8 kHz back into the speech band and
 * cost real accuracy. Returns {context, needsFallbackFilter}.
 */
function createCaptureContext() {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    try {
        const context = new Ctor({ sampleRate: LIVE_STT_SAMPLE_RATE, latencyHint: 'interactive' });
        if (context.sampleRate === LIVE_STT_SAMPLE_RATE) {
            return { context, needsFallbackFilter: false };
        }
        // Safari historically ignores the hint; fall through to the filtered path.
        try { context.close(); } catch { /* noop */ }
    } catch {
        // NotSupportedError on browsers that cannot honour an explicit rate.
    }
    return { context: new Ctor({ latencyHint: 'interactive' }), needsFallbackFilter: true };
}

/** Decimate to 16 kHz. Only used when the context rate could not be forced. */
function decimateTo16k(float32, inputRate) {
    if (inputRate === LIVE_STT_SAMPLE_RATE) return float32;
    const ratio = inputRate / LIVE_STT_SAMPLE_RATE;
    const outLength = Math.floor(float32.length / ratio);
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
        out[i] = float32[Math.floor(i * ratio)];
    }
    return out;
}

/** Send one PCM block, dropping it if the socket is already backed up. */
function sendLiveAudio(socket, buffer) {
    if (socket.readyState !== WebSocket.OPEN) return;
    if (socket.bufferedAmount > LIVE_STT_MAX_BUFFERED_BYTES) {
        // The network is not keeping up. Queueing more would only add latency
        // the user cannot see until they press Stop, so drop instead.
        liveSTT.droppedFrames++;
        return;
    }
    socket.send(buffer);
}

/** Build the capture graph, preferring an AudioWorklet over the main thread. */
async function startCapture(socket) {
    const { context, needsFallbackFilter } = createCaptureContext();
    liveSTT.audioContext = context;
    // Autoplay policy can hand back a suspended context.
    if (context.state === 'suspended') {
        try { await context.resume(); } catch { /* noop */ }
    }
    liveSTT.source = context.createMediaStreamSource(liveSTT.mediaStream);

    let head = liveSTT.source;
    if (needsFallbackFilter) {
        // Band-limit below the 8 kHz Nyquist of the target rate before decimating.
        liveSTT.filter = context.createBiquadFilter();
        liveSTT.filter.type = 'lowpass';
        liveSTT.filter.frequency.value = 7500;
        liveSTT.source.connect(liveSTT.filter);
        head = liveSTT.filter;
    }

    // Preferred path: capture on the audio rendering thread, where main-thread
    // work (rendering partials, layout, GC) cannot drop frames.
    if (!needsFallbackFilter && context.audioWorklet) {
        try {
            await context.audioWorklet.addModule(`/static/js/mic-worklet.js?v=${window.APP_VERSION || '1'}`);
            // numberOfOutputs: 1 routed through a muted gain node to the
            // destination, exactly like the ScriptProcessor path below. A
            // zero-output node is only rendered if the browser keeps it in the
            // graph, and that behaviour is not reliable across engines — if it
            // is skipped, process() never runs and the session silently sends
            // no audio at all with no error to fall back on.
            const node = new AudioWorkletNode(context, 'mic-capture', {
                numberOfInputs: 1,
                numberOfOutputs: 1,
                outputChannelCount: [1],
                processorOptions: { sampleRate: LIVE_STT_SAMPLE_RATE },
            });
            let gotAudio = false;
            node.port.onmessage = (event) => {
                if (event.data && event.data.type === 'error') {
                    console.warn('mic-worklet:', event.data.message);
                    return;
                }
                gotAudio = true;
                sendLiveAudio(socket, event.data);
            };
            head.connect(node);
            const sink = context.createGain();
            sink.gain.value = 0;
            node.connect(sink);
            sink.connect(context.destination);
            liveSTT.processor = node;
            liveSTT.sink = sink;

            // Belt and braces: if the worklet is never pulled, fall back rather
            // than leaving the user with a permanently silent session.
            liveSTT.worletWatchdog = setTimeout(() => {
                if (!gotAudio && liveSTT.processor === node) {
                    console.warn('AudioWorklet produced no audio; falling back to ScriptProcessor');
                    try { node.disconnect(); } catch { /* noop */ }
                    try { sink.disconnect(); } catch { /* noop */ }
                    liveSTT.processor = null;
                    liveSTT.sink = null;
                    attachScriptProcessor(context, head, socket);
                }
            }, 2000);
            return;
        } catch (err) {
            console.warn('AudioWorklet unavailable, falling back to ScriptProcessor:', err);
        }
    }

    attachScriptProcessor(context, head, socket);
}

/**
 * Fallback capture path: ScriptProcessorNode.
 *
 * Deprecated and main-thread, but universally supported. 1024 frames @ 16 kHz
 * is 64 ms of granularity (the old 4096 @ 48 kHz was 85 ms and ran on the main
 * thread regardless).
 */
function attachScriptProcessor(context, head, socket) {
    const bufferSize = context.sampleRate === LIVE_STT_SAMPLE_RATE ? 1024 : 4096;
    const processor = context.createScriptProcessor(bufferSize, 1, 1);
    processor.onaudioprocess = (event) => {
        const float32 = event.inputBuffer.getChannelData(0);
        sendLiveAudio(socket, encodePCM16(decimateTo16k(float32, context.sampleRate)));
    };
    head.connect(processor);
    // A ScriptProcessor only fires while it is part of a rendering graph, but
    // routing it to the speakers would be a feedback path. A muted gain node
    // keeps it pulling without making a sound.
    const sink = context.createGain();
    sink.gain.value = 0;
    processor.connect(sink);
    sink.connect(context.destination);
    liveSTT.processor = processor;
    liveSTT.sink = sink;
}

/** Start or stop the live microphone transcription session. */
async function toggleLiveTranscription() {
    if (liveSTT.active) {
        stopLiveTranscription();
        return;
    }
    // `active` is only set once the socket opens, two awaits later — without a
    // synchronous guard a double-click strands a mic stream, an AudioContext
    // and a socket that keeps decoding forever.
    if (liveSTT.starting) return;
    liveSTT.starting = true;
    const session = ++liveSTT.generation;

    const button = document.getElementById('live-stt-button');
    const transcript = document.getElementById('live-stt-transcript');
    const confirmedEl = document.getElementById('live-stt-confirmed');
    const pendingEl = document.getElementById('live-stt-pending');
    if (button) button.disabled = true;

    // Only the current session may clear the guard. The previous socket stays
    // open after Stop so the server can deliver its final transcript, and its
    // close can land *after* a new session has already started.
    const release = () => {
        if (liveSTT.generation !== session) return;
        liveSTT.starting = false;
        if (button) button.disabled = false;
    };

    // Go through the same-origin relay rather than dialling the STT container's
    // published port. getUserMedia needs a secure context, and under HTTPS a
    // ws:// handshake to another port is blocked as mixed content; behind
    // single-port ingress that port is not reachable at all.
    const wsScheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsScheme}://${window.location.host}/ws/stt?provider=whisper`;

    try {
        liveSTT.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });
    } catch (err) {
        showStatus('live-stt-status', 'error', `Microphone access denied: ${err.message}`);
        release();
        return;
    }

    confirmedEl.textContent = '';
    pendingEl.textContent = '';
    transcript.style.display = '';
    liveSTT.droppedFrames = 0;
    showStatus('live-stt-status', 'info', 'Connecting...');

    const socket = new WebSocket(wsUrl);
    socket.binaryType = 'arraybuffer';
    liveSTT.socket = socket;
    // Distinguishes "server delivered the final transcript and closed" from
    // "the connection died", which look identical at the onclose callback.
    let sawFinal = false;

    socket.onopen = async () => {
        if (liveSTT.generation !== session) return;
        const language = document.getElementById('stt-language')?.value || 'auto';
        socket.send(JSON.stringify({ language: language || 'auto' }));

        try {
            await startCapture(socket);
        } catch (err) {
            showStatus('live-stt-status', 'error', `Could not start capture: ${err.message}`);
            release();
            stopLiveTranscription(true);
            return;
        }

        // startCapture awaits (worklet module fetch), so the server may have
        // rejected and closed us in the meantime — e.g. "too many live
        // sessions". Publishing "Listening..." then would leave the microphone
        // hot on a dead socket.
        if (liveSTT.generation !== session || socket.readyState !== WebSocket.OPEN) {
            release();
            stopLiveTranscription(true);
            return;
        }

        liveSTT.active = true;
        release();
        button.textContent = 'Stop Live Transcription';
        showStatus('live-stt-status', 'info', 'Listening... speak into your microphone.');
    };

    socket.onmessage = (event) => {
        let message;
        try {
            message = JSON.parse(event.data);
        } catch {
            return;
        }
        if (message.type === 'partial') {
            // Read scroll state before writing — afterwards scrollHeight has
            // already changed and "was the user at the bottom" is unanswerable.
            const atBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 40;
            confirmedEl.textContent = message.confirmed ? message.confirmed + ' ' : '';
            pendingEl.textContent = message.pending || '';
            if (atBottom) transcript.scrollTop = transcript.scrollHeight;
            updateLiveLatency(message);
        } else if (message.type === 'final') {
            sawFinal = true;
            confirmedEl.textContent = message.text || '';
            pendingEl.textContent = '';
            transcript.scrollTop = transcript.scrollHeight;
            const dropped = liveSTT.droppedFrames
                ? ` — ${liveSTT.droppedFrames} audio blocks dropped (slow connection)` : '';
            showStatus(
                'live-stt-status',
                'success',
                `Final transcript ready (${escapeHtml(message.language || 'unknown')}, ${message.duration}s of audio)${dropped}.`
            );
        } else if (message.type === 'warning') {
            showStatus('live-stt-status', 'info', escapeHtml(message.message || 'Warning'));
        } else if (message.type === 'error') {
            showStatus('live-stt-status', 'error', escapeHtml(message.error || 'Streaming error'));
        }
    };

    socket.onerror = () => {
        if (liveSTT.generation !== session) return;
        showStatus('live-stt-status', 'error', 'WebSocket connection failed. Is the Whisper service reachable?');
        release();
        stopLiveTranscription(true);
    };

    socket.onclose = () => {
        if (liveSTT.generation !== session) return;
        release();
        if (liveSTT.active) {
            // Closed without us asking: the relay could not reach the STT
            // service, or it went away mid-session. Say so — the status
            // otherwise keeps claiming "Listening..." on a dead socket.
            if (!sawFinal) {
                showStatus('live-stt-status', 'error', 'Live transcription connection lost.');
            }
            stopLiveTranscription(true);
        }
    };
}

/** Surface the server's decode timings so latency is visible, not guessed. */
function updateLiveLatency(message) {
    const el = document.getElementById('live-stt-latency');
    if (!el || message.decode_ms === undefined) return;
    const parts = [`decode ${Math.round(message.decode_ms)} ms`];
    if (message.pending_seconds) parts.push(`${message.pending_seconds.toFixed(1)} s behind`);
    if (liveSTT.droppedFrames) parts.push(`${liveSTT.droppedFrames} dropped`);
    el.textContent = parts.join(' · ');
}

/** Tear down the microphone capture chain and (optionally) the socket. */
function stopLiveTranscription(skipStopMessage = false) {
    const button = document.getElementById('live-stt-button');
    liveSTT.active = false;

    if (liveSTT.worletWatchdog) { clearTimeout(liveSTT.worletWatchdog); liveSTT.worletWatchdog = null; }
    if (liveSTT.processor) {
        try { liveSTT.processor.disconnect(); } catch { /* noop */ }
        if (liveSTT.processor.port) liveSTT.processor.port.onmessage = null;
        liveSTT.processor.onaudioprocess = null;
        liveSTT.processor = null;
    }
    if (liveSTT.sink) { try { liveSTT.sink.disconnect(); } catch { /* noop */ } liveSTT.sink = null; }
    if (liveSTT.filter) { try { liveSTT.filter.disconnect(); } catch { /* noop */ } liveSTT.filter = null; }
    if (liveSTT.source) { try { liveSTT.source.disconnect(); } catch { /* noop */ } liveSTT.source = null; }
    if (liveSTT.audioContext) { try { liveSTT.audioContext.close(); } catch { /* noop */ } liveSTT.audioContext = null; }
    if (liveSTT.mediaStream) {
        liveSTT.mediaStream.getTracks().forEach(track => track.stop());
        liveSTT.mediaStream = null;
    }

    const socket = liveSTT.socket;
    if (socket && socket.readyState === WebSocket.OPEN && !skipStopMessage) {
        // Ask for the final transcript; the server closes after sending it.
        socket.send(JSON.stringify({ event: 'stop' }));
        showStatus('live-stt-status', 'info', 'Finishing final transcript...');
    } else if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
    }
    liveSTT.socket = null;

    if (button) { button.textContent = 'Start Live Transcription'; button.disabled = false; }
}

/** Copy the current full transcription text to the clipboard. */
function copyTranscription() {
    const el = document.getElementById('full-transcription');
    if (!el) return;

    const sttEngine = document.getElementById('stt-engine-select')?.value || providerRegistry.ui?.default_stt_provider || 'whisper';
    const messages = getProviderMessages(sttEngine)?.transcription || {};
    const text = el.textContent || el.innerText;
    navigator.clipboard.writeText(text).then(() => {
        showNotification(messages.copy_success || 'Transcription copied to clipboard', 'success');
    }).catch(() => {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showNotification(messages.copy_success || 'Transcription copied to clipboard', 'success');
    });
}

/** Show selected STT file metadata in the UI. */
function handleSTTFile(input) {
    const fileInfo = document.getElementById('stt-file-info');
    if (input.files.length > 0) {
        const file = input.files[0];
        fileInfo.innerHTML = `
            <div class="file-details">
                <strong>${escapeHtml(file.name)}</strong><br>
                <span>Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</span> |
                <span>Type: ${escapeHtml(file.type)}</span>
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

/** Start a new Piper training job from uploaded audio files. */
async function startTraining() {
    const voiceName = document.getElementById('training-voice-name').value.trim();
    const language = document.getElementById('training-language').value;
    const gender = document.getElementById('training-gender').value;
    const epochs = parseInt(document.getElementById('training-epochs').value);
    const batchSize = parseInt(document.getElementById('training-batch-size').value);
    const deploymentTarget = getSelectedTrainingDeploymentTarget('training-deployment-target');
    const fileInput = document.getElementById('training-files');
    const progressDiv = document.getElementById('training-progress');
    const messages = getProviderMessages(getTrainingProviderId())?.start_training || {};

    if (!voiceName) {
        showStatus('training-progress-status', 'error', messages.validation_name || 'Please enter a voice model name');
        return;
    }
    if (!fileInput.files.length) {
        showStatus('training-progress-status', 'error', messages.validation_files || 'Please select training audio files');
        return;
    }

    const formData = new FormData();
    formData.append('model_name', voiceName);
    formData.append('language', language);
    formData.append('gender', gender);
    formData.append('epochs', epochs);
    formData.append('batch_size', batchSize);
    if (deploymentTarget) formData.append('deployment_target', deploymentTarget);

    for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('audio_files', fileInput.files[i]);
    }

    try {
        showStatus(
            'training-progress-status',
            'info',
            formatMessage(messages.start, { deployment_target: getTrainingDeploymentLabel(deploymentTarget) }) || `Starting VITS training pipeline for ${getTrainingDeploymentLabel(deploymentTarget)}...`
        );
        progressDiv.innerHTML = '';

        const response = await fetch('/api/training/train', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `Training failed: ${response.statusText}`);
        }

        const result = await response.json();
        showStatus('training-progress-status', 'success', messages.success || 'Training started successfully!');

        progressDiv.innerHTML = `
            <div class="training-info">
                <p><strong>Job ID:</strong> ${result.job_id}</p>
                <p><strong>Voice Name:</strong> ${voiceName}</p>
                <p><strong>Language:</strong> ${language}</p>
                <p><strong>Epochs:</strong> ${epochs}</p>
                <p><strong>Deployment Target:</strong> ${getTrainingDeploymentLabel(deploymentTarget)}</p>
            </div>
        `;

        monitorTrainingProgress(result.job_id);
    } catch (error) {
        console.error('Training error:', error);
        showStatus(
            'training-progress-status',
            'error',
            formatMessage(messages.error, { error: error.message }) || `Training failed: ${error.message}`
        );
    }
}

/** Display the selected training files and their total size. */
function handleTrainingFiles(input) {
    const fileInfo = document.getElementById('training-files-info');
    if (input.files.length > 0) {
        const totalSize = Array.from(input.files).reduce((sum, f) => sum + f.size, 0);
        fileInfo.innerHTML = `
            <div class="file-details">
                <strong>${input.files.length} file(s) selected</strong><br>
                <span>Total Size: ${(totalSize / 1024 / 1024).toFixed(2)} MB</span><br>
                ${Array.from(input.files).map(f => `<span>- ${escapeHtml(f.name)}</span>`).join('<br>')}
            </div>
        `;
        fileInfo.style.display = 'block';
    } else {
        fileInfo.innerHTML = '';
        fileInfo.style.display = 'none';
    }
}

/** Poll the training service until a job completes, fails, or stops updating. */
async function monitorTrainingProgress(sessionId) {
    const progressDiv = document.getElementById('training-progress');
    const messages = getProviderMessages(getTrainingProviderId())?.start_training || {};
    let consecutiveFailures = 0;

    const checkProgress = async () => {
        try {
            const response = await fetchTrainingRequest(`/api/training/status/${sessionId}`, `/status/${sessionId}`);
            if (!response.ok) {
                throw new Error(`Status request failed: HTTP ${response.status}`);
            }
            const status = await response.json();
            consecutiveFailures = 0;

            if (status.status === 'completed') {
                const deploymentLabel = status.deployment_target_label || getTrainingDeploymentLabel(status.deployment_target);
                showStatus(
                    'training-progress-status',
                    'success',
                    formatMessage(messages.completed, { deployment_target: deploymentLabel }) || `Training completed. Deployment target: ${deploymentLabel}.`
                );
                refreshTrainingJobs();
                refreshModels();
                return;
            } else if (status.status === 'failed') {
                showStatus('training-progress-status', 'error', messages.failed || 'Training failed. Check training jobs for details.');
                refreshTrainingJobs();
                return;
            } else if (status.status === 'running' || status.status === 'training') {
                const progress = status.progress || 0;
                const progressMessage = formatMessage(messages.progress, {
                    progress: progress.toFixed(1),
                    current_epoch: status.current_epoch || 0,
                    total_epochs: status.total_epochs || 1000,
                }) || `Progress: ${progress.toFixed(1)}% (Epoch ${status.current_epoch || 0}/${status.total_epochs || 1000})`;
                progressDiv.innerHTML = `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <p>${progressMessage}</p>
                `;
                setTimeout(checkProgress, 5000);
            } else {
                // Unknown status — keep polling
                setTimeout(checkProgress, 5000);
            }
        } catch (error) {
            // Transient errors (service restart, network blip) shouldn't kill
            // monitoring permanently — retry a few times before giving up.
            consecutiveFailures += 1;
            console.error(`Progress monitoring error (attempt ${consecutiveFailures}):`, error);
            if (consecutiveFailures < 5) {
                setTimeout(checkProgress, 10000);
            } else {
                showStatus(
                    'training-progress-status',
                    'error',
                    formatMessage(messages.error, { error: error.message }) || `Lost contact with the training job: ${error.message}`
                );
            }
        }
    };

    checkProgress();
}

// ============================================================
// Model Management Functions
// ============================================================

/** Refresh the table of completed/exportable trained voice models. */
async function refreshModels() {
    const modelsList = document.getElementById('models-list');
    const messages = getProviderMessages(getTrainingProviderId())?.model_list || {};
    if (!modelsList) return;

    try {
        modelsList.innerHTML = `<p>${messages.loading || 'Loading trained models...'}</p>`;

        const response = await fetch('/api/training/jobs');
        if (!response.ok) throw new Error('Failed to fetch models');

        const data = await response.json();
        const jobs = Array.isArray(data) ? data : (data.jobs || []);
        const completedModels = jobs.filter(job => job.status === 'completed');

        if (completedModels.length === 0) {
            modelsList.innerHTML = `<p>${messages.empty || 'No trained models found. Start training to create your first model!'}</p>`;
            return;
        }

        let html = '<table class="data-table"><thead><tr><th>Voice Name</th><th>Job ID</th><th>Target</th><th>Created</th><th>Actions</th></tr></thead><tbody>';

        completedModels.forEach(job => {
            const voiceName = job.voice_name || job.model_name || job.job_id;
            const deploymentLabel = job.deployment_target_label || getTrainingDeploymentLabel(job.deployment_target);
            const createdAtLabel = job.created_at_display || formatTimestamp(job.created_at);
            html += `
                <tr>
                    <td>${voiceName}</td>
                    <td><code>${job.job_id}</code></td>
                    <td>${deploymentLabel}</td>
                    <td>${createdAtLabel}</td>
                    <td class="action-buttons">
                        <button class="btn-secondary btn-sm" onclick="deployExportedModel('${job.job_id}', '${voiceName}')">${messages.deploy_action || 'Deploy'}</button>
                        <button class="btn-secondary btn-sm" onclick="downloadModel('${job.job_id}')">${messages.download_action || 'Download'}</button>
                        <button class="btn-secondary btn-sm btn-danger" onclick="deleteModel('${job.job_id}')">${messages.delete_action || 'Delete'}</button>
                    </td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        modelsList.innerHTML = html;
    } catch (error) {
        console.error('Failed to refresh models:', error);
        modelsList.innerHTML = `<p style="color: var(--error);">${messages.error || 'Failed to load models. Is the training service running?'}</p>`;
    }
}

/** Refresh the training-jobs table with current job status information. */
async function refreshTrainingJobs() {
    const jobsList = document.getElementById('training-jobs-list');
    const messages = getProviderMessages(getTrainingProviderId())?.job_list || {};
    if (!jobsList) return;

    try {
        jobsList.innerHTML = `<p>${messages.loading || 'Loading training jobs...'}</p>`;

        const response = await fetchTrainingRequest('/api/training/jobs', '/jobs');
        if (!response.ok) throw new Error('Failed to fetch training jobs');

        const data = await response.json();
        const jobs = Array.isArray(data) ? data : (data.jobs || []);

        if (jobs.length === 0) {
            jobsList.innerHTML = `<p>${messages.empty || 'No training jobs found.'}</p>`;
            return;
        }

        let html = '<table class="data-table"><thead><tr><th>Voice Name</th><th>Status</th><th>Target</th><th>Progress</th><th>Created</th><th>Actions</th></tr></thead><tbody>';

        jobs.forEach(job => {
            const statusClass = job.status === 'completed' ? 'status-badge-success' :
                              job.status === 'failed' ? 'status-badge-error' :
                              job.status === 'training' || job.status === 'running' ? 'status-badge-active' :
                              job.status === 'interrupted' ? 'status-badge-warning' : 'status-badge-default';

            const voiceName = job.voice_name || job.model_name || job.job_id;
            const deploymentLabel = job.deployment_target_label || getTrainingDeploymentLabel(job.deployment_target);
            const createdAtLabel = job.created_at_display || formatTimestamp(job.created_at);
            let actionButtons = `<button class="btn-secondary btn-sm" onclick="viewJobDetails('${job.job_id}')">${messages.details_action || 'Details'}</button>`;
            if (job.status === 'interrupted') {
                actionButtons += ` <button class="btn-secondary btn-sm" onclick="resumeTraining('${voiceName}')">${messages.resume_action || 'Resume'}</button>`;
            } else if (job.status !== 'completed' && job.status !== 'failed') {
                actionButtons += ` <button class="btn-secondary btn-sm btn-danger" onclick="cancelJob('${job.job_id}')">${messages.cancel_action || 'Cancel'}</button>`;
            }

            html += `
                <tr>
                    <td>${voiceName}</td>
                    <td><span class="status-badge ${statusClass}">${job.status}</span></td>
                    <td>${deploymentLabel}</td>
                    <td>${(job.progress || 0).toFixed(1)}%</td>
                    <td>${createdAtLabel}</td>
                    <td class="action-buttons">${actionButtons}</td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        jobsList.innerHTML = html;
    } catch (error) {
        console.error('Failed to refresh training jobs:', error);
        jobsList.innerHTML = `<p style="color: var(--error);">${messages.error || 'Failed to load training jobs.'}</p>`;
    }
}

/** Export a completed training job into PiperTTS and refresh voice lists. */
async function deployExportedModel(jobId, modelName) {
    const messages = getProviderMessages(getTrainingProviderId())?.model_management || {};
    try {
        const deploymentTarget = getSelectedTrainingDeploymentTarget('model-deployment-target');
        const deploymentLabel = getTrainingDeploymentLabel(deploymentTarget);
        showNotification(
            formatMessage(messages.deploy_start, { model_name: modelName, deployment_target: deploymentLabel }) || `Deploying "${modelName}" to ${deploymentLabel}...`,
            'info'
        );

        const formData = new FormData();
        formData.append('model_name', modelName);
        if (deploymentTarget) formData.append('deployment_target', deploymentTarget);

        const response = await fetch(`/api/training/export/${jobId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Export failed' }));
            throw new Error(errorData.detail || 'Export failed');
        }

        const result = await response.json();
        const deploymentResult = result.deployment || {};
        const resolvedDeploymentLabel = deploymentResult.target_label || getTrainingDeploymentLabel(deploymentResult.target || deploymentTarget);
        showNotification(
            formatMessage(messages.deploy_success, {
                model_name: modelName,
                status: deploymentResult.status || 'ok',
                deployment_target: resolvedDeploymentLabel,
            }) || `Model "${modelName}" deployment status: ${deploymentResult.status || 'ok'} on ${resolvedDeploymentLabel}.`,
            'success'
        );

        if ((deploymentResult.target || deploymentTarget || '').startsWith('piper')) {
            refreshTTSVoices();
            refreshCustomVoices();
        }
    } catch (error) {
        console.error('Export error:', error);
        showNotification(
            formatMessage(messages.deploy_error, { error: error.message }) || `Export failed: ${error.message}`,
            'error'
        );
    }
}

/** Download the exported ONNX artifact for a completed training job. */
async function downloadModel(jobId) {
    const messages = getProviderMessages(getTrainingProviderId())?.model_management || {};
    try {
        const response = await fetch(`/api/training/download/${jobId}`);
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

        showNotification(messages.download_success || 'Model download started', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showNotification(messages.download_error || 'Failed to download model', 'error');
    }
}

/** Delete a trained model and its associated dataset/checkpoint artifacts. */
async function deleteModel(jobId) {
    const messages = getProviderMessages(getTrainingProviderId())?.model_management || {};
    if (!confirm(formatMessage(messages.delete_confirm, { job_id: jobId }) || `Delete model "${jobId}" and all training data? This cannot be undone.`)) return;

    try {
        const response = await fetch(`/api/training/model/${jobId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Delete failed');

        showNotification(messages.delete_success || 'Model deleted successfully', 'success');
        refreshModels();
        refreshTTSVoices();
    } catch (error) {
        console.error('Delete error:', error);
        showNotification(messages.delete_error || 'Failed to delete model', 'error');
    }
}

/** Cancel an in-progress training job. */
async function cancelJob(jobId) {
    const messages = getProviderMessages(getTrainingProviderId())?.model_management || {};
    if (!confirm(messages.cancel_confirm || 'Cancel this training job?')) return;

    try {
        const response = await fetch(`/api/training/job/${jobId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Cancel failed');

        showNotification(messages.cancel_success || 'Training job cancelled', 'success');
        refreshTrainingJobs();
    } catch (error) {
        console.error('Cancel error:', error);
        showNotification(messages.cancel_error || 'Failed to cancel job', 'error');
    }
}

/** Resume training for the voice name entered in the manual resume form. */
async function resumeTrainingManual() {
    const voiceName = document.getElementById('continue-voice-name').value.trim();
    const messages = getProviderMessages(getTrainingProviderId())?.resume_training || {};
    if (!voiceName) {
        showStatus('continue-status', 'error', messages.validation_name || 'Please enter a voice model name');
        return;
    }
    resumeTraining(voiceName, 'continue-status');
}

/** Start training from an already-prepared dataset on disk. */
async function trainFromDataset() {
    const voiceName = document.getElementById('continue-voice-name').value.trim();
    const epochs = parseInt(document.getElementById('continue-epochs').value) || 10000;
    const deploymentTarget = getSelectedTrainingDeploymentTarget('continue-deployment-target');
    const messages = getProviderMessages(getTrainingProviderId())?.train_from_dataset || {};
    if (!voiceName) {
        showStatus('continue-status', 'error', messages.validation_name || 'Please enter a voice model name');
        return;
    }

    if (!confirm(formatMessage(messages.confirm, { voice_name: voiceName }) || `Start training "${voiceName}" from the existing prepared dataset (train.json / val.json)?`)) return;

    try {
        showStatus(
            'continue-status',
            'info',
            formatMessage(messages.start, { voice_name: voiceName }) || `Starting training for "${voiceName}" from existing dataset...`
        );

        const formData = new FormData();
        formData.append('model_name', voiceName);
        formData.append('epochs', epochs);
        if (deploymentTarget) formData.append('deployment_target', deploymentTarget);

        const response = await fetchTrainingRequest('/api/training/train-from-dataset', '/train-from-dataset', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to start training' }));
            throw new Error(errorData.detail || 'Failed to start training');
        }

        const result = await response.json();
        showStatus(
            'continue-status',
            'success',
            formatMessage(messages.success_status, { job_id: result.job_id }) || `Training started! Job ID: ${result.job_id}`
        );
        showNotification(
            formatMessage(messages.success_notification, { voice_name: voiceName, deployment_target: getTrainingDeploymentLabel(deploymentTarget) }) || `Training started for "${voiceName}" with target ${getTrainingDeploymentLabel(deploymentTarget)}`,
            'success'
        );
        refreshTrainingJobs();
        monitorTrainingProgress(result.job_id);
    } catch (error) {
        console.error('Train from dataset error:', error);
        showStatus(
            'continue-status',
            'error',
            formatMessage(messages.error, { error: error.message }) || `Failed: ${error.message}`
        );
    }
}

/** Ask the backend to resume a training job from its latest checkpoint. */
async function resumeTraining(voiceName, statusElementId = 'training-progress-status') {
    const messages = getProviderMessages(getTrainingProviderId())?.resume_training || {};
    if (!confirm(formatMessage(messages.confirm, { voice_name: voiceName }) || `Resume training for voice "${voiceName}" from the last checkpoint?`)) return;

    try {
        const deploymentTarget = getSelectedTrainingDeploymentTarget('continue-deployment-target');
        showNotification(
            formatMessage(messages.start_notification, { voice_name: voiceName }) || `Resuming training for "${voiceName}"...`,
            'info'
        );

        const formData = new FormData();
        formData.append('model_name', voiceName);
        if (deploymentTarget) formData.append('deployment_target', deploymentTarget);

        const response = await fetchTrainingRequest('/api/training/resume', '/resume-training', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Resume failed' }));
            throw new Error(errorData.detail || 'Resume failed');
        }

        const result = await response.json();
        showNotification(
            formatMessage(messages.success_notification, { voice_name: voiceName, deployment_target: getTrainingDeploymentLabel(deploymentTarget) }) || `Training resumed for "${voiceName}" with target ${getTrainingDeploymentLabel(deploymentTarget)}`,
            'success'
        );
        showStatus(
            statusElementId,
            'info',
            formatMessage(messages.success_status, { voice_name: voiceName }) || `Resumed training for "${voiceName}" — monitoring progress...`
        );
        refreshTrainingJobs();
        monitorTrainingProgress(result.job_id);
    } catch (error) {
        console.error('Resume error:', error);
        showStatus(
            statusElementId,
            'error',
            formatMessage(messages.error_status, { error: error.message }) || `Resume failed: ${error.message}`
        );
        showNotification(
            formatMessage(messages.error_notification, { error: error.message }) || `Resume failed: ${error.message}`,
            'error'
        );
    }
}

/** Fetch and display detailed information for a specific training job. */
async function viewJobDetails(jobId) {
    const messages = getProviderMessages(getTrainingProviderId())?.job_details || {};
    try {
        const response = await fetchTrainingRequest(`/api/training/status/${jobId}`, `/status/${jobId}`);
        if (!response.ok) throw new Error('Failed to fetch job details');

        const job = await response.json();
        const notAvailable = messages.na || 'N/A';
        const voiceName = job.voice_name || job.model_name || job.job_id || jobId;
        const deploymentLabel = job.deployment_target_label || getTrainingDeploymentLabel(job.deployment_target);
        const configSummary = job.config_summary || {};
        const recentLogs = Array.isArray(job.recent_logs) ? job.recent_logs : [];

        let details = `${messages.job_label || 'Job'}: ${voiceName}\n`;
        details += `${messages.status_label || 'Status'}: ${job.status}\n`;
        details += `${messages.deployment_target_label || 'Deployment Target'}: ${deploymentLabel}\n`;
        details += `${messages.progress_label || 'Progress'}: ${(job.progress || 0).toFixed(1)}%\n`;
        details += `${messages.current_epoch_label || 'Current Epoch'}: ${job.current_epoch || 0}\n`;

        if (configSummary.epochs || configSummary.batch_size || configSummary.learning_rate) {
            details += `\n${messages.configuration_heading || 'Configuration'}:\n`;
            details += `  ${messages.epochs_label || 'Epochs'}: ${configSummary.epochs || notAvailable}\n`;
            details += `  ${messages.batch_size_label || 'Batch Size'}: ${configSummary.batch_size || notAvailable}\n`;
            details += `  ${messages.learning_rate_label || 'Learning Rate'}: ${configSummary.learning_rate || notAvailable}\n`;
        }

        if (job.best_loss_display) details += `\n${messages.best_loss_label || 'Best Loss'}: ${job.best_loss_display}\n`;

        if (recentLogs.length > 0) {
            details += `\n${messages.recent_logs_heading || 'Recent Logs'}:\n`;
            recentLogs.forEach(log => {
                details += `  ${log.display || log.message}\n`;
            });
        }

        alert(details);
    } catch (error) {
        console.error('Job details error:', error);
        showNotification(messages.fetch_error || 'Failed to fetch job details', 'error');
    }
}
