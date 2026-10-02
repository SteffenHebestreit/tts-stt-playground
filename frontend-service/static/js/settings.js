// The Settings page (placeholder until the page itself is built).
// It runs under a strict CSP (script-src 'self'), so no inline script, eval or
// innerHTML: text goes in through textContent only.
'use strict';

(function () {
    const status = document.getElementById('settings-status');
    if (status) {
        status.textContent = 'The Settings page is not built yet; its API answers at /api/settings.';
    }
})();
