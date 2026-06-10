"""Browser end-to-end test for the live microphone transcription UI.

Drives a real Chromium via Playwright with fake-media-device flags so
getUserMedia yields a looping German WAV instead of hardware audio, clicks
the Live Transcription button, and asserts that partial/final transcripts
appear. Requires a running frontend + whisper stack and the FAKE_MIC_WAV
file; skips otherwise.

Run:
    FRONTEND_URL=http://localhost:3000 FAKE_MIC_WAV=/path/german.wav \
        pytest tests/test_live_mic_browser.py -v
"""

import os
import time

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
FAKE_MIC_WAV = os.getenv("FAKE_MIC_WAV", "")


def _frontend_reachable() -> bool:
    import httpx

    try:
        return httpx.get(f"{FRONTEND_URL}/health", timeout=5.0).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not FAKE_MIC_WAV or not os.path.exists(FAKE_MIC_WAV), reason="FAKE_MIC_WAV not provided")
def test_live_mic_transcription_in_browser():
    if not _frontend_reachable():
        pytest.skip(f"frontend not reachable at {FRONTEND_URL}")

    console_errors: list[str] = []

    with playwright_sync.sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={FAKE_MIC_WAV}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = browser.new_page()
        page.on("pageerror", lambda err: console_errors.append(str(err)))
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )

        page.goto(FRONTEND_URL, wait_until="domcontentloaded")
        page.wait_for_selector("#live-stt-button", timeout=15000)
        page.click("#live-stt-button")

        # Wait for the first partial transcript (model decode of >=1s of audio)
        deadline = time.time() + 60
        partial_text = ""
        while time.time() < deadline:
            partial_text = (
                page.inner_text("#live-stt-confirmed") + " " + page.inner_text("#live-stt-pending")
            ).strip()
            if partial_text:
                break
            time.sleep(0.5)
        assert partial_text, f"no partial transcript appeared; console errors: {console_errors}"

        # Let a bit more audio stream, then stop and wait for the final transcript
        time.sleep(4)
        page.click("#live-stt-button")

        deadline = time.time() + 60
        final_status = ""
        while time.time() < deadline:
            final_status = page.inner_text("#live-stt-status")
            if "Final transcript" in final_status:
                break
            time.sleep(0.5)
        assert "Final transcript" in final_status, (
            f"final transcript never arrived; status='{final_status}', console errors: {console_errors}"
        )

        final_text = page.inner_text("#live-stt-confirmed").strip().lower()
        browser.close()

    # The fake mic loops a German sentence about realtime transcription
    assert any(word in final_text for word in ("transkription", "mikrofon", "echtzeit")), (
        f"unexpected final transcript: '{final_text}'; console errors: {console_errors}"
    )
