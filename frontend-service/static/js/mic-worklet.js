/**
 * Microphone capture worklet for live transcription.
 *
 * Runs on the dedicated audio rendering thread, so main-thread work (layout,
 * GC, rendering partial transcripts) can no longer drop capture frames the way
 * it did with ScriptProcessorNode.
 *
 * The node expects to run in an AudioContext already at 16 kHz — the caller is
 * responsible for that, and the constructor refuses to run otherwise rather
 * than silently emitting samples the server would misread as 16 kHz.
 *
 * Posts Int16 PCM (little-endian, mono) to the main thread in FRAMES_PER_POST
 * blocks, transferring the underlying buffer so nothing is copied.
 */

// 512 samples @ 16 kHz = 32 ms, and exactly 4 render quanta of 128 frames.
const FRAMES_PER_POST = 512;

class MicCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const expected = (options && options.processorOptions && options.processorOptions.sampleRate) || 16000;
        if (sampleRate !== expected) {
            this.port.postMessage({
                type: 'error',
                message: `AudioWorklet context is ${sampleRate} Hz, expected ${expected} Hz`,
            });
            this.disabled = true;
        }
        this.buffer = new Int16Array(FRAMES_PER_POST);
        this.offset = 0;
    }

    process(inputs) {
        if (this.disabled) return false;

        const channel = inputs[0] && inputs[0][0];
        // No input connected yet (or the track ended): stay alive and wait.
        if (!channel) return true;

        for (let i = 0; i < channel.length; i++) {
            const s = Math.max(-1, Math.min(1, channel[i]));
            this.buffer[this.offset++] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            if (this.offset === FRAMES_PER_POST) {
                // Transfer rather than copy; allocate a fresh buffer for the next block.
                this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
                this.buffer = new Int16Array(FRAMES_PER_POST);
                this.offset = 0;
            }
        }
        return true;
    }
}

registerProcessor('mic-capture', MicCaptureProcessor);
