# TTS-STT Studio

Self-hosted neural **Text-to-Speech** and **Speech-to-Text** in one web UI:

- **TTS:** PiperTTS (40+ voices, CPU) and Qwen3-TTS (voice cloning, GPU)
- **STT:** Whisper (faster-whisper), Qwen3-ASR, and optional Parakeet-TDT (realtime, 25 EU languages incl. German)
- **Voice training:** custom VITS pipeline (upload audio → segment → train → export)

All processing is local — no data leaves your server. NVIDIA GPU recommended.

After install, open the Web UI on the configured port (default **3000**). Set
**Browser-facing host** to your TrueNAS IP/hostname if you access it from another
machine, and point **Data dataset** at a pool dataset with room for models.
