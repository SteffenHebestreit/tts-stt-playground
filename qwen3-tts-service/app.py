import os
import io
import gc
import json
import re
import time
import uuid
import subprocess
import tempfile
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(
    title="Qwen3-TTS Service",
    description="Text-to-Speech and Voice Cloning using Qwen3-TTS"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# STT service URLs for auto-transcription
QWEN3_ASR_URL = os.getenv("QWEN3_ASR_SERVICE_URL", "http://qwen3-asr-service:5002")

# Global model state
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = None
model_loaded = False
current_model_name = ""

# Available Qwen3-TTS model variants
AVAILABLE_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {
        "name": "1.7B Base",
        "description": "General-purpose TTS and voice cloning (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "voice_clone"],
    },
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {
        "name": "0.6B Base",
        "description": "Smaller, faster model for TTS and voice cloning (2.5GB)",
        "size": "0.6B",
        "capabilities": ["tts", "voice_clone"],
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "name": "1.7B CustomVoice",
        "description": "Optimized for custom voice synthesis with built-in speakers (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "custom_voice"],
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "name": "1.7B VoiceDesign",
        "description": "Design voices via text description (4.5GB)",
        "size": "1.7B",
        "capabilities": ["tts", "voice_design"],
    },
}

# Supported languages
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian"
]

# Built-in speaker names (Qwen3-TTS CustomVoice speakers)
BUILTIN_SPEAKERS = [
    "Vivian", "Ryan", "Ethan", "Olivia", "Aria",
    "Liam", "Nova", "Atlas", "Aurora", "Kai"
]


def load_model(model_name=None):
    """Load a Qwen3-TTS model by name. Unloads the current model first if switching."""
    global tts_model, model_loaded, current_model_name

    if model_name is None:
        model_name = os.getenv("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    # Already loaded
    if tts_model is not None and current_model_name == model_name:
        return tts_model

    # Unload previous model
    if tts_model is not None:
        print(f"Unloading current model: {current_model_name}")
        del tts_model
        tts_model = None
        model_loaded = False
        current_model_name = ""
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"Loading Qwen3-TTS model: {model_name}")
    try:
        from qwen_tts import Qwen3TTSModel

        attn_impl = "flash_attention_2"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn_impl = "sdpa"
            print("flash-attn not available, using SDPA attention")

        tts_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=f"{device}:0" if device == "cuda" else "cpu",
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation=attn_impl,
        )
        model_loaded = True
        current_model_name = model_name
        print(f"Qwen3-TTS model '{model_name}' loaded on {device}")
    except Exception as e:
        print(f"Failed to load Qwen3-TTS model: {e}")
        raise
    return tts_model


def get_model():
    """Load or return cached Qwen3-TTS model."""
    if tts_model is None:
        return load_model()
    return tts_model


@app.on_event("startup")
async def startup():
    """Pre-load model on startup."""
    try:
        get_model()
    except Exception as e:
        print(f"Warning: Could not preload model: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device,
    }


@app.get("/status")
async def status():
    status_info = {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "current_model": current_model_name,
        "current_model_info": AVAILABLE_MODELS.get(current_model_name),
        "supported_languages": SUPPORTED_LANGUAGES,
        "builtin_speakers": BUILTIN_SPEAKERS,
    }
    if torch.cuda.is_available():
        status_info["gpu_name"] = torch.cuda.get_device_name(0)
        status_info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
        status_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    return status_info


@app.get("/models")
async def list_models():
    """List available Qwen3-TTS model variants."""
    return {
        "models": AVAILABLE_MODELS,
        "current_model": current_model_name,
    }


class LoadModelRequest(BaseModel):
    model: str


@app.post("/load_model")
async def switch_model(request: LoadModelRequest):
    """Switch to a different Qwen3-TTS model variant. Downloads if not cached."""
    model_name = request.model

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    if model_name == current_model_name:
        return {"status": "ok", "message": "Model already loaded", "model": model_name}

    try:
        load_model(model_name)
        return {
            "status": "ok",
            "message": f"Model switched to {model_name}",
            "model": model_name,
            "model_info": AVAILABLE_MODELS[model_name],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/speakers")
async def list_speakers():
    """List available built-in speakers."""
    return {
        "speakers": BUILTIN_SPEAKERS,
        "languages": SUPPORTED_LANGUAGES,
    }


def _cleanup_temp(path):
    """Safely remove a temp file."""
    if path and os.path.exists(path):
        os.unlink(path)


async def _auto_transcribe(audio_path: str, filename: str = "audio.wav") -> dict:
    """Auto-transcribe an audio file using the Qwen3-ASR service.
    Returns full response dict with text, segments, duration."""
    try:
        print(f"Auto-transcribing reference audio via Qwen3-ASR: {filename}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    f"{QWEN3_ASR_URL}/transcribe",
                    files={"audio": (filename, f)},
                )
            if response.status_code == 200:
                data = response.json()
                text = data.get("text", "").strip()
                print(f"Auto-transcription result: '{text[:100]}...'")
                return data
            else:
                print(f"Auto-transcription failed: HTTP {response.status_code}")
                return {}
    except Exception as e:
        print(f"Auto-transcription error: {e}")
        return {}


def _trim_audio_segment(input_path: str, start: float, end: float, output_path: str) -> bool:
    """Use ffmpeg to extract a segment from an audio file."""
    try:
        duration = end - start
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-ar", "24000", "-ac", "1",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"ffmpeg trim error: {e}")
        return False


def _pick_best_segment(segments: list, min_dur: float = 3.0, max_dur: float = 10.0) -> dict:
    """Pick the best reference segment from ASR timestamps.
    Prefers segments between min_dur and max_dur seconds."""
    if not segments:
        return {}
    # Try to find a single segment in the ideal range
    for seg in segments:
        dur = seg.get("end", 0) - seg.get("start", 0)
        if min_dur <= dur <= max_dur:
            return seg
    # Otherwise merge consecutive segments to reach min_dur
    merged_start = segments[0].get("start", 0)
    merged_end = segments[0].get("end", 0)
    merged_text = segments[0].get("text", "")
    for seg in segments[1:]:
        merged_end = seg.get("end", 0)
        merged_text += " " + seg.get("text", "")
        if (merged_end - merged_start) >= min_dur:
            break
    return {"start": merged_start, "end": min(merged_end, merged_start + max_dur), "text": merged_text.strip()}


# --- Voice Library: persistent speaker prompt cache ---

VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)


def _voice_dir(voice_id: str) -> Path:
    return VOICES_DIR / voice_id


def _load_voice_metadata(voice_id: str) -> dict:
    meta_path = _voice_dir(voice_id) / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _save_voice_prompt(voice_id: str, prompt_item, metadata: dict):
    """Save a VoiceClonePromptItem to disk as tensors + metadata."""
    vdir = _voice_dir(voice_id)
    vdir.mkdir(parents=True, exist_ok=True)
    # Save tensors
    torch.save(prompt_item.ref_spk_embedding, vdir / "ref_spk_embedding.pt")
    if prompt_item.ref_code is not None:
        torch.save(prompt_item.ref_code, vdir / "ref_code.pt")
    # Save metadata
    metadata.update({
        "x_vector_only_mode": prompt_item.x_vector_only_mode,
        "icl_mode": prompt_item.icl_mode,
        "ref_text": prompt_item.ref_text,
    })
    (vdir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


def _load_voice_prompt(voice_id: str):
    """Load a cached VoiceClonePromptItem from disk (x_vector_only for fast TTS)."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    vdir = _voice_dir(voice_id)
    if not vdir.exists():
        return None
    ref_spk = torch.load(vdir / "ref_spk_embedding.pt", map_location=device, weights_only=True)
    return VoiceClonePromptItem(
        ref_code=None,
        ref_spk_embedding=ref_spk,
        x_vector_only_mode=True,
        icl_mode=False,
        ref_text=None,
    )


def _list_voices() -> list:
    """List all saved voices."""
    voices = []
    if not VOICES_DIR.exists():
        return voices
    for vdir in sorted(VOICES_DIR.iterdir()):
        if vdir.is_dir() and (vdir / "metadata.json").exists():
            meta = json.loads((vdir / "metadata.json").read_text())
            meta["id"] = vdir.name
            voices.append(meta)
    return voices


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked generation.

    Keeps sentences together if they're very short (<20 chars) to avoid
    generating tiny audio fragments with bad prosody.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?;:])\s+', text.strip())
    if not raw:
        return [text]

    # Merge very short fragments with the next sentence
    merged = []
    buf = ""
    for s in raw:
        if buf:
            buf += " " + s
        else:
            buf = s
        if len(buf) >= 20:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged


def _generate_chunks(model, sentences: list[str], language: str, voice_clone_prompt, sample_rate: int = 24000) -> tuple[np.ndarray, int]:
    """Generate audio for sentences using batch mode for speed, then concatenate with gaps."""
    gap = np.zeros(int(sample_rate * 0.15), dtype=np.float32)  # 150ms gap

    # Batch mode: generate all sentences at once (2x+ faster than sequential)
    prompt_item = voice_clone_prompt[0] if voice_clone_prompt else None
    prompts = [prompt_item] * len(sentences)
    langs = [language] * len(sentences)

    print(f"  Generating {len(sentences)} sentences in batch mode...")
    start = time.time()
    wavs, sr = model.generate_voice_clone(
        text=sentences,
        language=langs,
        voice_clone_prompt=prompts,
    )
    elapsed = time.time() - start
    print(f"  Batch generation done in {elapsed:.2f}s")

    # Concatenate with gaps
    all_audio = []
    for i, wav in enumerate(wavs):
        all_audio.append(np.array(wav))
        if i < len(wavs) - 1:
            all_audio.append(gap)
    del wavs

    return np.concatenate(all_audio), sr


@app.get("/voices")
async def list_saved_voices():
    """List all saved voice profiles."""
    return {"voices": _list_voices()}


@app.post("/voices/save")
async def save_voice(
    name: str = Form(...),
    lang: str = Form("auto"),
    file: UploadFile = File(...),
):
    """Upload reference audio, extract speaker prompt, and save for reuse.
    Trims audio to best segment via ASR timestamps + ffmpeg."""
    model_info = AVAILABLE_MODELS.get(current_model_name, {})
    if "voice_clone" not in model_info.get("capabilities", []):
        raise HTTPException(status_code=400, detail="Current model does not support voice cloning. Switch to a Base model.")

    tmp_path = None
    trimmed_path = None
    try:
        model = get_model()
        start_time = time.time()

        # Save uploaded file
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Auto-transcribe to get text + segments for trimming
        asr_result = await _auto_transcribe(tmp_path, file.filename or "reference.wav")
        ref_text = asr_result.get("text", "").strip()
        segments = asr_result.get("segments", [])

        if not ref_text:
            raise HTTPException(status_code=400, detail="Could not transcribe reference audio. Please try a clearer recording.")

        # Trim to best segment using ffmpeg
        audio_path = tmp_path
        segment_text = ref_text
        best = _pick_best_segment(segments)
        if best and best.get("start") is not None:
            trimmed_path = tmp_path + "_trimmed.wav"
            if _trim_audio_segment(tmp_path, best["start"], best["end"], trimmed_path):
                audio_path = trimmed_path
                segment_text = best.get("text", ref_text)
                dur = best["end"] - best["start"]
                print(f"Trimmed reference to {dur:.1f}s segment: '{segment_text[:60]}...'")

        # Extract speaker embedding (x_vector_only for fast TTS)
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=audio_path,
            x_vector_only_mode=True,
        )
        prompt_item = prompt_items[0]

        # Save to disk
        voice_id = name.lower().replace(" ", "_").replace("/", "_")
        _save_voice_prompt(voice_id, prompt_item, {
            "name": name,
            "lang": lang,
            "original_filename": file.filename,
            "ref_text": segment_text,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_used": current_model_name,
        })

        elapsed = time.time() - start_time
        print(f"Voice '{name}' saved as '{voice_id}' in {elapsed:.1f}s")

        return {
            "status": "ok",
            "voice_id": voice_id,
            "name": name,
            "ref_text": segment_text,
            "processing_time": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Save voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_temp(tmp_path)
        _cleanup_temp(trimmed_path)


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a saved voice profile."""
    import shutil
    vdir = _voice_dir(voice_id)
    if not vdir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    shutil.rmtree(vdir)
    return {"status": "ok", "deleted": voice_id}


@app.post("/voices/{voice_id}/tts")
async def tts_with_saved_voice(
    voice_id: str,
    text: str = Form(...),
    lang: str = Form("English"),
):
    """Generate speech using a saved voice profile. Skips audio processing — fast."""
    prompt_item = _load_voice_prompt(voice_id)
    if prompt_item is None:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    if not text:
        raise HTTPException(status_code=400, detail="Text not provided")

    model_info = AVAILABLE_MODELS.get(current_model_name, {})
    if "voice_clone" not in model_info.get("capabilities", []):
        raise HTTPException(status_code=400, detail="Current model does not support voice cloning. Switch to a Base model.")

    try:
        model = get_model()
        start_time = time.time()

        sentences = _split_sentences(text)
        print(f"Saved-voice TTS: {len(sentences)} chunks for voice={voice_id}")

        if len(sentences) > 1:
            audio, sr = _generate_chunks(model, sentences, lang, [prompt_item])
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=lang,
                voice_clone_prompt=[prompt_item],
            )
            audio = np.array(wavs[0])
            del wavs

        generation_time = time.time() - start_time
        audio_duration = len(audio) / sr
        print(f"Saved-voice TTS done in {generation_time:.2f}s ({audio_duration:.1f}s audio, voice={voice_id})")

        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)

        del audio
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Audio-Duration": f"{audio_duration:.3f}",
                "X-Voice-ID": voice_id,
                "X-Language": lang,
            },
        )

    except Exception as e:
        print(f"Saved-voice TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TTSRequest(BaseModel):
    text: str
    lang: str = "English"
    speaker: str = "Vivian"
    instruct: str = ""


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech using a built-in speaker voice."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text not provided")

    try:
        model = get_model()
        start_time = time.time()

        # Check if model has custom voice generation (CustomVoice model)
        if hasattr(model, 'generate_custom_voice'):
            wavs, sr = model.generate_custom_voice(
                text=request.text,
                language=request.lang,
                speaker=request.speaker,
                instruct=request.instruct or "",
            )
        else:
            # Base model fallback — use voice cloning with empty ref
            # For base model, we need a reference audio; generate simple output
            wavs, sr = model.generate_voice_clone(
                text=request.text,
                language=request.lang,
                ref_audio=None,
                ref_text="",
            )

        generation_time = time.time() - start_time
        print(f"TTS generated in {generation_time:.2f}s")

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, np.array(wavs[0]), sr, format="WAV")
        buffer.seek(0)

        # Memory cleanup
        del wavs
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Speaker": request.speaker,
                "X-Language": request.lang,
            },
        )

    except Exception as e:
        print(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clone")
async def clone_voice(
    text: str = Form(...),
    lang: str = Form("English"),
    file: UploadFile = File(...),
):
    """
    Clone a voice from a reference audio file (3+ seconds recommended).
    Provide reference audio as 'file' and the text to synthesize as 'text'.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text not provided")

    tmp_path = None
    try:
        model = get_model()

        model_info = AVAILABLE_MODELS.get(current_model_name, {})
        capabilities = model_info.get("capabilities", [])
        if "voice_clone" not in capabilities:
            raise HTTPException(
                status_code=400,
                detail=f"Current model '{model_info.get('name', current_model_name)}' does not support voice cloning. "
                       f"Please switch to a Base model (1.7B Base or 0.6B Base)."
            )
        start_time = time.time()

        # Save uploaded reference audio
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        print(f"Voice clone request: text='{text[:50]}...', lang={lang}, ref={file.filename}")

        # Auto-transcribe the reference audio using Qwen3-ASR
        asr_result = await _auto_transcribe(tmp_path, file.filename or "reference.wav")
        ref_text = asr_result.get("text", "").strip() if asr_result else ""
        if not ref_text:
            print("Auto-transcription returned empty, trying clone without ref_text")

        # Extract speaker embedding first (fast), then use it for chunked generation
        try:
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=tmp_path,
                x_vector_only_mode=True,
            )
            prompt_item = prompt_items[0] if isinstance(prompt_items, list) else prompt_items
        except Exception as clone_err:
            err_msg = str(clone_err)
            if "does not support generate_voice_clone" in err_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Current model does not support voice cloning. "
                           f"Please switch to a Base model (1.7B Base or 0.6B Base)."
                )
            raise

        # Chunked generation for long texts
        sentences = _split_sentences(text)
        print(f"Voice clone: {len(sentences)} chunks, ref={file.filename}")

        try:
            if len(sentences) > 1:
                audio, sr = _generate_chunks(model, sentences, lang, [prompt_item])
            else:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    voice_clone_prompt=[prompt_item],
                )
                audio = np.array(wavs[0])
                del wavs
        except Exception as gen_err:
            err_msg = str(gen_err)
            if "ref_text" in err_msg.lower() or "icl mode" in err_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail="Voice cloning requires reference text but auto-transcription failed. "
                           "Please enable 'Provide reference text' and enter the transcript manually."
                )
            raise

        generation_time = time.time() - start_time
        audio_duration = len(audio) / sr
        print(f"Voice clone done in {generation_time:.2f}s ({audio_duration:.1f}s audio)")

        # Convert to WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)

        # Cleanup
        del audio
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Audio-Duration": f"{audio_duration:.3f}",
                "X-Clone-Source": file.filename or "unknown",
                "X-Language": lang,
            },
        )

    except Exception as e:
        print(f"Voice clone error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_temp(tmp_path)


@app.post("/clone-with-ref-text")
async def clone_voice_with_ref_text(
    text: str = Form(...),
    ref_text: str = Form(...),
    lang: str = Form("English"),
    file: UploadFile = File(...),
):
    """
    High-quality voice cloning with reference text.
    Provide reference audio + its transcript for best results.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text not provided")
    if not ref_text:
        raise HTTPException(status_code=400, detail="Reference text not provided")

    tmp_path = None
    try:
        model = get_model()
        start_time = time.time()

        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang,
            ref_audio=tmp_path,
            ref_text=ref_text,
        )

        generation_time = time.time() - start_time
        print(f"High-quality voice clone generated in {generation_time:.2f}s")

        buffer = io.BytesIO()
        sf.write(buffer, np.array(wavs[0]), sr, format="WAV")
        buffer.seek(0)

        del wavs
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Language": lang,
            },
        )

    except Exception as e:
        print(f"High-quality clone error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_temp(tmp_path)


class VoiceDesignRequest(BaseModel):
    text: str
    voice_description: str
    lang: str = "English"


@app.post("/voice_design")
async def voice_design(request: VoiceDesignRequest):
    """
    Generate speech with a voice designed from a text description.
    Requires the VoiceDesign model to be loaded.
    Example description: "A deep male voice with a warm, calm British accent"
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text not provided")
    if not request.voice_description:
        raise HTTPException(status_code=400, detail="Voice description not provided")

    try:
        model = get_model()
        start_time = time.time()

        if hasattr(model, 'generate_voice_design'):
            wavs, sr = model.generate_voice_design(
                text=request.text,
                language=request.lang,
                instruct=request.voice_description,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Current model does not support voice design. Please switch to the VoiceDesign model."
            )

        generation_time = time.time() - start_time
        print(f"Voice design generated in {generation_time:.2f}s")

        buffer = io.BytesIO()
        sf.write(buffer, np.array(wavs[0]), sr, format="WAV")
        buffer.seek(0)

        del wavs
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{generation_time:.3f}",
                "X-Language": request.lang,
                "X-Voice-Description": request.voice_description[:100],
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Voice design error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5004)
