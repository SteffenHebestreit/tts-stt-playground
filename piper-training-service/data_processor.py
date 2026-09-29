"""Dataset preparation utilities for Piper VITS training.

Handles audio loading, mel-spectrogram extraction, phonemisation, and
train/validation splitting.
"""

import os
import json
import io
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from audio_sources import AudioSourceError, max_upload_bytes, resolve_audio_source
from phonemization import normalize_language, phonemize_texts
from validation import split_train_val
import soundfile as sf
import asyncio
import aiofiles
import aiohttp
import subprocess
import logging

logger = logging.getLogger(__name__)

# ffprobe reads a header; it should answer immediately. A bound stops a wedged
# process from hanging the caller indefinitely.
FFPROBE_TIMEOUT_S = 30

class DataProcessor:
    """Prepare audio/text examples and mel features for VITS training."""

    def __init__(self):
        """Initialise default audio and spectrogram parameters."""
        self.sample_rate = 22050
        self.hop_length = 256
        self.n_fft = 1024
        self.n_mels = 80
        
    async def prepare_dataset(self, segments: List, model_name: str, language: str = "de",
                              stats: Optional[dict] = None) -> Path:
        """Build a training dataset from STT segments (audio + mel + phonemes).

        Every audio_path is vetted before anything is written (see
        ``audio_sources``): ``AudioSourceError`` for the first that is not
        permitted, ``PhonemizationError`` if phonemes cannot be produced.
        The decode / trim / mel work of each segment runs in a worker thread, so
        the event loop (health checks, status polls) stays responsive for the
        minutes a large dataset takes. *stats*, if given, receives the counts.
        """
        normalize_language(language)  # ValueError before any file is touched

        sources = []
        for idx, segment in enumerate(segments):
            try:
                sources.append(resolve_audio_source(segment.audio_path))
            except AudioSourceError as exc:
                raise AudioSourceError(f"segments[{idx}]: {exc}") from exc

        # Fails fast if phonemizer/espeak is missing, before minutes of audio work.
        await asyncio.to_thread(phonemize_texts, [], language)

        dataset_dir = Path(f"data/{model_name}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        (dataset_dir / "audio").mkdir(exist_ok=True)
        (dataset_dir / "mel").mkdir(exist_ok=True)
        
        metadata = []
        skipped = 0
        
        logger.info(f"Processing {len(segments)} segments for model {model_name}")
        
        for idx, (segment, (kind, source)) in enumerate(zip(segments, sources)):
            try:
                if kind == "url":
                    audio_source = io.BytesIO(await self._download_audio(source))
                else:
                    audio_source = str(source)

                entry = await asyncio.to_thread(
                    self._process_segment, idx, audio_source, segment, dataset_dir)
                if entry is None:
                    skipped += 1
                    continue
                metadata.append(entry)
                logger.info(f"Processed segment {idx+1}/{len(segments)}")
                
            except Exception as e:
                logger.error(f"Error processing segment {idx}: {e}")
                skipped += 1
                continue
        
        if not metadata:
            raise ValueError("No valid segments were processed. Please check your audio files and transcriptions.")

        # Phonemes for the segments that made it this far. A segment whose text
        # cannot be phonemised is dropped, never given its raw text as phonemes.
        all_phonemes = await asyncio.to_thread(
            phonemize_texts, [entry['text'] for entry in metadata], language)
        prepared = []
        for entry, phonemes in zip(metadata, all_phonemes):
            if phonemes is None:
                continue
            entry['phonemes'] = phonemes
            prepared.append(entry)
        metadata = prepared
        
        if stats is not None:
            stats.update({
                'received': len(segments),
                'prepared': len(metadata),
                'skipped': len(segments) - len(metadata),
            })
        
        # Save metadata
        metadata_path = dataset_dir / "metadata.json"
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        # Create train/val split
        await self._create_splits(metadata, dataset_dir)
        
        logger.info(f"Dataset prepared with {len(metadata)} samples at {dataset_dir}")
        return dataset_dir

    def _process_segment(self, idx: int, audio_source, segment, dataset_dir: Path) -> Optional[dict]:
        """Decode, trim, normalise and write one segment; ``None`` if it is unusable.

        Blocking (librosa, soundfile, numpy): call it through ``asyncio.to_thread``.
        Only the requested slice is decoded. Loading the whole file for every
        segment made a long recording cost one full decode per segment.
        """
        text = (segment.text or "").strip()
        if not text:
            logger.warning(f"Skipping segment {idx}: empty text")
            return None

        start = float(getattr(segment, 'start_time', 0.0) or 0.0)
        end = float(getattr(segment, 'end_time', 0.0) or 0.0)
        if start < 0 or end <= start:
            logger.warning(f"Skipping segment {idx}: invalid time range {start}-{end}")
            return None

        audio, sr = librosa.load(audio_source, sr=self.sample_rate, offset=start, duration=end - start)

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Skip if too short
        if len(audio) < sr * 0.5:  # Less than 0.5 seconds
            logger.warning(f"Skipping segment {idx}: too short ({len(audio)/sr:.2f}s)")
            return None
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        # Save processed audio
        output_audio_path = dataset_dir / "audio" / f"{idx:05d}.wav"
        sf.write(output_audio_path, audio, self.sample_rate)
        
        # Compute mel spectrogram
        mel_spec = self._compute_mel_spectrogram(audio)
        mel_path = dataset_dir / "mel" / f"{idx:05d}.npy"
        np.save(mel_path, mel_spec)
        
        return {
            'audio_path': str(output_audio_path.relative_to(dataset_dir)),
            'mel_path': str(mel_path.relative_to(dataset_dir)),
            'text': text,
            'duration': len(audio) / self.sample_rate,
            'segment_id': idx
        }
    
    async def _download_audio(self, url: str) -> bytes:
        """Download an audio file from *url* (already vetted) and return the raw bytes.

        Bounded in size and time, and redirects are refused: a redirect would let
        an allowed host send the request to one that is not.
        """
        limit = max_upload_bytes()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60),
                                   allow_redirects=False) as resp:
                if 300 <= resp.status < 400:
                    raise AudioSourceError(f"{url} redirected ({resp.status}); redirects are not followed")
                resp.raise_for_status()
                if resp.content_length is not None and resp.content_length > limit:
                    raise AudioSourceError(f"{url} is {resp.content_length} bytes; the limit is {limit}")
                chunks, received = [], 0
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    received += len(chunk)
                    if received > limit:
                        raise AudioSourceError(f"{url} exceeds the {limit} byte limit")
                    chunks.append(chunk)
                return b"".join(chunks)
    
    def _compute_mel_spectrogram(self, audio):
        """Compute a normalised log-mel spectrogram from a waveform array."""
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window='hann'
        )
        
        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(stft)**2,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sample_rate // 2
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        mel_spec = 2 * mel_spec - 1
        
        return mel_spec
    
    async def _create_splits(self, metadata: List[Dict], dataset_dir: Path):
        """Write train.json / val.json with a reproducible 90/10 split.

        Shares ``split_train_val`` with the segmenter rather than repeating the
        unseeded ``np.random.permutation`` that used to be here: an unseeded
        split means two runs over the same data validate against different sets
        and their loss curves cannot be compared. It also refuses a dataset too
        small to divide, instead of handing back an empty training set.
        """
        train_metadata, val_metadata = split_train_val(metadata)
        
        async with aiofiles.open(dataset_dir / "train.json", 'w') as f:
            await f.write(json.dumps(train_metadata, indent=2))
        
        async with aiofiles.open(dataset_dir / "val.json", 'w') as f:
            await f.write(json.dumps(val_metadata, indent=2))
        
        logger.info(f"Created splits: {len(train_metadata)} training, {len(val_metadata)} validation")

    def create_speaker_map(self, dataset_dir: Path) -> Dict:
        """Create and persist a speaker-name → integer-ID mapping."""
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        speakers = set()
        for item in metadata:
            speaker = item.get('speaker_name', 'default')
            speakers.add(speaker)
        
        speaker_map = {speaker: idx for idx, speaker in enumerate(sorted(speakers))}
        
        # Save speaker map
        speaker_map_path = dataset_dir / "speaker_map.json"
        with open(speaker_map_path, 'w') as f:
            json.dump(speaker_map, f, indent=2)
        
        return speaker_map

    def analyze_audio_with_ffmpeg(self, audio_path: str) -> dict:
        """Use ffprobe to extract codec, duration, sample-rate, etc."""
        try:
            # Use ffprobe to get audio information
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", audio_path
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=FFPROBE_TIMEOUT_S)
            info = json.loads(result.stdout)
            
            # Extract audio stream information
            audio_stream = None
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise ValueError("No audio stream found")
            
            analysis = {
                "format": info["format"]["format_name"],
                "duration": float(info["format"]["duration"]),
                "bitrate": int(info["format"].get("bit_rate", 0)),
                "codec": audio_stream["codec_name"],
                "sample_rate": int(audio_stream["sample_rate"]),
                "channels": int(audio_stream["channels"]),
                "channel_layout": audio_stream.get("channel_layout", "unknown"),
                "bits_per_sample": audio_stream.get("bits_per_sample", 0),
                "file_size": int(info["format"]["size"])
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing audio {audio_path}: {e}")
            return {}

    def preprocess_audio(self, audio_path: str, output_dir: str, target_sample_rate: int = 22050) -> str:
        """Resample and normalise an audio file, saving the result as 16-bit WAV."""
        try:
            # Analyze audio first
            analysis = self.analyze_audio_with_ffmpeg(audio_path)
            logger.info(f"Audio analysis for {audio_path}: {analysis}")
            
            # Check if audio needs conversion
            needs_conversion = (
                analysis.get("sample_rate", 0) != target_sample_rate or
                analysis.get("channels", 0) != 1 or
                analysis.get("format", "").lower() not in ["wav", "wave"]
            )
            
            if needs_conversion:
                logger.info(f"Converting audio: SR {analysis.get('sample_rate')} -> {target_sample_rate}, "
                           f"Channels {analysis.get('channels')} -> 1")
            
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Resample if necessary
            if sr != target_sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
            
            # Normalize audio. The +1e-6 matters: a fully silent file makes
            # np.max(np.abs(audio)) zero, and the division then writes a WAV of
            # NaN. prepare_dataset() guards this the same way.
            audio = audio / (np.max(np.abs(audio)) + 1e-6)
            
            # Save preprocessed audio
            filename = Path(audio_path).stem + ".wav"
            output_path = os.path.join(output_dir, filename)
            sf.write(output_path, audio, target_sample_rate)
            
            logger.info(f"Preprocessed audio saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
