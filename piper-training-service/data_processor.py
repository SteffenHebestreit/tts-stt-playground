import os
import json
import io
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict
import soundfile as sf
from phonemizer import phonemize
import pandas as pd
import asyncio
import aiofiles
import requests
import subprocess
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 256
        self.n_fft = 1024
        self.n_mels = 80
        
    async def prepare_dataset(self, segments: List, model_name: str, language: str = "en") -> Path:
        """Prepare dataset from STT segments"""
        
        dataset_dir = Path(f"data/{model_name}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        (dataset_dir / "audio").mkdir(exist_ok=True)
        (dataset_dir / "mel").mkdir(exist_ok=True)
        
        metadata = []
        
        logger.info(f"Processing {len(segments)} segments for model {model_name}")
        
        for idx, segment in enumerate(segments):
            try:
                # Handle both local paths and URLs
                audio_path = segment.audio_path
                if audio_path.startswith('http'):
                    # Download audio file
                    audio_data = await self._download_audio(audio_path)
                    audio, sr = librosa.load(io.BytesIO(audio_data), sr=self.sample_rate)
                else:
                    # Load local file
                    audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                
                # Extract segment if start_time and end_time are provided
                if hasattr(segment, 'start_time') and hasattr(segment, 'end_time'):
                    start_sample = int(segment.start_time * sr)
                    end_sample = int(segment.end_time * sr)
                    audio = audio[start_sample:end_sample]
                
                # Trim silence
                audio, _ = librosa.effects.trim(audio, top_db=20)
                
                # Skip if too short
                if len(audio) < sr * 0.5:  # Less than 0.5 seconds
                    logger.warning(f"Skipping segment {idx}: too short ({len(audio)/sr:.2f}s)")
                    continue
                
                # Normalize
                audio = audio / (np.max(np.abs(audio)) + 1e-6)
                
                # Save processed audio
                output_audio_path = dataset_dir / "audio" / f"{idx:05d}.wav"
                sf.write(output_audio_path, audio, self.sample_rate)
                
                # Compute mel spectrogram
                mel_spec = self._compute_mel_spectrogram(audio)
                mel_path = dataset_dir / "mel" / f"{idx:05d}.npy"
                np.save(mel_path, mel_spec)
                
                # Get phonemes - map language code to phonemizer format
                lang_map = {
                    'de': 'de', 'en': 'en-us', 'fr': 'fr-fr', 'es': 'es',
                    'it': 'it', 'nl': 'nl', 'pt': 'pt', 'ru': 'ru',
                }
                phonemizer_lang = lang_map.get(language, 'en-us')
                try:
                    phonemes = phonemize(
                        segment.text,
                        language=phonemizer_lang,
                        backend='espeak',
                        strip=True
                    )
                except Exception as e:
                    logger.warning(f"Phonemization error for segment {idx}: {e}")
                    phonemes = segment.text  # Fallback to text
                
                # Add to metadata
                metadata.append({
                    'audio_path': str(output_audio_path.relative_to(dataset_dir)),
                    'mel_path': str(mel_path.relative_to(dataset_dir)),
                    'text': segment.text,
                    'phonemes': phonemes,
                    'duration': len(audio) / self.sample_rate,
                    'segment_id': idx
                })
                
                logger.info(f"Processed segment {idx+1}/{len(segments)}")
                
            except Exception as e:
                logger.error(f"Error processing segment {idx}: {e}")
                continue
        
        if not metadata:
            raise ValueError("No valid segments were processed. Please check your audio files and transcriptions.")
        
        # Save metadata
        metadata_path = dataset_dir / "metadata.json"
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        # Create train/val split
        await self._create_splits(metadata, dataset_dir)
        
        logger.info(f"Dataset prepared with {len(metadata)} samples at {dataset_dir}")
        return dataset_dir
    
    async def _download_audio(self, url: str) -> bytes:
        """Download audio file from URL"""
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    
    def _compute_mel_spectrogram(self, audio):
        """Compute mel spectrogram from audio"""
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
        """Create train/validation splits"""
        n_samples = len(metadata)
        n_val = max(1, int(n_samples * 0.1))  # 10% validation
        
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_metadata = [metadata[i] for i in train_indices]
        val_metadata = [metadata[i] for i in val_indices]
        
        async with aiofiles.open(dataset_dir / "train.json", 'w') as f:
            await f.write(json.dumps(train_metadata, indent=2))
        
        async with aiofiles.open(dataset_dir / "val.json", 'w') as f:
            await f.write(json.dumps(val_metadata, indent=2))
        
        logger.info(f"Created splits: {len(train_metadata)} training, {len(val_metadata)} validation")

    def create_speaker_map(self, dataset_dir: Path) -> Dict:
        """Create speaker ID mapping for multi-speaker models"""
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
        """Analyze audio file using FFmpeg to get detailed information"""
        try:
            # Use ffprobe to get audio information
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
        """Preprocess audio file for training with format analysis"""
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
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Save preprocessed audio
            filename = Path(audio_path).stem + ".wav"
            output_path = os.path.join(output_dir, filename)
            sf.write(output_path, audio, target_sample_rate)
            
            logger.info(f"Preprocessed audio saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
