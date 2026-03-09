import asyncio
import subprocess
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import tempfile
from dataclasses import dataclass

from stt_processor import STTProcessor, SegmentInfo

logger = logging.getLogger(__name__)


@dataclass
class TrainingSegment:
    """A single training segment with audio path and metadata."""
    audio_path: Path
    text: str
    duration: float
    speaker_id: int = 0
    confidence: float = 1.0
    original_file: str = ""
    start_time: float = 0.0
    end_time: float = 0.0


class AudioSegmenter:
    """Segments audio files using ffmpeg based on STT timestamps."""

    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """Locate the ffmpeg executable in PATH."""
        for path in ['ffmpeg', 'ffmpeg.exe']:
            if shutil.which(path):
                return path
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")

    async def extract_audio_segment(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        sample_rate: int = 22050,
        channels: int = 1
    ) -> bool:
        """
        Extract a time-bounded segment from an audio file using ffmpeg.

        Args:
            input_path: Source audio file.
            output_path: Destination WAV file.
            start_time: Segment start in seconds.
            end_time: Segment end in seconds.
            sample_rate: Target sample rate (default 22050 Hz).
            channels: Output channels — 1 for mono.

        Returns:
            True on success, False on failure.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                self.ffmpeg_path,
                '-i', str(input_path),
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-acodec', 'pcm_s16le',
                '-y',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.error(f"ffmpeg error for {output_path.name}: {result.stderr}")
                return False
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"No output file created: {output_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error extracting segment {output_path.name}: {e}")
            return False

    async def create_training_segments(
        self,
        audio_file: Path,
        segments: List[SegmentInfo],
        output_dir: Path,
        model_name: str,
        sample_rate: int = 22050
    ) -> List[TrainingSegment]:
        """
        Cut audio into labelled training segments.

        Args:
            audio_file: Source audio file.
            segments: STT segment list with timestamps and text.
            output_dir: Output directory (audio/ sub-dir created automatically).
            model_name: Voice model name (used for file naming).
            sample_rate: Target sample rate.

        Returns:
            List of successfully created TrainingSegment objects.
        """
        logger.info(f"Creating training segments from {audio_file.name}")
        audio_output_dir = output_dir / "audio"
        audio_output_dir.mkdir(parents=True, exist_ok=True)

        training_segments = []
        base_name = audio_file.stem

        for i, segment in enumerate(segments):
            try:
                segment_filename = f"{model_name}_{base_name}_{i:04d}.wav"
                segment_path = audio_output_dir / segment_filename

                success = await self.extract_audio_segment(
                    audio_file, segment_path,
                    segment.start_time, segment.end_time, sample_rate
                )

                if success:
                    training_segments.append(TrainingSegment(
                        audio_path=segment_path,
                        text=segment.text.strip(),
                        duration=segment.end_time - segment.start_time,
                        speaker_id=0,
                        confidence=segment.confidence,
                        original_file=audio_file.name,
                        start_time=segment.start_time,
                        end_time=segment.end_time
                    ))
                    logger.debug(
                        f"Segment {i+1:3d}: {segment.start_time:.1f}s-{segment.end_time:.1f}s — {segment.text[:60]}"
                    )
                else:
                    logger.warning(f"Failed to create segment {i+1}")

            except Exception as e:
                logger.error(f"Error creating segment {i+1}: {e}")

        logger.info(f"Created {len(training_segments)}/{len(segments)} segments from {audio_file.name}")
        return training_segments

    async def process_multiple_audio_files(
        self,
        audio_files: List[Path],
        output_dir: Path,
        model_name: str,
        stt_service_url: str = "http://stt-service:8000",
        sample_rate: int = 22050,
        quality_filters: Optional[Dict] = None
    ) -> Tuple[List[TrainingSegment], Dict]:
        """
        Run STT + segmentation on a list of audio files.

        Args:
            audio_files: Source audio files.
            output_dir: Root output directory for training data.
            model_name: Voice model name (used for file naming).
            stt_service_url: Whisper STT service URL.
            sample_rate: Target sample rate for extracted segments.
            quality_filters: Optional overrides for duration/confidence filters.

        Returns:
            Tuple of (training_segments, processing_stats dict).
        """
        logger.info(f"Processing {len(audio_files)} audio files for training dataset")

        if quality_filters is None:
            quality_filters = {
                'min_duration': 1.0,
                'max_duration': 15.0,
                'min_confidence': 0.6,
                'min_text_length': 10
            }

        all_training_segments = []
        processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_segments_found': 0,
            'segments_after_quality_filter': 0,
            'segments_created': 0,
            'total_audio_duration': 0.0,
            'training_audio_duration': 0.0
        }

        async with STTProcessor(stt_service_url) as stt_processor:
            for audio_file in audio_files:
                try:
                    logger.info(f"Processing: {audio_file.name}")

                    import librosa
                    file_duration = librosa.get_duration(path=str(audio_file))
                    processing_stats['total_audio_duration'] += file_duration
                    logger.info(f"  Duration: {file_duration:.1f}s")

                    segments = await stt_processor.process_audio_file(audio_file)
                    processing_stats['total_segments_found'] += len(segments)

                    if not segments:
                        logger.warning(f"  No segments found in {audio_file.name}")
                        processing_stats['files_failed'] += 1
                        continue

                    filtered_segments = stt_processor.filter_segments_by_quality(segments, **quality_filters)
                    processing_stats['segments_after_quality_filter'] += len(filtered_segments)

                    if not filtered_segments:
                        logger.warning(f"  No segments passed quality filter for {audio_file.name}")
                        processing_stats['files_failed'] += 1
                        continue

                    training_segments = await self.create_training_segments(
                        audio_file, filtered_segments, output_dir, model_name, sample_rate
                    )

                    processing_stats['segments_created'] += len(training_segments)
                    processing_stats['training_audio_duration'] += sum(s.duration for s in training_segments)
                    all_training_segments.extend(training_segments)
                    processing_stats['files_processed'] += 1
                    logger.info(f"  {audio_file.name}: {len(training_segments)} training segments")

                except Exception as e:
                    logger.error(f"  Failed to process {audio_file.name}: {e}")
                    processing_stats['files_failed'] += 1

        logger.info(
            f"Processing complete — {processing_stats['files_processed']}/{len(audio_files)} files, "
            f"{processing_stats['segments_created']} segments, "
            f"{processing_stats['training_audio_duration']:.1f}s training audio"
        )
        return all_training_segments, processing_stats

    def generate_training_metadata(
        self,
        training_segments: List[TrainingSegment],
        output_dir: Path,
        model_name: str,
        language: str = "de"
    ) -> Path:
        """
        Write train.json, val.json, and metadata.json for the training dataset.

        Phonemizes each segment using espeak via the phonemizer library.
        Falls back to raw text if phonemizer is not available.

        Args:
            training_segments: Prepared audio segments.
            output_dir: Dataset root directory.
            model_name: Voice model name (stored in metadata).
            language: ISO language code for phonemization (e.g. 'de', 'en').

        Returns:
            Path to the generated train.json file.
        """
        logger.info(f"Generating training metadata for {len(training_segments)} segments")

        lang_map = {
            'de': 'de', 'en': 'en-us', 'fr': 'fr-fr', 'es': 'es',
            'it': 'it', 'nl': 'nl', 'pt': 'pt', 'ru': 'ru',
        }
        phonemizer_lang = lang_map.get(language, 'en-us')

        try:
            from phonemizer.backend import EspeakBackend
            from phonemizer import phonemize
            import ctypes.util as _cu
            _so = _cu.find_library('espeak-ng') or '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1'
            EspeakBackend.set_library(_so)
            has_phonemizer = True
        except ImportError:
            has_phonemizer = False
            logger.warning("phonemizer not available — using raw text as phonemes")

        metadata = []
        for i, segment in enumerate(training_segments):
            phonemes = segment.text
            if has_phonemizer:
                try:
                    from phonemizer import phonemize
                    phonemes = phonemize(segment.text, language=phonemizer_lang, backend='espeak', strip=True)
                except Exception as e:
                    logger.warning(f"Phonemization failed for segment {i}: {e}")

            metadata.append({
                "audio_path": f"audio/{segment.audio_path.name}",
                "text": segment.text,
                "phonemes": phonemes,
                "duration": segment.duration,
                "speaker_id": segment.speaker_id,
                "mel_path": f"mel/{segment.audio_path.stem}.npy",
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "original_file": segment.original_file,
                "confidence": segment.confidence
            })

        # 90/10 train/val split
        n_val = max(1, int(len(metadata) * 0.1))
        indices = np.random.permutation(len(metadata))
        train_metadata = [metadata[i] for i in indices[n_val:]]
        val_metadata = [metadata[i] for i in indices[:n_val]]

        train_path = output_dir / "train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_metadata, f, indent=2, ensure_ascii=False)

        with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_metadata, f, indent=2, ensure_ascii=False)

        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved: {len(train_metadata)} train, {len(val_metadata)} val")
        return train_path
