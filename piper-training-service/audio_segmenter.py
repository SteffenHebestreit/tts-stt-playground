"""FFmpeg-based audio segmentation driven by STT timestamps."""

import asyncio
import subprocess
import json
import logging
import shutil
from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple
import numpy as np
import tempfile
from dataclasses import dataclass

from phonemization import phonemize_texts
from stt_processor import STTProcessor, SegmentInfo, STTError, default_min_confidence
from validation import split_train_val

logger = logging.getLogger(__name__)

# Ceiling on a single ffmpeg cut. Generous — these are seconds of audio — but a
# wedged process must not stall the whole segmentation run indefinitely.
FFMPEG_TIMEOUT_S = 120


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
        """Initialise the segmenter and resolve the ffmpeg executable."""
        self.ffmpeg_path = self._find_ffmpeg()
        # Segments dropped by the last generate_training_metadata() call.
        self.phonemization_failures = 0

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
                # -ss BEFORE -i is input seeking: ffmpeg jumps to the position and
                # decodes only the requested slice. After -i it is *output*
                # seeking, which decodes the file from the beginning every time
                # and throws away everything before start_time. Cutting one
                # recording into N segments therefore cost N full decodes -- for
                # a one-hour file at a few hundred segments that is hours of CPU
                # doing nothing. Modern ffmpeg (>= 2.1) makes input seeking
                # accurate as well as fast, so nothing is traded away here.
                '-ss', str(start_time),
                '-i', str(input_path),
                '-t', str(end_time - start_time),
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-acodec', 'pcm_s16le',
                '-y',
                str(output_path)
            ]
            # Off the event loop, and bounded. This runs inside a BackgroundTask,
            # so a synchronous call here froze /health and /status for the whole
            # of segmentation -- long enough for Docker to mark the container
            # unhealthy -- and a wedged ffmpeg would have hung it for good.
            result = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, check=False, timeout=FFMPEG_TIMEOUT_S,
            )
            if result.returncode != 0:
                logger.error(f"ffmpeg error for {output_path.name}: {result.stderr}")
                return False
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"No output file created: {output_path}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s for {output_path.name}")
            return False
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
        quality_filters: Optional[Dict] = None,
        language: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
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
            language: Dataset language passed to the STT service ("de"); None
                lets the service detect it per request.
            should_stop: Polled between files; when it returns True the run
                ends early with what has been produced so far. A file can take
                many minutes of STT, so a cancelled job should not wait for the
                rest of the upload.

        Returns:
            Tuple of (training_segments, processing_stats dict).

        Raises:
            STTError: no file produced any segment and at least one failed
                because the STT service errored. Reported as such rather than
                as an empty dataset.
        """
        logger.info(f"Processing {len(audio_files)} audio files for training dataset")

        if quality_filters is None:
            quality_filters = {
                'min_duration': 1.0,
                'max_duration': 15.0,
                'min_confidence': default_min_confidence(),
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
            'training_audio_duration': 0.0,
            'stt_errors': [],
        }

        async with STTProcessor(stt_service_url, language=language) as stt_processor:
            for audio_file in audio_files:
                if should_stop is not None and should_stop():
                    logger.info("Stop requested; skipping the remaining audio files")
                    break
                try:
                    logger.info(f"Processing: {audio_file.name}")

                    import librosa
                    file_duration = await asyncio.to_thread(librosa.get_duration, path=str(audio_file))
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

                except STTError as e:
                    logger.error(f"  STT failed for {audio_file.name}: {e}")
                    processing_stats['files_failed'] += 1
                    processing_stats['stt_errors'].append(f"{audio_file.name}: {e}")
                except Exception as e:
                    logger.error(f"  Failed to process {audio_file.name}: {e}")
                    processing_stats['files_failed'] += 1

        if not all_training_segments and processing_stats['stt_errors']:
            errors = processing_stats['stt_errors']
            raise STTError(
                f"STT failed for {len(errors)} of {len(audio_files)} file(s) and no segments were "
                f"produced; first error: {errors[0]}"
            )

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

        Phonemizes each segment using espeak via the phonemizer library. A
        segment that cannot be phonemised is dropped and counted (see
        ``phonemization_failures``), never kept with its raw text as "phonemes":
        the text would be turned into ids by the same vocabulary builder and
        teach the model spellings it never sees at inference. If phonemizer is
        missing, or more than PHONEMIZER_MAX_FAILURE_FRACTION of the segments
        fail, ``PhonemizationError`` is raised and no dataset is written.

        Args:
            training_segments: Prepared audio segments.
            output_dir: Dataset root directory.
            model_name: Voice model name (stored in metadata).
            language: ISO language code for phonemization (e.g. 'de', 'en').

        Returns:
            Path to the generated train.json file.
        """
        logger.info(f"Generating training metadata for {len(training_segments)} segments")

        all_phonemes = phonemize_texts([segment.text for segment in training_segments], language)

        metadata = []
        self.phonemization_failures = 0
        for segment, phonemes in zip(training_segments, all_phonemes):
            if phonemes is None:
                self.phonemization_failures += 1
                continue

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

        train_metadata, val_metadata = split_train_val(metadata)

        train_path = output_dir / "train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_metadata, f, indent=2, ensure_ascii=False)

        with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_metadata, f, indent=2, ensure_ascii=False)

        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved: {len(train_metadata)} train, {len(val_metadata)} val")
        return train_path
