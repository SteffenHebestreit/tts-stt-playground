"""Async STT service client with retry logic and chunked processing."""

import asyncio
import aiohttp
import aiofiles
import json
import os
import tempfile
import math
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import librosa
import soundfile as sf
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The STT service answers 503 while it loads a model or is at capacity, and 502/504
# from a proxy in front of it: all worth another try. Any other status is final.
RETRY_STATUSES = frozenset({502, 503, 504})

DEFAULT_MIN_CONFIDENCE = 0.6


class STTError(RuntimeError):
    """The STT service could not transcribe a file; the message carries the cause.

    Distinct from "the file had no speech": every failure used to be turned into
    an empty segment list, so an unreachable or misconfigured STT service ended
    the job as "No valid training segments" with the actual error only in a log.
    """


def default_min_confidence() -> float:
    """STT_MIN_CONFIDENCE: the lowest segment confidence kept for training (0..1).

    Applied to :func:`confidence_from_segment`, i.e. to ``exp(avg_logprob)`` when
    the STT service reports only a log-probability. 0.6 corresponds to an
    ``avg_logprob`` of about -0.51; Whisper itself treats -1.0 (0.37) as a failed
    decode. Set 0 to keep everything.
    """
    raw = os.getenv("STT_MIN_CONFIDENCE", "").strip()
    if not raw:
        return DEFAULT_MIN_CONFIDENCE
    try:
        value = float(raw)
    except ValueError:
        logger.warning("STT_MIN_CONFIDENCE=%r is not a number; using %s", raw, DEFAULT_MIN_CONFIDENCE)
        return DEFAULT_MIN_CONFIDENCE
    if not 0.0 <= value <= 1.0:
        logger.warning("STT_MIN_CONFIDENCE=%r is outside 0..1; using %s", raw, DEFAULT_MIN_CONFIDENCE)
        return DEFAULT_MIN_CONFIDENCE
    return value


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def confidence_from_segment(segment: Dict) -> Optional[float]:
    """A 0..1 confidence for one STT segment, or ``None`` if the service gave none.

    stt-service returns ``avg_logprob`` (the mean token log-probability) and
    ``no_speech_prob`` and never a ``confidence`` field, so reading
    ``segment.get('confidence', 1.0)`` made the quality filter's confidence
    threshold a no-op: every segment scored 1.0. ``exp(avg_logprob)`` is the
    geometric-mean token probability, which is on the scale the threshold was
    written for.

    An ``avg_logprob`` that is present but null means the service met -inf/NaN
    and its JSON layer replaced it: a degenerate decode, scored 0.0 rather than
    treated as unknown.
    """
    explicit = segment.get("confidence")
    if _finite_number(explicit):
        return min(max(float(explicit), 0.0), 1.0)
    if "avg_logprob" in segment:
        logprob = segment["avg_logprob"]
        if _finite_number(logprob):
            return math.exp(min(float(logprob), 0.0))
        return 0.0
    return None


def confidence_from_result(result: Dict) -> Optional[float]:
    """Duration-weighted confidence of a whole transcription, or ``None`` if unknown."""
    weighted = 0.0
    total = 0.0
    for segment in result.get("segments") or []:
        confidence = confidence_from_segment(segment)
        if confidence is None:
            continue
        try:
            span = max(float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)), 0.0)
        except (TypeError, ValueError):
            span = 0.0
        weighted += confidence * (span or 1e-3)
        total += span or 1e-3
    return weighted / total if total else None


@dataclass
class SegmentInfo:
    """Information about an audio segment from STT processing"""
    text: str
    start_time: float
    end_time: float
    confidence: float
    words: Optional[List[Dict]] = None

class STTProcessor:
    """Handles STT service integration for audio segmentation"""
    
    def __init__(self, stt_service_url: str = "http://stt-service:8000", language: Optional[str] = None):
        """Store the STT base URL and defer session creation to async entry.

        *language* is what the dataset is in ("de"); without one the service
        detects per request, which on a short single-language clip sometimes
        picks the wrong language and returns text in it.
        """
        self.stt_service_url = stt_service_url.rstrip('/')
        self.language = (language or "").strip() or "auto"
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Create connector with proper DNS resolution and connection pooling
        connector = aiohttp.TCPConnector(
            ttl_dns_cache=300,  # Cache DNS for 5 minutes
            force_close=False,  # Reuse connections
            enable_cleanup_closed=True,
            family=0,  # Allow both IPv4 and IPv6
            ssl=False  # No SSL for internal Docker network
        )
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def transcribe_audio_file(self, audio_path: Path, return_segments: bool = True, max_retries: int = 3) -> Dict:
        """
        Transcribe an audio file using the STT service with retry logic
        
        Args:
            audio_path: Path to audio file
            return_segments: Whether to return segment-level timestamps
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with transcription results including segments

        Raises:
            STTError: the service was unreachable, kept answering 502/503/504,
                or rejected the request. The message names the cause.
        """
        if not self.session:
            raise RuntimeError("STTProcessor must be used as async context manager")

        # Read once, outside the retry loop: a missing or unreadable file is not
        # a transient STT problem (the old OSError retry waited 2+4 s for it) and
        # a large upload should not be re-read from disk on every attempt.
        try:
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.1f}MB")
            async with aiofiles.open(audio_path, 'rb') as f:
                audio_data = await f.read()
        except OSError as e:
            raise STTError(f"Cannot read {audio_path.name}: {e}") from e

        last_error = None
        for attempt in range(max_retries):
            if attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                logger.info(f"Retry {attempt}/{max_retries} - waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

            logger.info(f"Starting STT transcription for {audio_path.name} (attempt {attempt + 1}/{max_retries})")
            logger.info(f"STT Service URL: {self.stt_service_url}")

            # A FormData is consumed by sending it, so each attempt builds its own.
            data = aiohttp.FormData()
            data.add_field('audio', audio_data, filename=audio_path.name)
            data.add_field('task', 'transcribe')
            data.add_field('language', self.language)
            data.add_field('beam_size', '5')
            data.add_field('best_of', '5')

            logger.info(f"Making STT request to {self.stt_service_url}/transcribe (language={self.language})")
            # Very long timeout: a multi-hour recording is one request.
            timeout = aiohttp.ClientTimeout(total=3600)
            try:
                async with self.session.post(f"{self.stt_service_url}/transcribe", data=data, timeout=timeout) as response:
                    logger.info(f"STT response received: {response.status}")
                    if response.status in RETRY_STATUSES and attempt < max_retries - 1:
                        last_error = f"STT service answered {response.status}"
                        logger.warning(f"{last_error} on attempt {attempt + 1}; will retry")
                        continue
                    if response.status != 200:
                        error_text = (await response.text())[:500]
                        raise STTError(f"STT service error {response.status}: {error_text}")

                    result = await response.json()
                    logger.info("STT transcription completed successfully")
                    return result
            except STTError:
                raise
            except asyncio.TimeoutError as e:
                # Before the connection clause: aiohttp's ServerTimeoutError is both,
                # and a request that ran out its hour must not be started three times.
                raise STTError(f"STT request for {audio_path.name} timed out") from e
            except (aiohttp.ClientConnectionError, ConnectionError) as e:
                # Refused, reset, DNS failure: the service may simply be starting.
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
            except (aiohttp.ClientError, ValueError) as e:
                raise STTError(f"STT request for {audio_path.name} failed: {type(e).__name__}: {e}") from e

        logger.error(f"All {max_retries} attempts failed")
        raise STTError(
            f"STT service at {self.stt_service_url} unavailable after {max_retries} attempts: {last_error}"
        )
    
    async def process_large_audio_file(self, audio_path: Path, max_chunk_duration: float = 60.0) -> List[SegmentInfo]:
        """
        Process large audio files by chunking them before STT processing
        
        Args:
            audio_path: Path to large audio file
            max_chunk_duration: Maximum duration per chunk in seconds
            
        Returns:
            List of segment information
        """
        logger.info(f"Processing large audio file: {audio_path.name}")

        # Get audio duration without loading entire file
        total_duration = librosa.get_duration(path=str(audio_path))

        if total_duration <= max_chunk_duration:
            # File is not that large, process normally
            return await self.process_audio_file(audio_path)

        logger.info(f"Audio duration: {total_duration:.1f}s, chunking into {max_chunk_duration}s segments")
        
        # Now load audio for chunking
        y, sr = librosa.load(str(audio_path), sr=None)
        
        all_segments = []
        chunk_duration_samples = int(max_chunk_duration * sr)
        overlap_samples = int(2.0 * sr)  # 2 second overlap
        
        start_sample = 0
        chunk_index = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            while start_sample < len(y):
                end_sample = min(start_sample + chunk_duration_samples, len(y))
                
                # Extract chunk
                chunk_audio = y[start_sample:end_sample]
                chunk_start_time = start_sample / sr
                
                # Save temporary chunk
                chunk_path = Path(temp_dir) / f"chunk_{chunk_index:03d}.wav"
                sf.write(str(chunk_path), chunk_audio, sr)
                
                logger.info(f"Processing chunk {chunk_index + 1}: {chunk_start_time:.1f}s - {chunk_start_time + len(chunk_audio)/sr:.1f}s")
                
                # Transcribe chunk
                chunk_segments = await self.process_audio_file(chunk_path)
                
                # Adjust timestamps to global time
                for segment in chunk_segments:
                    segment.start_time += chunk_start_time
                    segment.end_time += chunk_start_time
                    if segment.words:
                        for word in segment.words:
                            if 'start' in word:
                                word['start'] += chunk_start_time
                            if 'end' in word:
                                word['end'] += chunk_start_time
                
                all_segments.extend(chunk_segments)
                chunk_index += 1

                # The last chunk ends at the end of the file. Stepping back by the
                # overlap from there stayed below len(y), so the same tail was
                # transcribed again for ever.
                if end_sample >= len(y):
                    break

                # Move to next chunk with overlap
                start_sample = end_sample - overlap_samples
        
        # Merge overlapping segments if needed
        merged_segments = self._merge_overlapping_segments(all_segments)
        
        logger.info(f"Processed {chunk_index} chunks, found {len(merged_segments)} segments")
        return merged_segments
    
    async def process_audio_file(self, audio_path: Path) -> List[SegmentInfo]:
        """
        Process a single audio file through STT service
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of segment information. An empty list means the service
            answered and found no speech.

        Raises:
            STTError: the service could not be reached or rejected the request.
                This used to be logged and returned as an empty list, which the
                caller could not tell from a silent recording.
        """
        logger.info(f"Sending {audio_path.name} to STT service...")

        # Get transcription with segments from STT service
        result = await self.transcribe_audio_file(audio_path, return_segments=True)

        try:
            segments = self._segments_from_result(result, audio_path)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise STTError(f"Unusable STT response for {audio_path.name}: {type(e).__name__}: {e}") from e

        logger.info(f"Processed {len(segments)} segments from {audio_path.name}")
        return segments

    def _segments_from_result(self, result: Dict, audio_path: Path) -> List[SegmentInfo]:
        """Turn one STT response into SegmentInfo objects."""
        segments = []
        unscored = 0

        # Parse segments from STT response
        if 'segments' in result:
            logger.info(f"STT returned {len(result['segments'])} segments")
            for segment_data in result['segments']:
                confidence = confidence_from_segment(segment_data)
                if confidence is None:
                    unscored += 1
                    confidence = 1.0
                segment = SegmentInfo(
                    text=(segment_data.get('text') or '').strip(),
                    start_time=float(segment_data.get('start', 0.0)),
                    end_time=float(segment_data.get('end', 0.0)),
                    confidence=confidence,
                    words=segment_data.get('words', [])
                )

                # Only add segments with meaningful text
                if len(segment.text) > 0 and segment.end_time > segment.start_time:
                    segments.append(segment)

        # Fallback: if no segments, use full text with duration
        elif 'text' in result and result['text'].strip():
            logger.warning("No segments returned, using full text as single segment")
            # Get audio duration efficiently
            duration = librosa.get_duration(path=str(audio_path))
            confidence = confidence_from_result(result)
            if confidence is None:
                unscored += 1
                confidence = 1.0

            segments.append(SegmentInfo(
                text=result['text'].strip(),
                start_time=0.0,
                end_time=duration,
                confidence=confidence
            ))

        if unscored:
            # Not an error (other STT backends may not report one) but the
            # operator should know the confidence filter is not filtering.
            logger.warning(
                f"{unscored} STT segment(s) for {audio_path.name} carry neither 'confidence' nor "
                f"'avg_logprob'; they are kept at confidence 1.0, so STT_MIN_CONFIDENCE cannot reject them"
            )
        return segments
    
    def _merge_overlapping_segments(self, segments: List[SegmentInfo]) -> List[SegmentInfo]:
        """
        Merge overlapping segments from chunked processing
        
        Args:
            segments: List of segments that may overlap
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda s: s.start_time)
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            # Check for overlap (within 1 second tolerance)
            if next_segment.start_time - current.end_time <= 1.0:
                # Merge segments
                if len(next_segment.text) > len(current.text):
                    # Use the longer/better text
                    current.text = next_segment.text
                current.end_time = max(current.end_time, next_segment.end_time)
                current.confidence = max(current.confidence, next_segment.confidence)
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = next_segment
        
        # Add the last segment
        merged.append(current)
        
        return merged
    
    async def process_multiple_files(self, audio_files: List[Path], max_chunk_duration: float = 60.0) -> Dict[str, List[SegmentInfo]]:
        """
        Process multiple audio files for STT segmentation
        
        Args:
            audio_files: List of audio file paths
            max_chunk_duration: Maximum duration per chunk for large files
            
        Returns:
            Dictionary mapping file names to their segments
        """
        logger.info(f"Starting STT processing for {len(audio_files)} audio files...")

        results = {}
        failed = 0
        first_error = None

        for i, audio_path in enumerate(audio_files):
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_path.name}")

            try:
                # Get file info
                y, sr = librosa.load(str(audio_path), sr=None)
                duration = len(y) / sr
                file_size_mb = audio_path.stat().st_size / (1024 * 1024)

                logger.info(f"Audio: {duration:.1f}s, {file_size_mb:.1f}MB, {sr}Hz")

                # Choose processing method based on file size
                if duration > max_chunk_duration or file_size_mb > 50:
                    logger.info(f"Large file detected - using chunked processing")
                    segments = await self.process_large_audio_file(audio_path, max_chunk_duration)
                else:
                    logger.info(f"Standard file - using direct processing")
                    segments = await self.process_audio_file(audio_path)

                results[audio_path.name] = segments

                logger.info(f"Processed {audio_path.name}: {len(segments)} segments")

            except Exception as e:
                logger.error(f"Failed to process {audio_path.name}: {e}")
                results[audio_path.name] = []
                failed += 1
                first_error = first_error or e

        if audio_files and failed == len(audio_files):
            # Every file failed: that is an outage or a misconfiguration, not a
            # dataset with no speech in it.
            raise STTError(f"STT failed for all {failed} file(s); first error: {first_error}")

        total_segments = sum(len(segments) for segments in results.values())
        logger.info(f"STT processing complete: {total_segments} total segments from {len(audio_files)} files")
        
        return results

    def filter_segments_by_quality(self, segments: List[SegmentInfo], 
                                 min_duration: float = 1.0, 
                                 max_duration: float = 15.0,
                                 min_confidence: float = 0.5,
                                 min_text_length: int = 10) -> List[SegmentInfo]:
        """
        Filter segments by quality criteria for training
        
        Args:
            segments: List of segments to filter
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            min_confidence: Minimum confidence score
            min_text_length: Minimum text length in characters
            
        Returns:
            Filtered list of high-quality segments
        """
        filtered = []
        rejected = {"duration": 0, "confidence": 0, "text": 0}
        
        for segment in segments:
            duration = segment.end_time - segment.start_time
            
            # Apply quality filters
            if not (min_duration <= duration <= max_duration):
                rejected["duration"] += 1
            elif segment.confidence < min_confidence:
                rejected["confidence"] += 1
            elif len(segment.text.strip()) < min_text_length:
                rejected["text"] += 1
            else:
                filtered.append(segment)
        
        logger.info(
            f"Quality filter: {len(filtered)}/{len(segments)} segments passed "
            f"(rejected for duration {rejected['duration']}, confidence "
            f"{rejected['confidence']} < {min_confidence:g}, text length {rejected['text']})"
        )
        return filtered

async def main():
    """Test the STT processor"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python stt_processor.py <audio_file_path>")
        return

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    async with STTProcessor() as processor:
        segments = await processor.process_audio_file(audio_path)

        print(f"\nResults for {audio_path.name}:")
        for i, segment in enumerate(segments):
            print(f"  {i+1:2d}: {segment.start_time:6.1f}s - {segment.end_time:6.1f}s ({segment.end_time-segment.start_time:4.1f}s) | {segment.text[:80]}...")

if __name__ == "__main__":
    asyncio.run(main())