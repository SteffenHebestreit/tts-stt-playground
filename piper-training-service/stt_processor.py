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
    
    def __init__(self, stt_service_url: str = "http://stt-service:8000"):
        self.stt_service_url = stt_service_url.rstrip('/')
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
        """
        if not self.session:
            raise RuntimeError("STTProcessor must be used as async context manager")
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.info(f"Retry {attempt}/{max_retries} - waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

                logger.info(f"Starting STT transcription for {audio_path.name} (attempt {attempt + 1}/{max_retries})")
                logger.info(f"STT Service URL: {self.stt_service_url}")

                # Check file size
                file_size_mb = audio_path.stat().st_size / (1024 * 1024)
                logger.info(f"File size: {file_size_mb:.1f}MB")

                # Prepare the multipart form data (fresh for each attempt)
                data = aiohttp.FormData()

                logger.info(f"Reading audio file...")
                # Add the audio file
                async with aiofiles.open(audio_path, 'rb') as f:
                    audio_data = await f.read()
                    data.add_field('audio', audio_data, filename=audio_path.name)

                logger.info(f"Adding STT parameters...")
                # Add parameters for transcription with segments
                data.add_field('task', 'transcribe')
                data.add_field('language', 'auto')  # Auto-detect language
                data.add_field('beam_size', '5')
                data.add_field('best_of', '5')

                logger.info(f"Making STT request to {self.stt_service_url}/transcribe")
                # Make the request with very long timeout for large files
                timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout for very large files
                async with self.session.post(f"{self.stt_service_url}/transcribe", data=data, timeout=timeout) as response:
                    logger.info(f"STT response received: {response.status}")
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"STT service error {response.status}: {error_text}")

                    result = await response.json()
                    logger.info(f"STT transcription completed successfully")
                    return result

            except (aiohttp.ClientConnectorError, ConnectionRefusedError, OSError) as e:
                last_error = e
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    raise Exception(f"Failed to connect to STT service after {max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Error transcribing {audio_path.name}: {e}")
                raise
    
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
                
                # Move to next chunk with overlap
                start_sample = end_sample - overlap_samples
                chunk_index += 1
        
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
            List of segment information
        """
        try:
            logger.info(f"Sending {audio_path.name} to STT service...")

            # Get transcription with segments from STT service
            result = await self.transcribe_audio_file(audio_path, return_segments=True)

            segments = []

            # Parse segments from STT response
            if 'segments' in result:
                logger.info(f"STT returned {len(result['segments'])} segments")
                for segment_data in result['segments']:
                    segment = SegmentInfo(
                        text=segment_data.get('text', '').strip(),
                        start_time=float(segment_data.get('start', 0.0)),
                        end_time=float(segment_data.get('end', 0.0)),
                        confidence=float(segment_data.get('confidence', 1.0)),
                        words=segment_data.get('words', [])
                    )

                    # Only add segments with meaningful text
                    if len(segment.text) > 0 and segment.end_time > segment.start_time:
                        segments.append(segment)

            # Fallback: if no segments, use full text with duration
            elif 'text' in result and result['text'].strip():
                logger.warning(f"No segments returned, using full text as single segment")
                # Get audio duration efficiently
                duration = librosa.get_duration(path=str(audio_path))

                segments.append(SegmentInfo(
                    text=result['text'].strip(),
                    start_time=0.0,
                    end_time=duration,
                    confidence=1.0
                ))

            logger.info(f"Processed {len(segments)} segments from {audio_path.name}")
            return segments

        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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
        
        for segment in segments:
            duration = segment.end_time - segment.start_time
            
            # Apply quality filters
            if (duration >= min_duration and 
                duration <= max_duration and
                segment.confidence >= min_confidence and
                len(segment.text.strip()) >= min_text_length):
                filtered.append(segment)
        
        logger.info(f"Quality filter: {len(filtered)}/{len(segments)} segments passed")
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