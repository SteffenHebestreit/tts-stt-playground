import torch
from torch.utils.data import Dataset
import json
import logging
import numpy as np
from pathlib import Path
import librosa
import random

logger = logging.getLogger(__name__)

class TTSDataset(Dataset):
    def __init__(self, data_dir: Path, config: dict, split: str = "train"):
        self.data_dir = data_dir
        self.config = config
        
        # Load metadata
        metadata_file = data_dir / f"{split}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        if len(self.metadata) == 0:
            raise ValueError(f"No data found in {metadata_file}")
        
        # Create phoneme vocabulary
        self.phoneme_to_id = self._create_phoneme_vocab()
        
        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")
        logger.info(f"Phoneme vocabulary size: {len(self.phoneme_to_id)}")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        try:
            # Load audio if available
            audio = None
            if 'audio_path' in item:
                audio_path = self.data_dir / item['audio_path']
                if audio_path.exists():
                    # Use sample_rate from config, with fallback to 22050
                    sample_rate = self.config.get('sample_rate', 22050)
                    audio, _ = librosa.load(audio_path, sr=sample_rate)
                    audio = torch.FloatTensor(audio)
            
            # Load mel spectrogram
            mel_path = self.data_dir / item['mel_path']
            if not mel_path.exists():
                raise FileNotFoundError(f"Mel spectrogram not found: {mel_path}")
            
            mel_spec = np.load(mel_path)
            mel_spec = torch.FloatTensor(mel_spec)
            
            # Convert phonemes to IDs
            phonemes = item.get('phonemes', item['text'])
            phoneme_ids = self._phonemes_to_ids(phonemes)
            
            # Ensure minimum length
            if len(phoneme_ids) == 0:
                phoneme_ids = [self.phoneme_to_id.get('<unk>', 0)]
            
            # Create duration target
            duration_target = self._estimate_durations(
                len(phoneme_ids),
                mel_spec.shape[1]
            )
            
            return {
                'audio': audio if audio is not None else torch.zeros(mel_spec.shape[1] * 256),
                'mel_spec': mel_spec,
                'text': torch.LongTensor(phoneme_ids),
                'text_lengths': len(phoneme_ids),
                'duration_target': torch.FloatTensor(duration_target)
            }
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            # Return None, collate_fn will filter it out
            return None
    
    def _create_phoneme_vocab(self):
        """Create phoneme to ID mapping"""
        phonemes = set()
        for item in self.metadata:
            phoneme_text = item.get('phonemes', item['text'])
            if phoneme_text:
                phonemes.update(list(phoneme_text))
        
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>', ' ']
        all_phonemes = sorted(phonemes.union(special_tokens))
        
        phoneme_to_id = {p: i for i, p in enumerate(all_phonemes)}
        
        return phoneme_to_id
    
    def _phonemes_to_ids(self, phonemes: str):
        """Convert phoneme string to ID sequence"""
        if not phonemes:
            return [self.phoneme_to_id.get('<unk>', 0)]
        
        ids = []
        for p in phonemes:
            ids.append(self.phoneme_to_id.get(p, self.phoneme_to_id.get('<unk>', 0)))
        
        return ids
    
    def _estimate_durations(self, text_len: int, mel_len: int):
        """Estimate phoneme durations (simplified)"""
        if text_len == 0:
            return [1.0]
        
        avg_duration = mel_len / text_len
        durations = np.ones(text_len) * avg_duration
        
        # Add some variation
        variation = np.random.normal(0, avg_duration * 0.1, text_len)
        durations += variation
        durations = np.maximum(durations, 1.0)
        
        # Ensure they sum to mel_len
        durations = durations * (mel_len / durations.sum())
        
        return durations

def collate_fn(batch):
    """Custom collate function for batching"""
    # Filter out None items
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Sort by text length for better batching
    batch.sort(key=lambda x: x['text_lengths'], reverse=True)
    
    # Get max lengths
    max_text_len = max(item['text_lengths'] for item in batch)
    max_mel_len = max(item['mel_spec'].shape[1] for item in batch)
    max_audio_len = max(item['audio'].shape[0] for item in batch)
    
    # Pad sequences
    batch_size = len(batch)
    
    padded_audio = torch.zeros(batch_size, max_audio_len)
    padded_mel = torch.zeros(batch_size, batch[0]['mel_spec'].shape[0], max_mel_len)
    padded_text = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    padded_durations = torch.zeros(batch_size, max_text_len)
    text_lengths = torch.LongTensor([item['text_lengths'] for item in batch])
    mel_lengths = torch.LongTensor([item['mel_spec'].shape[1] for item in batch])

    for i, item in enumerate(batch):
        # Audio
        audio_len = item['audio'].shape[0]
        padded_audio[i, :audio_len] = item['audio']
        
        # Mel spectrogram
        mel_len = item['mel_spec'].shape[1]
        padded_mel[i, :, :mel_len] = item['mel_spec']
        
        # Text
        text_len = item['text_lengths']
        padded_text[i, :text_len] = item['text']
        
        # Duration
        dur_len = len(item['duration_target'])
        padded_durations[i, :dur_len] = item['duration_target']
    
    return {
        'audio': padded_audio,
        'mel_spec': padded_mel,
        'text': padded_text,
        'text_lengths': text_lengths,
        'mel_lengths': mel_lengths,
        'duration_target': padded_durations
    }
