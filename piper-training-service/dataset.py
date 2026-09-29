"""PyTorch Dataset for the mel-flow trainer with phoneme vocabulary handling."""

import torch
from torch.utils.data import Dataset
import json
import logging
import numpy as np
from pathlib import Path

from validation import phoneme_id_map_from_entries, validate_phoneme_id_map

logger = logging.getLogger(__name__)

class TTSDataset(Dataset):
    """Dataset wrapper that loads metadata, audio, mel features, and phoneme IDs."""

    def __init__(self, data_dir: Path, config: dict, split: str = "train",
                 phoneme_to_id: dict = None):
        """Load the requested dataset split and resolve its phoneme vocabulary.

        ``phoneme_to_id`` is the vocabulary a run was started with (or resumed
        with). When given it is used as is: the ids are row numbers of the trained
        embedding, so rebuilding them from the current metadata would renumber the
        symbols whenever the data changed. The validation split must be given the
        training vocabulary for the same reason; built on its own it would number
        its symbols differently.
        """
        self.data_dir = data_dir
        self.config = config

        # Load metadata
        metadata_file = data_dir / f"{split}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            entries = json.load(f)

        if len(entries) == 0:
            raise ValueError(f"No data found in {metadata_file}")

        # Wrap every sequence in <start>/<end>. The runtime (piper-tts-service
        # _custom_onnx_infer) always feeds [<start>, ..., <end>], but training
        # used to feed the bare symbols, so those two rows were never trained.
        self.boundary_tokens = bool(config.get('boundary_tokens', True))

        # Vocabulary from every entry of the split, including the ones dropped
        # below for lack of a mel: it stays the same function of the metadata
        # file, as the exporter's rebuild used to be.
        n_vocab = self.config.get('n_vocab', 256)
        if phoneme_to_id is None:
            self.phoneme_to_id = phoneme_id_map_from_entries(entries)
        else:
            self.phoneme_to_id = validate_phoneme_id_map(phoneme_to_id)

        # The text embedding is sized from config['n_vocab'] (256), while the
        # vocabulary is whatever distinct symbols the phonemizer produced. Any
        # id at or above n_vocab is an out-of-range embedding lookup, which on
        # GPU is a device-side assert — an opaque crash, hours into a run, with
        # a traceback pointing at an unrelated kernel. Checked here instead, in
        # the first seconds of the job, where the message can say what to fix.
        if len(self.phoneme_to_id) > n_vocab or max(self.phoneme_to_id.values()) >= n_vocab:
            raise ValueError(
                f"Dataset needs {len(self.phoneme_to_id)} phoneme symbols but the "
                f"model is configured for n_vocab={n_vocab}. Raise n_vocab, or "
                f"check the transcripts for text the phonemizer passed through "
                f"unconverted (mixed scripts and stray symbols are the usual cause)."
            )

        # An entry whose mel file is missing can only ever produce a None item
        # (and one error line per epoch). Dropped up front, loudly, so batches
        # are full and the caller can tell "too little usable data" from a
        # transient failure.
        self.metadata = [item for item in entries if self._has_mel(item)]
        dropped = len(entries) - len(self.metadata)
        if dropped:
            logger.warning(
                f"{split}: dropped {dropped} of {len(entries)} entries whose mel "
                f"spectrogram file is missing under {data_dir} "
                f"(POST /generate-missing-mels regenerates them)"
            )
        if not self.metadata:
            raise ValueError(
                f"None of the {len(entries)} entries in {metadata_file} has a mel "
                f"spectrogram file under {data_dir}"
            )

        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")
        logger.info(f"Phoneme vocabulary size: {len(self.phoneme_to_id)}/{n_vocab}")

    def _has_mel(self, item) -> bool:
        """True when the entry names a mel file that exists."""
        mel_path = item.get('mel_path') if isinstance(item, dict) else None
        return bool(mel_path) and (self.data_dir / mel_path).exists()

    def __len__(self):
        """Return the number of examples in the current split."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """Load one training example, returning audio, mel, text, and duration targets."""
        item = self.metadata[idx]

        try:
            # Load audio if available
            audio = None
            if 'audio_path' in item:
                audio_path = self.data_dir / item['audio_path']
                if audio_path.exists():
                    # Imported here: it is the only use, and librosa drags in
                    # numba, which the model/export code and its tests do not need.
                    import librosa
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
        return phoneme_id_map_from_entries(self.metadata)

    def _phonemes_to_ids(self, phonemes: str):
        """Convert phoneme string to ID sequence"""
        unk = self.phoneme_to_id.get('<unk>', 0)
        ids = [self.phoneme_to_id.get(p, unk) for p in (phonemes or "")]
        if not ids:
            ids = [unk]

        if self.boundary_tokens:
            pad = self.phoneme_to_id.get('<pad>', 0)
            ids = ([self.phoneme_to_id.get('<start>', pad)] + ids
                   + [self.phoneme_to_id.get('<end>', pad)])
        return ids

    def _estimate_durations(self, text_len: int, mel_len: int):
        """Synthetic duration target: the mel frames split evenly over the tokens.

        Not an alignment, and the reason the duration head can only learn a speaking
        rate. It used to add Gaussian noise before rescaling to the same sum, so the
        target for an identical example changed on every epoch and worker without
        carrying any information.
        """
        if text_len == 0:
            return [1.0]

        return np.full(text_len, mel_len / text_len, dtype=np.float64)

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
