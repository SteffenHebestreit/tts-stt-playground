"""Export a trained VITS checkpoint to ONNX with a runtime-compatible config bundle."""

import torch
from pathlib import Path
import json
import logging
from datetime import datetime

from validation import phoneme_id_map_from_entries

logger = logging.getLogger(__name__)

class ModelExporter:
    """Export trained checkpoints to ONNX bundles without deploying them."""

    def __init__(self):
        """Create the export directory used for generated model artifacts."""
        self.export_dir = Path("models")
        self.export_dir.mkdir(exist_ok=True)
    
    async def export_to_onnx(self, job_id: str) -> Path:
        """Export a training checkpoint to ONNX and write its companion config bundle."""
        
        checkpoint_path = Path(f"checkpoints/{job_id}/final_model.pt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create export directory
        export_path = self.export_dir / job_id
        export_path.mkdir(exist_ok=True)
        
        # Load configuration from checkpoint (preferred) or fallback
        config = checkpoint.get('config', self._load_config(job_id))
        logger.info(f"Using training config: hidden_channels={config.get('hidden_channels', 192)}")
        logger.info(f"Model architecture: {config.get('n_layers', 6)} layers")
        
        # Export to ONNX
        onnx_path = export_path / f"{job_id}.onnx"
        
        try:
            # Create simplified inference model with exact training config
            from vits_model import VITS, VITSConfig
            
            # Create VITSConfig object with exact training parameters
            vits_config = VITSConfig(**config)
            logger.info(f"VITS Config - hidden_channels: {vits_config.hidden_channels}")
            logger.info(f"VITS Config - inter_channels: {vits_config.inter_channels}")
            logger.info(f"VITS Config - n_layers: {vits_config.n_layers}")
            
            model = VITS(vits_config)
            
            # Load state dict with strict=False to handle any missing keys gracefully
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()
            
            # Create dummy input for tracing
            batch_size = 1
            max_seq_len = 100
            
            dummy_text = torch.randint(0, config.get('n_vocab', 256), (batch_size, max_seq_len), dtype=torch.long)
            dummy_text_lengths = torch.tensor([max_seq_len], dtype=torch.long)
            
            # Export with simplified inputs
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (dummy_text, dummy_text_lengths),
                    onnx_path,
                    input_names=['text', 'text_lengths'],
                    output_names=['audio'],
                    dynamic_axes={
                        'text': {0: 'batch_size', 1: 'sequence'},
                        'text_lengths': {0: 'batch_size'},
                        'audio': {0: 'batch_size', 1: 'time'}
                    },
                    opset_version=15,
                    do_constant_folding=True,
                    verbose=False
                )
        
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise RuntimeError(f"ONNX export failed: {e}. Training may need more epochs or the model architecture has issues.")
        
        # Build phoneme vocab from dataset (must happen before writing config)
        lang = config.get('language', 'en')
        lang_map = {
            'de': 'de', 'en': 'en-us', 'fr': 'fr-fr', 'es': 'es',
            'it': 'it', 'nl': 'nl', 'pt': 'pt', 'ru': 'ru',
        }
        phonemizer_lang = lang_map.get(lang, 'en-us')
        phoneme_id_map = await self._create_phoneme_map(job_id, export_path, config)

        # Create Piper config file (includes phoneme vocab for custom inference)
        piper_config = {
            "audio": {
                "sample_rate": config['sample_rate'],
                "quality": config.get('quality', 'medium')
            },
            "espeak": {
                "voice": lang
            },
            "inference": {
                "noise_scale": 0.667,
                "length_scale": 1.0,
                "noise_w": 0.8
            },
            "phonemizer_language": phonemizer_lang,
            "phoneme_id_map": phoneme_id_map,
            "model_card": {
                "name": job_id,
                "language": lang,
                "dataset": "custom",
                "version": "1.0.0",
                "speaker": config.get('speaker_name', 'default')
            }
        }

        config_path = export_path / f"{job_id}.json"
        with open(config_path, 'w') as f:
            json.dump(piper_config, f, indent=2)

        # Free the model from CPU RAM — it's been exported to disk
        import gc
        del model
        gc.collect()

        logger.info(f"Model exported to: {onnx_path}")
        logger.info(f"Config saved to: {config_path}")
        
        return onnx_path
    
    def _load_config(self, job_id: str) -> dict:
        """Load training configuration"""
        config_path = Path(f"checkpoints/{job_id}/config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default config if not found
        return {
            'sample_rate': 22050,
            'hidden_channels': 192,
            'inter_channels': 192,
            'n_layers': 6,
            'n_vocab': 256,
            'n_heads': 2,
            'dropout_p': 0.1,
            'n_mels': 80,
            'quality': 'medium',
            'language': 'de',
            'speaker_name': 'stst'
        }
    
    async def _create_phoneme_map(self, job_id: str, export_path: Path, config: dict) -> dict:
        """Build phoneme->id mapping that matches the training vocabulary, and save it.

        Training builds its vocabulary from the ``train.json`` split via
        ``TTSDataset._create_phoneme_vocab()`` — ``sorted(phonemes ∪ special_tokens)``.
        To produce an identical id mapping for inference we must read the SAME
        split of the SAME dataset, located by the model name stored in the
        training config (the dataset dir is named after the model, not the uuid
        job id). Falls back to the job id, then a scan, then ``metadata.json``.

        Returns the mapping dict so callers can embed it in the model config.
        """
        model_name = (config or {}).get('speaker_name')

        # Prefer the dataset whose directory matches the trained model name.
        candidates = []
        if model_name:
            candidates.append(Path("data") / model_name)
        candidates.append(Path(f"data/{job_id}"))

        dataset_path = next((c for c in candidates if c.exists()), None)
        if dataset_path is None:
            data_root = Path("data")
            if data_root.exists():
                for candidate in sorted(data_root.iterdir()):
                    if candidate.is_dir() and (candidate / "train.json").exists():
                        dataset_path = candidate
                        break

        phoneme_map = {}
        if dataset_path is not None and dataset_path.exists():
            # train.json matches the split used to build the training vocab;
            # metadata.json (train+val superset) is only a fallback.
            source_path = dataset_path / "train.json"
            if not source_path.exists():
                source_path = dataset_path / "metadata.json"

            if source_path.exists():
                with open(source_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)

                # Build the vocab exactly as TTSDataset._create_phoneme_vocab() does
                phoneme_map = phoneme_id_map_from_entries(entries)
                logger.info(f"Phoneme map built from {source_path} ({len(phoneme_map)} symbols)")
            else:
                logger.warning(f"No train.json/metadata.json under {dataset_path}; phoneme map will be empty")
        else:
            logger.warning("Dataset directory not found for export; phoneme map will be empty")

        # Save as standalone file for debugging / manual inspection
        phoneme_map_path = export_path / "phonemes.json"
        with open(phoneme_map_path, 'w', encoding='utf-8') as f:
            json.dump(phoneme_map, f, indent=2, ensure_ascii=False)

        return phoneme_map
    
