import torch
import onnx
from pathlib import Path
import json
import logging
import numpy as np
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelExporter:
    def __init__(self):
        self.export_dir = Path("models")
        self.export_dir.mkdir(exist_ok=True)
    
    async def export_to_onnx(self, job_id: str) -> Path:
        """Export PyTorch model to ONNX format for Piper"""
        
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
        phoneme_id_map = await self._create_phoneme_map(job_id, export_path)

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

        # Copy model to PiperTTS service if shared models directory exists
        await self._copy_to_piper_service(job_id, onnx_path, config_path)

        logger.info(f"Model exported to: {onnx_path}")
        logger.info(f"Config saved to: {config_path}")
        
        return onnx_path
    
    async def _copy_to_piper_service(self, job_id: str, onnx_path: Path, config_path: Path):
        """Copy exported model to PiperTTS service"""
        try:
            shared_models_dir = Path("/app/shared_models")
            if shared_models_dir.exists():
                piper_models_dir = shared_models_dir / "custom" / job_id
                piper_models_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy ONNX model
                import shutil
                shutil.copy2(onnx_path, piper_models_dir / f"{job_id}.onnx")
                shutil.copy2(config_path, piper_models_dir / f"{job_id}.json")
                
                logger.info(f"Model copied to PiperTTS service: {piper_models_dir}")
        except Exception as e:
            logger.warning(f"Failed to copy model to PiperTTS service: {e}")
    
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
    
    async def _create_phoneme_map(self, job_id: str, export_path: Path) -> dict:
        """Build phoneme->id mapping from dataset metadata and save it.

        Returns the mapping dict so callers can embed it in the model config.
        The ordering matches TTSDataset._create_phoneme_vocab() exactly:
        sorted(phonemes ∪ special_tokens).
        """
        # Try to find dataset directory - check by job_id first, then scan data/
        dataset_path = Path(f"data/{job_id}")
        if not dataset_path.exists():
            data_root = Path("data")
            if data_root.exists():
                for candidate in data_root.iterdir():
                    if candidate.is_dir() and (candidate / "metadata.json").exists():
                        dataset_path = candidate
                        break

        phoneme_map = {}
        if dataset_path.exists():
            metadata_path = dataset_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Collect every character seen in phonemized text
                phonemes = set()
                for item in metadata:
                    phoneme_text = item.get('phonemes', item.get('text', ''))
                    if phoneme_text:
                        phonemes.update(list(phoneme_text))

                # Replicate TTSDataset._create_phoneme_vocab() ordering
                special_tokens = ['<pad>', '<unk>', '<start>', '<end>', ' ']
                all_phonemes = sorted(phonemes.union(special_tokens))
                phoneme_map = {p: i for i, p in enumerate(all_phonemes)}

        # Save as standalone file for debugging / manual inspection
        phoneme_map_path = export_path / "phonemes.json"
        with open(phoneme_map_path, 'w') as f:
            json.dump(phoneme_map, f, indent=2, ensure_ascii=False)

        return phoneme_map
    
    def create_model_info(self, job_id: str, export_path: Path, config: dict):
        """Create model information file"""
        info = {
            "model_id": job_id,
            "created_at": str(datetime.now()),
            "model_type": "VITS",
            "framework": "PyTorch",
            "exported_format": "ONNX",
            "sample_rate": config['sample_rate'],
            "language": config.get('language', 'en'),
            "speaker": config.get('speaker_name', 'custom'),
            "quality": config.get('quality', 'medium'),
            "model_size": {
                "hidden_channels": config['hidden_channels'],
                "layers": config['n_layers'],
                "vocab_size": config.get('n_vocab', 256)
            }
        }
        
        info_path = export_path / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        return info_path
