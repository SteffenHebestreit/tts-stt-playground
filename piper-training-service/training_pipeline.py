import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Optional, Callable
import asyncio
import gc
from datetime import datetime

logger = logging.getLogger(__name__)

from vits_model import VITS, VITSConfig
from dataset import TTSDataset, collate_fn
from training_utils import get_optimizer, get_scheduler


class OptimizedTrainingPipeline:
    def __init__(self):
        self.device, self.device_info = self._detect_compute_device()
        self.memory_limit = self._get_memory_limit()
        logger.info(f"Training device: {self.device}")
        logger.info(f"GPU memory available: {self.memory_limit:.2f} GB")

        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            test_tensor = torch.randn(100, 100).to(self.device)
            test_result = torch.mm(test_tensor, test_tensor)
            logger.info(f"GPU test passed — result shape: {test_result.shape}")
            del test_tensor, test_result
            torch.cuda.empty_cache()

        # FP16 mixed precision is disabled: VITS normalizing flow overflows FP16
        # (max ~65504), causing NaN losses. FP32 is used instead.
        self.enable_mixed_precision = False
        self.gradient_accumulation_steps = 2
        self.memory_cleanup_interval = 5

        if self.enable_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled (FP16)")
        else:
            self.scaler = None
            logger.info("Using FP32 training (mixed precision disabled)")

    def _detect_compute_device(self):
        """Detect the best available compute device."""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            device_info = {
                'type': 'CUDA GPU',
                'name': torch.cuda.get_device_name(0),
                'memory': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
            return device, device_info

        device = torch.device('cpu')
        device_info = {'type': 'CPU', 'cores': os.cpu_count()}
        return device, device_info

    def _get_memory_limit(self):
        """Return usable memory limit (80% of GPU, or 8 GB for CPU)."""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_properties(0).total_memory / 1024**3 * 0.8
        return 8.0

    def _create_optimized_dataloader(self, dataset, batch_size):
        """Create DataLoader with persistent workers and prefetching."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=2
        )

    def _aggressive_memory_cleanup(self):
        """Free GPU cache and run GC without blocking the training loop."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def _log_memory_usage(self, prefix=""):
        """Log GPU memory usage when it exceeds 15 GB."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > 15.0:
                logger.info(f"{prefix} GPU memory: {allocated:.1f} GB")

    def train_sync(
        self,
        job_id: str,
        request,
        callback: Optional[Callable] = None,
        override_config: Optional[dict] = None,
        resume_from: Optional[str] = None,
    ):
        """Main training loop. Runs in a thread to avoid blocking the event loop."""
        logger.info(f"Training starting for job {job_id}")

        config = self._get_config(request)
        if override_config:
            config.update(override_config)

        # Cap batch size for stability
        optimized_batch_size = min(config.get('batch_size', 8), 4)
        config['batch_size'] = optimized_batch_size
        logger.info(f"Batch size: {optimized_batch_size}")

        checkpoint_dir = Path(f"checkpoints/{job_id}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        dataset = TTSDataset(data_dir=Path("data") / request.model_name, config=config)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        dataloader = self._create_optimized_dataloader(dataset, optimized_batch_size)
        logger.info(f"DataLoader: {len(dataloader)} batches/epoch")

        vits_config = VITSConfig(**config)
        model = VITS(vits_config).to(self.device)

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)
        logger.info(f"Optimizer: {type(optimizer).__name__}, Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")

        # Resume from checkpoint if requested
        start_epoch = 0
        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location=self.device)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            logger.info(f"Resumed at epoch {start_epoch}, previous loss: {ckpt.get('loss', 'n/a')}")

        epochs = config['epochs']
        logger.info(f"Training: epochs {start_epoch+1}–{epochs}, {len(dataloader)} batches each")

        model.train()

        for epoch in range(start_epoch, epochs):
            if callback:
                current_status = callback({'check_status': True})
                if current_status is not None and getattr(current_status, 'status', None) == 'cancelled':
                    logger.info(f"Training cancelled at epoch {epoch}")
                    break

            logger.info(f"=== EPOCH {epoch+1}/{epochs} ===")
            epoch_losses = []
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                try:
                    if batch is None:
                        logger.warning(f"Skipping batch {batch_idx} — all items failed to load")
                        continue

                    gpu_batch = {}
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            gpu_batch[k] = v.to(self.device, non_blocking=True)
                        else:
                            gpu_batch[k] = v

                    model.train()

                    if self.enable_mixed_precision and self.scaler:
                        with torch.cuda.amp.autocast():
                            loss_components = model.compute_loss(gpu_batch)
                            total_loss = sum(loss_components.values())
                            scaled_loss = total_loss / self.gradient_accumulation_steps

                        current_loss = total_loss.item()
                        if not torch.isfinite(total_loss):
                            logger.warning(f"NaN/Inf loss at batch {batch_idx} — skipping, resetting scaler")
                            optimizer.zero_grad()
                            self.scaler = torch.cuda.amp.GradScaler()
                            continue

                        self.scaler.scale(scaled_loss).backward()

                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad()

                    else:
                        loss_components = model.compute_loss(gpu_batch)
                        total_loss = sum(loss_components.values())
                        current_loss = total_loss.item()

                        if not torch.isfinite(total_loss):
                            logger.warning(f"NaN/Inf loss at batch {batch_idx} — skipping")
                            optimizer.zero_grad()
                            continue

                        scaled_loss = total_loss / self.gradient_accumulation_steps
                        scaled_loss.backward()

                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()

                    epoch_losses.append(current_loss)

                    avg_loss = sum(epoch_losses[-10:]) / min(len(epoch_losses), 10)
                    gpu_info = ""
                    if self.device.type == 'cuda':
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        gpu_info = f" | GPU: {gpu_memory:.1f}GB"
                    logger.info(f"Batch {batch_idx+1}/{len(dataloader)}: Loss={avg_loss:.4f}{gpu_info}")

                    if callback:
                        callback({
                            'current_epoch': epoch + 1,
                            'total_epochs': epochs,
                            'progress': ((epoch * len(dataloader) + batch_idx) /
                                         (epochs * len(dataloader))) * 100,
                            'loss': avg_loss,
                            'message': f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}{gpu_info}"
                        })

                    if batch_idx % self.memory_cleanup_interval == 0:
                        self._aggressive_memory_cleanup()

                    if batch_idx % 50 == 0:
                        self._log_memory_usage(f"Batch {batch_idx}")

                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM at batch {batch_idx} — attempting batch size reduction")
                    optimizer.zero_grad()
                    model.zero_grad()
                    self._aggressive_memory_cleanup()

                    if optimized_batch_size > 2:
                        optimized_batch_size = max(2, optimized_batch_size // 2)
                        logger.warning(f"Batch size reduced to {optimized_batch_size}")
                        dataloader = self._create_optimized_dataloader(dataset, optimized_batch_size)
                        break
                    else:
                        raise RuntimeError("Cannot reduce batch size further — insufficient GPU memory")

                except Exception as e:
                    logger.error(f"Training error at batch {batch_idx}: {e}")
                    continue

            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                logger.info(f"Epoch {epoch+1} complete — average loss: {avg_epoch_loss:.6f}")
                if scheduler:
                    scheduler.step()

            if (epoch + 1) % config.get('save_interval', 5) == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss if epoch_losses else 0.0,
                    'config': config
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

                job_state = {
                    'job_id': job_id,
                    'model_name': config.get('speaker_name', request.model_name),
                    'language': config.get('language', 'de'),
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'loss': avg_epoch_loss if epoch_losses else 0.0,
                    'latest_checkpoint': str(checkpoint_path),
                    'status': 'training',
                    'config': config,
                }
                state_path = checkpoint_dir / "job_state.json"
                with open(state_path, 'w') as f:
                    json.dump(job_state, f, indent=2)
                logger.info(f"Job state saved: {state_path}")

        final_path = checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_complete': True
        }, final_path)
        logger.info(f"Training complete — model saved: {final_path}")

        state_path = checkpoint_dir / "job_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    job_state = json.load(f)
                job_state['status'] = 'completed'
                with open(state_path, 'w') as f:
                    json.dump(job_state, f, indent=2)
            except Exception:
                pass

        del model, optimizer, scheduler, dataset, dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Training memory freed")

        if callback:
            callback({'status': 'completed', 'message': 'Training completed successfully'})

    def _get_config(self, request):
        """Build training configuration for PiperTTS-compatible VITS model."""
        return {
            'batch_size': min(request.batch_size, 32) if hasattr(request, 'batch_size') else 2,
            'epochs': request.epochs if hasattr(request, 'epochs') else 1000,
            'learning_rate': 0.0001,   # Conservative: 2e-4 causes NaN on VITS normalizing flow
            'save_interval': 5,
            'sample_rate': 22050,
            'hidden_channels': 192,
            'inter_channels': 192,
            'n_layers': 6,
            'n_heads': 2,
            'dropout_p': 0.1,
            'n_vocab': 256,
            'n_mels': 80,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'language': request.language if hasattr(request, 'language') else 'de',
            'speaker_name': request.model_name if hasattr(request, 'model_name') else 'custom',
            'quality': 'medium',
            'vocoder_type': 'hifigan',
            'use_pitch': False,
            'use_energy': False,
        }

    async def start_training(self, request, job_id: str, jobs_dict: dict):
        """Start a training job (async wrapper)."""
        try:
            await self.train(job_id, request,
                             callback=lambda x: self._update_job_status(job_id, jobs_dict, x))
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if job_id in jobs_dict:
                jobs_dict[job_id].status = "failed"
                jobs_dict[job_id].message = str(e)

    def _update_job_status(self, job_id: str, jobs_dict: dict, update_data: dict):
        """Update job status from callback data."""
        if job_id in jobs_dict:
            job = jobs_dict[job_id]
            if 'check_status' in update_data:
                return job
            for key, value in update_data.items():
                if hasattr(job, key):
                    setattr(job, key, value)

    def cleanup_old_checkpoints(self, checkpoint_dir: Path, keep_latest: int = 5):
        """Remove all but the latest N checkpoints to save disk space."""
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > keep_latest:
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for old_checkpoint in checkpoints[:-keep_latest]:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint.name}")

    async def generate_training_metadata(self, dataset_path: Path, audio_files_info: list, model_name: str):
        """Write a metadata.json summary file for a prepared dataset."""
        metadata = {
            "dataset_name": model_name,
            "audio_files": len(audio_files_info),
            "total_duration": 0,
            "sample_rate": 22050,
            "created_at": str(datetime.now())
        }
        metadata_path = dataset_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Training metadata saved: {metadata_path}")
        return metadata
