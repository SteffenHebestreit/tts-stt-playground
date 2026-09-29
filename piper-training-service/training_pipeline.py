"""Training loop of the experimental mel-flow trainer, with GPU memory management.

Supports checkpoint resume, gradient accumulation and automatic batch-size
reduction on OOM. Failures are loud: a run that cannot train raises instead of
completing with untrained weights (see ``TrainingFailed``).

The objective is NOT a real VITS/Piper fine-tune; see ``training_utils.TRAINER_KIND``.
"""

import os
import json
import math
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Callable
import gc
from datetime import datetime

logger = logging.getLogger(__name__)

from vits_model import VITS, VITSConfig
from dataset import TTSDataset, collate_fn
from training_utils import (
    get_optimizer, get_scheduler, atomic_torch_save, TRAINER_KIND, TRAINER_CAVEAT,
)
from validation import (
    VOCAB_FILENAME, env_int, prune_old_checkpoints, save_vocab, vocab_for_checkpoint,
)


class TrainingCancelled(Exception):
    """Raised by ``train_sync`` when the caller cancelled the job mid-run.

    Distinct from a failure on purpose. The callers export and deploy whatever
    ``train_sync`` returns, so cancellation had to become something they cannot
    fall through — and it must not be reported as "failed" either, since the
    checkpoints are intact and the job can be resumed.
    """


class TrainingFailed(RuntimeError):
    """Raised by ``train_sync`` when the run cannot produce a usable model.

    Every caller already treats any exception from ``train_sync`` as a failed job
    and skips export and deployment; what this adds is a message that says why.
    Before, per-batch exceptions were logged and skipped, so a run in which every
    batch failed (or that had no batch at all) still "completed", saved
    untrained weights, exported them and deployed the result as a voice.
    """


class _FailureTracker:
    """Counts failed batches and aborts the run once too many come in a row."""

    def __init__(self, limit: int):
        self.limit = limit
        self.consecutive = 0
        self.in_epoch = 0
        self.last = None

    def start_epoch(self) -> None:
        self.in_epoch = 0

    def ok(self) -> None:
        self.consecutive = 0

    def fail(self, reason: str, where: str) -> None:
        """Record one failed batch; raise ``TrainingFailed`` at the limit."""
        self.consecutive += 1
        self.in_epoch += 1
        self.last = reason
        logger.warning(
            f"Batch {where} failed ({reason}); {self.consecutive} in a row, limit {self.limit}"
        )
        if self.consecutive >= self.limit:
            raise TrainingFailed(
                f"Aborting after {self.consecutive} consecutive failed batches "
                f"(TRAIN_MAX_CONSECUTIVE_FAILURES={self.limit}). Last error: {reason}"
            )


class OptimizedTrainingPipeline:
    """Manage dataset loading, checkpointing, and stable training execution."""

    def __init__(self):
        """Initialise device selection and training-safety defaults."""
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

        # FP32 only. FP16 mixed precision was disabled long ago because the
        # normalizing flow overflows FP16 (max ~65504) and gives NaN losses; the
        # dead GradScaler branch that remained has been removed.
        self.gradient_accumulation_steps = 2

        # Hard cap on batch size for training stability. The from-scratch
        # loss is sensitive to large batches; override via MAX_TRAIN_BATCH_SIZE
        # if you have GPU headroom. OOM still triggers automatic reduction.
        try:
            self.max_batch_size = max(1, int(os.getenv("MAX_TRAIN_BATCH_SIZE", "4")))
        except ValueError:
            self.max_batch_size = 4
        logger.info(f"Max training batch size: {self.max_batch_size}")

        # Abort after this many batches in a row that failed to load, raised, or
        # produced a non-finite loss/gradient.
        self.max_consecutive_failures = env_int("TRAIN_MAX_CONSECUTIVE_FAILURES", 5, minimum=1)
        # Periodic checkpoints kept per job (best and final are never pruned);
        # 0 keeps them all.
        self.keep_last_checkpoints = env_int("KEEP_LAST_CHECKPOINTS", 5, minimum=0)
        # DataLoader worker processes; 0 loads in the training thread.
        self.num_workers = env_int("TRAIN_NUM_WORKERS", 2, minimum=0)

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

    def _create_optimized_dataloader(self, dataset, batch_size, shuffle=True, drop_last=True,
                                     num_workers=None):
        """Create DataLoader with persistent workers and prefetching."""
        workers = self.num_workers if num_workers is None else num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=drop_last,
            collate_fn=collate_fn,
            # Both are errors with num_workers=0.
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
        )

    def _aggressive_memory_cleanup(self):
        """Free GPU cache and run GC. For OOM recovery and teardown, not per step:
        emptying the cache every few batches made the allocator give memory back to
        the driver and re-request it, a device-wide sync each time."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def _log_memory_usage(self, prefix=""):
        """Log GPU memory usage when it exceeds 15 GB."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > 15.0:
                logger.info(f"{prefix} GPU memory: {allocated:.1f} GB")

    @staticmethod
    def _optimizer_step(model, optimizer) -> bool:
        """Clip and apply the accumulated gradient; False (and no step) if it is not finite.

        A NaN gradient passed through ``clip_grad_norm_`` unchanged and was applied,
        turning every weight into NaN; the loss check before ``backward`` cannot
        see that.
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        stepped = bool(torch.isfinite(grad_norm))
        if stepped:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return stepped

    def _to_device(self, batch):
        """Move the tensors of a collated batch to the training device."""
        return {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    def _evaluate(self, model, loader):
        """Mean total loss over the validation loader, or None if nothing was scored."""
        total, count = 0.0, 0
        model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    if batch is None:
                        continue
                    loss = float(sum(model.compute_loss(self._to_device(batch)).values()))
                    if math.isfinite(loss):
                        total += loss
                        count += 1
        finally:
            model.train()
        return total / count if count else None

    def train_sync(
        self,
        job_id: str,
        request,
        callback: Optional[Callable] = None,
        override_config: Optional[dict] = None,
        resume_from: Optional[str] = None,
    ):
        """Main training loop. Runs in a thread to avoid blocking the event loop.

        Returns normally only when at least one optimizer step ran and the saved
        weights are finite. Raises ``TrainingCancelled`` if the job was cancelled
        and ``TrainingFailed`` (or another exception) if it could not train.
        """
        logger.info(f"Training starting for job {job_id}")
        logger.warning("Trainer kind '%s'. %s", TRAINER_KIND, TRAINER_CAVEAT)
        if callback:
            # Ignored by update_training_status unless the job record has these
            # fields; harmless otherwise.
            callback({'trainer_kind': TRAINER_KIND, 'trainer_caveat': TRAINER_CAVEAT})

        config = self._get_config(request)
        if override_config:
            config.update(override_config)

        # Cap batch size for stability (configurable via MAX_TRAIN_BATCH_SIZE)
        requested_batch_size = config.get('batch_size', 8)
        optimized_batch_size = min(requested_batch_size, self.max_batch_size)
        if optimized_batch_size < requested_batch_size:
            logger.warning(
                f"Requested batch size {requested_batch_size} capped to "
                f"{optimized_batch_size} (MAX_TRAIN_BATCH_SIZE={self.max_batch_size})"
            )
        config['batch_size'] = optimized_batch_size
        logger.info(f"Batch size: {optimized_batch_size}")

        checkpoint_dir = Path(f"checkpoints/{job_id}")

        # Resume state is read first: it carries the vocabulary the dataset has to use.
        ckpt = None
        if resume_from:
            if not Path(resume_from).exists():
                # Used to fall through and silently train from scratch under the
                # same job id, overwriting the run the caller meant to continue.
                raise TrainingFailed(f"Resume checkpoint not found: {resume_from}")
            logger.info(f"Resuming from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location='cpu', weights_only=True)
            # A checkpoint from before boundary tokens existed was trained
            # without them; keep feeding it what it was trained on.
            config['boundary_tokens'] = bool((ckpt.get('config') or {}).get('boundary_tokens', False))

        vocab = None
        if ckpt is not None:
            vocab, vocab_source = vocab_for_checkpoint(
                ckpt, Path(resume_from).parent, config.get('n_vocab')
            )
            if vocab is None:
                logger.warning(
                    "Checkpoint %s predates persisted vocabularies. The vocabulary is "
                    "rebuilt from the current train.json; if the dataset changed since "
                    "the checkpoint was written, symbol ids no longer match the trained "
                    "embedding.", resume_from
                )
            else:
                logger.info("Vocabulary restored from %s (%d symbols)", vocab_source, len(vocab))

        data_dir = Path("data") / request.model_name
        dataset = TTSDataset(data_dir=data_dir, config=config, phoneme_to_id=vocab)
        vocab = dataset.phoneme_to_id
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        # drop_last=True discards an incomplete batch, so with fewer examples than
        # the batch size the loader is EMPTY: every epoch runs zero steps in
        # milliseconds, and the job used to "complete" and deploy random weights.
        if len(dataset) < optimized_batch_size:
            raise TrainingFailed(
                f"The training split has only {len(dataset)} usable example(s) but "
                f"the effective batch size is {optimized_batch_size}, so not a single "
                f"batch could be formed. Add more audio, or request a smaller "
                f"batch_size (MAX_TRAIN_BATCH_SIZE={self.max_batch_size} also caps it)."
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_vocab(
            checkpoint_dir / VOCAB_FILENAME, vocab,
            n_vocab=config.get('n_vocab', 256), boundary_tokens=config.get('boundary_tokens', True),
        )

        # The validation split scores held-out examples when a checkpoint is
        # written. It must use the TRAINING vocabulary (see TTSDataset).
        val_loader = None
        if (data_dir / "val.json").exists():
            try:
                val_dataset = TTSDataset(
                    data_dir=data_dir, config=config, split="val", phoneme_to_id=vocab
                )
                val_loader = self._create_optimized_dataloader(
                    val_dataset, max(1, min(optimized_batch_size, len(val_dataset))),
                    shuffle=False, drop_last=False,
                    # Small and read once per checkpoint: not worth a second set
                    # of persistent worker processes.
                    num_workers=0,
                )
            except (ValueError, FileNotFoundError) as exc:
                logger.warning("No usable validation split, scoring on training loss: %s", exc)

        dataloader = self._create_optimized_dataloader(dataset, optimized_batch_size)
        logger.info(f"DataLoader: {len(dataloader)} batches/epoch")
        if len(dataloader) == 0:
            raise TrainingFailed("The DataLoader yields no batches; nothing to train on.")

        vits_config = VITSConfig(**config)
        model = VITS(vits_config).to(self.device)

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)
        logger.info(f"Optimizer: {type(optimizer).__name__}, Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")

        # Optimizer steps this job has taken in total, across resumes. A model
        # with none is the initial random weights and must never be exported.
        optimizer_steps = 0
        best_metric, best_metric_name = float('inf'), None
        start_epoch = 0
        if ckpt is not None:
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            if 'optimizer_steps' in ckpt:
                optimizer_steps = int(ckpt['optimizer_steps'])
            elif (ckpt.get('optimizer_state_dict') or {}).get('state'):
                # Older checkpoint: Adam keeps per-parameter state only after a step.
                optimizer_steps = 1
            best_metric = ckpt.get('best_metric', float('inf'))
            best_metric_name = ckpt.get('best_metric_name')
            logger.info(f"Resumed at epoch {start_epoch}, previous loss: {ckpt.get('loss', 'n/a')}")
            ckpt = None  # release the CPU copy of the weights and optimizer state

        epochs = config['epochs']
        logger.info(f"Training: epochs {start_epoch+1}–{epochs}, {len(dataloader)} batches each")

        def _write_job_state(state: dict) -> None:
            """Write job_state.json atomically; the resume endpoint reads it."""
            state_path = checkpoint_dir / "job_state.json"
            tmp = state_path.with_name(state_path.name + ".tmp")
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, state_path)

        def _mark_state(status: str, error: Optional[str] = None) -> None:
            """Record the terminal status in job_state.json, if one was written."""
            state_path = checkpoint_dir / "job_state.json"
            if not state_path.exists():
                return
            try:
                with open(state_path) as f:
                    job_state = json.load(f)
                job_state['status'] = status
                if error:
                    job_state['error'] = error
                _write_job_state(job_state)
            except Exception:
                pass

        def _free_memory() -> None:
            # `nonlocal` is load-bearing, not decoration. Rebinding (or `del`) on
            # a name inside a nested function makes that name LOCAL to it unless
            # it is declared nonlocal, so without it this would only rebind
            # locals and the enclosing bindings would keep the GPU tensors alive
            # past gc.collect(). Rebinding to None rather than `del` also keeps
            # it safe on the failure paths, where not every name is bound yet.
            nonlocal model, optimizer, scheduler, dataset, dataloader, val_loader
            model = optimizer = scheduler = dataset = dataloader = val_loader = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Training memory freed")

        def _checkpoint_payload(epoch_done: int, loss: float, with_optimizer: bool) -> dict:
            payload = {
                'epoch': epoch_done,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'config': config,
                'phoneme_to_id': vocab,
                'trainer_kind': TRAINER_KIND,
                'optimizer_steps': optimizer_steps,
                'best_metric': best_metric,
                'best_metric_name': best_metric_name,
            }
            if with_optimizer:
                payload['optimizer_state_dict'] = optimizer.state_dict()
            return payload

        model.train()

        cancelled_at_epoch = None
        failures = _FailureTracker(self.max_consecutive_failures)
        try:
            for epoch in range(start_epoch, epochs):
                if callback:
                    current_status = callback({'check_status': True})
                    if current_status is not None and getattr(current_status, 'status', None) == 'cancelled':
                        logger.info(f"Training cancelled at epoch {epoch}")
                        cancelled_at_epoch = epoch
                        break

                logger.info(f"=== EPOCH {epoch+1}/{epochs} ===")
                epoch_losses = []
                epoch_steps = 0
                pending = 0  # batches whose gradient is accumulated but not yet applied
                batch_size_reduced = False
                n_batches = len(dataloader)
                failures.start_epoch()
                optimizer.zero_grad(set_to_none=True)

                for batch_idx, batch in enumerate(dataloader):
                    where = f"{batch_idx+1}/{n_batches}"
                    reason = None
                    try:
                        if batch is None:
                            reason = "every item of the batch failed to load"
                        else:
                            gpu_batch = self._to_device(batch)
                            model.train()

                            loss_components = model.compute_loss(gpu_batch)
                            total_loss = sum(loss_components.values())

                            if not torch.isfinite(total_loss):
                                reason = "non-finite loss"
                            else:
                                current_loss = total_loss.item()
                                (total_loss / self.gradient_accumulation_steps).backward()
                                pending += 1

                                if pending >= self.gradient_accumulation_steps:
                                    pending = 0
                                    if self._optimizer_step(model, optimizer):
                                        epoch_steps += 1
                                        optimizer_steps += 1
                                    else:
                                        reason = "non-finite gradient"

                                if reason is None:
                                    epoch_losses.append(current_loss)

                    except torch.cuda.OutOfMemoryError:
                        logger.error(f"OOM at batch {batch_idx} — attempting batch size reduction")
                        optimizer.zero_grad(set_to_none=True)
                        self._aggressive_memory_cleanup()

                        if optimized_batch_size > 2:
                            optimized_batch_size = max(2, optimized_batch_size // 2)
                            logger.warning(f"Batch size reduced to {optimized_batch_size}")
                            dataloader = self._create_optimized_dataloader(dataset, optimized_batch_size)
                            batch_size_reduced = True
                            break
                        else:
                            raise RuntimeError("Cannot reduce batch size further — insufficient GPU memory")

                    except Exception as e:
                        # Not "skip and carry on": the loop used to log and continue
                        # for any exception, so a bug that hit every batch produced
                        # an endless run of identical errors and a job that finished.
                        logger.error(f"Training error at batch {batch_idx}: {e}", exc_info=True)
                        # The failure may have come after a partial backward.
                        optimizer.zero_grad(set_to_none=True)
                        pending = 0
                        reason = f"{type(e).__name__}: {e}"

                    if reason is not None:
                        failures.fail(reason, where)
                        continue

                    failures.ok()

                    avg_loss = sum(epoch_losses[-10:]) / min(len(epoch_losses), 10)
                    gpu_info = ""
                    if self.device.type == 'cuda':
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        gpu_info = f" | GPU: {gpu_memory:.1f}GB"
                    logger.info(f"Batch {batch_idx+1}/{n_batches}: Loss={avg_loss:.4f}{gpu_info}")

                    if callback:
                        callback({
                            'current_epoch': epoch + 1,
                            'total_epochs': epochs,
                            'progress': ((epoch * n_batches + batch_idx) /
                                         (epochs * n_batches)) * 100,
                            'loss': avg_loss,
                            'message': f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{n_batches}{gpu_info}"
                        })

                    if batch_idx % 50 == 0:
                        self._log_memory_usage(f"Batch {batch_idx}")

                # Gradient left over from an odd number of batches. It used to be
                # thrown away by the next epoch's zero_grad, so with one batch per
                # epoch the model never took a single step.
                if pending > 0 and not batch_size_reduced:
                    if self._optimizer_step(model, optimizer):
                        epoch_steps += 1
                        optimizer_steps += 1
                    else:
                        failures.fail("non-finite gradient", "at the end of the epoch")

                if epoch_steps == 0 and not batch_size_reduced:
                    raise TrainingFailed(
                        f"Epoch {epoch+1} performed zero optimizer steps "
                        f"({failures.in_epoch} of {n_batches} batches failed"
                        f"{'; last error: ' + failures.last if failures.last else ''}). "
                        f"The model has not been trained."
                    )

                avg_epoch_loss = 0.0
                if epoch_losses:
                    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                    logger.info(f"Epoch {epoch+1} complete — average loss: {avg_epoch_loss:.6f}")
                    if scheduler:
                        scheduler.step()

                if (epoch + 1) % config.get('save_interval', 5) == 0:
                    # Best is judged on held-out loss when there is a validation
                    # split, on training loss otherwise. A failure to score must not
                    # kill a run that is otherwise training.
                    metric_name, metric = 'train_loss', avg_epoch_loss
                    if val_loader is not None:
                        try:
                            val_loss = self._evaluate(model, val_loader)
                        except Exception as exc:
                            logger.warning("Validation failed, scoring on training loss from now on: %s", exc)
                            val_loader = None
                            val_loss = None
                        if val_loss is not None:
                            metric_name, metric = 'val_loss', val_loss
                            logger.info(f"Validation loss: {val_loss:.6f}")

                    if metric_name != best_metric_name:
                        best_metric = float('inf')
                    if epoch_losses and math.isfinite(metric) and metric < best_metric:
                        best_metric, best_metric_name = metric, metric_name
                        atomic_torch_save(
                            {**_checkpoint_payload(epoch + 1, avg_epoch_loss, False),
                             'metric': metric, 'metric_name': metric_name},
                            checkpoint_dir / "best_model.pt",
                        )
                        logger.info(f"New best model ({metric_name}={metric:.6f}) at epoch {epoch+1}")

                    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                    atomic_torch_save(
                        _checkpoint_payload(epoch + 1, avg_epoch_loss, True), checkpoint_path
                    )
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

                    _write_job_state({
                        'job_id': job_id,
                        'model_name': config.get('speaker_name', request.model_name),
                        'language': config.get('language', 'de'),
                        'epoch': epoch + 1,
                        'total_epochs': epochs,
                        'loss': avg_epoch_loss,
                        'latest_checkpoint': str(checkpoint_path),
                        'status': 'training',
                        'config': config,
                        'trainer_kind': TRAINER_KIND,
                        'trainer_caveat': TRAINER_CAVEAT,
                        'optimizer_steps': optimizer_steps,
                        'best_metric': best_metric if math.isfinite(best_metric) else None,
                        'best_metric_name': best_metric_name,
                    })
                    logger.info(f"Job state saved: {checkpoint_dir / 'job_state.json'}")

                    # After the state file points at the new checkpoint, so a crash
                    # in between never leaves it pointing at a deleted one.
                    pruned = prune_old_checkpoints(
                        checkpoint_dir, self.keep_last_checkpoints, protect=(checkpoint_path,)
                    )
                    if pruned:
                        logger.info(
                            f"Pruned {len(pruned)} old checkpoint(s), keeping the last "
                            f"{self.keep_last_checkpoints} (KEEP_LAST_CHECKPOINTS)"
                        )
        except TrainingFailed as failed:
            logger.error(f"Training failed for job {job_id}: {failed}")
            _mark_state('failed', str(failed))
            _free_memory()
            raise
        except Exception:
            # Free the GPU before the caller reports the failure; the exception is
            # the caller's to handle.
            _free_memory()
            raise

        # Cancellation used to `break` and then fall straight into the block
        # below, which saves the final model, stamps the job state as finished and
        # calls back with status='completed' — overwriting the 'cancelled' the
        # user had just set. The caller then exported and DEPLOYED the
        # half-trained model. So DELETE /job/{id} appeared to work, and the voice
        # it was meant to stop went live anyway.
        #
        # Periodic checkpoints are already on disk, so the job stays resumable;
        # only the "this is a finished model" artefacts are withheld.
        if cancelled_at_epoch is not None:
            _mark_state('cancelled')
            _free_memory()
            raise TrainingCancelled(
                f"Training cancelled at epoch {cancelled_at_epoch + 1}/{epochs}"
            )

        # Nothing below may run for weights that were never trained or that
        # diverged: the callers export and deploy whatever this returns.
        if optimizer_steps == 0:
            message = (
                "No optimizer step was taken (nothing left to train after the "
                "resume point), so the weights are still the random initialisation."
            )
            _mark_state('failed', message)
            _free_memory()
            raise TrainingFailed(message)
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            message = "The weights contain NaN/Inf after training; the run diverged."
            _mark_state('failed', message)
            _free_memory()
            raise TrainingFailed(message)

        final_path = checkpoint_dir / "final_model.pt"
        atomic_torch_save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_complete': True,
            'phoneme_to_id': vocab,
            'trainer_kind': TRAINER_KIND,
            'trainer_caveat': TRAINER_CAVEAT,
            'optimizer_steps': optimizer_steps,
        }, final_path)
        logger.info(f"Training complete — model saved: {final_path}")

        _mark_state('completed')
        _free_memory()

        if callback:
            callback({'status': 'completed', 'message': 'Training completed successfully'})

    def _get_config(self, request):
        """Build training configuration for the experimental mel-flow model."""
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
            # Sequences are wrapped in <start>/<end>, as the runtime does at inference.
            'boundary_tokens': True,
        }

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
