"""Export a trained checkpoint to ONNX with a runtime-compatible config bundle."""

import asyncio
import inspect
import torch
from pathlib import Path
import json
import logging
import warnings
from datetime import datetime

from phonemization import exported_voice
from validation import (
    is_safe_name, phoneme_id_map_from_entries, validate_phoneme_id_map, vocab_for_checkpoint,
)
from training_utils import TRAINER_KIND, TRAINER_CAVEAT

logger = logging.getLogger(__name__)

ONNX_OPSET = 15

# Text lengths (in symbols) the exported graph is run with straight after
# export. Two different lengths, so a graph whose output length was frozen at
# export time cannot pass.
VERIFY_LENGTHS = (16, 48)


def onnx_export_kwargs() -> dict:
    """Extra ``torch.onnx.export`` arguments for the installed torch.

    ``dynamo=False`` selects the TorchScript tracer. From torch 2.9 the default is
    the ``torch.export``-based exporter, which needs ``onnxscript`` and, for the
    duration expansion, symbolic reasoning about a data-dependent output length;
    the tracer needs neither. The Dockerfiles pin the torch this was tested on.
    """
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        return {"dynamo": False}
    return {}


def verify_onnx_export(model, onnx_path, n_vocab: int, lengths=VERIFY_LENGTHS) -> dict:
    """Run the exported graph on CPU for several text lengths and compare with eager.

    Raises ``RuntimeError`` if an output is not finite or if its length differs
    from what the PyTorch model produces for the same text. The length is the
    check that matters: the previous export baked the durations of a random dummy
    input into the graph, so every text produced the same number of frames.
    Sample values are only compared loosely (logged, not fatal), because the flow
    divides by a learned scale and can amplify float differences in a trained model.

    Returns ``{"onnxruntime": version, "samples": {text_length: output_samples}}``,
    or ``{"skipped": reason}`` when onnxruntime is not installed.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        logger.warning("ONNX export not verified (%s)", exc)
        return {"skipped": str(exc)}

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    frame = model.samples_per_frame
    generator = torch.Generator().manual_seed(0)
    samples = {}
    for length in lengths:
        ids = torch.randint(1, n_vocab, (1, length), generator=generator)
        with torch.no_grad():
            expected = model(ids, torch.tensor([length])).numpy()
        produced = session.run(
            None, {"text": ids.numpy(), "text_lengths": np.array([length], dtype=np.int64)}
        )[0]

        if not np.isfinite(produced).all():
            raise RuntimeError(f"The exported graph produced NaN/Inf for a {length}-symbol text.")
        # A tie in the rounding of a duration can move the output by a frame.
        allowed = max(2 * frame, expected.shape[-1] // 100)
        if abs(produced.shape[-1] - expected.shape[-1]) > allowed:
            raise RuntimeError(
                f"The exported graph is not length dependent as the model is: for a "
                f"{length}-symbol text PyTorch produces {expected.shape[-1]} samples "
                f"and the ONNX graph {produced.shape[-1]}."
            )
        overlap = min(produced.shape[-1], expected.shape[-1])
        worst = float(np.abs(produced[..., :overlap] - expected[..., :overlap]).max()) if overlap else 0.0
        if worst > 0.05:
            logger.warning(
                "ONNX output differs from PyTorch by up to %.4f for a %d-symbol text", worst, length
            )
        samples[length] = int(produced.shape[-1])

    return {"onnxruntime": ort.__version__, "samples": samples}


class ModelExporter:
    """Export trained checkpoints to ONNX bundles without deploying them."""

    def __init__(self):
        """Record where exports go. Deliberately creates nothing.

        This class is instantiated at module scope in ``app.py``, so a ``mkdir``
        here ran at *import* time — the same failure ``VOICES_DIR.mkdir()`` had
        in qwen3-tts: PermissionError for any caller without write access to the
        working directory, and a hard stop on a read-only rootfs. The directory
        is created in ``export_to_onnx``, where it is used and where a failure
        can be reported against a request.
        """
        self.export_dir = Path("models")

    async def export_to_onnx(self, job_id: str) -> Path:
        """Export a training checkpoint to ONNX and write its companion config bundle."""

        checkpoint_path = Path(f"checkpoints/{job_id}/final_model.pt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        # Load checkpoint (off the event loop: it is a few hundred MB)
        checkpoint = await asyncio.to_thread(
            torch.load, checkpoint_path, map_location='cpu', weights_only=True
        )
        self._require_trained(checkpoint, job_id)

        # Create export directory
        export_path = self.export_dir / job_id
        export_path.mkdir(parents=True, exist_ok=True)

        # Load configuration from checkpoint (preferred) or fallback
        config = checkpoint.get('config') or self._load_config(job_id)
        logger.info(f"Using training config: hidden_channels={config.get('hidden_channels', 192)}")
        logger.info(f"Model architecture: {config.get('n_layers', 6)} layers")

        # The vocabulary the model was trained with. Resolved before the slow part
        # so a checkpoint without one fails in a second, not after the export.
        phoneme_id_map = self._resolve_phoneme_map(checkpoint, checkpoint_path, job_id, config)

        # Export to ONNX
        onnx_path = export_path / f"{job_id}.onnx"

        # Build the model and load its weights OUTSIDE the try below. That
        # handler relabels everything as an "ONNX export failure", which is the
        # wrong diagnosis for a checkpoint/architecture mismatch and would bury
        # the only message that says what to fix.
        from vits_model import VITS, VITSConfig

        vits_config = VITSConfig(**config)
        logger.info(f"VITS Config - hidden_channels: {vits_config.hidden_channels}")
        logger.info(f"VITS Config - inter_channels: {vits_config.inter_channels}")
        logger.info(f"VITS Config - n_layers: {vits_config.n_layers}")

        model = VITS(vits_config)

        # strict=False, but the result is inspected rather than discarded.
        #
        # Training and export build the same VITS class, so a MISSING key means
        # the checkpoint has no weights for that layer and the export would carry
        # its random initialisation instead — an ONNX that loads, runs, and emits
        # noise, reported as a successful export. That is the failure this module
        # already refuses elsewhere ("raise rather than write an invalid
        # placeholder ONNX").
        #
        # UNEXPECTED keys are the benign direction: the checkpoint carries more
        # than inference needs. Logged, not fatal.
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if result.unexpected_keys:
            logger.info(
                "Checkpoint has %d tensor(s) the inference model does not use "
                "(e.g. %s)", len(result.unexpected_keys), result.unexpected_keys[0]
            )
        if result.missing_keys:
            raise RuntimeError(
                f"Checkpoint is missing weights for {len(result.missing_keys)} "
                f"tensor(s), so the export would contain untrained random values "
                f"for them and produce noise. First few: "
                f"{result.missing_keys[:5]}. This usually means the training "
                f"config and the checkpoint disagree on the architecture."
            )
        model.eval()

        # The dummy input only fixes the ranks and dtypes for tracing; every length
        # is dynamic in the graph (see VITS._expand). Not 1: size-1 dimensions get
        # specialised by the tracer.
        n_vocab = vits_config.n_vocab
        dummy_length = 16
        dummy_text = torch.randint(1, n_vocab, (1, dummy_length), dtype=torch.long)
        dummy_text_lengths = torch.tensor([dummy_length], dtype=torch.long)

        try:
            # Off the event loop: tracing a full-size model takes seconds to
            # minutes, and /health must keep answering meanwhile.
            with warnings.catch_warnings():
                # The tracer warns about every tensor->int conversion it records,
                # and torch >= 2.9 warns that this exporter is legacy. Both are
                # known and pinned; the verification below is the actual check.
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    await asyncio.to_thread(
                        torch.onnx.export,
                        model,
                        (dummy_text, dummy_text_lengths),
                        str(onnx_path),
                        input_names=['text', 'text_lengths'],
                        output_names=['audio'],
                        dynamic_axes={
                            'text': {0: 'batch_size', 1: 'sequence'},
                            'text_lengths': {0: 'batch_size'},
                            'audio': {0: 'batch_size', 1: 'time'}
                        },
                        opset_version=ONNX_OPSET,
                        do_constant_folding=True,
                        verbose=False,
                        **onnx_export_kwargs(),
                    )

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            onnx_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"ONNX export failed with torch {torch.__version__}: {e}"
            ) from e

        # Refuse an artifact that does not behave like the model. A file that
        # loads but ignores its input is worse than no file: it gets deployed.
        try:
            verification = await asyncio.to_thread(verify_onnx_export, model, onnx_path, n_vocab)
        except Exception:
            onnx_path.unlink(missing_ok=True)
            raise

        # The voice the runtime phonemises with. Not a table of its own: training
        # phonemised with phonemization.espeak_voice(), and a second mapping here
        # exported pt-BR (trained as 'pt-br') as English.
        lang = config.get('language', 'en')
        phonemizer_lang = exported_voice(lang)

        # Save the vocabulary as a standalone file for debugging / manual inspection
        with open(export_path / "phonemes.json", 'w', encoding='utf-8') as f:
            json.dump(phoneme_id_map, f, indent=2, ensure_ascii=False)

        trainer_kind = checkpoint.get('trainer_kind', 'legacy-unknown')

        # Create Piper config file (includes phoneme vocab for custom inference)
        piper_config = {
            "audio": {
                "sample_rate": config.get('sample_rate', 22050),
                "quality": config.get('quality', 'medium')
            },
            "espeak": {
                "voice": phonemizer_lang
            },
            "inference": {
                "noise_scale": 0.667,
                "length_scale": 1.0,
                "noise_w": 0.8
            },
            "phonemizer_language": phonemizer_lang,
            "phoneme_id_map": phoneme_id_map,
            # Whether the ids are to be wrapped in <start>/<end>, as they were in
            # training. Models trained before this key existed were not.
            "phoneme_boundary_tokens": bool(config.get('boundary_tokens', False)),
            "limits": {
                # The positional encoding table; longer input cannot be encoded.
                "max_input_symbols": int(model.text_encoder.pos_encoding.pe.shape[1]),
            },
            "model_card": {
                "name": job_id,
                "language": lang,
                "dataset": "custom",
                "version": "1.0.0",
                "speaker": config.get('speaker_name', 'default'),
                # What produced these weights; see training_utils.TRAINER_CAVEAT.
                "trainer_kind": trainer_kind,
                "trainer_caveat": (
                    TRAINER_CAVEAT if trainer_kind == TRAINER_KIND
                    else "Trained before the trainer kind was recorded. " + TRAINER_CAVEAT
                ),
            },
            "export": {
                "torch": torch.__version__,
                "opset": ONNX_OPSET,
                "exporter": "torchscript-trace",
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "verification": verification,
            },
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

    @staticmethod
    def _require_trained(checkpoint: dict, job_id: str) -> None:
        """Refuse a final checkpoint that records zero optimizer steps."""
        steps = checkpoint.get('optimizer_steps')
        if steps is None:
            logger.warning(
                "Checkpoint of job %s predates step counting; cannot verify it was "
                "trained before exporting it.", job_id
            )
        elif int(steps) <= 0:
            raise RuntimeError(
                f"Job {job_id} recorded no optimizer step: its weights are the random "
                f"initialisation and would export as noise. Nothing was exported."
            )

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

    def _resolve_phoneme_map(self, checkpoint: dict, checkpoint_path: Path,
                             job_id: str, config: dict) -> dict:
        """The phoneme->id map the model was trained with, validated against its embedding.

        Read from the checkpoint (or ``vocab.json`` beside it), never rebuilt from
        a dataset that may have changed since. Only a checkpoint written before
        vocabularies were persisted falls back to the dataset, and that is logged.
        """
        rows = checkpoint['model_state_dict'].get('text_encoder.embedding.weight')
        n_vocab = int(rows.shape[0]) if rows is not None else config.get('n_vocab', 256)

        phoneme_map, source = vocab_for_checkpoint(checkpoint, checkpoint_path.parent, n_vocab)
        if phoneme_map is None:
            phoneme_map = self._legacy_phoneme_map(job_id, config)
            source = "the current dataset (legacy checkpoint)"
            logger.warning(
                "Checkpoint of job %s carries no vocabulary; rebuilt it from the "
                "dataset. If that dataset changed after training, the ids are wrong.", job_id
            )
        phoneme_map = validate_phoneme_id_map(phoneme_map, n_vocab)
        logger.info("Phoneme map from %s (%d symbols)", source, len(phoneme_map))
        return phoneme_map

    def _legacy_phoneme_map(self, job_id: str, config: dict) -> dict:
        """Rebuild the vocabulary of a checkpoint that predates persisted ones.

        Reads the split training used (``train.json``) of the dataset named after
        the trained model. Falls back to the job id, and to nothing else: it used
        to scan ``data/`` and take the first directory that had a ``train.json``,
        which is another voice's dataset whenever this one is gone, and exported
        that voice's alphabet with this voice's weights.
        """
        model_name = (config or {}).get('speaker_name')
        if model_name and not is_safe_name(model_name):
            # Read back from the checkpoint, so not trusted to still be a name:
            # data/<this> must stay a direct child of data/.
            logger.warning("Ignoring speaker_name %r from the checkpoint config: not a valid name", model_name)
            model_name = None

        candidates = []
        if model_name:
            candidates.append(Path("data") / model_name)
        candidates.append(Path(f"data/{job_id}"))

        for dataset_path in candidates:
            for name in ("train.json", "metadata.json"):
                source_path = dataset_path / name
                if source_path.exists():
                    with open(source_path, 'r', encoding='utf-8') as f:
                        entries = json.load(f)
                    logger.info(f"Phoneme map built from {source_path}")
                    return phoneme_id_map_from_entries(entries)

        raise RuntimeError(
            f"Cannot export job {job_id}: its checkpoint has no persisted vocabulary and "
            f"the dataset it was trained on is gone (looked for "
            f"{', '.join(str(c) for c in candidates)}). Without the phoneme map the "
            f"exported voice cannot be used; restore the dataset or retrain."
        )
