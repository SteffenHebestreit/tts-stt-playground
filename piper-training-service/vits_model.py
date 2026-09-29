"""Experimental mel-flow-vocoder text-to-speech model (historically named "VITS").

This is NOT a VITS in the sense of the paper or of Piper. Training reconstructs
the waveform from the ground-truth mel through ``mel_projection -> flow ->
HiFi-GAN generator``, so the reconstruction loss never reaches the text encoder;
the text encoder is trained only by a duration regressor whose targets are a
synthetic uniform split (see ``TTSDataset._estimate_durations``). There is no
monotonic alignment search, no posterior encoder, no prior/KL term and no
discriminator. ``training_utils.TRAINER_KIND`` / ``TRAINER_CAVEAT`` carry the
same statement into job state, checkpoints and the exported model.

The class names are kept so existing checkpoints and imports keep working.

Inference (``forward`` without a mel) is written with tensor ops only, so the
ONNX export is length dependent: the output length follows the text length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Bounds on the number of output frames one token may occupy at inference.
# Same values the eager implementation used; ``forward`` clamps log-durations to
# roughly [0.13, 33] frames before they get here, so 50 is a backstop.
MIN_FRAMES_PER_TOKEN = 1
MAX_FRAMES_PER_TOKEN = 50


class VITS(nn.Module):
    """Mel-flow-vocoder model with a duration predictor (see the module docstring)."""

    def __init__(self, config):
        """Build the encoder, duration predictor, flow, and vocoder modules."""
        super().__init__()
        # Accept both dict and VITSConfig
        if isinstance(config, VITSConfig):
            config = config.to_dict()
        self.config = config

        self.text_encoder = TextEncoder(
            n_vocab=config.get('n_vocab', 256),
            hidden_channels=config['hidden_channels'],
            filter_channels=config['inter_channels'],
            n_heads=config.get('n_heads', 4),
            n_layers=config['n_layers']
        )

        self.duration_predictor = DurationPredictor(
            hidden_channels=config['hidden_channels'],
            filter_channels=config['inter_channels']
        )

        # Mel spectrogram projection: n_mels -> hidden_channels
        self.mel_projection = nn.Conv1d(
            config.get('n_mels', 80),
            config['hidden_channels'],
            1
        )

        self.flow = ResidualCouplingBlock(
            channels=config['hidden_channels'],
            hidden_channels=config['hidden_channels'],
            kernel_size=5,
            n_layers=4
        )

        upsample_rates = [8, 8, 2, 2]
        self.vocoder = HiFiGANGenerator(
            hidden_channels=config['hidden_channels'],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=[16, 16, 4, 4],
            initial_channels=config.get('vocoder_channels', 512),
        )
        # Samples produced per mel frame; the reconstruction loss needs it to
        # mask the padding of a batch.
        self.samples_per_frame = int(np.prod(upsample_rates))

    def forward(self, text, text_lengths, mel_spec=None, mel_lengths=None):
        """Run training-mode or inference-mode synthesis depending on mel inputs."""
        text_encoded, text_mask = self.text_encoder(text, text_lengths)
        log_duration = self.duration_predictor(text_encoded, text_mask)

        if mel_spec is not None:
            # Training mode — use ground-truth mel alignment
            expected_mels = self.config.get('n_mels', 80)
            if mel_spec.shape[1] != expected_mels:
                if mel_spec.shape[2] == expected_mels:
                    mel_spec = mel_spec.transpose(1, 2)

            mel_mask = None
            if mel_lengths is not None:
                max_mel_len = mel_spec.shape[2]
                mel_mask = torch.arange(max_mel_len, device=mel_spec.device)[None, :] < mel_lengths[:, None]

            mel_projected = self.mel_projection(mel_spec)
            flow_mask = mel_mask if mel_mask is not None else text_mask
            z, log_det = self.flow(mel_projected, flow_mask)

            # No empty_cache() before the vocoder: it ran on every step, forcing
            # the allocator to hand memory back to the driver and re-request it
            # for the next batch (a device-wide sync each time).
            #
            # An out-of-memory error is deliberately NOT handled here. It used to
            # be caught and the vocoder re-run on the CPU under no_grad, which
            # returns a waveform without a graph: the step then trained only the
            # duration head, silently, and moved the vocoder between devices on
            # every affected step. The training loop halves the batch on OOM
            # instead, and fails the job if that is not enough.
            audio = self.vocoder(z)

            return audio, log_duration, log_det
        else:
            # Inference mode — predict durations, expand, decode
            # Clamp log_duration to prevent exploding repeat counts (0.13–33 frames)
            duration = torch.exp(log_duration.clamp(-2.0, 3.5)) * text_mask
            frames = self.frames_per_token(duration, text_mask)
            expanded = self._expand(text_encoded, frames)
            # Zero the frames past each item's own length, as training does with
            # its mel mask; a no-op for the batch of one the exported graph serves.
            frame_mask = (
                torch.arange(expanded.shape[1], device=expanded.device)[None, :]
                < frames.sum(dim=1)[:, None]
            )
            z = self.flow.inverse(expanded.transpose(1, 2), frame_mask)
            audio = self.vocoder(z)
            return audio

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute reconstruction, duration, and log-determinant training losses.

        Padding is excluded from both the reconstruction and the duration mean.
        Before, a padded duration slot (prediction 0.0) was compared with
        ``log(0 + 1e-6)`` = -13.8, so every padded position added a constant
        ~190 to the squared error: a batch of unequal lengths reported a loss
        dominated by padding, and the real positions were down-weighted by the
        share of padding.
        """
        dev = self.get_device()

        text = batch['text']
        if not isinstance(text, torch.Tensor):
            text = torch.tensor(text, device=dev)

        text_lengths = batch['text_lengths']
        if not isinstance(text_lengths, torch.Tensor):
            text_lengths = torch.tensor(text_lengths, device=dev)

        mel_spec = batch.get('mel_spec')
        if mel_spec is not None and not isinstance(mel_spec, torch.Tensor):
            mel_spec = torch.tensor(mel_spec, device=dev)

        mel_lengths = batch.get('mel_lengths')
        if mel_lengths is not None and not isinstance(mel_lengths, torch.Tensor):
            mel_lengths = torch.tensor(mel_lengths, device=dev)

        audio = batch.get('audio')
        if audio is not None and not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, device=dev)

        pred_audio, log_duration, log_det = self(text, text_lengths, mel_spec, mel_lengths)

        if audio is not None:
            min_len = min(pred_audio.shape[-1], audio.shape[-1])
            sq_err = (pred_audio[..., :min_len] - audio[..., :min_len]) ** 2
            if mel_lengths is not None:
                sample_mask = (
                    torch.arange(min_len, device=sq_err.device)[None, :]
                    < (mel_lengths * self.samples_per_frame)[:, None]
                )
                recon_loss = (sq_err * sample_mask).sum() / sample_mask.sum().clamp(min=1)
            else:
                recon_loss = sq_err.mean()
        elif mel_spec is not None:
            recon_loss = F.l1_loss(
                pred_audio,
                mel_spec.squeeze(1) if len(mel_spec.shape) == 3 else mel_spec
            )
        else:
            recon_loss = torch.tensor(0.0, device=dev)

        duration_target = batch.get('duration_target')
        if duration_target is None:
            duration_target = torch.ones_like(log_duration)
        elif not isinstance(duration_target, torch.Tensor):
            duration_target = torch.tensor(duration_target, device=dev)

        if duration_target.shape != log_duration.shape:
            min_len = min(duration_target.shape[1], log_duration.shape[1])
            duration_target = duration_target[:, :min_len]
            log_duration = log_duration[:, :min_len]

        duration_mask = (
            torch.arange(log_duration.shape[1], device=log_duration.device)[None, :]
            < text_lengths[:, None]
        )
        duration_sq_err = (log_duration - torch.log(duration_target + 1e-6)) ** 2
        duration_loss = (
            (duration_sq_err * duration_mask).sum() / duration_mask.sum().clamp(min=1)
        )

        # Clamp log_det to prevent FP16 overflow → NaN. Note the clamp also makes
        # this term a constant for any realistic input (the summed log-determinant
        # is in the tens of thousands), so it contributes no gradient; it is kept
        # only so logged totals stay comparable with earlier runs.
        kl_loss = -torch.mean(log_det.float().clamp(-100.0, 100.0))

        return {
            'reconstruction': recon_loss,
            'duration': duration_loss * 0.1,
            'kl': kl_loss * 0.01
        }

    def get_device(self):
        """Return the device the model parameters reside on."""
        return next(self.parameters()).device

    @staticmethod
    def frames_per_token(durations, mask=None):
        """Integer number of output frames for each token, ``[B, T]`` int64.

        Rounded to the nearest frame. The eager code used ``int()``, which
        truncates and so lost half a frame per token on average: output biased
        short, and every token shorter than two frames counted as one. Masked
        (padding) positions get 0 frames; before, a padding slot was forced up to
        1 frame, so a shorter text in a padded batch was followed by silence.
        """
        frames = torch.floor(durations + 0.5).clamp(MIN_FRAMES_PER_TOKEN, MAX_FRAMES_PER_TOKEN).long()
        if mask is not None:
            frames = frames * mask.long()
        return frames

    @staticmethod
    def _expand(encodings, frames):
        """Repeat token ``t`` ``frames[b, t]`` times, as tensor ops only.

        The previous implementation looped over tokens and called ``.item()``.
        That is data dependent control flow: tracing froze the durations of the
        random dummy input into the graph (the same output length for every text,
        and nothing past the dummy's 100 tokens). Here the output frame ``p``
        belongs to the token whose cumulative end is the first one above ``p``,
        which is a comparison and a sum, so the output length stays a function of
        the input in eager mode, when traced and when exported.

        The comparison is ``[B, L, T]``. That is small for the sentence-sized
        input the exported graph serves; it is not meant for whole documents.
        """
        batch_size, seq_len, hidden_size = encodings.shape
        ends = torch.cumsum(frames, dim=1)
        total = ends[:, -1]
        out_len = torch.clamp(total.max(), min=1)
        position = torch.arange(out_len, device=encodings.device)
        token_index = (position[None, :, None] >= ends[:, None, :]).sum(dim=-1)
        token_index = token_index.clamp(max=seq_len - 1)
        gathered = torch.gather(
            encodings, 1, token_index.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        valid = position[None, :] < total[:, None]
        return gathered * valid.unsqueeze(-1).to(encodings.dtype)

    def expand_encodings(self, encodings, durations, mask=None):
        """Repeat each encoder frame according to its predicted duration.

        ``mask`` (``[B, T]`` bool, True for real tokens) keeps padding out of the
        output; without it every position counts as a token.
        """
        return self._expand(encodings, self.frames_per_token(durations, mask))


class TextEncoder(nn.Module):
    """Transformer-based text encoder."""

    def __init__(self, n_vocab, hidden_channels, filter_channels, n_heads, n_layers):
        """Create token embeddings, positional encoding, and transformer layers."""
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, hidden_channels)
        self.pos_encoding = PositionalEncoding(hidden_channels)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_channels, filter_channels, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, text, text_lengths):
        """Encode token IDs into contextual hidden states and a padding mask."""
        x = self.embedding(text)
        x = self.pos_encoding(x)
        mask = torch.arange(text.shape[1], device=text.device)[None, :] < text_lengths[:, None]
        for layer in self.layers:
            x = layer(x, mask)
        return x, mask


class DurationPredictor(nn.Module):
    """Predicts log-scale phoneme durations from encoder output."""

    def __init__(self, hidden_channels, filter_channels):
        """Construct the convolutional duration prediction stack."""
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_channels, filter_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        """Predict per-token log durations and apply the sequence mask."""
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.proj(x)
        x = x.squeeze(1) * mask.float()
        return x


class ResidualCouplingBlock(nn.Module):
    """Stack of normalizing-flow coupling layers."""

    def __init__(self, channels, hidden_channels, kernel_size, n_layers):
        """Create the sequence of affine coupling layers used by the flow."""
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.flows = nn.ModuleList([
            CouplingLayer(channels, hidden_channels, kernel_size)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        """Apply each coupling layer in forward mode and sum log determinants."""
        log_det_total = 0
        for flow in self.flows:
            x, log_det = flow(x, mask)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z, mask=None):
        """Run the flow in reverse to decode latent frames back to hidden states."""
        for flow in reversed(self.flows):
            z = flow.inverse(z, mask)
        return z


class CouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""

    def __init__(self, channels, hidden_channels, kernel_size):
        """Initialise the transformation network for one coupling step."""
        super().__init__()
        self.half_channels = channels // 2
        self.transform_net = nn.Sequential(
            nn.Conv1d(self.half_channels, hidden_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, self.half_channels * 2, 1)
        )

    def forward(self, x, mask=None):
        """Transform half the channels conditioned on the other half."""
        x0, x1 = x[:, :self.half_channels], x[:, self.half_channels:]
        params = self.transform_net(x0)
        shift = params[:, :self.half_channels]
        scale = torch.sigmoid(params[:, self.half_channels:])
        x1 = x1 * scale + shift
        if mask is not None:
            x1 = x1 * mask[:, None, :]
        log_det = torch.sum(torch.log(scale + 1e-6), dim=[1, 2])
        return torch.cat([x0, x1], dim=1), log_det

    def inverse(self, z, mask=None):
        """Undo the affine coupling transform during inference."""
        z0, z1 = z[:, :self.half_channels], z[:, self.half_channels:]
        params = self.transform_net(z0)
        shift = params[:, :self.half_channels]
        scale = torch.sigmoid(params[:, self.half_channels:])
        z1 = (z1 - shift) / (scale + 1e-6)
        if mask is not None:
            z1 = z1 * mask[:, None, :]
        return torch.cat([z0, z1], dim=1)


class HiFiGANGenerator(nn.Module):
    """Simplified HiFi-GAN vocoder generator."""

    def __init__(self, hidden_channels, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_kernel_sizes, initial_channels=512):
        """Build the upsampling stack and residual blocks for waveform generation."""
        super().__init__()
        self.pre_conv = nn.Conv1d(hidden_channels, initial_channels, 7, padding=3)
        self.upsample_layers = nn.ModuleList()
        self.resblock_layers = nn.ModuleList()

        channel_size = initial_channels
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsample_layers.append(
                nn.ConvTranspose1d(channel_size, channel_size // 2, k, u, padding=(k-u)//2)
            )
            channel_size = channel_size // 2
            resblocks = nn.ModuleList()
            for rk, rd in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                resblocks.append(ResBlock(channel_size, rk, rd))
            self.resblock_layers.append(resblocks)

        self.post_conv = nn.Conv1d(channel_size, 1, 7, padding=3)

    def forward(self, x):
        """Convert latent acoustic features into a time-domain waveform."""
        x = self.pre_conv(x)
        for i, (upsample, resblocks) in enumerate(zip(self.upsample_layers, self.resblock_layers)):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            res_outputs = [resblock(x) for resblock in resblocks]
            x = torch.stack(res_outputs, dim=0).mean(dim=0)
        x = F.leaky_relu(x, 0.1)
        x = self.post_conv(x)
        return torch.tanh(x).squeeze(1)


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN."""

    def __init__(self, channels, kernel_size, dilations):
        """Create the dilated convolution pairs used in one residual block."""
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            self.convs1.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                          padding=dilation*(kernel_size-1)//2)
            )
            self.convs2.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=(kernel_size-1)//2)
            )

    def forward(self, x):
        """Apply the residual convolution stack and return the updated activations."""
        for conv1, conv2 in zip(self.convs1, self.convs2):
            res = x
            x = F.leaky_relu(x, 0.1)
            x = conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = conv2(x)
            x = x + res
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with manual multi-head attention (ONNX-compatible)."""

    def __init__(self, hidden_channels, filter_channels, n_heads, dropout=0.1):
        """Initialise ONNX-friendly attention projections and feed-forward layers."""
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_channels // n_heads

        # Manual Q/K/V projections wrapped in a sub-module named 'self_attn'
        # for checkpoint compatibility
        self.self_attn = nn.Module()
        self.self_attn.in_proj_weight = nn.Parameter(torch.empty(3 * hidden_channels, hidden_channels))
        self.self_attn.in_proj_bias = nn.Parameter(torch.empty(3 * hidden_channels))
        self.self_attn.out_proj = nn.Linear(hidden_channels, hidden_channels)
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.zeros_(self.self_attn.in_proj_bias)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_channels, filter_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(filter_channels, hidden_channels)
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.attn_scale = self.head_dim ** -0.5

    def forward(self, x, mask):
        """Run self-attention and feed-forward sublayers with residual connections."""
        B, T, C = x.shape

        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        if mask is not None:
            attn_mask = (~mask).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        x2 = torch.matmul(attn, v)
        x2 = x2.transpose(1, 2).contiguous().view(B, T, C)
        x2 = self.self_attn.out_proj(x2)

        x = self.norm1(x + self.dropout(x2))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the transformer."""

    def __init__(self, d_model, max_len=5000):
        """Precompute sinusoidal encodings up to ``max_len`` positions."""
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encodings to an embedded token sequence."""
        return x + self.pe[:, :x.size(1)]


class VITSConfig:
    """Configuration for the VITS model."""

    def __init__(self, **kwargs):
        """Populate configuration fields with provided overrides or sane defaults."""
        self.sample_rate = kwargs.get('sample_rate', 22050)
        self.n_fft = kwargs.get('n_fft', 1024)
        self.hop_length = kwargs.get('hop_length', 256)
        self.win_length = kwargs.get('win_length', 1024)
        self.n_mels = kwargs.get('n_mels', 80)
        self.hidden_channels = kwargs.get('hidden_channels', 192)
        self.inter_channels = kwargs.get('inter_channels', 192)
        self.n_layers = kwargs.get('n_layers', 6)
        self.n_vocab = kwargs.get('n_vocab', 256)
        self.n_heads = kwargs.get('n_heads', 2)
        # Width of the first vocoder stage (halved at each upsampling stage). The
        # default is the architecture existing checkpoints were trained with; it
        # is a knob so tests can build a small model.
        self.vocoder_channels = kwargs.get('vocoder_channels', 512)
        self.language = kwargs.get('language', 'en')
        self.speaker_name = kwargs.get('speaker_name', 'default')

    def to_dict(self):
        """Return the configuration as a plain serialisable dictionary."""
        return {k: v for k, v in self.__dict__.items()}
