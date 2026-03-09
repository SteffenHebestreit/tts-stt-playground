import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


def monotonic_align(attn_map: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    """
    Monotonic alignment via dynamic programming.
    Replaces the external monotonic_align dependency.
    """
    b, t_x, t_y = attn_map.shape
    device = attn_map.device

    alignment = torch.zeros_like(attn_map)

    for i in range(b):
        x_len = x_mask[i].sum().item()
        y_len = y_mask[i].sum().item()

        if x_len == 0 or y_len == 0:
            continue

        attn_slice = attn_map[i, :x_len, :y_len]

        path = torch.zeros(x_len, dtype=torch.long, device=device)
        for j in range(x_len):
            if j == 0:
                path[j] = 0
            else:
                min_pos = path[j-1]
                max_pos = min(y_len - 1, min_pos + 3)
                if max_pos > min_pos:
                    path[j] = min_pos + torch.argmax(attn_slice[j, min_pos:max_pos+1])
                else:
                    path[j] = min_pos

        for j, k in enumerate(path):
            if j < t_x and k < t_y:
                alignment[i, j, k] = 1.0

    return alignment


class VITS(nn.Module):
    """VITS model for TTS training."""

    def __init__(self, config):
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

        self.vocoder = HiFiGANGenerator(
            hidden_channels=config['hidden_channels'],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4]
        )

    def forward(self, text, text_lengths, mel_spec=None, mel_lengths=None):
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

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                audio = self.vocoder(z)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Vocoder GPU OOM — falling back to CPU (z shape: {z.shape})")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    z_cpu = z.detach().cpu()
                    vocoder_device = next(self.vocoder.parameters()).device
                    if vocoder_device != torch.device('cpu'):
                        self.vocoder = self.vocoder.cpu()
                    with torch.no_grad():
                        audio_cpu = self.vocoder(z_cpu)
                        audio = audio_cpu.to(z.device)
                    if vocoder_device != torch.device('cpu'):
                        self.vocoder = self.vocoder.to(vocoder_device)
                    logger.info("CPU vocoder fallback completed")
                else:
                    raise

            return audio, log_duration, log_det
        else:
            # Inference mode — predict durations, expand, decode
            # Clamp log_duration to prevent exploding repeat counts (0.13–33 frames)
            duration = torch.exp(log_duration.clamp(-2.0, 3.5)) * text_mask
            expanded = self.expand_encodings(text_encoded, duration)
            expanded = expanded.transpose(1, 2)
            z = self.flow.inverse(expanded)
            audio = self.vocoder(z)
            return audio

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute reconstruction, duration, and KL training losses."""
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
            recon_loss = F.mse_loss(pred_audio[..., :min_len], audio[..., :min_len])
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

        duration_loss = F.mse_loss(log_duration, torch.log(duration_target + 1e-6))

        # Clamp log_det to prevent FP16 overflow → NaN
        kl_loss = -torch.mean(log_det.float().clamp(-100.0, 100.0))

        return {
            'reconstruction': recon_loss,
            'duration': duration_loss * 0.1,
            'kl': kl_loss * 0.01
        }

    def get_device(self):
        """Return the device the model parameters reside on."""
        return next(self.parameters()).device

    def expand_encodings(self, encodings, durations):
        """Repeat each encoder frame according to its predicted duration."""
        batch_size, seq_len, hidden_size = encodings.shape
        expanded = []

        for b in range(batch_size):
            expanded_seq = []
            for t in range(seq_len):
                # Cap at 50 frames per phoneme to prevent runaway expansion
                duration = max(1, min(50, int(durations[b, t].item())))
                expanded_seq.append(encodings[b, t:t+1].repeat(duration, 1))

            if expanded_seq:
                expanded.append(torch.cat(expanded_seq, dim=0))
            else:
                expanded.append(torch.zeros(1, hidden_size, device=encodings.device))

        max_len = max(e.shape[0] for e in expanded) if expanded else 1

        padded = []
        for e in expanded:
            pad_len = max_len - e.shape[0]
            if pad_len > 0:
                e = torch.cat([e, torch.zeros(pad_len, hidden_size, device=e.device)], dim=0)
            padded.append(e)

        if not padded:
            logger.error("expand_encodings: padded list is empty — returning None")
            return None
        return torch.stack(padded)


class TextEncoder(nn.Module):
    """Transformer-based text encoder."""

    def __init__(self, n_vocab, hidden_channels, filter_channels, n_heads, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, hidden_channels)
        self.pos_encoding = PositionalEncoding(hidden_channels)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_channels, filter_channels, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, text, text_lengths):
        x = self.embedding(text)
        x = self.pos_encoding(x)
        mask = torch.arange(text.shape[1], device=text.device)[None, :] < text_lengths[:, None]
        for layer in self.layers:
            x = layer(x, mask)
        return x, mask


class DurationPredictor(nn.Module):
    """Predicts log-scale phoneme durations from encoder output."""

    def __init__(self, hidden_channels, filter_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_channels, filter_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
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
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.flows = nn.ModuleList([
            CouplingLayer(channels, hidden_channels, kernel_size)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        log_det_total = 0
        for flow in self.flows:
            x, log_det = flow(x, mask)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z, mask=None):
        for flow in reversed(self.flows):
            z = flow.inverse(z, mask)
        return z


class CouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""

    def __init__(self, channels, hidden_channels, kernel_size):
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
                 upsample_rates, upsample_kernel_sizes):
        super().__init__()
        self.pre_conv = nn.Conv1d(hidden_channels, 512, 7, padding=3)
        self.upsample_layers = nn.ModuleList()
        self.resblock_layers = nn.ModuleList()

        channel_size = 512
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
        x = self.pre_conv(x)
        for i, (upsample, resblocks) in enumerate(zip(self.upsample_layers, self.resblock_layers)):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            res_outputs = [resblock(x) for resblock in resblocks]
            x = torch.stack(res_outputs, dim=0).mean(dim=0)
            if i % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        x = F.leaky_relu(x, 0.1)
        x = self.post_conv(x)
        return torch.tanh(x).squeeze(1)


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN."""

    def __init__(self, channels, kernel_size, dilations):
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
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VITSConfig:
    """Configuration for the VITS model."""

    def __init__(self, **kwargs):
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
        self.language = kwargs.get('language', 'en')
        self.speaker_name = kwargs.get('speaker_name', 'default')

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
