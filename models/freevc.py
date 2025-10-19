"""
FreeVC Model Architectures for Vietnamese Voice Conversion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x + residual


class ContentEncoder(nn.Module):
    """
    Content encoder using pre-trained WavLM features.
    Projects WavLM features to a lower-dimensional space.
    """
    def __init__(self, hidden_channels: int = 768, out_channels: int = 256):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Project WavLM features
        self.pre_proj = nn.Conv1d(hidden_channels, out_channels, 1)
        
        # Residual blocks for feature refinement
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels, kernel_size=3, dilation=1),
            ResidualBlock(out_channels, kernel_size=3, dilation=3),
            ResidualBlock(out_channels, kernel_size=3, dilation=9),
        ])
        
        self.post_proj = nn.Conv1d(out_channels, out_channels, 1)
    
    def forward(self, wavlm_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavlm_features: WavLM features [B, T, C]
        Returns:
            Content features [B, C, T]
        """
        # Transpose to [B, C, T]
        x = wavlm_features.transpose(1, 2)
        
        # Project
        x = self.pre_proj(x)
        
        # Refine with residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.post_proj(x)
        return x


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder for extracting speaker embeddings.
    Uses temporal pooling to create fixed-size embeddings.
    """
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, 
                 embedding_dim: int = 256):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.ReLU()
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel-spectrogram [B, n_mels, T]
        Returns:
            Speaker embedding [B, embedding_dim]
        """
        x = mel
        for conv in self.conv_layers:
            x = self.activation(conv(x))
        
        # Temporal pooling
        x = self.pool(x).squeeze(-1)
        
        # Final projection
        x = self.fc(x)
        return x


class Generator(nn.Module):
    """
    Generator (Decoder) based on HiFi-GAN architecture.
    Converts content and speaker features to audio waveform.
    """
    def __init__(self, content_dim: int = 256, speaker_dim: int = 256,
                 hidden_channels: int = 512,
                 upsample_rates: List[int] = [8, 8, 2, 2],
                 upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
                 resblock_kernel_sizes: List[int] = [3, 7, 11],
                 resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Input projection
        self.input_proj = nn.Conv1d(content_dim + speaker_dim, hidden_channels, 7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    hidden_channels // (2 ** i),
                    hidden_channels // (2 ** (i + 1)),
                    kernel_size,
                    stride=rate,
                    padding=(kernel_size - rate) // 2
                )
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hidden_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResidualBlock(ch, k, d[0]))
        
        # Output projection
        self.output_proj = nn.Conv1d(
            hidden_channels // (2 ** len(upsample_rates)),
            1,
            7,
            padding=3
        )
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, content: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: Content features [B, C, T]
            speaker_emb: Speaker embedding [B, D]
        Returns:
            Generated audio waveform [B, 1, T']
        """
        # Expand speaker embedding to match content length
        speaker_emb = speaker_emb.unsqueeze(-1).expand(-1, -1, content.size(-1))
        
        # Concatenate content and speaker
        x = torch.cat([content, speaker_emb], dim=1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Upsample and apply residual blocks
        for i, up in enumerate(self.ups):
            x = self.activation(x)
            x = up(x)
            
            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        # Output projection
        x = self.activation(x)
        x = self.output_proj(x)
        x = torch.tanh(x)
        
        return x


class PeriodDiscriminator(nn.Module):
    """Discriminator that processes audio with a specific period"""
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Audio waveform [B, 1, T]
        Returns:
            Score and list of intermediate features
        """
        features = []
        
        # Reshape to 2D with period
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = self.activation(conv(x))
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator.
    Uses multiple discriminators with different periods.
    """
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])
    
    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> Tuple[
        List[torch.Tensor], List[torch.Tensor],
        List[List[torch.Tensor]], List[List[torch.Tensor]]
    ]:
        """
        Args:
            real: Real audio waveform [B, 1, T]
            fake: Fake audio waveform [B, 1, T]
        Returns:
            Real scores, fake scores, real features, fake features
        """
        real_scores = []
        fake_scores = []
        real_features = []
        fake_features = []
        
        for disc in self.discriminators:
            real_score, real_feat = disc(real)
            fake_score, fake_feat = disc(fake)
            
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            real_features.append(real_feat)
            fake_features.append(fake_feat)
        
        return real_scores, fake_scores, real_features, fake_features


class FreeVC(nn.Module):
    """
    Complete FreeVC model for Vietnamese voice conversion.
    """
    def __init__(self, config: dict):
        super().__init__()
        
        model_config = config.get('model', {})
        encoder_config = model_config.get('encoder', {})
        decoder_config = model_config.get('decoder', {})
        speaker_config = model_config.get('speaker_encoder', {})
        
        # Content encoder
        self.content_encoder = ContentEncoder(
            hidden_channels=encoder_config.get('hidden_channels', 768),
            out_channels=encoder_config.get('out_channels', 256)
        )
        
        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            input_dim=config.get('audio', {}).get('n_mels', 80),
            hidden_dim=speaker_config.get('hidden_dim', 256),
            embedding_dim=speaker_config.get('embedding_dim', 256)
        )
        
        # Generator
        self.generator = Generator(
            content_dim=encoder_config.get('out_channels', 256),
            speaker_dim=speaker_config.get('embedding_dim', 256),
            hidden_channels=decoder_config.get('hidden_channels', 512),
            upsample_rates=decoder_config.get('upsample_rates', [8, 8, 2, 2]),
            upsample_kernel_sizes=decoder_config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
            resblock_kernel_sizes=decoder_config.get('resblock_kernel_sizes', [3, 7, 11]),
            resblock_dilation_sizes=decoder_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        )
    
    def forward(self, wavlm_features: torch.Tensor, 
                target_speaker_mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavlm_features: WavLM features from source audio [B, T, C]
            target_speaker_mel: Mel-spectrogram from target speaker [B, n_mels, T']
        Returns:
            Converted audio waveform [B, 1, T'']
        """
        # Extract content
        content = self.content_encoder(wavlm_features)
        
        # Extract target speaker embedding
        speaker_emb = self.speaker_encoder(target_speaker_mel)
        
        # Generate converted audio
        converted_audio = self.generator(content, speaker_emb)
        
        return converted_audio
