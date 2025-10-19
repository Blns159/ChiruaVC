"""
Audio processing utilities for Vietnamese voice conversion
"""

import torch
import torchaudio
import numpy as np
import librosa
from typing import Tuple, Optional


def load_audio(path: str, sample_rate: int = 22050) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if necessary.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
    Returns:
        Audio waveform and sample rate
    """
    waveform, sr = torchaudio.load(path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    return waveform, sample_rate


def compute_mel_spectrogram(audio: torch.Tensor, 
                            sample_rate: int = 22050,
                            n_fft: int = 1024,
                            hop_length: int = 256,
                            win_length: int = 1024,
                            n_mels: int = 80,
                            fmin: float = 0.0,
                            fmax: float = 8000.0) -> torch.Tensor:
    """
    Compute mel-spectrogram from audio waveform.
    
    Args:
        audio: Audio waveform [B, 1, T] or [1, T]
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
    Returns:
        Mel-spectrogram [B, n_mels, T']
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=1.0
    )
    
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)
    
    mel = mel_transform(audio.squeeze(1))
    
    # Convert to log scale
    mel = torch.log(torch.clamp(mel, min=1e-5))
    
    return mel


def extract_f0(audio: np.ndarray, 
               sample_rate: int = 22050,
               hop_length: int = 256,
               fmin: float = 50.0,
               fmax: float = 500.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fundamental frequency (F0) using PYIN algorithm.
    Important for Vietnamese tonal feature preservation.
    
    Args:
        audio: Audio waveform as numpy array
        sample_rate: Audio sample rate
        hop_length: Hop length for F0 extraction
        fmin: Minimum F0
        fmax: Maximum F0
    Returns:
        F0 contour and voiced/unvoiced flags
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop_length
    )
    
    # Replace NaN with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return f0, voiced_flag


def compute_prosody_features(audio: np.ndarray,
                             sample_rate: int = 22050,
                             hop_length: int = 256) -> np.ndarray:
    """
    Compute prosody features (energy, duration patterns).
    Important for Vietnamese speech naturalness.
    
    Args:
        audio: Audio waveform as numpy array
        sample_rate: Audio sample rate
        hop_length: Hop length
    Returns:
        Prosody features
    """
    # Compute energy
    energy = librosa.feature.rms(
        y=audio,
        hop_length=hop_length
    )[0]
    
    # Compute zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=hop_length
    )[0]
    
    # Stack features
    prosody = np.stack([energy, zcr], axis=0)
    
    return prosody


def normalize_vietnamese_tones(audio: torch.Tensor,
                               sample_rate: int = 22050) -> torch.Tensor:
    """
    Apply Vietnamese-specific tone normalization.
    Helps preserve tonal characteristics during conversion.
    
    Args:
        audio: Audio waveform [B, 1, T]
        sample_rate: Audio sample rate
    Returns:
        Normalized audio
    """
    # Simple normalization - can be extended with more sophisticated methods
    audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
    return audio


class AudioProcessor:
    """
    Audio processor for Vietnamese voice conversion.
    Handles all audio preprocessing and feature extraction.
    """
    def __init__(self, config: dict):
        self.config = config
        audio_config = config.get('audio', {})
        
        self.sample_rate = audio_config.get('sampling_rate', 22050)
        self.n_fft = audio_config.get('n_fft', 1024)
        self.hop_length = audio_config.get('hop_length', 256)
        self.win_length = audio_config.get('win_length', 1024)
        self.n_mels = audio_config.get('n_mels', 80)
        self.fmin = audio_config.get('fmin', 0.0)
        self.fmax = audio_config.get('fmax', 8000.0)
        
        vietnamese_config = config.get('vietnamese', {})
        self.use_tone_normalization = vietnamese_config.get('use_tone_normalization', True)
    
    def load_and_process(self, audio_path: str) -> dict:
        """
        Load and process audio file for voice conversion.
        
        Args:
            audio_path: Path to audio file
        Returns:
            Dictionary containing processed features
        """
        # Load audio
        audio, sr = load_audio(audio_path, self.sample_rate)
        
        # Vietnamese tone normalization
        if self.use_tone_normalization:
            audio = normalize_vietnamese_tones(audio, self.sample_rate)
        
        # Compute mel-spectrogram
        mel = compute_mel_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Extract F0 for Vietnamese tone preservation
        audio_np = audio.squeeze().numpy()
        f0, voiced_flag = extract_f0(
            audio_np,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Compute prosody features
        prosody = compute_prosody_features(
            audio_np,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return {
            'audio': audio,
            'mel': mel,
            'f0': torch.from_numpy(f0).float(),
            'voiced_mask': torch.from_numpy(voiced_flag).float(),
            'prosody': torch.from_numpy(prosody).float()
        }
