"""
FreeVC Loss Functions Optimized for Vietnamese Voice Conversion

This module implements optimized loss functions for Vietnamese voice conversion,
including standard losses and Vietnamese-specific components for tonal preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss between generated and target audio.
    Higher weight for Vietnamese to preserve tonal characteristics.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: Generated audio features [B, C, T]
            target: Target audio features [B, C, T]
        Returns:
            Reconstruction loss value
        """
        return F.l1_loss(generated, target)


class MelSpectrogramLoss(nn.Module):
    """
    Mel-spectrogram loss for perceptual quality.
    Critical for Vietnamese tonal feature preservation.
    """
    def __init__(self, sample_rate: int = 22050, n_fft: int = 1024,
                 hop_length: int = 256, n_mels: int = 80):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: Generated mel-spectrogram [B, n_mels, T]
            target: Target mel-spectrogram [B, n_mels, T]
        Returns:
            Mel-spectrogram loss value
        """
        return F.l1_loss(generated, target)


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for intermediate discriminator features.
    Enhanced for Vietnamese phonetic feature preservation.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, real_features: List[torch.Tensor], 
                fake_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            real_features: List of real feature maps from discriminator
            fake_features: List of fake feature maps from discriminator
        Returns:
            Feature matching loss value
        """
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss / len(real_features)


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for generator and discriminator.
    """
    def __init__(self, loss_type: str = "hinge"):
        super().__init__()
        self.loss_type = loss_type
    
    def generator_loss(self, fake_scores: List[torch.Tensor]) -> torch.Tensor:
        """
        Generator adversarial loss.
        Args:
            fake_scores: List of discriminator scores for fake samples
        Returns:
            Generator loss value
        """
        loss = 0.0
        for fake_score in fake_scores:
            if self.loss_type == "hinge":
                loss += -fake_score.mean()
            elif self.loss_type == "mse":
                loss += F.mse_loss(fake_score, torch.ones_like(fake_score))
        return loss / len(fake_scores)
    
    def discriminator_loss(self, real_scores: List[torch.Tensor], 
                          fake_scores: List[torch.Tensor]) -> torch.Tensor:
        """
        Discriminator adversarial loss.
        Args:
            real_scores: List of discriminator scores for real samples
            fake_scores: List of discriminator scores for fake samples
        Returns:
            Discriminator loss value
        """
        loss = 0.0
        for real_score, fake_score in zip(real_scores, fake_scores):
            if self.loss_type == "hinge":
                loss += F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
            elif self.loss_type == "mse":
                loss += F.mse_loss(real_score, torch.ones_like(real_score)) + \
                       F.mse_loss(fake_score, torch.zeros_like(fake_score))
        return loss / len(real_scores)


class F0Loss(nn.Module):
    """
    Fundamental frequency (F0) loss for Vietnamese tonal preservation.
    This is crucial for maintaining Vietnamese tones during voice conversion.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, generated_f0: torch.Tensor, target_f0: torch.Tensor,
                voiced_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            generated_f0: Generated F0 contour [B, T]
            target_f0: Target F0 contour [B, T]
            voiced_mask: Mask for voiced regions [B, T]
        Returns:
            F0 loss value
        """
        if voiced_mask is not None:
            # Only compute loss on voiced regions
            generated_f0 = generated_f0 * voiced_mask
            target_f0 = target_f0 * voiced_mask
            loss = F.l1_loss(generated_f0, target_f0, reduction='sum')
            loss = loss / (voiced_mask.sum() + 1e-8)
        else:
            loss = F.l1_loss(generated_f0, target_f0)
        
        return loss


class ProsodyLoss(nn.Module):
    """
    Prosody loss for Vietnamese speech naturalness.
    Captures energy, duration, and rhythm patterns specific to Vietnamese.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, generated_prosody: torch.Tensor, 
                target_prosody: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated_prosody: Generated prosody features [B, D, T]
            target_prosody: Target prosody features [B, D, T]
        Returns:
            Prosody loss value
        """
        # Combine L1 and L2 losses for better prosody matching
        l1_loss = F.l1_loss(generated_prosody, target_prosody)
        l2_loss = F.mse_loss(generated_prosody, target_prosody)
        return 0.5 * l1_loss + 0.5 * l2_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function optimized for Vietnamese voice conversion.
    Integrates all loss components with Vietnamese-specific weights.
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # Standard losses
        self.reconstruction_loss = ReconstructionLoss()
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=config.get('audio', {}).get('sampling_rate', 22050),
            n_fft=config.get('audio', {}).get('n_fft', 1024),
            hop_length=config.get('audio', {}).get('hop_length', 256),
            n_mels=config.get('audio', {}).get('n_mels', 80)
        )
        self.feature_matching_loss = FeatureMatchingLoss()
        self.adversarial_loss = AdversarialLoss()
        
        # Vietnamese-specific losses
        self.f0_loss = F0Loss()
        self.prosody_loss = ProsodyLoss()
        
        # Loss weights from config
        loss_config = config.get('loss', {})
        self.reconstruction_weight = loss_config.get('reconstruction_weight', 45.0)
        self.mel_loss_weight = loss_config.get('mel_loss_weight', 50.0)
        self.feature_matching_weight = loss_config.get('feature_matching_weight', 3.0)
        self.adversarial_weight = loss_config.get('adversarial_weight', 1.0)
        self.f0_loss_weight = loss_config.get('f0_loss_weight', 15.0)
        self.prosody_loss_weight = loss_config.get('prosody_loss_weight', 5.0)
    
    def generator_loss(self, 
                       generated_audio: torch.Tensor,
                       target_audio: torch.Tensor,
                       generated_mel: torch.Tensor,
                       target_mel: torch.Tensor,
                       fake_scores: List[torch.Tensor],
                       real_features: List[torch.Tensor],
                       fake_features: List[torch.Tensor],
                       generated_f0: Optional[torch.Tensor] = None,
                       target_f0: Optional[torch.Tensor] = None,
                       voiced_mask: Optional[torch.Tensor] = None,
                       generated_prosody: Optional[torch.Tensor] = None,
                       target_prosody: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss with all components.
        
        Args:
            generated_audio: Generated audio waveform
            target_audio: Target audio waveform
            generated_mel: Generated mel-spectrogram
            target_mel: Target mel-spectrogram
            fake_scores: Discriminator scores for generated samples
            real_features: Discriminator features for real samples
            fake_features: Discriminator features for generated samples
            generated_f0: Generated F0 contour (optional)
            target_f0: Target F0 contour (optional)
            voiced_mask: Voiced regions mask (optional)
            generated_prosody: Generated prosody features (optional)
            target_prosody: Target prosody features (optional)
        
        Returns:
            Total loss and dictionary of individual loss components
        """
        losses = {}
        
        # Standard losses
        losses['reconstruction'] = self.reconstruction_loss(generated_audio, target_audio)
        losses['mel'] = self.mel_loss(generated_mel, target_mel)
        losses['feature_matching'] = self.feature_matching_loss(real_features, fake_features)
        losses['adversarial'] = self.adversarial_loss.generator_loss(fake_scores)
        
        # Vietnamese-specific losses
        if generated_f0 is not None and target_f0 is not None:
            losses['f0'] = self.f0_loss(generated_f0, target_f0, voiced_mask)
        else:
            losses['f0'] = torch.tensor(0.0, device=generated_audio.device)
        
        if generated_prosody is not None and target_prosody is not None:
            losses['prosody'] = self.prosody_loss(generated_prosody, target_prosody)
        else:
            losses['prosody'] = torch.tensor(0.0, device=generated_audio.device)
        
        # Weighted sum
        total_loss = (
            self.reconstruction_weight * losses['reconstruction'] +
            self.mel_loss_weight * losses['mel'] +
            self.feature_matching_weight * losses['feature_matching'] +
            self.adversarial_weight * losses['adversarial'] +
            self.f0_loss_weight * losses['f0'] +
            self.prosody_loss_weight * losses['prosody']
        )
        
        losses['total'] = total_loss
        return total_loss, losses
    
    def discriminator_loss(self,
                          real_scores: List[torch.Tensor],
                          fake_scores: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss.
        
        Args:
            real_scores: Discriminator scores for real samples
            fake_scores: Discriminator scores for fake samples
        
        Returns:
            Total loss and dictionary of loss components
        """
        losses = {}
        losses['discriminator'] = self.adversarial_loss.discriminator_loss(real_scores, fake_scores)
        losses['total'] = losses['discriminator']
        
        return losses['total'], losses
