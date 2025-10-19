"""
Training script for FreeVC with optimized loss for Vietnamese voice conversion
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.freevc import FreeVC, MultiPeriodDiscriminator
from models.loss import CombinedLoss
from utils.audio import AudioProcessor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: torch.device) -> tuple:
    """
    Create FreeVC model and discriminator.
    
    Args:
        config: Configuration dictionary
        device: Device to create models on
    Returns:
        Tuple of (generator, discriminator)
    """
    # Create generator (FreeVC)
    generator = FreeVC(config).to(device)
    
    # Create discriminator
    disc_config = config.get('model', {}).get('discriminator', {})
    discriminator = MultiPeriodDiscriminator(
        periods=disc_config.get('periods', [2, 3, 5, 7, 11])
    ).to(device)
    
    return generator, discriminator


def create_optimizers(generator: nn.Module, 
                     discriminator: nn.Module,
                     config: dict) -> tuple:
    """
    Create optimizers for generator and discriminator.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        config: Configuration dictionary
    Returns:
        Tuple of (gen_optimizer, disc_optimizer)
    """
    train_config = config.get('training', {})
    lr = train_config.get('learning_rate', 0.0002)
    beta1 = train_config.get('adam_beta1', 0.8)
    beta2 = train_config.get('adam_beta2', 0.99)
    
    gen_optimizer = optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=(beta1, beta2)
    )
    
    disc_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(beta1, beta2)
    )
    
    return gen_optimizer, disc_optimizer


def train_step(generator: nn.Module,
              discriminator: nn.Module,
              criterion: CombinedLoss,
              gen_optimizer: optim.Optimizer,
              disc_optimizer: optim.Optimizer,
              batch: dict,
              device: torch.device,
              config: dict) -> tuple:
    """
    Single training step.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        criterion: Combined loss function
        gen_optimizer: Generator optimizer
        disc_optimizer: Discriminator optimizer
        batch: Batch of training data
        device: Device
    Returns:
        Tuple of (gen_loss_dict, disc_loss_dict)
    """
    # Move batch to device
    wavlm_features = batch['wavlm_features'].to(device)
    target_mel = batch['target_mel'].to(device)
    target_audio = batch['target_audio'].to(device)
    target_f0 = batch.get('target_f0', None)
    voiced_mask = batch.get('voiced_mask', None)
    target_prosody = batch.get('target_prosody', None)
    
    if target_f0 is not None:
        target_f0 = target_f0.to(device)
    if voiced_mask is not None:
        voiced_mask = voiced_mask.to(device)
    if target_prosody is not None:
        target_prosody = target_prosody.to(device)
    
    # Generate audio
    generated_audio = generator(wavlm_features, target_mel)
    
    # Compute mel-spectrogram of generated audio
    from utils.audio import compute_mel_spectrogram
    audio_config = config.get('audio', {})
    generated_mel = compute_mel_spectrogram(
        generated_audio,
        sample_rate=audio_config.get('sampling_rate', 22050),
        n_fft=audio_config.get('n_fft', 1024),
        hop_length=audio_config.get('hop_length', 256),
        win_length=audio_config.get('win_length', 1024),
        n_mels=audio_config.get('n_mels', 80),
        fmin=audio_config.get('fmin', 0.0),
        fmax=audio_config.get('fmax', 8000.0)
    )
    
    # Train discriminator
    disc_optimizer.zero_grad()
    
    with torch.no_grad():
        generated_audio_detach = generated_audio.detach()
    
    real_scores, fake_scores, real_features, fake_features = discriminator(
        target_audio, generated_audio_detach
    )
    
    disc_loss, disc_loss_dict = criterion.discriminator_loss(real_scores, fake_scores)
    disc_loss.backward()
    disc_optimizer.step()
    
    # Train generator
    gen_optimizer.zero_grad()
    
    # Get discriminator features for generated audio
    _, fake_scores, _, fake_features = discriminator(target_audio, generated_audio)
    
    # Compute generator loss
    gen_loss, gen_loss_dict = criterion.generator_loss(
        generated_audio=generated_audio,
        target_audio=target_audio,
        generated_mel=generated_mel,
        target_mel=target_mel,
        fake_scores=fake_scores,
        real_features=real_features,
        fake_features=fake_features,
        generated_f0=None,  # Would need to extract F0 from generated audio
        target_f0=target_f0,
        voiced_mask=voiced_mask,
        generated_prosody=None,  # Would need to extract prosody from generated audio
        target_prosody=target_prosody
    )
    
    gen_loss.backward()
    gen_optimizer.step()
    
    return gen_loss_dict, disc_loss_dict


def train_epoch(generator: nn.Module,
               discriminator: nn.Module,
               criterion: CombinedLoss,
               gen_optimizer: optim.Optimizer,
               disc_optimizer: optim.Optimizer,
               dataloader: DataLoader,
               device: torch.device,
               epoch: int,
               config: dict,
               writer: SummaryWriter,
               log_interval: int = 10) -> tuple:
    """
    Train for one epoch.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        criterion: Combined loss function
        gen_optimizer: Generator optimizer
        disc_optimizer: Discriminator optimizer
        dataloader: Training data loader
        device: Device
        epoch: Current epoch number
        writer: Tensorboard writer
        log_interval: Logging interval
    Returns:
        Tuple of average losses
    """
    generator.train()
    discriminator.train()
    
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for step, batch in enumerate(pbar):
        gen_loss_dict, disc_loss_dict = train_step(
            generator, discriminator, criterion,
            gen_optimizer, disc_optimizer,
            batch, device, config
        )
        
        gen_loss = gen_loss_dict['total'].item()
        disc_loss = disc_loss_dict['total'].item()
        
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss
        
        pbar.set_postfix({
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        })
        
        # Log to tensorboard
        if step % log_interval == 0:
            global_step = epoch * len(dataloader) + step
            
            # Log generator losses
            for key, value in gen_loss_dict.items():
                writer.add_scalar(f'train/gen_{key}', value.item(), global_step)
            
            # Log discriminator losses
            for key, value in disc_loss_dict.items():
                writer.add_scalar(f'train/disc_{key}', value.item(), global_step)
    
    avg_gen_loss = total_gen_loss / len(dataloader)
    avg_disc_loss = total_disc_loss / len(dataloader)
    
    return avg_gen_loss, avg_disc_loss


def main():
    parser = argparse.ArgumentParser(description='Train FreeVC for Vietnamese voice conversion')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create models
    print('Creating models...')
    generator, discriminator = create_model(config, device)
    
    # Create optimizers
    gen_optimizer, disc_optimizer = create_optimizers(generator, discriminator, config)
    
    # Create loss function
    criterion = CombinedLoss(config)
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    train_config = config.get('training', {})
    max_epochs = train_config.get('max_epochs', 1000)
    save_interval = train_config.get('save_interval', 10)
    log_interval = train_config.get('log_interval', 10)
    
    print('Starting training...')
    print(f'Max epochs: {max_epochs}')
    print(f'Optimized loss weights for Vietnamese:')
    print(f'  - Reconstruction: {criterion.reconstruction_weight}')
    print(f'  - Mel-spectrogram: {criterion.mel_loss_weight}')
    print(f'  - Feature matching: {criterion.feature_matching_weight}')
    print(f'  - F0 (tone preservation): {criterion.f0_loss_weight}')
    print(f'  - Prosody: {criterion.prosody_loss_weight}')
    
    # Note: In a real implementation, you would create a dataset and dataloader here
    # For now, this is a skeleton that demonstrates the training structure
    print('\nNote: This is a skeleton implementation.')
    print('To train the model, you need to:')
    print('1. Prepare Vietnamese voice conversion dataset')
    print('2. Implement dataset class that loads audio and extracts WavLM features')
    print('3. Create dataloader with the dataset')
    print('4. Run the training loop')
    
    writer.close()


if __name__ == '__main__':
    main()
