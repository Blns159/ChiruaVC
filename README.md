# ChiruaVC - Vietnamese Voice Conversion with FreeVC

Advanced voice conversion system for Vietnamese language based on FreeVC architecture with optimized loss functions for Vietnamese tonal preservation.

## Features

- **Vietnamese-Optimized Loss Functions**: Enhanced loss weights specifically tuned for Vietnamese tonal characteristics
- **F0 Loss**: Fundamental frequency loss for accurate Vietnamese tone preservation (6 tones)
- **Prosody Loss**: Captures Vietnamese speech rhythm and naturalness
- **Enhanced Mel-Spectrogram Loss**: Higher weight for better acoustic quality in Vietnamese
- **Feature Matching Loss**: Improved phonetic feature preservation
- **Multi-Period Discriminator**: For high-quality audio generation

## Loss Function Optimization for Vietnamese

### Key Optimizations

1. **Reconstruction Loss Weight**: 45.0 (increased from standard 30.0)
   - Higher weight ensures better preservation of original content
   
2. **Mel-Spectrogram Loss Weight**: 50.0 (increased from standard 45.0)
   - Critical for Vietnamese tonal feature accuracy
   
3. **Feature Matching Loss Weight**: 3.0 (increased from standard 2.0)
   - Better phonetic matching for Vietnamese sounds
   
4. **F0 Loss Weight**: 15.0 (NEW - Vietnamese-specific)
   - Preserves fundamental frequency patterns
   - Essential for maintaining Vietnamese tones during conversion
   
5. **Prosody Loss Weight**: 5.0 (NEW - Vietnamese-specific)
   - Captures energy and rhythm patterns
   - Ensures natural Vietnamese speech patterns

### Why These Optimizations Matter for Vietnamese

Vietnamese is a tonal language with 6 distinct tones that change word meanings:
- Level tone (ngang)
- Rising tone (sắc)
- Falling tone (huyền)
- Question tone (hỏi)
- Tumbling tone (ngã)
- Heavy tone (nặng)

Our optimized loss functions ensure these tonal characteristics are preserved during voice conversion, which is critical for intelligibility and naturalness in Vietnamese.

## Architecture

```
Source Audio → WavLM Features → Content Encoder
                                      ↓
                                  Generator → Converted Audio
                                      ↑
Target Speaker → Mel-Spectrogram → Speaker Encoder
```

### Components

1. **Content Encoder**: Extracts linguistic content from WavLM features
2. **Speaker Encoder**: Extracts speaker identity from mel-spectrogram
3. **Generator**: HiFi-GAN-based decoder with residual blocks
4. **Multi-Period Discriminator**: Ensures high-quality audio generation

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to adjust model and training parameters:

```yaml
# Vietnamese-specific settings
vietnamese:
  use_tone_normalization: true
  preserve_tones: true
  num_tones: 6

# Optimized loss weights
loss:
  reconstruction_weight: 45.0
  mel_loss_weight: 50.0
  feature_matching_weight: 3.0
  f0_loss_weight: 15.0
  prosody_loss_weight: 5.0
```

## Usage

### Training

```bash
python scripts/train.py \
    --config config.yaml \
    --checkpoint_dir checkpoints \
    --log_dir logs
```

### Inference (Voice Conversion)

```bash
python scripts/inference.py \
    --config config.yaml \
    --checkpoint checkpoints/model.pth \
    --source source_audio.wav \
    --target target_speaker_reference.wav \
    --output converted.wav
```

## Project Structure

```
ChiruaVC/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── models/
│   ├── __init__.py
│   ├── freevc.py           # FreeVC model architecture
│   └── loss.py             # Optimized loss functions
├── utils/
│   ├── __init__.py
│   └── audio.py            # Audio processing utilities
└── scripts/
    ├── train.py            # Training script
    └── inference.py        # Inference script
```

## Technical Details

### Loss Function Implementation

The `CombinedLoss` class in `models/loss.py` implements:

- **ReconstructionLoss**: L1 loss between generated and target audio
- **MelSpectrogramLoss**: L1 loss in mel-spectrogram domain
- **FeatureMatchingLoss**: Matches intermediate discriminator features
- **AdversarialLoss**: Hinge loss for GAN training
- **F0Loss**: L1 loss for fundamental frequency (Vietnamese tones)
- **ProsodyLoss**: Combined L1/L2 loss for prosody features

### Vietnamese-Specific Processing

- **Tone Normalization**: Pre-processing to normalize Vietnamese tonal patterns
- **F0 Extraction**: PYIN algorithm for accurate pitch tracking
- **Prosody Features**: Energy and zero-crossing rate for rhythm patterns
- **Voiced/Unvoiced Detection**: Masks for F0 loss computation

## Testing Voice Conversion

To test the voice conversion system:

1. **Prepare Data**:
   - Source audio: Vietnamese speech you want to convert
   - Target audio: Reference audio from target speaker
   
2. **Run Inference**:
   ```bash
   python scripts/inference.py \
       --source source.wav \
       --target target_reference.wav \
       --output converted.wav
   ```

3. **Evaluate**:
   - Check if Vietnamese tones are preserved
   - Verify speaker similarity to target
   - Assess naturalness of converted speech

## Key Benefits for Vietnamese

✓ **Tonal Accuracy**: F0 loss ensures correct tone reproduction
✓ **Natural Prosody**: Prosody loss maintains Vietnamese rhythm
✓ **High Quality**: Enhanced mel-spectrogram loss for better acoustics
✓ **Speaker Similarity**: Strong feature matching for target speaker characteristics
✓ **Linguistic Content**: WavLM-based content encoding preserves meaning

## Citation

This implementation is based on FreeVC architecture with optimizations for Vietnamese language characteristics.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
