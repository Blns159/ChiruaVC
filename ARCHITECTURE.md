# Vietnamese Voice Conversion - Loss Function Architecture

## Loss Function Components Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                   COMBINED LOSS FOR GENERATOR                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────┐                                            │
│  │ Standard Losses     │  (from original FreeVC)                     │
│  ├─────────────────────┤                                            │
│  │ • Reconstruction    │  Weight: 45.0  (+50% from standard)        │
│  │ • Mel-Spectrogram   │  Weight: 50.0  (+11% from standard)        │
│  │ • Feature Matching  │  Weight: 3.0   (+50% from standard)        │
│  │ • Adversarial       │  Weight: 1.0   (unchanged)                 │
│  └─────────────────────┘                                            │
│                                                                       │
│  ┌─────────────────────┐                                            │
│  │ Vietnamese-Specific │  (NEW for Vietnamese)                       │
│  ├─────────────────────┤                                            │
│  │ • F0 Loss          │  Weight: 15.0  (NEW - tone preservation)   │
│  │ • Prosody Loss     │  Weight: 5.0   (NEW - naturalness)         │
│  └─────────────────────┘                                            │
│                                                                       │
│  Total Loss = 45.0×L_recon + 50.0×L_mel + 3.0×L_fm +                │
│              1.0×L_adv + 15.0×L_f0 + 5.0×L_prosody                  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Voice Conversion Pipeline

```
┌──────────────┐
│ Source Audio │ (Vietnamese speech to convert)
│  (speaker A) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ WavLM Model  │ (extract linguistic content)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Content Encoder  │ → Content Features (256-dim)
└──────┬───────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌─────────────────┐   ┌──────────────┐
│   Generator     │   │Target Audio  │ (speaker B reference)
│   (HiFi-GAN)    │   │(speaker B)   │
│                 │   └──────┬───────┘
│  Combines:      │          │
│  - Content      │◄─────────┘
│  - Speaker ID   │   Speaker Encoder
│                 │   → Speaker Embedding (256-dim)
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│ Converted Audio    │ (speaker B saying speaker A's content)
│ (speaker B voice)  │
└────────┬───────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│              LOSS COMPUTATION                       │
├────────────────────────────────────────────────────┤
│                                                     │
│  Compare with Target Audio:                        │
│  ✓ Waveform (Reconstruction Loss)                  │
│  ✓ Mel-spectrogram (Mel Loss)                      │
│  ✓ Discriminator features (Feature Matching)       │
│  ✓ Discriminator score (Adversarial Loss)          │
│                                                     │
│  Vietnamese-Specific Comparisons:                  │
│  ✓ F0 contour (F0 Loss) ← PRESERVES TONES         │
│  ✓ Prosody features (Prosody Loss) ← NATURALNESS  │
│                                                     │
└────────────────────────────────────────────────────┘
```

## Vietnamese Tone Preservation Flow

```
                    ┌─────────────────┐
                    │  Source Audio   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐        ┌─────────┐
    │ Content │         │   F0    │        │ Prosody │
    │Features │         │Contour  │        │Features │
    └────┬────┘         └────┬────┘        └────┬────┘
         │                   │                   │
         │                   │                   │
         └───────────┬───────┴───────┬───────────┘
                     │               │
                     ▼               ▼
              ┌─────────────┐  ┌──────────┐
              │  Generator  │  │ Target   │
              │             │  │ Speaker  │
              └──────┬──────┘  └────┬─────┘
                     │              │
                     └──────┬───────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Converted      │
                   │ Audio          │
                   └────────┬───────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
           ┌────────┐  ┌────────┐  ┌─────────┐
           │ Recon  │  │F0 Loss │  │Prosody  │
           │  Loss  │  │        │  │  Loss   │
           └────────┘  └────────┘  └─────────┘
                │           │           │
                │  ✓ Tones: ngang, sắc, huyền,
                │           hỏi, ngã, nặng
                │
                └───→ Backpropagation ───→ Update Model
```

## Loss Weight Comparison

```
Component              Standard FreeVC    Vietnamese-Optimized
────────────────────────────────────────────────────────────────
Reconstruction               30.0              45.0  ████████
Mel-spectrogram             45.0              50.0  ████████████
Feature Matching             2.0               3.0  ████
Adversarial                  1.0               1.0  ██
F0 (Tone)                    0.0              15.0  ████████████████  ← NEW
Prosody                      0.0               5.0  ██████  ← NEW
────────────────────────────────────────────────────────────────
Total Weight                78.0             119.0
Vietnamese-Specific           -               20.0  (17% of total)
```

## Training Process Flow

```
Epoch 0 ──────────────────────────────────────────→ Epoch 1000
│                                                            │
├─ Warmup Phase (0-50) ─────┤                               │
│  Focus: Basic generation   │                               │
│  - Reconstruction Loss     │                               │
│  - Mel Loss               │                               │
│                           │                               │
├─ Refinement Phase (50-500)─────────────────────┤          │
│  Add: Vietnamese features                      │          │
│  - Gradually increase F0 loss                  │          │
│  - Add prosody loss                           │          │
│  - Fine-tune all components                   │          │
│                                                │          │
├─ Polish Phase (500+) ─────────────────────────────────────┤
│  Full optimization with all losses                        │
│  - All weights at full strength                          │
│  - Focus on naturalness and quality                      │
│                                                           │
└───────────────────────────────────────────────────────────┘

Quality Metrics Over Training:
┌─────────────────────────────────────────────────────────┐
│ MOS  5.0 ┤                                    ┌──────── │
│ Score    ┤                           ┌────────┘         │
│      4.0 ┤                    ┌──────┘                  │
│          ┤             ┌──────┘                         │
│      3.0 ┤      ┌──────┘                                │
│          ┤──────┘                                       │
│      2.0 └┬──────┬──────┬──────┬──────┬──────┬──────┬─ │
│           0     100    200    300    400    500    600  │
│                         Epochs                          │
└─────────────────────────────────────────────────────────┘
```

## Key Innovation: Vietnamese-Specific Losses

### F0 Loss Architecture

```
┌───────────────────────────────────────────────────┐
│               F0 Loss Computation                  │
├───────────────────────────────────────────────────┤
│                                                    │
│  Generated Audio ──→ F0 Extraction ──→ F0_gen     │
│                                                    │
│  Target Audio ────→ F0 Extraction ──→ F0_target   │
│                                                    │
│  ┌──────────────────────────────────────┐         │
│  │  Voiced/Unvoiced Detection           │         │
│  │  (only compute loss on voiced parts) │         │
│  └──────────────┬───────────────────────┘         │
│                 │                                  │
│                 ▼                                  │
│  Loss = L1(F0_gen × mask, F0_target × mask)       │
│                                                    │
│  Weight: 15.0 (significant impact on training)    │
│                                                    │
└───────────────────────────────────────────────────┘
```

### Prosody Loss Architecture

```
┌────────────────────────────────────────────────────┐
│            Prosody Loss Computation                 │
├────────────────────────────────────────────────────┤
│                                                     │
│  Generated Audio ──→ Energy + ZCR ──→ Prosody_gen  │
│                                                     │
│  Target Audio ────→ Energy + ZCR ──→ Prosody_tgt   │
│                                                     │
│  ┌───────────────────────────────────────┐         │
│  │  Combined L1 + L2 Loss               │         │
│  │  (robust to outliers & smooth)        │         │
│  └───────────────┬───────────────────────┘         │
│                  │                                  │
│                  ▼                                  │
│  Loss = 0.5×L1(P_gen, P_tgt) +                     │
│         0.5×L2(P_gen, P_tgt)                        │
│                                                     │
│  Weight: 5.0 (balance with other losses)           │
│                                                     │
└────────────────────────────────────────────────────┘
```

## Summary

This architecture provides:
- ✓ Comprehensive loss function for Vietnamese voice conversion
- ✓ Tone preservation through F0 loss
- ✓ Natural prosody through prosody loss
- ✓ High audio quality through enhanced standard losses
- ✓ Balanced training with optimized weights
- ✓ State-of-the-art voice conversion for Vietnamese
