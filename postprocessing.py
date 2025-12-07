import os
import torch
import soundfile as sf
import librosa
from resemble_enhance.enhancer.inference import enhance, load_enhancer

indir = "TEMP"
outdir = "TEMP"
os.makedirs(outdir, exist_ok=True)

device = torch.device("cpu")
print("Using:", device)

enhancer = load_enhancer(run_dir=None, device=device)
enhancer.to(device)

for fname in os.listdir(indir):
    if not fname.lower().endswith(".wav"):
        continue

    inpath = os.path.join(indir, fname)
    outpath = os.path.join(outdir, fname)

    # --- Bước 1: load về mono 48kHz cho đúng mô hình ---
    wav, sr = librosa.load(inpath, sr=48000, mono=True)

    wav = torch.tensor(wav, dtype=torch.float32, device=device)

    # --- Bước 2: enhance ở 48kHz ---
    enhanced = enhance(
        dwav=wav,
        sr=48000,          # Bắt buộc 48k!
        device=device,
        nfe=32,
        solver="midpoint",
        lambd=0.5,
        tau=0.5
    )

    if isinstance(enhanced, tuple):
        enhanced = enhanced[0]

    enhanced = enhanced.detach().cpu().numpy()

    # --- Bước 3: Resample xuống 16kHz ---
    enhanced_16k = librosa.resample(enhanced, orig_sr=48000, target_sr=16000)

    sf.write(outpath, enhanced_16k, 16000)

    print("Saved:", outpath)
