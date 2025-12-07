import torch
import soundfile as sf
from resemble_enhance.enhancer.inference import enhance, load_enhancer

enh = None

def enhance_audio(path):
    global enh
    device = torch.device("cpu")
    if enh is None:
        enh = load_enhancer(run_dir=None, device=device).to(device)
    w, sr = sf.read(path)
    if w.ndim == 2:
        w = w.mean(axis=1)
    w = torch.tensor(w, dtype=torch.float32, device=device)
    e = enhance(dwav=w, sr=sr, device=device, nfe=32, solver="midpoint", lambd=0.5, tau=0.5)
    if isinstance(e, tuple):
        e = e[0]
    o = e.detach().cpu().numpy()
    out = path.replace(".wav", "_enh.wav")
    sf.write(out, o, 48000)
    return out