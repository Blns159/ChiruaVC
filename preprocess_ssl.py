import os
import torch
import librosa
from glob import glob
from tqdm import tqdm

import utils
from wavlm import WavLM, WavLMConfig

# ==================== CONFIGURATION ====================
sr = 16000  # sampling rate
in_dir = "dataset/audio_vlsp"  # path to input dir (chứa SPEAKER_* folders)
out_dir = "dataset/wavlm"  # path to output dir
wavlm_checkpoint = 'wavlm/WavLM-Large.pt'  # WavLM checkpoint
# =======================================================


def process(filename):
    # Extract speaker ID từ đường dẫn
    # VD: dataset/audio_vlsp/SPEAKER_001/SPEAKER_001_042.wav
    speaker_dir = os.path.dirname(filename)
    speaker_id = os.path.basename(speaker_dir)
    
    # Tạo output directory
    save_dir = os.path.join(out_dir, speaker_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load và process audio
    wav, _ = librosa.load(filename, sr=sr)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav)
    
    # Save với tên file giống input
    basename = os.path.basename(filename)
    save_name = os.path.join(save_dir, basename.replace(".wav", ".pt"))
    torch.save(c.cpu(), save_name)


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)

    print("Configuration:")
    print(f"  Input directory: {in_dir}")
    print(f"  Output directory: {out_dir}")
    print(f"  Sample rate: {sr}")
    print(f"  WavLM checkpoint: {wavlm_checkpoint}")
    print()

    print("Loading WavLM for content...")
    checkpoint = torch.load(wavlm_checkpoint)
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")
    print()
    
    # Tìm tất cả wav files trong các thư mục SPEAKER_*
    filenames = glob(f'{in_dir}/SPEAKER_*/*.wav', recursive=False)
    
    print(f"Found {len(filenames)} audio files to process")
    print()
    
    for filename in tqdm(filenames, desc="Processing"):
        process(filename)
    