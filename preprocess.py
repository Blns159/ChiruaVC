import logging
import warnings
import os
import torch
import librosa
import numpy as np
from glob import glob
from tqdm import tqdm
import torchaudio
import torch.nn.functional as F
import utils
from wavlm import WavLM, WavLMConfig
from hyperpyyaml import load_hyperpyyaml
from random import shuffle

# ==================== CÀI ĐẶT CHUNG ====================
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.WARNING, force=True)
logging.getLogger('speechbrain').setLevel(logging.ERROR)
logging.getLogger('speechbrain.utils.checkpoints').setLevel(logging.ERROR)
logging.getLogger('speechbrain.utils.seed').setLevel(logging.ERROR)

SAMPLE_RATE = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IN_DIR = 'dataset/wav'
WAVLM_OUT_DIR = 'dataset/wavlm'
ECAPA_OUT_DIR = 'dataset/spk'
FILELIST_DIR = 'filelists'
TRAIN_LIST_PATH = os.path.join(FILELIST_DIR, 'train.txt')
VAL_LIST_PATH = os.path.join(FILELIST_DIR, 'val.txt')

WAVLM_CHECKPOINT = 'wavlm/WavLM-Large.pt'
ECAPA_CKPT = 'ECAPA_TDNN/embedding_model.ckpt'
ECAPA_HPARAMS = 'ECAPA_TDNN/hyperparams.yaml'

os.makedirs(WAVLM_OUT_DIR, exist_ok=True)
os.makedirs(ECAPA_OUT_DIR, exist_ok=True)
os.makedirs(FILELIST_DIR, exist_ok=True)

# ==================== WAVLM ====================
print("Đang xử lý WavLM...")

def process_wavlm(filename, model):
    speaker_dir = os.path.dirname(filename)
    speaker_id = os.path.basename(speaker_dir)
    save_dir = os.path.join(WAVLM_OUT_DIR, speaker_id)
    os.makedirs(save_dir, exist_ok=True)
    
    basename = os.path.basename(filename)
    save_name = os.path.join(save_dir, basename.replace(".wav", ".pt"))
    
    if os.path.exists(save_name):
        return
        
    wav, _ = librosa.load(filename, sr=SAMPLE_RATE)
    wav = torch.from_numpy(wav).unsqueeze(0).to(device)
    c = utils.get_content(model, wav)
    torch.save(c.cpu(), save_name)

checkpoint = torch.load(WAVLM_CHECKPOINT)
cfg = WavLMConfig(checkpoint['cfg'])
cmodel = WavLM(cfg).to(device)
cmodel.load_state_dict(checkpoint['model'])
cmodel.eval()

filenames = glob(f'{IN_DIR}/SPEAKER_*/*.wav', recursive=False)
for filename in tqdm(filenames, desc="WavLM"):
    process_wavlm(filename, cmodel)
print(f"[WavLM]: Đã xử lý và lưu tại {WAVLM_OUT_DIR}")
# ===============================================


# ==================== ECAPA-TDNN ====================
print("\nĐang xử lý ECAPA-TDNN...")
with open(ECAPA_HPARAMS, 'r', encoding='utf-8') as fin:
    params = load_hyperpyyaml(fin)
ecapa = params['embedding_model']
ecapa.eval()
ecapa.to(device)
ckpt = torch.load(ECAPA_CKPT, map_location=device)
ecapa.load_state_dict(ckpt, strict=False)
def encode_batch_vn(wavs, model, device, params):
    if len(wavs.shape) == 1:
        wavs = wavs.unsqueeze(0)
    wav_lens = torch.ones(wavs.shape[0], device=device)
    wavs = wavs.to(device).float()
    wav_lens = wav_lens.to(device)
    feats = params['compute_features'](wavs)
    feats = params['mean_var_norm'](feats, wav_lens)
    with torch.no_grad():
        embeddings = model(feats, wav_lens)
    return F.normalize(embeddings, dim=-1)
speaker_dirs = sorted(glob(f'{IN_DIR}/SPEAKER_*'))
for speaker_dir in tqdm(speaker_dirs, desc="ECAPA-TDNN"):
    speaker_id = os.path.basename(speaker_dir)
    npy_path = f'{ECAPA_OUT_DIR}/{speaker_id}.npy'
    
    if os.path.exists(npy_path):
        continue
    
    wav_files = glob(f'{speaker_dir}/*.wav') + glob(f'{speaker_dir}/*.mp3')
    if not wav_files:
        continue
    all_embeds = []
    for wav_path in wav_files:
        sig, sr = torchaudio.load(wav_path)
        if sig.dim() > 1:
            sig = sig[0]
        if sr != SAMPLE_RATE:
            sig = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(sig)
        embed = encode_batch_vn(sig, ecapa, device, params)
        all_embeds.append(embed.squeeze().cpu().numpy())
    if all_embeds:
        global_embed = np.mean(all_embeds, axis=0)
        global_embed = global_embed / np.linalg.norm(global_embed)
        np.save(npy_path, global_embed, allow_pickle=False)

print(f"[ECAPA]: Đã xử lý và lưu tại {ECAPA_OUT_DIR}")
# ====================================================


# ==================== SPLIT ====================
print("\nĐang tạo file train/val...")
train_files = []
val_files = []
for speaker_dir_path in tqdm(speaker_dirs, desc="Splitting"):
    speaker_id = os.path.basename(speaker_dir_path)
    wavs = [f for f in os.listdir(speaker_dir_path) if f.endswith((".wav", ".mp3"))]
    if not wavs:
        continue
    shuffle(wavs)
    num_wavs = len(wavs)
    val_size = max(1, int(num_wavs * 0.02))
    current_val = wavs[:val_size]
    current_train = wavs[val_size:]
    for fname in current_train:
        train_files.append(os.path.join(IN_DIR, speaker_id, fname))
    for fname in current_val:
        val_files.append(os.path.join(IN_DIR, speaker_id, fname))
shuffle(train_files)
shuffle(val_files)
with open(TRAIN_LIST_PATH, "w", encoding='utf-8') as f:
    for wavpath in train_files:
        f.write(wavpath + "\n")
with open(VAL_LIST_PATH, "w", encoding='utf-8') as f:
    for wavpath in val_files:
        f.write(wavpath + "\n")
print(f"[Split]: Đã tạo {TRAIN_LIST_PATH} và {VAL_LIST_PATH}")
print("Hoàn tất toàn bộ quá trình tiền xử lý!")
# ===============================================