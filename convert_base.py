import os
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from hyperpyyaml import load_hyperpyyaml
from ECAPA_TDNN.utils import encode_batch_vn
import logging

logging.getLogger('numba').setLevel(logging.WARNING)


if __name__ == "__main__":
    hpfile = "ECAPA_TDNN/config.json"
    ptfile = "ECAPA_TDNN/G_280000.pth"
    txtpath = "convert.txt"
    outdir = "./"
    use_timestamp = False
    
    os.makedirs(outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        with open('ECAPA_TDNN/hyperparams.yaml', 'r', encoding='utf-8') as fin:
            params = load_hyperpyyaml(fin)
        ecapa = params['embedding_model']
        ecapa.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ecapa.to(device)
        ckpt = torch.load('ECAPA_TDNN/embedding_model.ckpt', map_location=device)
        ecapa.load_state_dict(ckpt, strict=False)


    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(txtpath, "r") as f:
        for rawline in f.readlines():
            parts = rawline.strip().split("|")
            if len(parts) < 3: continue
            titles.append(parts[0])
            srcs.append(parts[1])
            tgts.append(parts[2])

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts), total=len(titles)):
            title, src, tgt = line
            
            # Load Target Audio
            wav_tgt_np, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt_np, _ = librosa.effects.trim(wav_tgt_np, top_db=20)

            if hps.model.use_spk:
                g_tgt = encode_batch_vn(
                    torch.from_numpy(wav_tgt_np).unsqueeze(0), 
                    ecapa, 
                    device, 
                    params
                )
                g_tgt = g_tgt.squeeze(1).cuda()
            else:
                wav_tgt = torch.from_numpy(wav_tgt_np).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt, 
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
            
            # Load Source Audio
            wav_src_np, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src_np).unsqueeze(0).cuda()
            
            # Extract Content
            c = utils.get_content(cmodel, wav_src)
            
            # Inference
            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            
            audio = audio[0][0].data.cpu().float().numpy()
            
            # Save output
            if use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                out_filename = f"{timestamp}_{title}.wav"
            else:
                out_filename = f"{title}.wav"
                
            write(os.path.join(outdir, out_filename), hps.data.sampling_rate, audio)