import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="configs/freevc.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="checkpoints/freevc.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="convert.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="output/freevc", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            
            # ========== XỬ LÝ TGT (SPEAKER) ==========
            if hps.model.use_spk:
                # Kiểm tra xem tgt là file hay folder
                if os.path.isdir(tgt):
                    # TGT LÀ FOLDER → Dùng embed_speaker
                    print(f"Loading multiple reference audios from {tgt}...")
                    wav_tgt_list = []
                    # Lấy tất cả .wav files trong folder
                    for wav_file in sorted(os.listdir(tgt)):
                        if wav_file.endswith('.wav'):
                            wav_path = os.path.join(tgt, wav_file)
                            wav, _ = librosa.load(wav_path, sr=hps.data.sampling_rate)
                            wav, _ = librosa.effects.trim(wav, top_db=20)
                            wav_tgt_list.append(wav)
                    
                    if len(wav_tgt_list) == 0:
                        print(f"Warning: No .wav files found in {tgt}, skipping...")
                        continue
                    
                    print(f"Computing speaker embedding from {len(wav_tgt_list)} files...")
                    # Dùng embed_speaker để average embedding từ nhiều file
                    g_tgt = smodel.embed_speaker(wav_tgt_list)
                    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
                    
                elif os.path.isfile(tgt):
                    # TGT LÀ FILE → Dùng embed_utterance (như cũ)
                    print(f"Loading single reference audio from {tgt}...")
                    wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
                    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
                    g_tgt = smodel.embed_utterance(wav_tgt)
                    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
                else:
                    print(f"Error: {tgt} is neither file nor directory, skipping...")
                    continue
            else:
                # Không dùng speaker embedding (dùng mel)
                if os.path.isdir(tgt):
                    # Lấy file đầu tiên trong folder
                    wav_files = [f for f in os.listdir(tgt) if f.endswith('.wav')]
                    if len(wav_files) == 0:
                        print(f"Warning: No .wav files found in {tgt}, skipping...")
                        continue
                    tgt = os.path.join(tgt, wav_files[0])
                
                wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
                wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
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
            
            # ========== XỬ LÝ SRC (CONTENT) ==========
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            c = utils.get_content(cmodel, wav_src)
            
            # ========== INFERENCE ==========
            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            
            # ========== SAVE OUTPUT ==========
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(timestamp+"_"+title)), hps.data.sampling_rate, audio)
            else:
                write(os.path.join(args.outdir, f"{title}.wav"), hps.data.sampling_rate, audio)