import torch
import torch.nn.functional as F

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
    print('embeddings', embeddings.shape )
    return F.normalize(embeddings, dim=-1)