import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

# Flow Container. Kết hợp nhiều flow (ResidualCouplingLayer + Flip)
# Vì sao lại cần nhiều flow?
# Giải thích: Mỗi flow giúp mô hình nắm bắt các mối quan hệ phi tuyến tính trong dữ liệu, từ đó cải thiện khả năng biểu diễn và tổng quát hóa của mô hình. Khi kết hợp nhiều flow, mô hình có thể học được các đặc trưng đa dạng và phức tạp hơn, giúp tăng cường hiệu suất trong các nhiệm vụ như sinh dữ liệu, chuyển đổi giọng nói, hoặc các ứng dụng khác liên quan đến xử lý tín hiệu.
class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x

# Input [B, in_channels, T]
#     ↓
# Pre-Conv: in_channels → hidden_channels (192)
#     ↓
# WaveNet: context modeling + conditioning
#     ↓
# Proj-Conv: hidden_channels → out_channels*2 (384)
#     ↓
# Split: [mean, logs] each [B, 192, T]
#     ↓
# Sampling: z = mean + noise * exp(logs)
#     ↓
# Output: z, mean, logs, mask
# NÀY LÀ BOTTLE NECK, TRÍCH THÔNG TIN TỪ WAVLM
class Encoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels # 1024
    self.out_channels = out_channels # 192
    self.hidden_channels = hidden_channels # 192
    self.kernel_size = kernel_size # 5
    self.dilation_rate = dilation_rate # 1
    self.n_layers = n_layers # 1616
    self.gin_channels = gin_channels # 256

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    # x_mask: [batch_size, 1, seq_len]. Ni là thêm chiều, này chắc là độ dài thật của câu
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    # x: [batch_size, hidden_channels, seq_len]
    # Với kernel_size=1, Conv1D hoạt động như:
    # output[b, c_out, t] = sum(weight[c_out, c_in] * input[b, c_in, t]) + bias[c_out]
    x = self.pre(x) * x_mask # Conv1d: in_channels -> hidden_channels


    # x: [batch_size, hidden_channels, seq_len]
    x = self.enc(x, x_mask, g=g)  # WaveNet layers

    # =======Variational Autoencoder (VAE)========
    # Vì sao lại tăng thêm số đặc trưng, rồi tách ra rồi lại gộp lại thành z. Vì sao không chỉ để 1 cái thôi rồi chuyển thành z
    # stats: [batch_size, out_channels*2, seq_len] [192*2, seq_len] rồi chia đôi hàng hấn ra
    stats = self.proj(x) * x_mask

    # m, logs: [batch_size, out_channels, seq_len]. Coi hấn như có shape [192, seq_len]
    m, logs = torch.split(stats, self.out_channels, dim=1)

    # z: [batch_size, out_channels, seq_len] - reparameterization trick
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

# DiscriminatorP (Period-based) Phân tích audio theo các khoảng thời gian định kỳ khác nhau
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# DiscriminatorS (Scale-based) Phân tích audio ở các mức độ khác nhau
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        
        
class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)
            
        return mel_slices
    
    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:,-partial_frames:]
        
        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:,s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
        
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            #embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        
        return embed


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    spec_channels, # Chiều là n_fft/2+1 = 641
    segment_size, # 8960 (segment_size ?) / 320 (hop_len) = 28 (segment size)
    inter_channels, # 192
    hidden_channels, # 192
    filter_channels, # 768
    n_heads, # 2
    n_layers, # 6
    kernel_size, # 3
    p_dropout, # 0.1
    resblock, # "1"
    resblock_kernel_sizes, # [3,7,11]
    resblock_dilation_sizes, # [[1,3,5], [1,3,5], [1,3,5]]
    upsample_rates, # [10,8,2,2]
    upsample_initial_channel, # 512
    upsample_kernel_sizes, # [16,16,4,4]
    gin_channels, # 256256
    ssl_dim, # này có thể đổi tùy vào Audio LM model khác -> WavLM-Large là 1024
    use_spk, # có dùng
    **kwargs):

    super().__init__()
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.gin_channels = gin_channels
    self.ssl_dim = ssl_dim
    self.use_spk = use_spk

    self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels) 
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    
    if not self.use_spk:
      self.enc_spk = SpeakerEncoder(model_hidden_size=gin_channels, model_embedding_size=gin_channels)

  def forward(self, c, spec, g=None, mel=None, c_lengths=None, spec_lengths=None):
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    if spec_lengths == None:
      spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)
      
    if not self.use_spk:
      g = self.enc_spk(mel.transpose(1,2))
    # Thêm chiều cuối cùng của g, từ chiều 256 thành 256x1
    g = g.unsqueeze(-1)
      
    # Content encoder, chỉ lấy mean và log ? 
    _, m_p, logs_p, _ = self.enc_p(c, c_lengths)

    # Audio encoder, SPEC là số phức, tui nhiên ở đây chỉ lấy phần thực tức là biên độ, không lấy phapha
    z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g) 

    z_p = self.flow(z, spec_mask, g=g)

    z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)

    o = self.dec(z_slice, g=g)
    
    return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, c, g=None, mel=None, c_lengths=None):
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    if not self.use_spk:
      g = self.enc_spk.embed_utterance(mel.transpose(1,2))
    g = g.unsqueeze(-1)

    z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
    z = self.flow(z_p, c_mask, g=g, reverse=True)
    o = self.dec(z * c_mask, g=g)
    
    return o
