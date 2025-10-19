import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss, # FIX VUNV
  cyclic_consistency_loss # FIX VUNV
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

torch.backends.cudnn.benchmark = True
global_step = 0
#os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."
  hps = utils.get_hparams()

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = hps.train.port

  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000], # Hỗ trợ đa batch size à ?
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate(hps)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=True,
        batch_size=hps.train.batch_size, pin_memory=False,
        drop_last=False, collate_fn=collate_fn)

  # Mạng Generator và Discriminator
  net_g = SynthesizerTrn(
      hps.data.filter_length // 2 + 1, # số tần số trong phổ
      hps.train.segment_size // hps.data.hop_length, # kích thước đoạn
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])#, find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank]) # Dùng để train đa GPU

  try:
    # Resume từ checkpoint mới nhất nếu có
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    # Train from scratch
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  # FIX VUNV: Load SSL model một lần thay vì mỗi iteration
  cmodel = None
  if hasattr(hps.train, 'c_cyclic') and hps.train.c_cyclic > 0:
    cmodel = utils.get_cmodel(rank)
    cmodel.eval()  # Set to eval mode

  net_g.train()
  net_d.train()
  for batch_idx, items in enumerate(train_loader):
    # =======TRONG MỖI BATCH========
    # đưa spec, spk, y và content sang GPU
    if hps.model.use_spk:
      c, spec, y, spk = items
      g = spk.cuda(rank, non_blocking=True)
    else:
      c, spec, y = items
      g = None
    spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
    c = c.cuda(rank, non_blocking=True)
    mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
    # ==============================

    # ==========BẬT FP16==============
    with autocast(enabled=hps.train.fp16_run):
      # Ta có 2 phân phối z và z'
      # TODO slide ở đâu?
      y_hat, ids_slice, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(c, spec, g=g, mel=mel)

      # Trích xuất mel từ y_hat
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator output shape: (B, 1, T) 
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        # loss_disc là tổng loss của discriminator
        # loss_disc_r là loss thật
        # loss_disc_g là loss giả
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    # ==========BẬT FP16==============
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    # ==========BẬT FP16==============
    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat) # fmap_r 
      with autocast(enabled=False):
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        
        # FIX VUNV
        # ==========CYCLIC CONSISTENCY LOSS==========
        # Chỉ tính cyclic loss nếu có config và batch size >= 2
        loss_cyclic = 0
        if hasattr(hps.train, 'c_cyclic') and hps.train.c_cyclic > 0 and c.size(0) >= 2:
          # Random permutation để tạo cross-speaker pairs
          perm_idx = torch.randperm(c.size(0)).to(c.device)
          
          # Content từ sample gốc, speaker từ sample khác (random permutation)
          g_permuted = g[perm_idx] if hps.model.use_spk else None
          mel_permuted = mel[perm_idx] if not hps.model.use_spk else None
          
          # Forward: A → B (content_A + speaker_B)
          # Dùng infer để generate audio B
          with torch.no_grad():
            # Generate intermediate audio với cross-speaker
            y_ab, _, _, _ = net_g(c, spec, g=g_permuted, mel=mel_permuted)
          
          # Extract content từ generated audio (backward pass)
          # Dùng SSL model để extract content từ y_ab
          with torch.no_grad():
            c_ab = utils.get_content(cmodel, y_ab)
          
          # Backward: B → A' (content_B + speaker_A) 
          y_aba, _, _, _ = net_g(c_ab, spec, g=g, mel=mel)
          
          # Cyclic consistency loss: ||A - A'||
          # Slice để match lengths
          min_len = min(y.size(-1), y_aba.size(-1))
          loss_cyclic = cyclic_consistency_loss(
            y[:, :, :min_len], 
            y_aba[:, :, :min_len]
          ) * hps.train.c_cyclic
        # ===========================================
        
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_cyclic
        # FIX VUNV
    # ==========BẬT FP16==============

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    # grad norm là tổng độ lớn của gradient. Sqrt(sum(grad_i^2))
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})
        
        # FIX VUNV
        # Log cyclic loss nếu có
        if hasattr(hps.train, 'c_cyclic') and hps.train.c_cyclic > 0:
          scalar_dict.update({"loss/g/cyclic": loss_cyclic})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, items in enumerate(eval_loader):
        if hps.model.use_spk:
          c, spec, y, spk = items
          g = spk[:1].cuda(0)
        else:
          c, spec, y = items
          g = None
        spec, y = spec[:1].cuda(0), y[:1].cuda(0)
        c = c[:1].cuda(0)
        break
      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat = generator.module.infer(c, g=g, mel=mel)
      
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
      "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0],
      "gt/audio": y[0]
    }
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
