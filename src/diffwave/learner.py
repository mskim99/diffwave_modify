# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt
import io
from PIL import Image

from diffwave.dataset import from_path, from_gtzan
from diffwave.model import DiffWave
# from diffwave.model_snn import DiffWave # train_step > predicted()
# from diffwave.model_snn_sj import DiffWave
# from diffwave.model_KAFE_SResnet import SResnet1D
# from diffwave.model_SEW_SResnet import sew_resnet18_1d, sew_resnet34_1d, sew_resnet50_1d

from spikingjelly.clock_driven import functional

import traceback

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

def plot_waveform_to_image(waveform, title="Waveform"):
  """ 1D 오디오 파형(Numpy 배열)을 PyTorch 텐서 형태의 RGB 이미지로 변환합니다. """
  fig, ax = plt.subplots(1, 1, figsize=(12, 3))
  ax.plot(waveform)
  ax.set_title(title)
  ax.set_ylim([-1.1, 1.1])
  ax.grid(True)

  # 그래프를 인-메모리 버퍼에 PNG 이미지로 렌더링합니다.
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  buf.seek(0)

  # 버퍼에서 이미지를 읽어 RGB로 변환합니다.
  image = Image.open(buf).convert('RGB')

  # matplotlib figure 객체를 닫아 메모리를 해제합니다.
  plt.close(fig)

  # 이미지를 numpy 배열로 변환하고, TensorBoard에 맞게 (C, H, W) 형태로 축을 변경합니다.
  image_np = np.array(image)
  image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

  return image_tensor

class DiffWaveLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.noise_level = self.noise_level.cuda().detach()
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None


  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        # torch.cuda.reset_peak_memory_stats()
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss, predicted_audio = self.train_step(features)
        # print(predicted_audio.shape)
        # print(loss.item())
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss, predicted_audio)
          if self.step % 1000 == 0:
            self.save_to_checkpoint()
        self.step += 1

        # max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB 단위로 변환
        # print(f"Peak GPU Memory Allocated: {max_memory_allocated:.2f} MB")

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    functional.reset_net(self.model)

    audio = features['audio']
    spectrogram = features['spectrogram']

    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise

      predicted = self.model(noisy_audio, t)
      # predicted = self.model(noisy_audio, t, spectrogram, False)
      # predicted = self.model(noisy_audio, t, spectrogram, num_steps=self.params.num_steps)
      # loss = self.loss_fn(noise, predicted.squeeze(1).squeeze(1))
      loss = self.loss_fn(noise, predicted.squeeze(1))

      predicted_audio = (noisy_audio - (1.0 - noise_scale) ** 0.5 * predicted[0].squeeze(1)) / noise_scale_sqrt
    try:
      self.scaler.scale(loss).backward()
    except RuntimeError as e:
      print("\n[RuntimeError during backward pass! Full traceback below]")
      traceback.print_exc()
      print("\n[Tensor status before backward]")
      for name, param in self.model.named_parameters():
        if param.grad_fn is not None:
          print(f"{name}: grad_fn = {type(param.grad_fn)}")
      raise e  # 예외 다시 던져서 학습 중단
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9).item()
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss, predicted_audio

  def _write_summary(self, step, features, loss, predicted_audio):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    # writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)

    gt_waveform = features['audio'][0].cpu().numpy()
    gt_image = plot_waveform_to_image(gt_waveform, title="GT Waveform")
    writer.add_image('result/GT_waveform', gt_image, step)

    generated_waveform = predicted_audio[0].detach().cpu().numpy().squeeze()
    generated_image = plot_waveform_to_image(generated_waveform, title="Gen Waveform (one-step denoised)")
    writer.add_image('result/gen_waveform', generated_image, step)

    if not self.params.unconditional:
      writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss.item(), step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)


def train(args, params):
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params)
  else:
    dataset = from_path(args.data_dirs, params)
    model = DiffWave(params).cuda()
  # model = SResnet1D(leak_mem=0.99).cuda()
  # model = sew_resnet18_1d().cuda()
  # model = torch.compile(model)
  _train_impl(0, model, dataset, args, params)

'''
def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params, is_distributed=False)
  else:
    dataset = from_path(args.data_dirs, params, is_distributed=False)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, args, params)
'''