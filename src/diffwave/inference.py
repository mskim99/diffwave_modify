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
import torchaudio

from argparse import ArgumentParser

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave
# from diffwave.model_snn import DiffWave
# from diffwave.model_snn_opt import DiffWave
# from diffwave.model_snn_sj import DiffWave
# from diffwave.model_snn_sj_opt import DiffWave

import os
from tqdm import tqdm

models = {}

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'


def remap_keys_ssn_sj(old_state_dict):
    new_state_dict = {}
    for k, v in old_state_dict.items():
        if not k.startswith("_orig_mod."):
            continue

        # 기본적으로 _orig_mod. 제거
        new_k = k.replace("_orig_mod.", "")

        # nested diffusion_embedding 처리
        new_k = new_k.replace("diffusion_embedding.diffusion_embedding.0", "diffusion_embedding.0")
        new_k = new_k.replace("diffusion_embedding.diffusion_embedding.2", "diffusion_embedding.2")

        # output_projection.1 → res_out + skip_out 구조로 나눠 저장되던 것 해결
        if ".output_projection.1" in new_k:
            if "residual_layers" in new_k:
                res_layer_id = new_k.split(".")[1]
                if "weight" in new_k:
                    new_state_dict[f"residual_layers.{res_layer_id}.res_out.weight"] = v
                    new_state_dict[f"residual_layers.{res_layer_id}.skip_out.weight"] = v
                elif "bias" in new_k:
                    new_state_dict[f"residual_layers.{res_layer_id}.res_out.bias"] = v
                    new_state_dict[f"residual_layers.{res_layer_id}.skip_out.bias"] = v
                continue  # 이미 저장했으므로 건너뜀

        # skip_out 처리
        if new_k == "skip_out.weight" or new_k == "skip_out.bias":
            continue  # 위에서 처리하므로 무시

        # 나머지 정상 키
        new_state_dict[new_k] = v

    return new_state_dict


def predict(spectrogram=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False, num_steps=1):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model = DiffWave(AttrDict(base_params)).to(device)
    compile_state_dict = checkpoint['model']
    state_dict = remap_keys_ssn_sj(compile_state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    models[model_dir] = model

  model = models[model_dir]
  model.params.override(params)
  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    if not model.params.unconditional:
      if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
      audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    else:
      audio = torch.randn(1, params.audio_len, device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device)).squeeze(1))
      # audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      # audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram, num_steps).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate


def main(args):
  if args.spectrogram_path:
    spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
  else:
    spectrogram = None
  idx = 0
  for _ in range(0, 100):

    if idx > 16:
        exit(0)

    audio, sr = predict(spectrogram, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params)
    # audio, sr = predict(spectrogram, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params, num_steps=1)
    # torchaudio.save(args.output, audio.cpu(), sample_rate=sr)
    try:
        np.save(args.output + '_' + str(idx).zfill(2) + '.npy', audio.cpu())
    except Exception as e:
        print(f"Failed to save sample {idx}: {e}")
        continue  # 다음 루프로 넘어감
    idx = idx + 1
    print(f"Succeed to save sample {idx}")


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--spectrogram_path', '-s',
      help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--output', '-o', default='output',
      help='output file name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  main(parser.parse_args())
