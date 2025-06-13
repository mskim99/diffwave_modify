import torch.profiler
from diffwave.model_snn import DiffWave
# from diffwave.model import DiffWave
from diffwave.params import params
from diffwave.dataset import from_path
import os

def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

model = DiffWave(params).cuda()
# dataset = from_path('/data/jionkim/Test/npy_dir_X/', params)
# features = dataset[0]
# features = _nested_map(features, lambda x: x.cuda() if isinstance(x, torch.Tensor) else x)

# audio = features['audio']
# spectrogram = features['spectrogram']
audio = torch.rand([8, 48000]).cuda()
spectrogram = None
N, T = audio.shape
t = torch.randint(0, len(params.noise_schedule), [N], device=audio.device)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,
    record_shapes=True,
    with_stack=True
) as prof:
    model(audio, t, spectrogram, num_steps=1)
    # model(audio, t, spectrogram)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))