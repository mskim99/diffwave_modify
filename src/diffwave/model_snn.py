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
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from math import sqrt

def print_cuda_memory(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB 단위
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


class EfficientLeaky(nn.Module):
    def __init__(self, beta=0.99, threshold=0.5, spike_scale=1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.spike_scale = spike_scale

    def forward(self, x, mem):
        mem.mul_(self.beta).add_(x)  # in-place update
        spk = (mem >= self.threshold).float() * self.spike_scale
        mem.masked_fill_(spk.bool(), 0.0)  # reset if spike
        return spk, mem


# spike_grad = surrogate.fast_sigmoid(slope=10)
spike_grad = surrogate.atan()

# --- Original PyTorch Layers and Functions ---
Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    # nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


# --- SNNtorch Adapted Model Components ---

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / 63.0)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        # Using standard ConvTranspose2d as synaptic layers
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad, threshold=0.5)  # Using Leaky neuron for activation
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, threshold=0.5)

    def forward(self, x, num_steps):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        x = torch.unsqueeze(x, 1)

        # Propagate through time
        spk_out_list = []
        for step in range(num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_out_list.append(spk2)

        return torch.stack(spk_out_list, dim=0).mean(dim=0).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        # self.lif_conv = snn.Leaky(beta=0.95, spike_grad=spike_grad, threshold=0.5)  # Spiking activation for conv
        self.lif_conv = EfficientLeaky(beta=0.99, threshold=0.5, spike_scale=10.0)
        self.diffusion_projection = Linear(512, residual_channels)

        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1) if not uncond else None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        # print_cuda_memory("Entering residual block")
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        # print_cuda_memory("After diffusion_projection")
        y = x + diffusion_step
        # print_cuda_memory("After adding diffusion_projection")

        if self.conditioner_projection is not None:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner
        else:
            y = self.dilated_conv(y)
        # print_cuda_memory("After conditioner_projection")

        mem_conv = torch.zeros_like(y)

        y, mem_conv = self.lif_conv(y, mem_conv)  # Spiking activation
        # print_cuda_memory("After lif_conv")

        # The original gated activation is simplified here for SNN compatibility
        gate, filter = torch.chunk(y, 2, dim=1)
        # print_cuda_memory("After chunk1")
        y = gate * filter  # Simplified element-wise multiplication
        # print_cuda_memory("After multiplication")

        y = self.output_projection(y)
        # print_cuda_memory("After output_projection")
        residual, skip = torch.chunk(y, 2, dim=1)
        # print_cuda_memory("After chunk2")
        return (x + residual) / sqrt(2.0), skip, mem_conv


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        # self.lif_input = snn.Leaky(beta=0.95, spike_grad=spike_grad, threshold=0.5)
        self.lif_input = EfficientLeaky(beta=0.99, threshold=0.5, spike_scale=10.0)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

        self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels) if not params.unconditional else None

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
                          uncond=params.unconditional)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        # self.lif_skip = snn.Leaky(beta=0.95, spike_grad=spike_grad, threshold=0.5)
        self.lif_skip = EfficientLeaky(beta=0.99, threshold=0.5, spike_scale=10.0)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram, num_steps):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)

        # print_cuda_memory("Start of forward")
        x = audio.unsqueeze(1)
        # print_cuda_memory("After unsqueeze")
        # print(x.mean())

        # Initialize membrane potentials
        '''
        mem_input = self.lif_input.init_leaky()
        mem_skip = self.lif_skip.init_leaky()
        mem_residual = [layer.lif_conv.init_leaky() for layer in self.residual_layers]
        '''

        # Time-step loop
        output_spikes_list = []

        cur_input = self.input_projection(x)
        # print(cur_input.mean())

        mem_input = torch.zeros_like(cur_input)
        mem_skip = torch.zeros_like(cur_input)
        # mem_residual = [torch.zeros_like(cur_input) for layer in self.residual_layers]

        spk_input, mem_input = self.lif_input(cur_input, mem_input)
        # print_cuda_memory("After lif_input")
        # print(spk_input.mean())

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        # print_cuda_memory("After diffusion_embedding")
        # print(diffusion_emb.mean())

        spectrogram_upsampled = None
        if self.spectrogram_upsampler:
            # For simplicity, we pass the upsampled spectrogram as a constant input over time
            spectrogram_upsampled = self.spectrogram_upsampler(spectrogram, num_steps)

        skip_connections = []
        x_res = spk_input
        for i, layer in enumerate(self.residual_layers):
            x_res, skip, _ = layer(x_res, diffusion_emb, spectrogram_upsampled)
            skip_connections.append(skip)
            # print(x_res.mean())
            # print(skip.mean())

        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0) / sqrt(len(self.residual_layers))
        cur_skip = self.skip_projection(skip_sum)
        # print(cur_skip.mean())
        spk_skip, mem_skip = self.lif_skip(cur_skip, mem_skip)
        # print(spk_skip.mean())

        output_current = self.output_projection(spk_skip)
        # output_spikes_list.append(output_current)
        # print(output_current.mean())
        # print('end')

        # Return the average output over time
        return output_current