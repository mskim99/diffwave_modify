import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, layer
from spikingjelly.clock_driven.encoding import PoissonEncoder

class MultiStepSNNResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.diffusion_projection = layer.Linear(diffusion_channels, channels)

        self.conv = layer.Conv1d(channels, 2 * channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.lif_conv = neuron.MultiStepLIFNode(tau=1.2, v_threshold=0.5, detach_reset=True)

        self.skip_out = layer.Conv1d(channels, channels, kernel_size=1)
        self.res_out = layer.Conv1d(channels, channels, kernel_size=1)

        self.diffusion_embedding = nn.Sequential(
            layer.Linear(1, channels),
            nn.SiLU(),
            layer.Linear(channels, channels)
        )

    def forward(self, x, diffusion_step):
        # x: [T, B, C, L]
        T, B, _, L = x.shape
        diffusion_emb = self.diffusion_embedding(diffusion_step.view(B, 1).float())  # [B, C]
        diffusion_proj = self.diffusion_projection(diffusion_emb).view(1, B, -1, 1).repeat(T, 1, 1, L)

        y = x + diffusion_proj
        y = self.conv(y)
        y = self.lif_conv(y)

        gate, filt = y.chunk(2, dim=2)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        skip = self.skip_out(y)
        residual = self.res_out(y)
        return x + residual, skip

class FullSNNDiffusionModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        channels = params.residual_channels
        layers = params.residual_layers
        cycle = params.dilation_cycle_length

        self.encoder = PoissonEncoder(T=params.time_steps)
        self.input_projection = layer.Conv1d(1, channels, kernel_size=1)
        self.input_lif = neuron.MultiStepLIFNode(tau=1.2, v_threshold=0.5, detach_reset=True)

        self.residual_blocks = nn.ModuleList()
        for i in range(layers):
            dilation = 2 ** (i % cycle)
            block = MultiStepSNNResidualBlock(channels, channels, kernel_size=3, dilation=dilation)
            self.residual_blocks.append(block)

        self.output_projection = nn.Sequential(
            layer.Conv1d(channels, 1, kernel_size=1),
            neuron.MultiStepLIFNode(tau=1.2, v_threshold=0.5, detach_reset=True)
        )

    def forward(self, audio, diffusion_step, reset=True):
        # audio: [B, L], diffusion_step: [B]
        if reset:
            functional.reset_net(self)

        # Encode to spike sequence
        spike_seq = self.encoder(audio.unsqueeze(1))  # [T, B, 1, L]
        x = self.input_projection(spike_seq)
        x = self.input_lif(x)

        total_skip = 0
        for block in self.residual_blocks:
            x, skip = block(x, diffusion_step)
            total_skip = total_skip + skip

        total_skip = F.relu(total_skip)
        out = self.output_projection(total_skip)

        # Optionally: sum across time steps to get final prediction
        return out.sum(dim=0)  # [B, 1, L]
