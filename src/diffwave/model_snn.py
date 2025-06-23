
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

spike_grad = surrogate.atan()

# EfficientLeaky neuron
class EfficientLeaky(nn.Module):
    def __init__(self, beta=0.99, threshold=0.5, spike_scale=5.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.spike_scale = spike_scale

    def forward(self, x, mem):
        mem.mul_(self.beta).add_(x)
        spk = (mem >= self.threshold).float() * self.spike_scale
        mem.masked_fill_(spk.bool(), 0.0)
        return spk, mem

# Conv1d with custom init
def Conv1d(in_channels, out_channels, kernel_size, dilation=1):
    padding = (kernel_size - 1) * dilation // 2
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    nn.init.kaiming_normal_(conv.weight, a=1.0, mode='fan_in', nonlinearity='leaky_relu')
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv

# ResidualBlock with SNN
class ResidualBlock(nn.Module):
    def __init__(self, channels, cond_channels, diffusion_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(channels, 2 * channels, kernel_size=3, dilation=dilation)
        self.conditioner_projection = Conv1d(cond_channels, 2 * channels, kernel_size=1)
        self.diffusion_projection = nn.Linear(diffusion_channels, channels)
        # self.lif_conv = EfficientLeaky(beta=0.99, spike_scale=5.0)
        # self.lif_skip = EfficientLeaky(beta=0.99, spike_scale=5.0)
        self.lif_conv = snn.Leaky(beta=0.95, spike_grad=spike_grad, threshold=0.5)
        self.lif_skip = snn.Leaky(beta=0.95, spike_grad=spike_grad, threshold=0.5)
        self.res_out = Conv1d(channels, channels, kernel_size=1)
        self.skip_out = Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, diffusion_step):

        mem_conv = self.lif_conv.init_leaky()
        mem_skip = self.lif_skip.init_leaky()

        # mem_conv = torch.zeros_like(x)
        # mem_skip = torch.zeros_like(x)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        y = self.dilated_conv(y)

        gate, filter = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y, mem_conv = self.lif_conv(y, mem_conv)
        skip = self.skip_out(y)
        skip, mem_skip = self.lif_skip(skip, mem_skip)
        residual = self.res_out(y)

        # return x + residual, skip, mem_conv, mem_skip
        return x + residual, skip

# DiffWave-style model with SNN using AttrDict params
class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, kernel_size=1)
        # self.lif_input = EfficientLeaky(beta=0.99, spike_scale=5.0)
        self.lif_input = snn.Leaky(beta=0.95, spike_grad=spike_grad, threshold=0.5)
        self.diffusion_embedding = nn.Sequential(
            nn.Linear(1, params.residual_channels),
            nn.SiLU(),
            nn.Linear(params.residual_channels, params.residual_channels)
        )

        cond_channels = 0 if getattr(params, 'unconditional', False) else params.n_mels
        self.residual_layers = nn.ModuleList()
        for i in range(params.residual_layers):
            dilation = 2 ** (i % params.dilation_cycle_length)
            block = ResidualBlock(params.residual_channels, cond_channels, params.residual_channels, dilation)
            self.residual_layers.append(block)

        self.output_projection = nn.Sequential(
            nn.ReLU(),
            Conv1d(params.residual_channels, 1, kernel_size=1)
        )

    def forward(self, audio, diffusion_step):

        mem_input = self.lif_input.init_leaky()
        
        # audio = audio * 10.0
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        # mem_input = torch.zeros_like(x)
        x, mem_input = self.lif_input(x, mem_input)

        diffusion_step = diffusion_step.unsqueeze(-1).float()
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # mem_convs = [torch.zeros_like(x) for _ in self.residual_layers]
        # mem_skips = [torch.zeros_like(x) for _ in self.residual_layers]

        skip_connections = []
        for i, block in enumerate(self.residual_layers):
            x, skip = block(x, diffusion_emb)
            skip_connections.append(skip)

        total_skip = sum(skip_connections) / len(skip_connections)
        output = self.output_projection(total_skip)
        return output