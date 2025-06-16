import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate, functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# spike_grad = surrogate.ATan()

def Conv1d(in_channels, out_channels, kernel_size, dilation=1):
    padding = (kernel_size - 1) * dilation // 2
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    nn.init.kaiming_normal_(conv.weight, nonlinearity='leaky_relu')
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv

class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(channels, 2 * channels, kernel_size=3, dilation=dilation)
        self.diffusion_projection = nn.Linear(diffusion_channels, channels)

        # self.lif_conv = neuron.LIFNode(tau=1.2, surrogate_function=spike_grad, detach_reset=True, v_threshold=0.5)
        # self.lif_skip = neuron.LIFNode(tau=1.2, surrogate_function=spike_grad, detach_reset=True, v_threshold=0.5)
        self.lif_conv = neuron.LIFNode(tau=1.2, detach_reset=True)
        self.lif_skip = neuron.LIFNode(tau=1.2, detach_reset=True)

        self.res_out = Conv1d(channels, channels, kernel_size=1)
        self.skip_out = Conv1d(channels, channels, kernel_size=1)

        self.diffusion_embedding = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x, diffusion_step):

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        y = x + self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = self.dilated_conv(y)

        gate, filt = y.split(y.size(1) // 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        y = self.lif_conv(y)
        skip = self.lif_skip(self.skip_out(y))
        residual = self.res_out(y)

        return x + residual, skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        channels = params.residual_channels
        layers = params.residual_layers
        cycle = params.dilation_cycle_length

        self.input_projection = Conv1d(1, channels, kernel_size=1)
        # self.lif_input = neuron.LIFNode(tau=1.2, surrogate_function=spike_grad, detach_reset=True, v_threshold=0.5)
        self.lif_input = neuron.MultiStepLIFNode(tau=1.2, detach_reset=True)

        self.residual_layers = nn.ModuleList()
        for i in range(layers):
            dilation = 2 ** (i % cycle)
            block = ResidualBlock(channels, channels, dilation)
            self.residual_layers.append(block)

        self.output_projection = nn.Sequential(
            nn.ReLU(),
            Conv1d(channels, 1, kernel_size=1)
        )

    # def forward(self, audio, diffusion_step):
    def forward(self, audio, diffusion_step, reset=True):
        if reset:
            functional.reset_net(self)

        x = self.input_projection(audio.unsqueeze(1))
        x = self.lif_input(x)

        diffusion_step = diffusion_step.unsqueeze(-1).float()
        # print(diffusion_step.mean())

        skip_connections = []
        for block in self.residual_layers:
            x, skip = block(x, diffusion_step)
            skip_connections.append(skip)
            # print(x.mean())

        # total_skip = sum(skip_connections) / len(skip_connections)
        # print(total_skip.mean())
        total_skip = sum(skip_connections)
        total_skip = F.relu(total_skip)
        total_skip = self.output_projection(total_skip)
        # print(total_skip.mean())

        # return total_skip
        return total_skip.squeeze(1)