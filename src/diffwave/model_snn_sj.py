import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional
from torch.utils.checkpoint import checkpoint

def Conv1d(in_channels, out_channels, kernel_size, dilation=1):
    """
    Kaiming 초기화를 적용한 1D Convolution 레이어를 생성합니다.
    """
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

        self.lif_conv = neuron.LIFNode(tau=1.2, v_threshold=0.5, detach_reset=True)
        self.lif_skip = neuron.LIFNode(tau=1.2, v_threshold=0.5, detach_reset=True)

        self.res_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_out = nn.Conv1d(channels, channels, kernel_size=1)

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

        '''
        y = self.lif_conv(y)
        skip = self.lif_skip(self.skip_out(y))
        residual = self.res_out(y)
        '''

        # LIF 뉴런을 한 번만 통과시킵니다.
        y = self.lif_conv(y)
        # lif_skip을 제거하고, lif_conv의 출력을 두 경로에서 공유합니다.
        skip = self.skip_out(y)
        residual = self.res_out(y)

        return x + residual, skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        channels = params.residual_channels
        layers = params.residual_layers
        cycle = params.dilation_cycle_length

        # 그래디언트 체크포인팅 사용 여부를 파라미터에서 가져옵니다.
        # 기본값은 False로 설정하여 이전 버전과의 호환성을 유지합니다.
        self.use_checkpointing = getattr(params, 'use_checkpointing', False)

        self.input_projection = Conv1d(1, channels, kernel_size=1)
        self.lif_input = neuron.MultiStepLIFNode(tau=1.2, v_threshold=0.5, detach_reset=True)

        self.residual_layers = nn.ModuleList()
        for i in range(layers):
            dilation = 2 ** (i % cycle)
            block = ResidualBlock(channels, channels, dilation)
            self.residual_layers.append(block)

        self.output_projection = nn.Sequential(
            nn.ReLU(),
            Conv1d(channels, 1, kernel_size=1)
        )

    def forward(self, audio, diffusion_step, reset=True):
        if reset:
            with torch.no_grad():
                functional.reset_net(self)

        x = self.input_projection(audio.unsqueeze(1))
        # print(x.mean())
        x = self.lif_input(x)
        # print(x.mean())

        diffusion_step = diffusion_step.unsqueeze(-1).float()
        # print(diffusion_step.mean())

        # 최적화 1: skip connection을 리스트에 저장하지 않고 바로 합산합니다.
        # 이렇게 하면 중간 텐서들을 저장할 필요가 없어 메모리가 절약됩니다.
        total_skip = 0

        for block in self.residual_layers:
            # 최적화 2: 그래디언트 체크포인팅을 적용합니다.
            # 훈련 중에만 활성화되며, 순전파 시 중간 활성화 값을 저장하는 대신
            # 역전파 시에 다시 계산하여 메모리를 절약합니다.
            if self.use_checkpointing and self.training:
                x, skip = checkpoint(block, x, diffusion_step, use_reentrant=False)
            else:
                x, skip = block(x, diffusion_step)
            # print(x.size())
            # print(skip.size())
            # print(x.mean())
            # print(skip.mean())

            # skip 출력을 점진적으로 더해줍니다.
            total_skip = total_skip + skip
            # print(total_skip.mean())

        total_skip = F.relu(total_skip)
        # print(total_skip.mean())
        total_skip = self.output_projection(total_skip)
        # print(total_skip.mean())

        return total_skip