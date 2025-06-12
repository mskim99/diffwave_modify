# SpikingJelly 기반으로 수정된 모델 코드
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
import torch.nn.functional as F
from math import sqrt

# SpikingJelly 임포트
from spikingjelly.activation_based import neuron, surrogate, functional

# --- 기존 PyTorch 레이어 및 함수 ---
Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


# --- SpikingJelly 기반 모델 구성 요소 ---

class DiffusionEmbedding(nn.Module):
    # 이 모듈은 스파이킹과 관련 없으므로 변경 없음
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
        return low + (high - low) * (t - low_idx).unsqueeze(-1)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / 63.0)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    # 이 모듈은 스파이킹 신호가 아닌 정적 조건 신호를 생성하므로,
    # 스파이킹 뉴런 없이 기존 ANN 방식으로 구현
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        # SpikingJelly의 LIF 뉴런 사용
        self.lif_conv = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.diffusion_projection = Linear(512, residual_channels)

        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1) if not uncond else None
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner):
        # x의 shape: [T, N, C, L] (T: num_steps, N: batch_size)
        T, N, C, L = x.shape

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        # diffusion_step을 시간 축(T)으로 확장
        diffusion_step = diffusion_step.unsqueeze(0).repeat(T, 1, 1, 1)

        y = x + diffusion_step

        # Conv1d는 (N, C, L) 형태의 3D 텐서를 기대하므로, T와 N 차원을 합쳤다가 다시 분리
        y = self.dilated_conv(y.view(T * N, C, L))

        if self.conditioner_projection:
            conditioner = self.conditioner_projection(conditioner)
            # conditioner도 시간 축(T)으로 확장
            conditioner = conditioner.unsqueeze(0).repeat(T, 1, 1, 1)
            y = y.view(T, N, y.shape[1], y.shape[2]) + conditioner

        # LIF 뉴런은 (T, N, ...) 형태의 입력을 받아 처리
        y = self.lif_conv(y.view(T, N, y.shape[1], y.shape[2]))

        # 게이트 활성화 부분
        gate, filter = torch.chunk(y, 2, dim=2)  # 채널(C) 차원에서 분리
        y = torch.sigmoid(gate) * torch.tanh(filter)  # SpikingJelly에서도 이 부분은 유사하게 유지

        # Output Projection
        y = self.output_projection(y.view(T * N, y.shape[2], y.shape[3]))
        y = y.view(T, N, y.shape[1], y.shape[2])

        residual, skip = torch.chunk(y, 2, dim=2)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        # SpikingJelly의 LIF 뉴런으로 교체
        self.lif_input = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

        self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels) if not params.unconditional else None

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
                          uncond=params.unconditional)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.lif_skip = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram, num_steps):
        # 1. 뉴런 상태 초기화
        # 학습 배치마다 이 함수를 호출하여 모든 뉴런의 멤브레인 전위를 0으로 리셋
        functional.reset_net(self)

        # --- 입력 및 조건 신호 준비 ---
        x = audio.unsqueeze(1)
        x = self.input_projection(x)  # shape: [N, C, L]

        # SpikingJelly를 위해 시간 축(T)을 추가하고 반복
        x = x.unsqueeze(0).repeat(num_steps, 1, 1, 1)  # shape: [T, N, C, L]
        x = self.lif_input(x)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        spectrogram_upsampled = None
        if self.spectrogram_upsampler:
            spectrogram_upsampled = self.spectrogram_upsampler(spectrogram)

        # --- Residual Layer 통과 (for-loop 없음) ---
        skip_connections = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb, spectrogram_upsampled)
            skip_connections.append(skip_connection)

        # --- 출력 계산 ---
        # skip_connection들을 합침
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        skip_sum = skip_sum / sqrt(len(self.residual_layers))

        # Conv1d를 위해 T와 N 차원을 합침
        T, N, C, L = skip_sum.shape
        skip_sum_reshaped = skip_sum.view(T * N, C, L)

        x = self.skip_projection(skip_sum_reshaped)
        x = F.relu(x)  # ReLU는 그대로 사용
        x = self.lif_skip(x.view(T, N, x.shape[1], x.shape[2]))  # LIF 뉴런 통과

        # 최종 출력
        T, N, C, L = x.shape
        x = x.view(T * N, C, L)
        x = self.output_projection(x)
        x = x.view(T, N, x.shape[1], x.shape[2])  # shape: [T, N, 1, L]

        # 시간 축에 대해 평균내어 최종 출력 생성
        return x.mean(dim=0)