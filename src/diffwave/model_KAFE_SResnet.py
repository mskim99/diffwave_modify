import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.utils import spectral_norm

class SurrogateBPFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def poisson_gen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


class SResnet1D(nn.Module):
    """
    SResnet을 1D 데이터(오디오)를 처리하도록 수정한 클래스.
    DiffWave learner와 호환되도록 설계되었습니다.
    """

    def __init__(self, n=6, nFilters=32, num_steps=1, leak_mem=0.95):
        super(SResnet1D, self).__init__()

        self.n = n
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps

        print(">>>>>>>>>>>>>>>>>>> S-ResNet-1D for Audio >>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False
        self.nFilters = nFilters

        # 1D 연산으로 변경
        self.conv1 = spectral_norm(nn.Conv1d(1, self.nFilters, kernel_size=3, stride=1, padding=1, bias=bias_flag))
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm1d(self.nFilters, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)])

        self.conv_list = nn.ModuleList([self.conv1])
        self.bntt_list = nn.ModuleList([self.bntt1])

        for block in range(3):
            for layer in range(2 * n):
                if block != 0 and layer == 0:
                    stride = 2
                    prev_nFilters = -1
                else:
                    stride = 1
                    prev_nFilters = 0

                # 1D Conv 레이어로 변경
                self.conv_list.append(
                    spectral_norm(nn.Conv1d(self.nFilters * (2 ** (block + prev_nFilters)), self.nFilters * (2 ** block),
                              kernel_size=3, stride=stride, padding=1, bias=bias_flag)))
                self.bntt_list.append(nn.ModuleList(
                    [nn.BatchNorm1d(self.nFilters * (2 ** block), eps=1e-4, momentum=0.1, affine=affine_flag) for _ in
                     range(self.batch_num)]))

        # 1D Resize Conv 레이어로 변경
        self.conv_resize_1 = spectral_norm(nn.Conv1d(self.nFilters, self.nFilters * 2, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag))
        self.resize_bn_1 = nn.ModuleList(
            [nn.BatchNorm1d(self.nFilters * 2, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in
             range(self.batch_num)])
        self.conv_resize_2 = spectral_norm(nn.Conv1d(self.nFilters * 2, self.nFilters * 4, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag))
        self.resize_bn_2 = nn.ModuleList(
            [nn.BatchNorm1d(self.nFilters * 4, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in
             range(self.batch_num)])

        # 1D Pool 레이어로 변경
        self.pool1d = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            # 첫 번째 업샘플링: 길이 L/4 -> L/2
            spectral_norm(nn.ConvTranspose1d(self.nFilters * 4, self.nFilters * 2, kernel_size=4, stride=2, padding=1,
                               bias=bias_flag)),
            nn.BatchNorm1d(self.nFilters * 2),
            nn.ReLU(),
            # 두 번째 업샘플링: 길이 L/2 -> L
            spectral_norm(nn.ConvTranspose1d(self.nFilters * 2, self.nFilters, kernel_size=4, stride=2, padding=1, bias=bias_flag)),
            nn.BatchNorm1d(self.nFilters),
            nn.ReLU()
        )

        # 출력을 다시 [B, 1, T] 형태로 만들기 위한 최종 conv 레이어
        self.final_conv = spectral_norm(nn.Conv1d(self.nFilters, 1, kernel_size=1, bias=bias_flag))

        # self.output_scale = nn.Parameter(torch.tensor([1.0]))

        self.conv1x1_list = nn.ModuleList([self.conv_resize_1, self.conv_resize_2])
        self.bn_conv1x1_list = nn.ModuleList([self.resize_bn_1, self.resize_bn_2])

        for bn_temp in self.bntt_list:
            if hasattr(bn_temp, 'bias') and bn_temp.bias is not None:
                bn_temp.bias = None

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp, t):
        # inp shape: [B, T] -> [B, 1, T] for Conv1D
        if inp.dim() == 2:
            inp = inp.unsqueeze(1)

        batch_size, _, seq_len = inp.shape

        out_prev = inp
        skip = None
        index_1x1 = 0

        for i in range(len(self.conv_list)):
            # 1. 컨볼루션과 배치 정규화 수행
            bn_t = self.bntt_list[i][0]
            current_val = bn_t(self.conv_list[i](out_prev))

            # 2. 잔차 연결(Residual Connection) 로직 수정
            #    첫 번째 레이어 이후, 짝수 번째 레이어마다 잔차 연결을 더해줍니다.
            if i > 0:
                # 다운샘플링이 필요한 블록에서 skip 연결을 변환합니다.
                # (i == 2*n) 또는 (i == 2*n + 2*n) 조건으로 볼 수 있음
                if self.conv_list[i].stride[0] > 1:
                    bn_1x1_t = self.bn_conv1x1_list[index_1x1][0]
                    skip = bn_1x1_t(self.conv1x1_list[index_1x1](skip))
                    index_1x1 += 1

                # 잔차(skip)를 더합니다.
                if skip is not None:
                    current_val = (current_val + skip) / 2.


            # 3. 표준 ReLU 활성화 함수를 잔차 연결 이후에 적용합니다.
            out = F.relu(current_val)

            # print(i)
            # print(out.mean())
            # if skip is not None:
                # print(skip.mean())

            # 4. 다음 잔차 연결을 위해 현재 출력을 skip 변수에 저장합니다.
            skip = out.clone()

            # 5. 현재 출력을 다음 레이어의 입력으로 사용합니다.
            out_prev = out.clone()

            # 최종 출력단을 통과
        upsampled_output = self.decoder(out_prev)

        # print(upsampled_output.mean())
        output_voltage = self.final_conv(upsampled_output)
        # print(output_voltage.mean())

        # 원본 시퀀스 길이로 복원
        # output_voltage = F.interpolate(final_output, size=seq_len, mode='linear', align_corners=False)
        # print(output_voltage.shape)

        # output_voltage = torch.tanh(output_voltage)
        # print(output_voltage.mean())

        # 최종 출력에 학습 가능한 스케일 파라미터를 곱해 진폭을 조절합니다.
        # final_scaled_output = output_voltage * self.output_scale
        # print(final_scaled_output.mean())

        return output_voltage