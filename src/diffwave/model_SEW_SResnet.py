import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
from spikingjelly.activation_based import neuron

# --- 1D 모델을 위한 새로운 정의 (for learner.py) ---

__all__ = ['SEWResNet1D', 'sew_resnet18_1d', 'sew_resnet34_1d', 'sew_resnet50_1d']


def conv3x3_1d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 1D convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1_1d(in_planes, out_planes, stride=1):
    """1x1 1D convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(BasicBlock1D, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3_1d(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = neuron.IFNode(detach_reset=True)

        self.conv2 = layer.SeqToANNContainer(
            conv3x3_1d(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = neuron.IFNode(detach_reset=True)

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        # Inplace(+=)가 아닌 Out-of-place(+) 연산으로 수정
        if self.connect_f == 'ADD':
            out = out + identity
        elif self.connect_f == 'AND':
            out = out * identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(Bottleneck1D, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = layer.SeqToANNContainer(
            conv1x1_1d(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = neuron.IFNode(detach_reset=True)
        self.conv2 = layer.SeqToANNContainer(
            conv3x3_1d(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = neuron.IFNode(detach_reset=True)
        self.conv3 = layer.SeqToANNContainer(
            conv1x1_1d(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = neuron.IFNode(detach_reset=True)

    def forward(self, x):
        identity = x
        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        out = self.sn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out = out + identity
        elif self.connect_f == 'AND':
            out = out * identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)
        return out


class SEWResNet1D(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f='ADD'):  # connect_f 기본값을 'ADD'로 변경 권장
        super(SEWResNet1D, self).__init__()
        self.T = T
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = neuron.IFNode(detach_reset=True)
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)

        self.final_conv = conv1x1_1d(512 * block.expansion, 1)
        # self.final_act = nn.Hardtanh(min_val=-1, max_val=1)

        # self.output_scale = nn.Parameter(torch.tensor([1.0]))

        # 표준 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.conv3.module[1].weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.conv2.module[1].weight, 0)

        # --- 최종 핵심 수정: '죽은 뉴런' 방지를 위한 BatchNorm Bias 재초기화 ---
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0.5)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample 경로에 IFNode가 없는 것을 확인
            downsample = layer.SeqToANNContainer(
                conv1x1_1d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        original_length = x.shape[-1]

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)

        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)

        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(dim=0)
        x = self.final_conv(x)
        x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)

        # 네트워크는 모양을 학습하고, 이 파라미터는 크기를 학습합니다.
        # x = x * self.output_scale

        x = torch.tanh(x)

        return x

    def forward(self, x, t=None):
        return self._forward_impl(x)


def _sew_resnet_1d(block, layers, **kwargs):
    model = SEWResNet1D(block, layers, **kwargs)
    return model


def sew_resnet18_1d(**kwargs):
    return _sew_resnet_1d(BasicBlock1D, [2, 2, 2, 2], **kwargs)


def sew_resnet34_1d(**kwargs):
    return _sew_resnet_1d(BasicBlock1D, [3, 4, 6, 3], **kwargs)


def sew_resnet50_1d(**kwargs):
    return _sew_resnet_1d(Bottleneck1D, [3, 4, 6, 3], **kwargs)