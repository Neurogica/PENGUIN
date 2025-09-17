import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample

from utils.help_func import register_baseline


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, groups=self.groups
        )

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=self.stride, groups=self.groups
        )

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1DMoE(nn.Module):
    """
    ResNet1D with Two Mixture of Experts (MoE) Regression Heads

    Parameters:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larger to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        n_experts: number of expert models in the MoE head
    """

    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        stride,
        groups,
        n_block,
        n_classes,
        n_experts=2,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
        use_projection=False,
    ):
        super(ResNet1DMoE, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_projection = use_projection
        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model
        self.n_experts = n_experts

        # First block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # Residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            is_first_block = i_block == 0
            downsample = i_block % self.downsample_gap == 1

            in_channels = base_filters if is_first_block else int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
            out_channels = in_channels * 2 if (i_block % self.increasefilter_gap == 0 and i_block != 0) else in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block,
            )
            self.basicblock_list.append(tmp_block)

        if self.use_projection:
            self.projector = nn.Sequential(nn.Linear(out_channels, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))
        # Final layers
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)
        # Mixture of Experts (MoE) Head 1 (mt_regression 1)
        self.expert_layers_1 = nn.ModuleList(
            [nn.Sequential(nn.Linear(out_channels, out_channels // 2), nn.ReLU(), nn.Linear(out_channels // 2, 1)) for _ in range(self.n_experts)]
        )
        self.gating_network_1 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts),
            nn.Softmax(dim=1),  # Softmax to produce weights for the experts
        )

        # Mixture of Experts (MoE) Head 2 (mt_regression 2)
        self.expert_layers_2 = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(out_channels, out_channels // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(out_channels // 2, 1))
                for _ in range(self.n_experts)
            ]
        )
        self.gating_network_2 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts),
            nn.Softmax(dim=1),  # Softmax to produce weights for the experts
        )

    def forward(self, x):
        out = x

        # First conv layer
        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # Residual blocks
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print(
                    "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                        i_block, net.in_channels, net.out_channels, net.downsample
                    )
                )
            out = net(out)
            if self.verbose:
                print(out.shape)

        # Final layers
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print("final pooling", out.shape)

        if self.use_projection:
            out_class = self.projector(out)
        else:
            out_class = self.dense(out)

        # Mixture of Experts (MoE) Head 1 (mt_regression 1)
        expert_outputs_1 = torch.stack([expert(out) for expert in self.expert_layers_1], dim=1)  # (batch_size, n_experts, 1)
        gate_weights_1 = self.gating_network_1(out)  # (batch_size, n_experts)
        out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)  # Weighted sum of experts

        # Mixture of Experts (MoE) Head 2 (mt_regression 2)
        expert_outputs_2 = torch.stack([expert(out) for expert in self.expert_layers_2], dim=1)  # (batch_size, n_experts, 1)
        gate_weights_2 = self.gating_network_2(out)  # (batch_size, n_experts)
        out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)  # Weighted sum of experts

        return out_class, out_moe1, out_moe2, out


@register_baseline("PaPaGei-S")
class PaPaGei_S(nn.Module):
    def __init__(
        self,
        segment_len=8,
        sample_rate=128,
        model_sample_rate=125,
        model_segment_len=10,
        ckpt_path="./ckpt/PaPaGei-S.pth",
        base_filters=32,
        kernel_size=3,
        stride=2,
        groups=1,
        n_block=18,
        n_classes=512,
        n_experts=3,
        **kwargs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.model_sample_rate = model_sample_rate
        self.model_segment_len = model_segment_len

        self.encoder = ResNet1DMoE(
            in_channels=1,
            base_filters=base_filters,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            n_block=n_block,
            n_classes=n_classes,
            n_experts=n_experts,
        )
        ckpt = torch.load(ckpt_path)
        state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                new_key = k[7:]  # Remove `module.` prefix
            else:
                new_key = k
            state_dict[new_key] = v
        self.encoder.load_state_dict(state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Linear(n_classes, sample_rate * segment_len),
        )

    def forward(self, ppg_signal, **kwargs):
        B, _ = ppg_signal.shape
        device = ppg_signal.device

        # Resample
        ppg_signal = ppg_signal.cpu().numpy()
        resampled = resample(ppg_signal, int(self.model_sample_rate * self.segment_len), axis=-1)
        ppg_signal = np.zeros((B, self.model_sample_rate * self.model_segment_len))
        ppg_signal[:, : self.model_sample_rate * self.segment_len] = resampled
        ppg_signal = torch.from_numpy(ppg_signal).to(device).to(torch.float32)

        ppg_signal = ppg_signal.unsqueeze(1)
        ppg_emb = self.encoder(ppg_signal)[0]
        target_signal = self.decoder(ppg_emb)

        return target_signal

    def optimize(self, pred_signal, target_signal, optimizer, *args, **kwargs):
        loss = F.mse_loss(pred_signal, target_signal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
