import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.S5 import S5
from utils.help_func import register_baseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TimestepEmbedder(nn.Module):
    def __init__(self, h_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, h_dim, bias=True),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Flow_SSM_Layer(nn.Module):
    def __init__(self, h_dim=8, ssm_ratio=2.0, mlp_ratio=4.0):
        super().__init__()

        # PPG/Target Components
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(h_dim, 12 * h_dim, bias=True))
        self.cross_attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=1, batch_first=True)

        # PPG Components
        self.norm1_ppg = nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-6)
        self.norm2_ppg = nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-6)
        self.ssm_ppg = S5(h_dim, int(h_dim * ssm_ratio), bidir=True)
        self.pre_attn_ppg = nn.Sequential(
            nn.Linear(h_dim, int(h_dim * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(h_dim * mlp_ratio), h_dim, bias=True),
        )
        self.mlp_ppg = nn.Sequential(
            nn.Linear(h_dim, int(h_dim * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(h_dim * mlp_ratio), h_dim, bias=True),
        )

        # Target Components
        self.norm1_target = nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-6)
        self.norm2_target = nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-6)
        self.ssm_target = S5(h_dim, int(h_dim * ssm_ratio), bidir=True)
        self.pre_attn_target = nn.Sequential(
            nn.Linear(h_dim, int(h_dim * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(h_dim * mlp_ratio), h_dim, bias=True),
        )
        self.post_attn_target = nn.Sequential(
            nn.Linear(h_dim, int(h_dim * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(h_dim * mlp_ratio), h_dim, bias=True),
        )
        self.mlp_target = nn.Sequential(
            nn.Linear(h_dim, int(h_dim * mlp_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(h_dim * mlp_ratio), h_dim, bias=True),
        )

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, ppg_signal, x_t, cond):
        ppg_signal, x_t = ppg_signal.transpose(1, 2), x_t.transpose(1, 2)
        (
            shift_ssm_ppg,
            scale_ssm_ppg,
            gate_ssm_ppg,
            shift_mlp_ppg,
            scale_mlp_ppg,
            gate_mlp_ppg,
            shift_ssm_target,
            scale_ssm_target,
            gate_ssm_target,
            shift_mlp_target,
            scale_mlp_target,
            gate_mlp_target,
        ) = self.adaLN_modulation(cond).chunk(12, dim=1)

        # PPG Stream
        res_ppg = ppg_signal
        ppg_cond = gate_ssm_ppg.unsqueeze(1) * self.ssm_ppg(self.modulate(self.norm1_ppg(ppg_signal), shift_ssm_ppg, scale_ssm_ppg))
        ppg_signal = res_ppg + ppg_cond

        ppg_signal = gate_mlp_ppg.unsqueeze(1) * self.mlp_ppg(self.modulate(self.norm2_ppg(ppg_signal), shift_mlp_ppg, scale_mlp_ppg))
        ppg_signal = res_ppg + ppg_signal

        # Target Stream
        res_target = x_t
        target_cond = gate_ssm_target.unsqueeze(1) * self.ssm_target(self.modulate(self.norm1_target(x_t), shift_ssm_target, scale_ssm_target))
        ppg_cond, target_cond = self.pre_attn_ppg(ppg_cond), self.pre_attn_target(target_cond)
        target_cond = target_cond + ppg_cond
        target_cond = self.post_attn_target(target_cond)
        x_t = res_target + target_cond

        target_cond = gate_mlp_target.unsqueeze(1) * self.mlp_target(self.modulate(self.norm2_target(x_t), shift_mlp_target, scale_mlp_target))
        dx_t = res_target + target_cond

        ppg_signal, dx_t = ppg_signal.transpose(1, 2), dx_t.transpose(1, 2)
        return ppg_signal, dx_t


class FinalLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(h_dim, 1, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(h_dim, 2 * h_dim, bias=True))

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, dx_t, cond):
        dx_t = dx_t.transpose(1, 2)
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        dx_t = self.modulate(self.norm_final(dx_t), shift, scale)
        dx_t = self.linear(dx_t)
        dx_t = dx_t.transpose(1, 2)
        return dx_t


@register_baseline("PENGUIN")
class PENGUIN(nn.Module):
    def __init__(
        self,
        n_step=25,
        sample_rate=128,
        h_dim=16,
        ssm_block_num=4,
        ssm_ratio=2.0,
        mlp_ratio=4.0,
        **kwargs,
    ):
        super().__init__()
        self.n_step = n_step
        self.mean, self.std = None, None
        self.sample_rate = sample_rate

        # Convolutional layers before Flow-SSM blocks
        self.revin = nn.Parameter(torch.zeros(2))
        self.pre_conv_ppg = nn.Sequential(
            nn.Conv1d(1, h_dim, kernel_size=sample_rate // 4, padding="same"),
            nn.SiLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=sample_rate // 4, padding="same"),
        )
        self.pre_conv_target = nn.Sequential(
            nn.Conv1d(1, h_dim, kernel_size=sample_rate // 4, padding="same"),
            nn.SiLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=sample_rate // 4, padding="same"),
        )
        self.timestep_embedder = TimestepEmbedder(h_dim)

        # Flow-SSM
        self.flow_ssm_list = nn.ModuleList([Flow_SSM_Layer(h_dim, ssm_ratio, mlp_ratio) for _ in range(ssm_block_num)])

        # Final layer
        self.final_layer = FinalLayer(h_dim)

        self.init_adaLN_modulation()

    def init_adaLN_modulation(self):
        # Zero-out adaLN modulation layers in Flow-SSM blocks:
        for block in self.flow_ssm_list:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward_step(self, x_t, ppg_signal, timestep):
        ppg_signal = self.pre_conv_ppg(ppg_signal)  # [B, h_dim, T]
        x_t_emb = self.pre_conv_target(x_t)  # [B, h_dim, T]
        timestep_emb = self.timestep_embedder(timestep.reshape(-1))  # [B, h_dim]
        cond = timestep_emb

        all_dx_t = torch.zeros_like(x_t_emb).to(x_t_emb.device)
        for flow_ssm in self.flow_ssm_list:
            ppg_signal, pred_dx_t = flow_ssm(ppg_signal, x_t_emb, cond)  # [B, h_dim, T], [B, h_dim, T]
            all_dx_t += pred_dx_t

        all_dx_t = self.final_layer(all_dx_t, cond)  # [B, 1, T]
        return all_dx_t

    def heun_step(self, x_t, ppg_signal, t_start, t_end):
        # Heun's method
        t_start, t_end = t_start.unsqueeze(-1), t_end.unsqueeze(-1)
        dx_t = self.forward_step(x_t, ppg_signal, t_start).detach()
        pre_x_t1 = x_t + (t_end - t_start).unsqueeze(-1) * dx_t
        x_t1 = x_t + (t_end - t_start).unsqueeze(-1) / 2 * (dx_t + self.forward_step(pre_x_t1, ppg_signal, t_end).detach())

        return x_t1

    def train_flow(self, ppg_signal, target_signal, **kwargs):
        B, T = ppg_signal.shape
        device = ppg_signal.device

        # Velocity preparation
        timestep = torch.rand(B, 1).to(device)  # [B,1]
        ppg_signal = ppg_signal.unsqueeze(1)  # [B,1,T]
        x_1 = target_signal.unsqueeze(1)  # [B,1,T]
        x_0 = torch.randn_like(x_1).to(device)  # [B,1,T]
        x_t = (1 - timestep.reshape(-1, 1, 1)) * x_0 + timestep.reshape(-1, 1, 1) * x_1  # [B,1,T]
        self.dx_t = x_1 - x_0  # [B,1,T]

        # Forward step
        self.pred_dx_t = self.forward_step(x_t, ppg_signal, timestep)  # [B, 1, T]
        pred_x_1 = x_t + (1 - timestep).unsqueeze(-1) * self.pred_dx_t  # Euler step for monitoring training
        pred_x_1 = pred_x_1.squeeze(1)

        return pred_x_1

    def sample(self, ppg_signal, **kwargs):
        B, T = ppg_signal.shape
        device = ppg_signal.device
        ppg_signal = ppg_signal.unsqueeze(1)  # [B,1,T]
        x_0 = torch.randn(B, 1, T).to(device)  # [B,1,T]

        # Sampling
        time_steps = torch.linspace(0, 1.0, self.n_step + 1).repeat(B, 1).to(device)  # [B, n_step + 1]
        x_t = x_0
        for i in range(self.n_step):
            x_t = self.heun_step(x_t, ppg_signal, time_steps[:, i], time_steps[:, i + 1])
        pred_x_1 = x_t

        return pred_x_1.squeeze(1)

    def forward(self, ppg_signal, target_signal=None, **kwargs):
        if target_signal is not None:
            return self.train_flow(ppg_signal, target_signal)
        else:
            return self.sample(ppg_signal)

    def optimize(self, pred_target, target_signal, optimizer, *args, **kwargs):
        loss = F.mse_loss(self.pred_dx_t, self.dx_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


if __name__ == "__main__":
    model = PENGUIN().to(DEVICE)
    sample_ppg = torch.randn(32, 1024).to(DEVICE)
    sample_target = torch.randn(32, 1024).to(DEVICE)

    output = model(sample_ppg, sample_target)
    # output = model(sample_ppg)
    print(output.shape)
