import neurokit2 as nk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.RDDM_components import (
    CrossAttentionBlock,
    DoubleConv,
    Down,
    OutConv,
    SegmentUp,
    Up,
)
from models.layers.RDDM_diffusion import ddpm_schedule
from utils.help_func import register_baseline


class ConditionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc_c = DoubleConv(1, 64)
        self.inc_freq = DoubleConv(1, 64)

        self.down1_c = Down(64, 128)
        self.down2_c = Down(128, 256)
        self.down3_c = Down(256, 512)
        self.down4_c = Down(512, 1024)
        self.down5_c = Down(1024, 2048 // 2)

        self.up1_c = SegmentUp(1024, 512)
        self.up2_c = SegmentUp(512, 256)
        self.up3_c = SegmentUp(256, 128)
        self.up4_c = SegmentUp(128, 64)
        self.up5_c = SegmentUp(64, 32)

        # self.down1_c = Down(64, 128)
        # self.down2_c = Down(128, 256)
        # self.down3_c = Down(256, 512 // 2)

        # self.up1_c = SegmentUp(256, 128)
        # self.up2_c = SegmentUp(128, 64)
        # self.up3_c = SegmentUp(64, 32)

    def forward(self, ppg_signal):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """

        d1 = self.inc_c(ppg_signal)
        d2 = self.down1_c(d1)
        d3 = self.down2_c(d2)
        d4 = self.down3_c(d3)
        d5 = self.down4_c(d4)
        d6 = self.down5_c(d5)

        u1 = self.up1_c(d6)
        u2 = self.up2_c(u1)
        u3 = self.up3_c(u2)
        u4 = self.up4_c(u3)
        u5 = self.up5_c(u4)

        return {
            "down_conditions": [d1, d2, d3, d4, d5, d6],
            "up_conditions": [u1, u2, u3, u4, u5],
        }

        # d1 = self.inc_c(ppg_signal)
        # d2 = self.down1_c(d1)
        # d3 = self.down2_c(d2)
        # d4 = self.down3_c(d3)

        # u1 = self.up1_c(d4)
        # u2 = self.up2_c(u1)
        # u3 = self.up3_c(u2)

        # return {
        #     "down_conditions": [d1, d2, d3, d4],
        #     "up_conditions": [u1, u2, u3],
        # }


class DiffusionUNetCrossAttention(nn.Module):
    def __init__(self, in_size, channels, device, num_heads=8):
        super().__init__()
        self.in_size = in_size
        self.channels = channels
        self.device = device

        self.inc_x = DoubleConv(channels, 64)
        self.inc_freq = DoubleConv(channels, 64)

        self.down1_x = Down(64, 128)
        self.down2_x = Down(128, 256)
        self.down3_x = Down(256, 512)
        self.down4_x = Down(512, 1024)
        self.down5_x = Down(1024, 2048 // 2)

        self.up1_x = Up(1024, 512)
        self.up2_x = Up(512, 256)
        self.up3_x = Up(256, 128)
        self.up4_x = Up(128, 64)
        self.up5_x = Up(64, 32)

        self.cross_attention_down1 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_down2 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_down3 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_down4 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_down5 = CrossAttentionBlock(1024, num_heads)
        self.cross_attention_down6 = CrossAttentionBlock(1024, num_heads)

        self.cross_attention_up1 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_up2 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_up3 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_up4 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_up5 = CrossAttentionBlock(32, num_heads)

        # self.down1_x = Down(64, 128)
        # self.down2_x = Down(128, 256)
        # self.down3_x = Down(256, 512 // 2)

        # self.up1_x = Up(256, 128)
        # self.up2_x = Up(128, 64)
        # self.up3_x = Up(64, 32)

        # self.cross_attention_down1 = CrossAttentionBlock(64, num_heads)
        # self.cross_attention_down2 = CrossAttentionBlock(128, num_heads)
        # self.cross_attention_down3 = CrossAttentionBlock(256, num_heads)
        # self.cross_attention_down4 = CrossAttentionBlock(256, num_heads)

        # self.cross_attention_up1 = CrossAttentionBlock(128, num_heads)
        # self.cross_attention_up2 = CrossAttentionBlock(64, num_heads)
        # self.cross_attention_up3 = CrossAttentionBlock(32, num_heads)

        self.outc_x = OutConv(32, channels)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1).repeat(1, 1, embed_size)

    def forward(self, x, c, t):
        """
        Model is U-Net with added positional encodings and cross-attention layers.
        """
        t = t.unsqueeze(-1)

        # Level 1
        x1 = self.inc_x(x)
        x1 = self.cross_attention_down1(x1, c["down_conditions"][0])

        x2 = self.down1_x(x1) + self.pos_encoding(t, 128, x1.shape[-1] // 2)
        x2 = self.cross_attention_down2(x2, c["down_conditions"][1])

        # Level 2
        x3 = self.down2_x(x2) + self.pos_encoding(t, 256, x1.shape[-1] // 4)
        x3 = self.cross_attention_down3(x3, c["down_conditions"][2])

        # Level 3
        x4 = self.down3_x(x3) + self.pos_encoding(t, 512, x1.shape[-1] // 8)
        x4 = self.cross_attention_down4(x4, c["down_conditions"][3])

        # Level 4
        x5 = self.down4_x(x4) + self.pos_encoding(t, 1024, x1.shape[-1] // 16)
        x5 = self.cross_attention_down5(x5, c["down_conditions"][4])

        # Level 5
        x6 = self.down5_x(x5) + self.pos_encoding(t, 1024, x1.shape[-1] // 32)
        x6 = self.cross_attention_down6(x6, c["down_conditions"][5])

        # Upward path
        x = self.up1_x(x6, x5) + self.pos_encoding(t, 512, x1.shape[-1] // 16)
        x = self.cross_attention_up1(x, c["up_conditions"][0])

        x = self.up2_x(x, x4) + self.pos_encoding(t, 256, x1.shape[-1] // 8)
        x = self.cross_attention_up2(x, c["up_conditions"][1])

        x = self.up3_x(x, x3) + self.pos_encoding(t, 128, x1.shape[-1] // 4)
        x = self.cross_attention_up3(x, c["up_conditions"][2])

        x = self.up4_x(x, x2) + self.pos_encoding(t, 64, x1.shape[-1] // 2)
        x = self.cross_attention_up4(x, c["up_conditions"][3])

        x = self.up5_x(x, x1) + self.pos_encoding(t, 32, x1.shape[-1])
        x = self.cross_attention_up5(x, c["up_conditions"][4])

        # # Level 1
        # x1 = self.inc_x(x)
        # x1 = self.cross_attention_down1(x1, c["down_conditions"][0])

        # x2 = self.down1_x(x1) + self.pos_encoding(t, 128, x1.shape[-1] // 2)
        # x2 = self.cross_attention_down2(x2, c["down_conditions"][1])

        # # Level 2
        # x3 = self.down2_x(x2) + self.pos_encoding(t, 256, x1.shape[-1] // 4)
        # x3 = self.cross_attention_down3(x3, c["down_conditions"][2])

        # # Level 3
        # x4 = self.down3_x(x3) + self.pos_encoding(t, 256, x1.shape[-1] // 8)
        # x4 = self.cross_attention_down4(x4, c["down_conditions"][3])

        # # Upward path
        # x = self.up1_x(x4, x3) + self.pos_encoding(t, 128, x1.shape[-1] // 4)
        # x = self.cross_attention_up1(x, c["up_conditions"][0])

        # x = self.up2_x(x, x2) + self.pos_encoding(t, 64, x1.shape[-1] // 2)
        # x = self.cross_attention_up2(x, c["up_conditions"][1])

        # x = self.up3_x(x, x1) + self.pos_encoding(t, 32, x1.shape[-1])
        # x = self.cross_attention_up3(x, c["up_conditions"][2])

        output = self.outc_x(x)

        return output.view(-1, self.channels, output.shape[-1])


@register_baseline("RDDM")
class RDDM(nn.Module):
    def __init__(
        self,
        betas=(1e-4, 0.02),
        alphas=(100, 1),
        n_T=10,
        num_heads=8,
        roi_size=32,
        sample_rate=128,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.condition_net_1 = ConditionNet()
        self.condition_net_2 = ConditionNet()
        self.eps_model = DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads)
        self.region_model = DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads)

        self.roi_size = roi_size
        self.sample_rate = sample_rate
        self.alphas = alphas
        self.n_T = n_T
        self.eta = 0
        self.beta1 = betas[0]
        self.beta_diff = betas[1] - betas[0]
        ## register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedule(self.beta1, self.beta1 + self.beta_diff, n_T).items():
            self.register_buffer(k, v)

    def create_noise_in_regions(self, patch_labels):
        patch_roi = torch.round(patch_labels)

        mask = patch_roi == 1
        random_noise = torch.randn_like(patch_roi)
        masked_noise = random_noise * mask.float()

        return masked_noise, random_noise

    def forward(self, ppg_signal, target_signal=None, peak_roi=None):
        ppg_signal = ppg_signal.unsqueeze(1)
        cond1 = self.condition_net_1(ppg_signal)
        cond2 = self.condition_net_2(ppg_signal)
        state = "train" if target_signal is not None else "sample"
        device = ppg_signal.device
        B, _, T = ppg_signal.shape

        if state == "train":
            # Calc ECG ROI regions
            target_signal = target_signal.unsqueeze(1)
            _ts = torch.randint(1, self.n_T, (B,)).to(device)
            eps, unmasked_eps = self.create_noise_in_regions(peak_roi)
            x_t = self.sqrtab[_ts, None, None] * target_signal + self.sqrtmab[_ts, None, None] * eps
            x_t_unmasked = self.sqrtab[_ts, None, None] * target_signal + self.sqrtmab[_ts, None, None] * unmasked_eps

            pred_x_t = self.region_model(x_t_unmasked, cond2, _ts / self.n_T)
            pred_masked_eps = self.eps_model(x_t, cond1, _ts / self.n_T)

            self.ddpm_loss = F.mse_loss(eps, pred_masked_eps)
            self.region_loss = F.mse_loss(pred_x_t, x_t)

            reconst_target = x_t_unmasked - pred_x_t - pred_masked_eps
            return reconst_target

        elif state == "sample":
            n_sample = cond1["down_conditions"][-1].shape[0]

            x_i = torch.randn(n_sample, 1, T).to(device)
            for i in range(self.n_T, 0, -1):
                if i > 1:
                    z = torch.randn(n_sample, 1, T).to(device)
                else:
                    z = 0

                # rho_phi estimates the trajectory from Gaussian manifold to Masked Gaussian manifold
                x_i = self.region_model(x_i, cond2, torch.tensor(i / self.n_T).to(device).repeat(n_sample))

                # epsilon_theta predicts the noise that needs to be removed to move from Masked Gaussian manifold to ECG manifold
                eps = self.eps_model(x_i, cond1, torch.tensor(i / self.n_T).to(device).repeat(n_sample))

                x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

            reconst_target = x_i.squeeze(1)
            return reconst_target

    def calc_loss(self, *args):
        loss = self.alphas[0] * self.ddpm_loss + self.alphas[1] * self.region_loss
        loss = loss.mean()
        return loss

    def optimize(self, pred_signal, target_signal, optimizer):
        loss = self.calc_loss(pred_signal, target_signal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
