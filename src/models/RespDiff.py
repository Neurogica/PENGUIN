import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from models.layers.RespDiff_components import FFTLoss, diffusion_basemodel
from utils.help_func import register_baseline


@register_baseline("RespDiff")
class RespDiff(nn.Module):
    def __init__(
        self,
        input_dim=384,
        hidden_dim=1024,
        num_layers=6,
        output_dim=128,
        n_samples=100,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.diffusion_model = diffusion_basemodel(input_dim, hidden_dim, num_layers, output_dim)

        self.num_steps = 50
        self.beta = np.linspace(0.0001, 0.5, self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
        self.register_buffer("alpha_torch", alpha_torch)
        self.loss_fft = FFTLoss()
        self.n_samples = n_samples
        self.device = device

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([-((x - window_size // 2) ** 2) / float(2 * sigma**2) for x in range(window_size)])
        gauss = torch.exp(gauss)
        return gauss / gauss.sum()

    def create_window(self, window_size, sigma):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(0).unsqueeze(0)
        return _1D_window

    def ssim_1d(self, x, y, window_size=11, sigma=1.5, size_average=True):
        # Ensure the inputs have the right shape: (batch_size, 1, length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if len(y.shape) == 2:
            y = y.unsqueeze(1)

        window = self.create_window(window_size, sigma).to(self.device)
        mu_x = F.conv1d(x, window, padding=window_size // 2, groups=1)
        mu_y = F.conv1d(y, window, padding=window_size // 2, groups=1)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x = F.conv1d(x * x, window, padding=window_size // 2, groups=1) - mu_x_sq
        sigma_y = F.conv1d(y * y, window, padding=window_size // 2, groups=1) - mu_y_sq
        sigma_xy = F.conv1d(x * y, window, padding=window_size // 2, groups=1) - mu_xy

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2))

        if size_average:
            return 1 - ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1)

    def train_process(self, ppg, co2):
        B, K, L = ppg.shape

        t = torch.randint(0, self.num_steps, [B])
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn(B, K, L).to(self.device)
        noisy_co2 = (current_alpha**0.5) * co2 + (1.0 - current_alpha) ** 0.5 * noise
        predicted = self.diffusion_model(ppg, noisy_co2, t).permute(0, 2, 1)
        self.residual = noise - predicted  # (B, 1, L)
        predicted_co2 = (noisy_co2 - (1.0 - current_alpha) ** 0.5 * predicted) / (current_alpha**0.5)

        return predicted_co2

    def calc_loss(self, predicted_co2, co2):
        loss1 = (self.residual**2).sum() / co2.shape[-1]
        loss2 = self.loss_fft(predicted_co2, co2)
        return loss1 + loss2 * 1e-2

    def optimize(self, pred_signal, target_signal, optimizer, *args, **kwargs):
        loss = self.calc_loss(pred_signal, target_signal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def imputation(self, ppg):
        self.n_samples = 1
        B, K, L = ppg.shape
        imputed_samples = torch.ones(B, self.n_samples, K, L)  # (B, N, K, L)
        for i in range(self.n_samples):
            sample_noise = torch.randn(B, K, L).to(self.device)
            for t in range(self.num_steps - 1, -1, -1):
                t = torch.full((B,), t)
                noise_predicted = self.diffusion_model(ppg, sample_noise, t.to(self.device)).permute(0, 2, 1)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                coeff1 = torch.tensor(coeff1)
                coeff2 = torch.tensor(coeff2)
                noise = torch.randn_like(noise_predicted)
                sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                sample_noise = coeff1.item() * (sample_noise - coeff2.item() * noise_predicted)
                if t[0] > 0:
                    sample_noise += sigma.item() * noise

            imputed_samples[:, i] = sample_noise.detach()
        imputed_samples = imputed_samples.mean(dim=1).squeeze(1).to(ppg.device)
        return imputed_samples

    def forward(self, ppg, target_signal=None, **kwargs):
        ppg = ppg.unsqueeze(1)
        if target_signal is not None:
            target_signal = target_signal.unsqueeze(1)
            pred_signal = self.train_process(ppg, target_signal)
            pred_signal = pred_signal.squeeze(1)  # (B, L)
            return pred_signal
        elif target_signal is None:
            return self.imputation(ppg)
