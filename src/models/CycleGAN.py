import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.help_func import register_baseline


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_downsampling: bool = True, add_activation: bool = True, **kwargs):
        super().__init__()
        if is_downsampling:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm1d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm1d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
            ConvolutionalBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels: int, num_features: int = 64, num_residuals: int = 6):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv1d(img_channels, num_features, kernel_size=7, stride=1, padding="same", padding_mode="reflect"),
            nn.InstanceNorm1d(num_features),
            nn.ReLU(inplace=True),
        )
        self.downsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(num_features, num_features * 2, is_downsampling=True, kernel_size=3, stride=2, padding=1),
                ConvolutionalBlock(num_features * 2, num_features * 4, is_downsampling=True, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.residual_layers = nn.Sequential(*[ResidualBlock(num_features * 4) for _ in range(num_residuals)])
        self.upsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(num_features * 4, num_features * 2, is_downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvolutionalBlock(num_features * 2, num_features * 1, is_downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )
        self.last_layer = nn.Conv1d(num_features * 1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.residual_layers(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        x = torch.tanh(self.last_layer(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv1d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvolutionalBlock(
                    in_channels,
                    feature,
                    is_downsampling=True,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = feature

        layers.append(nn.Conv1d(in_channels, 1, kernel_size=4, stride=1, padding="same", padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.model(x)

        return x


@register_baseline("CycleGAN")
class CycleGAN(nn.Module):
    def __init__(self, lambdas: tuple = (10, 10), **kwargs):
        super().__init__()
        self.ppg_gen = Generator(img_channels=1)
        self.target_gen = Generator(img_channels=1)
        self.ppg_disc = Discriminator(in_channels=1)
        self.target_disc = Discriminator(in_channels=1)
        self.lambdas = lambdas

        self.opt_gen = optim.Adam(itertools.chain(self.ppg_gen.parameters(), self.target_gen.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.opt_disc_ppg = optim.Adam(self.ppg_disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_disc_target = optim.Adam(self.target_disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, ppg_signal, target_signal=None, **kwargs):
        ppg_signal = ppg_signal.unsqueeze(1)
        self.fake_target = self.target_gen(ppg_signal)
        self.real_ppg = ppg_signal

        if target_signal is not None:
            target_signal = target_signal.unsqueeze(1)
            self.fake_ppg = self.ppg_gen(target_signal)
            self.real_target = target_signal

        return self.fake_target.squeeze(1)

    def optimize(self, *args, **kwargs):
        ###### Generator Optimizer ######
        self.opt_gen.zero_grad()

        # GAN Loss
        fake_ppg_disc = self.ppg_disc(self.fake_ppg)
        fake_target_disc = self.target_disc(self.fake_target)
        gan_loss_ppg = F.mse_loss(fake_ppg_disc, torch.ones_like(fake_ppg_disc))
        gan_loss_target = F.mse_loss(fake_target_disc, torch.ones_like(fake_target_disc))

        # Cycle Loss
        cyclic_fake_target = self.target_gen(self.fake_ppg)
        cyclic_loss_ppg = F.l1_loss(self.ppg_gen(self.fake_target), self.real_ppg) * self.lambdas[0]
        cyclic_loss_target = F.l1_loss(cyclic_fake_target, self.real_target) * self.lambdas[0]

        # R-R Loss
        cyclic_fake_target_fft = torch.fft.fft(cyclic_fake_target.squeeze(1), dim=-1)
        real_target_fft = torch.fft.fft(self.real_target.squeeze(1), dim=-1)

        freq_bin = torch.fft.fftfreq(cyclic_fake_target.shape[-1], d=1 / 128).to(cyclic_fake_target.device)
        cyclic_fake_target_mag = cyclic_fake_target_fft.abs()
        real_target_mag = real_target_fft.abs()

        mask = (freq_bin > 0) & (freq_bin < 1.0)
        freq_bin = freq_bin[mask]
        cyclic_fake_target_mag = cyclic_fake_target_mag[:, mask]
        real_target_mag = real_target_mag[:, mask]

        cyclic_fake_target_bpm = 60 * freq_bin[torch.argmax(cyclic_fake_target_mag, dim=-1)]
        real_target_bpm = 60 * freq_bin[torch.argmax(real_target_mag, dim=-1)]

        resp_rate_loss = F.l1_loss(cyclic_fake_target_bpm, real_target_bpm) * self.lambdas[1]

        # Total loss
        loss_gen = gan_loss_ppg + gan_loss_target + cyclic_loss_ppg + cyclic_loss_target + resp_rate_loss
        loss_gen.backward()
        self.opt_gen.step()

        ###### PPG Discriminator Optimizer ######
        self.opt_disc_ppg.zero_grad()

        # Real Loss
        real_ppg_disc = self.ppg_disc(self.real_ppg)
        real_ppg_disc_loss = F.mse_loss(real_ppg_disc, torch.ones_like(real_ppg_disc))

        # Fake Loss
        fake_ppg_disc = self.ppg_disc(self.fake_ppg.detach())
        fake_ppg_disc_loss = F.mse_loss(fake_ppg_disc, torch.zeros_like(fake_ppg_disc))

        # Total loss
        disc_loss_ppg = (real_ppg_disc_loss + fake_ppg_disc_loss) * 0.5
        disc_loss_ppg.backward()
        self.opt_disc_ppg.step()

        ###### Target Discriminator Optimizer ######
        self.opt_disc_target.zero_grad()

        # Real Loss
        real_target_disc = self.target_disc(self.real_target)
        real_target_disc_loss = F.mse_loss(real_target_disc, torch.ones_like(real_target_disc))

        # Fake Loss
        fake_target_disc = self.target_disc(self.fake_target.detach())
        fake_target_disc_loss = F.mse_loss(fake_target_disc, torch.zeros_like(fake_target_disc))

        # Total loss
        disc_loss_target = (real_target_disc_loss + fake_target_disc_loss) * 0.5
        disc_loss_target.backward()
        self.opt_disc_target.step()

        return torch.tensor(loss_gen.item() + disc_loss_ppg.item() + disc_loss_target.item())


if __name__ == "__main__":
    batch_size = 32
    segment_len = 900
    sample_input = torch.randn(batch_size, segment_len)
    sample_target = torch.randn(batch_size, segment_len)

    model = CycleGAN()
    output = model(sample_input, sample_target)
    model.optimize()
    print(output.shape)
