import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTLoss(nn.Module):
    def __init__(self, loss_type="magnitude", reduction="mean"):
        super(FFTLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, predicted, target):
        # Compute the FFT of both predicted and target signals
        pred_fft = torch.fft.fft(predicted, dim=-1, norm="ortho")
        target_fft = torch.fft.fft(target, dim=-1, norm="ortho")

        if self.loss_type == "magnitude":
            # Calculate the magnitude spectra
            pred_magnitude = torch.abs(pred_fft)
            target_magnitude = torch.abs(target_fft)
            loss = torch.mean((pred_magnitude - target_magnitude) ** 2, dim=-1)

        elif self.loss_type == "complex":
            # Calculate the difference in complex spectra
            loss = torch.mean((pred_fft - target_fft).abs() ** 2, dim=-1)

        elif self.loss_type == "phase":
            # Calculate the phase spectra
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)
            loss = torch.mean((pred_phase - target_phase) ** 2, dim=-1)

        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ConditionalTimeGrad(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, nonlinearity="tanh", batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        x = torch.nn.functional.relu(self.fc1(rnn_out))
        output = self.fc2(x)
        return output


class decoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=512, out_dimension=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, out_dimension)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SignalEncoder(nn.Module):
    def __init__(self, kernel_sizes=[1, 3, 5, 7, 9, 11]):
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv1d(1, 32, kernel_size=ks, padding=ks // 2) for ks in kernel_sizes])

        # Initialize weights using kaiming normal initialization
        for layer in self.conv_layers:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        outputs = [conv(x) for conv in self.conv_layers]
        concatenated_output = torch.cat(outputs, dim=1)  # Shape: (B, 6*N, L)

        return concatenated_output


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class dilated_conv(nn.Module):
    def __init__(self, kernel_size, in_channels=1, out_channels=48):
        super().__init__()
        self.layer0 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=1, padding=kernel_size // 2 * 1)
        self.layer1 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=2, padding=kernel_size // 2 * 2)
        self.layer2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=4, padding=kernel_size // 2 * 4)
        self.bn0 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.bn1 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU()
        for layer in [self.layer0, self.layer1, self.layer2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        self.bottle = Conv1d_with_init(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.bottle(x)
        x = self.bn0(self.relu(self.layer0(x)) + x)
        x = self.bn1(self.relu(self.layer1(x)) + x)
        x = self.bn2(self.relu(self.layer2(x)) + x)
        return x


class SignalEncoder_dil(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, kernel_sizes=[3, 5, 7, 9, 11, 13]):
        super().__init__()
        # Store the number of output channels per convolution
        self.N = out_channels

        # Create multiple convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([dilated_conv(kernel_size=ks, in_channels=1, out_channels=32) for ks in kernel_sizes])

    def forward(self, x):
        # Apply convolutions to corresponding slices
        outputs = [conv(x) for conv in self.conv_layers]
        concatenated_output = torch.cat(outputs, dim=1)  # Shape: (B, 6*N, L)
        return concatenated_output


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diffusion_basemodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.diff_model = ConditionalTimeGrad(input_dim, hidden_dim, num_layers, output_dim)
        self.diffusion_embedding = DiffusionEmbedding(num_steps=50, embedding_dim=128, projection_dim=192)
        self.ppg_encoder1 = SignalEncoder_dil()
        self.ppg_encoder2 = SignalEncoder()
        self.noise_encoder = SignalEncoder()
        self.de = decoder(input_dimension=output_dim)
        self.weight = torch.nn.parameter.Parameter(torch.ones(1, 192, 1), requires_grad=True)

    def forward(self, ppg, noise, step):
        embedding = self.diffusion_embedding(step).unsqueeze(2)
        f1 = self.ppg_encoder2(ppg) + embedding + self.weight * self.ppg_encoder1(ppg)
        # f2 = self.noise_encoder(noise)
        f2 = self.noise_encoder(noise)
        f = torch.concat([f1, f2], dim=1).permute(0, 2, 1)
        noise_predicted = self.de(torch.nn.functional.relu(self.diff_model(f)))
        return noise_predicted
