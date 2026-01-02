import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CNN 모델 (클라이언트 용) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN_Fashion(nn.Module):
    def __init__(self):
        super(SimpleCNN_Fashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # input: 1x28x28
        self.pool = nn.MaxPool2d(2, 2)  # output: 16x14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # output: 32x14x14 → 32x7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → 16x14x14
        x = self.pool(F.relu(self.conv2(x)))  # → 32x7x7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

import torch.nn as nn

class EMNIST_CNN(nn.Module):
    def __init__(self, num_classes=47):  # 47 classes for "balanced"
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    

class DAGMM(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(DAGMM, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: feature -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, latent_dim)
        )

        # Decoder: latent -> reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, input_dim)
        )

        # Estimation network: latent + reconstruction error -> gamma (energy)
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim + 1, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
            nn.Sigmoid()  # Optional depending on energy interpretation
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        recon_error = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
        z = torch.cat([latent, recon_error], dim=1)
        energy = self.estimation(z)
        return latent, reconstructed, energy

    def compute_loss(self, x, reconstructed, energy):
        recon_loss = F.mse_loss(reconstructed, x, reduction="mean")
        energy_loss = energy.mean()
        return recon_loss + 0.1 * energy_loss  # λ 조정 가능

