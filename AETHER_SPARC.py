import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


# =========================
# Dataset
# =========================

class SyntheticBurstAudio:
    def __init__(self, length=20000, burst_probability=0.01, noise_std=0.01):
        self.length = length
        self.burst_probability = burst_probability
        self.noise_std = noise_std

    def generate(self):
        t = np.linspace(0, 1, self.length)
        clean = np.zeros_like(t)

        i = 0
        while i < self.length:
            if np.random.rand() < self.burst_probability:
                freq = np.random.uniform(200, 1500)
                duration = np.random.randint(200, 800)
                end = min(self.length, i + duration)
                clean[i:end] += 0.7 * np.sin(2 * np.pi * freq * t[i:end])
                i = end
            else:
                i += 1

        noisy = clean + np.random.normal(0, self.noise_std, self.length)
        return noisy.astype(np.float32), clean.astype(np.float32)


# =========================
# Models
# =========================

class DenseDSP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AetherSparcNet(nn.Module):
    def __init__(self, hidden=128, threshold=0.02):
        super().__init__()
        self.threshold = threshold
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        diff = torch.abs(x - torch.roll(x, 1, 0))
        mask = (diff > self.threshold).float()

        active_indices = mask.squeeze().nonzero(as_tuple=False).squeeze()

        if active_indices.numel() == 0:
            return torch.zeros_like(x), 0

        x_active = x[active_indices]

        out_active = self.relu(self.fc1(x_active))
        out_active = self.relu(self.fc2(out_active))
        out_active = self.fc3(out_active)

        output = torch.zeros_like(x)
        output[active_indices] = out_active

        return output, len(active_indices)


# =========================
# Training
# =========================

@dataclass
class Result:
    loss: float
    macs: int
    active_ratio: float


def estimate_macs(hidden, samples):
    mac_per_sample = (1 * hidden) + (hidden * hidden) + (hidden * 1)
    return mac_per_sample * samples


def train_dense(model, x, y, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    total_macs = 0
    hidden = model.fc1.out_features

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_macs += estimate_macs(hidden, len(x))

    return Result(loss.item(), total_macs, 1.0)


def train_sparse(model, x, y, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    total_macs = 0
    total_active = 0
    hidden = model.fc1.out_features

    for _ in range(epochs):
        optimizer.zero_grad()
        out, active = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_active += active
        total_macs += estimate_macs(hidden, active)

    active_ratio = total_active / (len(x) * epochs)

    return Result(loss.item(), total_macs, active_ratio)


# =========================
# Experiment
# =========================

def run_experiment():
    dataset = SyntheticBurstAudio()
    noisy, clean = dataset.generate()

    x = torch.from_numpy(noisy).unsqueeze(1)
    y = torch.from_numpy(clean).unsqueeze(1)

    dense_model = DenseDSP()
    sparse_model = AetherSparcNet()

    dense_result = train_dense(dense_model, x, y)
    sparse_result = train_sparse(sparse_model, x, y)

    savings = 1 - (sparse_result.macs / dense_result.macs)

    return {
        "dense_loss": dense_result.loss,
        "sparse_loss": sparse_result.loss,
        "dense_macs": dense_result.macs,
        "sparse_macs": sparse_result.macs,
        "active_ratio": sparse_result.active_ratio,
        "mac_savings": savings
    }


if __name__ == "__main__":
    print(run_experiment())
