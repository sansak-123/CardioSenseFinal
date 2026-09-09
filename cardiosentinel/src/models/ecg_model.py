"""
Experiment 2 — ECG risk classifier. Three independent branches over the SAME
patients (signal, PTB-XL+ engineered features, demographics), concatenated and
projected to a multi-label sigmoid over the 5 PTB-XL diagnostic superclasses
(NORM/MI/STTC/CD/HYP). Kept separate from Experiment 1 entirely.
"""
import torch
import torch.nn as nn

PTBXL_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class SignalBranch(nn.Module):
    """Small 1D ResNet over a [n_leads, n_samples] waveform, e.g. [12, 1000]
    for PTB-XL at 100Hz/10s."""

    def __init__(self, n_leads: int = 12, base_channels: int = 32,
                 n_blocks_per_stage: int = 2, n_stages: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        stages = []
        channels = base_channels
        for stage in range(n_stages):
            for _ in range(n_blocks_per_stage):
                stages.append(ResidualBlock1D(channels))
            if stage < n_stages - 1:
                next_channels = channels * 2
                stages.append(nn.Sequential(
                    nn.Conv1d(channels, next_channels, kernel_size=1),
                    nn.BatchNorm1d(next_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                ))
                channels = next_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.pool(x).squeeze(-1)


class MLPBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECGRiskClassifier(nn.Module):
    def __init__(self, n_leads: int, n_ecg_features: int, n_demo_features: int,
                 n_classes: int = len(PTBXL_SUPERCLASSES), signal_channels: int = 32,
                 feature_hidden=(64, 64), demo_hidden=(16, 16), fusion_hidden=(128,)):
        super().__init__()
        self.signal_branch = SignalBranch(n_leads=n_leads, base_channels=signal_channels)
        self.feature_branch = MLPBranch(n_ecg_features, hidden_dims=feature_hidden)
        self.demo_branch = MLPBranch(n_demo_features, hidden_dims=demo_hidden)

        fused_dim = (self.signal_branch.output_dim + self.feature_branch.output_dim
                     + self.demo_branch.output_dim)
        layers = []
        prev = fused_dim
        for h in fusion_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, signal: torch.Tensor, features: torch.Tensor,
                demographics: torch.Tensor) -> torch.Tensor:
        s = self.signal_branch(signal)
        f = self.feature_branch(features)
        d = self.demo_branch(demographics)
        fused = torch.cat([s, f, d], dim=-1)
        return self.classifier(fused)  # logits, [B, n_classes]
