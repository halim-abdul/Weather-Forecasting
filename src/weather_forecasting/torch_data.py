from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        features,
        target,
        sequence_length: int = 48,
        horizon: int = 1,
    ) -> None:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(target, dtype=np.float32)
        if len(x) != len(y):
            raise ValueError("features and target must have identical length")
        if sequence_length < 2 or horizon < 1:
            raise ValueError("sequence_length >= 2 and horizon >= 1 required")
        if len(x) <= sequence_length + horizon - 1:
            raise ValueError("not enough samples for requested sequence_length and horizon")
        self.x = x
        self.y = y
        self.sequence_length = sequence_length
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.x) - self.sequence_length - self.horizon + 1

    def __getitem__(self, idx: int):
        end = idx + self.sequence_length
        target_end = end + self.horizon
        x = torch.from_numpy(self.x[idx:end])
        y = torch.from_numpy(self.y[end:target_end])
        return x, y
