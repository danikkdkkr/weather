"""PyTorch Dataset, DataLoader construction, and hyperparameter utilities.

Contains: SequenceDataset, Params, build_loaders, hyperparameter_grid.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline import train_test_split


class SequenceDataset(Dataset):
    """Sliding-window dataset.  First 11 columns of each target window are the targets."""

    def __init__(self, data, seq_len: int = 30, horizon: int = 3, step: int = 1):
        self.X = data.values if hasattr(data, "values") else np.asarray(data)
        self.seq_len = seq_len
        self.horizon = horizon
        self.step = step
        self.n = len(self.X)
        self.max_start = self.n - seq_len - horizon + 1

    def __len__(self):
        return max(0, (self.max_start + self.step - 1) // self.step)

    def __getitem__(self, idx):
        i = idx * self.step
        x = self.X[i : i + self.seq_len]
        y = self.X[i + self.seq_len : i + self.seq_len + self.horizon, :11]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class Params:
    """Hyper-parameter bundle."""

    def __init__(self, seq_len, horizon, hidden_dim, num_layers, lr, dropout):
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.dropout = dropout


def build_loaders(
    df_scaled: pd.DataFrame,
    params: Params,
    split: float = 0.8,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Split + wrap into DataLoaders."""
    train_df, test_df = train_test_split(split, df_scaled)
    train_ds = SequenceDataset(train_df, seq_len=params.seq_len, horizon=params.horizon)
    test_ds = SequenceDataset(test_df, seq_len=params.seq_len, horizon=params.horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def hyperparameter_grid(*,
    seq_lens=(30,),
    horizons=(3,),
    hidden_dims=(32,),
    num_layers_list=(1,),
    lrs=(1e-3,),
    dropouts=(0.0,),
) -> list[Params]:
    """Cartesian product of hyper-parameters."""
    result = []
    for sl in seq_lens:
        for h in horizons:
            for hd in hidden_dims:
                for nl in num_layers_list:
                    for lr in lrs:
                        for do in dropouts:
                            result.append(Params(sl, h, hd, nl, lr, do))
    return result
