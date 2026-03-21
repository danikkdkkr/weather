"""Training loop, prediction, and evaluation utilities.

Contains: run_training, predict_and_compare, evaluate_full_test.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from data_pipeline import TARGET_COLS, inverse_transform_cols
from dataset import Params


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    lr: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 30,
    checkpoint_path: str = "best_model.pt",
) -> float:
    """Train with early stopping.  Returns best validation loss."""
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    trigger = 0

    for _epoch in range(max_epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        n = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= n

        # --- validate ---
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for X, y in test_loader:
                val_loss += criterion(model(X), y).item()
                n += 1
        val_loss /= n

        # --- early stopping ---
        if val_loss < best_val:
            best_val = val_loss
            trigger = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            trigger += 1
            if trigger >= patience:
                break

    return best_val


def predict_and_compare(
    model: nn.Module,
    scaler: MinMaxScaler,
    test_loader: DataLoader,
    params: Params,
    target_cols: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Run inference on the last batch and return per-variable MAE dict."""
    if target_cols is None:
        target_cols = TARGET_COLS

    model.eval()
    y_hat = None
    y = None
    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X)

    if y_hat is None or y is None:
        return {}

    y_hat_np = y_hat[-1].cpu().numpy()
    y_true_np = y[-1].cpu().numpy()

    pred_df = pd.DataFrame(y_hat_np, columns=target_cols)
    true_df = pd.DataFrame(y_true_np, columns=target_cols)

    scaled_cols = list(scaler.feature_names_in_)
    pred_scaled = pred_df[[c for c in target_cols if c in scaled_cols]].copy()
    true_scaled = true_df[[c for c in target_cols if c in scaled_cols]].copy()

    inv_pred_scaled = inverse_transform_cols(scaler, pred_scaled)
    inv_true_scaled = inverse_transform_cols(scaler, true_scaled)

    inv_pred = pred_df.copy()
    inv_true = true_df.copy()
    for c in inv_pred_scaled.columns:
        inv_pred[c] = inv_pred_scaled[c]
        inv_true[c] = inv_true_scaled[c]

    # Post-processing: clip precipitation/snow
    for col in ("prcp", "snow"):
        if col in inv_pred.columns:
            inv_pred[col] = np.clip(inv_pred[col], 0, None)
            inv_pred.loc[inv_pred[col] < 0.3, col] = 0.0

    # Per-variable MAE
    maes: dict[str, float] = {}
    for col in target_cols:
        pred_vals = np.asarray(inv_pred[col].values, dtype=np.float64)
        true_vals = np.asarray(inv_true[col].values, dtype=np.float64)
        maes[col] = float(np.abs(pred_vals - true_vals).mean())

    if verbose:
        print("\n--- Prediction vs True (last sample) ---")
        for i, name in enumerate(target_cols):
            for h in range(params.horizon):
                p = float(inv_pred.iloc[h, i])
                t = float(inv_true.iloc[h, i])
                print(
                    f"Day +{h+1:>2}: {name:<10s} | "
                    f"Pred: {p:8.3f} | True: {t:8.3f} | Diff: {p-t:+8.3f}"
                )

    return maes


def evaluate_full_test(
    model: nn.Module,
    scaler: MinMaxScaler,
    test_loader: DataLoader,
    target_cols: list[str] | None = None,
) -> dict[str, float]:
    """Compute per-variable MAE over the *entire* test set (all batches, all samples)."""
    if target_cols is None:
        target_cols = TARGET_COLS

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X)
            all_preds.append(y_hat.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # (N, horizon, 11)
    trues = np.concatenate(all_trues, axis=0)

    # Flatten horizon dimension for inverse transform
    *_, D = preds.shape
    preds_flat = preds.reshape(-1, D)
    trues_flat = trues.reshape(-1, D)

    pred_df = pd.DataFrame(preds_flat, columns=target_cols)
    true_df = pd.DataFrame(trues_flat, columns=target_cols)

    scaled_cols = list(scaler.feature_names_in_)
    for df in (pred_df, true_df):
        sub = df[[c for c in target_cols if c in scaled_cols]].copy()
        inv = inverse_transform_cols(scaler, sub)
        for c in inv.columns:
            df[c] = inv[c]

    for col in ("prcp", "snow"):
        if col in pred_df.columns:
            pred_df[col] = np.clip(pred_df[col], 0, None)
            pred_df.loc[pred_df[col] < 0.3, col] = 0.0

    maes: dict[str, float] = {}
    for col in target_cols:
        pred_vals = np.asarray(pred_df[col].values, dtype=np.float64)
        true_vals = np.asarray(true_df[col].values, dtype=np.float64)
        maes[col] = float(np.abs(pred_vals - true_vals).mean())

    return maes
