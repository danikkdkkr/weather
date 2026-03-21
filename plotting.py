"""Error-analysis plots for weather forecasting models.

Generates per-variable error distributions, per-horizon breakdowns,
and summary bar charts. All plots are saved to a specified directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from data_pipeline import TARGET_COLS, inverse_transform_cols


def collect_errors(
    model: torch.nn.Module,
    scaler: MinMaxScaler,
    test_loader: DataLoader,
    target_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on entire test set and return (preds, trues, errors) in original scale.

    Returns arrays of shape (N, horizon, n_targets).
    """
    if target_cols is None:
        target_cols = TARGET_COLS

    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X)
            all_preds.append(y_hat.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # (N, horizon, 11)
    trues = np.concatenate(all_trues, axis=0)

    N, H, D = preds.shape

    # Inverse-transform to original scale
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

    preds_inv = pred_df.values.reshape(N, H, D)
    trues_inv = true_df.values.reshape(N, H, D)
    errors = preds_inv - trues_inv

    return preds_inv, trues_inv, errors


def plot_error_distributions(
    errors: np.ndarray,
    target_cols: list[str],
    out_dir: str | Path,
    model_name: str = "model",
) -> list[str]:
    """Per-variable error histograms across all horizons.

    Args:
        errors: (N, horizon, n_targets) array of (pred - true).
        out_dir: directory to save plots.
        model_name: label for titles.

    Returns list of saved file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, col in enumerate(target_cols):
        ax = axes[i]
        col_errors = errors[:, :, i].flatten()
        ax.hist(col_errors, bins=60, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        mean_err = np.mean(col_errors)
        mae = np.mean(np.abs(col_errors))
        ax.set_title(f"{col}\nmean={mean_err:.3f}, MAE={mae:.3f}", fontsize=10)
        ax.set_xlabel("Error (pred - true)")
        ax.set_ylabel("Count")

    # Hide unused subplot
    for j in range(len(target_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{model_name} — Error Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"{model_name}_error_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))
    return saved


def plot_error_by_horizon(
    errors: np.ndarray,
    target_cols: list[str],
    out_dir: str | Path,
    model_name: str = "model",
) -> list[str]:
    """Per-variable MAE broken down by forecast horizon (day+1, day+2, day+3).

    Returns list of saved file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    N, H, D = errors.shape

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, H))

    for i, col in enumerate(target_cols):
        ax = axes[i]
        for h in range(H):
            col_errors = errors[:, h, i]
            ax.hist(col_errors, bins=40, alpha=0.5, label=f"Day +{h+1}",
                    color=colors[h], edgecolor="black", linewidth=0.2)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Error")
        ax.legend(fontsize=7)

    for j in range(len(target_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{model_name} — Error by Horizon", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"{model_name}_error_by_horizon.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))
    return saved


def plot_mae_summary(
    errors: np.ndarray,
    target_cols: list[str],
    out_dir: str | Path,
    model_name: str = "model",
) -> list[str]:
    """Bar chart of MAE per variable, with per-horizon breakdown.

    Returns list of saved file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    N, H, D = errors.shape

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(target_cols))
    width = 0.8 / H

    for h in range(H):
        maes = [np.mean(np.abs(errors[:, h, i])) for i in range(D)]
        ax.bar(x + h * width, maes, width, label=f"Day +{h+1}", alpha=0.8)

    ax.set_xticks(x + width * (H - 1) / 2)
    ax.set_xticklabels(target_cols, rotation=45, ha="right")
    ax.set_ylabel("MAE (original scale)")
    ax.set_title(f"{model_name} — MAE per Variable by Horizon")
    ax.legend()
    plt.tight_layout()
    path = out_dir / f"{model_name}_mae_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))
    return saved


def plot_scatter_pred_vs_true(
    preds: np.ndarray,
    trues: np.ndarray,
    target_cols: list[str],
    out_dir: str | Path,
    model_name: str = "model",
    max_points: int = 2000,
) -> list[str]:
    """Scatter plots of predicted vs true for each variable (day+1 only).

    Returns list of saved file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    # Use day+1 predictions only
    p = preds[:, 0, :]
    t = trues[:, 0, :]

    # Subsample if too many points
    if len(p) > max_points:
        idx = np.random.default_rng(42).choice(len(p), max_points, replace=False)
        p = p[idx]
        t = t[idx]

    for i, col in enumerate(target_cols):
        ax = axes[i]
        ax.scatter(t[:, i], p[:, i], alpha=0.15, s=5, c="steelblue")
        lo = min(t[:, i].min(), p[:, i].min())
        hi = max(t[:, i].max(), p[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="perfect")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(col, fontsize=10)

    for j in range(len(target_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{model_name} — Predicted vs True (Day +1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"{model_name}_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))
    return saved


def generate_all_plots(
    model: torch.nn.Module,
    scaler: MinMaxScaler,
    test_loader: DataLoader,
    out_dir: str | Path,
    model_name: str = "model",
    target_cols: list[str] | None = None,
) -> list[str]:
    """Generate all error-analysis plots and return list of saved file paths."""
    if target_cols is None:
        target_cols = TARGET_COLS

    preds, trues, errors = collect_errors(model, scaler, test_loader, target_cols)

    saved = []
    saved += plot_error_distributions(errors, target_cols, out_dir, model_name)
    saved += plot_error_by_horizon(errors, target_cols, out_dir, model_name)
    saved += plot_mae_summary(errors, target_cols, out_dir, model_name)
    saved += plot_scatter_pred_vs_true(preds, trues, target_cols, out_dir, model_name)

    print(f"Saved {len(saved)} plots to {out_dir}/")
    return saved
