"""Efficient hyperparameter search for LSTM and Transformer weather models.

Algorithm: Successive Halving (SHA) + ternary search on learning rate.

SHA replaces full grid search:
  - Round 0: train ALL N configs for `min_epochs` each
  - Round 1: keep top 1/eta, train for min_epochs * eta epochs
  - Round 2: keep top 1/eta², train for min_epochs * eta² epochs
  - ...until one config remains
  Total compute ≈ N * min_epochs * eta/(eta-1)
  vs grid search:  N * max_epochs
  Typical speedup: 10-100×  (LSTM grid was 360 configs × 1000 epochs)

After SHA picks the best architecture, a ternary search (binary-search-style)
on the log-LR axis finds the optimal learning rate with O(log N) evaluations
instead of trying a dense grid.

Usage:
    from hyperparameter_search import search_transformer_hyperparams, search_lstm_hyperparams
    cfg = search_transformer_hyperparams(train_loader, test_loader, n_features=13, ...)
    cfg = search_lstm_hyperparams(train_loader, test_loader, n_features=13, ...)
"""

from __future__ import annotations

import copy
import itertools
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn
from torch.utils.data import DataLoader

from training import run_training


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

@dataclass
class TransformerSearchSpace:
    """Hyperparameter axes to search over.

    Discrete axes are combined in a cartesian product; the learning rate is
    handled separately via ternary search after SHA picks the best architecture.
    """
    d_models: list[int] = field(default_factory=lambda: [64, 128, 256])
    num_layers_list: list[int] = field(default_factory=lambda: [2, 4])
    patch_sizes: list[int] = field(default_factory=lambda: [3, 5])
    dropouts: list[float] = field(default_factory=lambda: [0.1, 0.2])

    # Initial LR grid (log-spaced) fed into SHA; ternary search refines afterwards
    lr_lo: float = 1e-4
    lr_hi: float = 5e-3
    lr_points: int = 2  # 2 = {lr_lo, lr_hi}; ternary search fills the middle

    # SHA settings
    min_epochs: int = 10   # budget per config in round 0
    max_epochs: int = 200  # cap for final training after search
    eta: int = 3           # elimination factor: keep top 1/eta each round

    # Ternary LR search settings (after SHA)
    lr_search_iters: int = 4   # log-space bisection steps
    lr_search_epochs: int = 50 # epochs per LR candidate during ternary search


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nhead_for(d_model: int) -> int:
    """Largest power-of-2 ≤ 8 that divides d_model."""
    nhead = min(8, d_model)
    while d_model % nhead != 0:
        nhead //= 2
    return nhead


def _make_configs(space: TransformerSearchSpace) -> list[dict]:
    """Cartesian product of discrete axes × initial LR grid."""
    lrs = [
        math.exp(
            math.log(space.lr_lo)
            + i * (math.log(space.lr_hi) - math.log(space.lr_lo))
            / max(space.lr_points - 1, 1)
        )
        for i in range(space.lr_points)
    ]
    configs = []
    for d_model, num_layers, patch_size, dropout, lr in itertools.product(
        space.d_models, space.num_layers_list, space.patch_sizes, space.dropouts, lrs
    ):
        configs.append({
            "d_model": d_model,
            "nhead": _nhead_for(d_model),
            "num_layers": num_layers,
            "patch_size": patch_size,
            "dropout": dropout,
            "lr": lr,
        })
    return configs


def _build_model(
    cfg: dict,
    n_features: int,
    output_dim: int,
    horizon: int,
    seq_len: int,
) -> nn.Module:
    """Instantiate a TransformerModel from a config dict."""
    from weather_transformer import TransformerModel
    return TransformerModel(
        input_dim=n_features,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        output_dim=output_dim,
        horizon=horizon,
        patch_size=cfg["patch_size"],
        seq_len=seq_len,
    )


def _train_config(
    cfg: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    build_fn: Callable,
    budget: int,
    patience_ratio: float = 0.35,
    checkpoint_prefix: str = "sha_ckpt",
    cfg_idx: int = 0,
) -> float:
    """Train a single config for `budget` epochs; returns best val_loss."""
    model = build_fn(cfg)
    patience = max(3, int(budget * patience_ratio))
    ckpt = f"{checkpoint_prefix}_{cfg_idx}.pt"
    val_loss = run_training(
        model, train_loader, test_loader,
        lr=cfg["lr"],
        max_epochs=budget,
        patience=patience,
        checkpoint_path=ckpt,
        grad_clip=1.0,
    )
    return val_loss


# ---------------------------------------------------------------------------
# Successive Halving
# ---------------------------------------------------------------------------

def successive_halving(
    configs: list[dict],
    train_loader: DataLoader,
    test_loader: DataLoader,
    build_fn: Callable,
    *,
    min_epochs: int = 10,
    max_epochs: int = 200,
    eta: int = 3,
) -> dict:
    """Successive Halving Algorithm (SHA).

    Trains all configs for `min_epochs`, keeps top 1/eta, trains those for
    min_epochs*eta, and so on.  Returns the surviving config dict.

    Binary-search intuition: each elimination round is analogous to discarding
    the "wrong half" of the search space based on observed performance.

    Args:
        configs:     candidate hyperparameter dicts.
        build_fn:    (cfg) -> nn.Module factory.
        min_epochs:  initial budget per config.
        max_epochs:  hard cap (last round won't exceed this).
        eta:         elimination factor; 3 means keep top 1/3 each round.

    Returns:
        Best surviving config dict.
    """
    n = len(configs)
    n_rounds = math.ceil(math.log(n, eta)) if n > 1 else 0
    active_idx = list(range(n))

    scores: dict[int, float] = {}

    for rnd in range(n_rounds + 1):
        budget = min(max_epochs, min_epochs * (eta ** rnd))
        if len(active_idx) <= 1:
            break

        print(f"\n[SHA] Round {rnd}: {len(active_idx)} configs × {budget} epochs")

        round_scores: list[tuple[float, int]] = []
        for cfg_idx in active_idx:
            val = _train_config(
                configs[cfg_idx], train_loader, test_loader, build_fn,
                budget=budget, cfg_idx=cfg_idx,
            )
            scores[cfg_idx] = val
            round_scores.append((val, cfg_idx))
            cfg = configs[cfg_idx]
            print(
                f"  [{cfg_idx:>3}] val={val:.5f}  d_model={cfg['d_model']}  "
                f"layers={cfg['num_layers']}  patch={cfg['patch_size']}  "
                f"dropout={cfg['dropout']}  lr={cfg['lr']:.2e}"
            )

        round_scores.sort(key=lambda t: t[0])
        n_keep = max(1, len(active_idx) // eta)
        eliminated = [idx for _, idx in round_scores[n_keep:]]
        active_idx = [idx for _, idx in round_scores[:n_keep]]

        # Clean up checkpoint files for eliminated configs
        for idx in eliminated:
            ckpt = f"sha_ckpt_{idx}.pt"
            if os.path.exists(ckpt):
                os.remove(ckpt)

    best_idx = active_idx[0]
    best_cfg = configs[best_idx]
    print(f"\n[SHA] Best config: {best_cfg}  val_loss={scores.get(best_idx, '?'):.5f}")

    # Clean up remaining SHA checkpoints
    for idx in active_idx:
        ckpt = f"sha_ckpt_{idx}.pt"
        if os.path.exists(ckpt):
            os.remove(ckpt)

    return best_cfg


# ---------------------------------------------------------------------------
# Ternary search on learning rate
# ---------------------------------------------------------------------------

def ternary_search_lr(
    base_cfg: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    build_fn: Callable,
    *,
    lo: float | None = None,
    hi: float | None = None,
    n_iter: int = 4,
    epochs_per_trial: int = 50,
) -> dict:
    """Ternary search on log-LR space around the SHA winner's learning rate.

    Each iteration evaluates three LR candidates (lo, geometric-mid, hi) and
    narrows the bracket to the inner 2/3 around the best.  This is the
    binary-search equivalent for unimodal continuous objectives.

    O(n_iter × 3) model trainings vs O(N) for a dense grid.

    Args:
        base_cfg:   winning config from SHA (architecture params reused).
        lo / hi:    search bracket; defaults to winner_lr / 10 … winner_lr * 10.
        n_iter:     number of bisection steps.
        epochs_per_trial: training budget per LR candidate.

    Returns:
        Updated config dict with the refined learning rate.
    """
    winner_lr = base_cfg["lr"]
    lo = lo if lo is not None else winner_lr / 10
    hi = hi if hi is not None else winner_lr * 10
    patience = max(5, epochs_per_trial // 4)

    print(f"\n[LR Search] Ternary search: [{lo:.2e}, {hi:.2e}]  ({n_iter} iters)")

    for iteration in range(n_iter):
        mid = math.exp((math.log(lo) + math.log(hi)) / 2)  # geometric midpoint
        candidates = [lo, mid, hi]
        losses = []

        for lr in candidates:
            cfg = copy.deepcopy(base_cfg)
            cfg["lr"] = lr
            model = build_fn(cfg)
            val = run_training(
                model, train_loader, test_loader,
                lr=lr, max_epochs=epochs_per_trial, patience=patience,
                checkpoint_path=f"lr_search_{iteration}.pt",
                grad_clip=1.0,
            )
            losses.append(val)
            print(f"  iter {iteration}: lr={lr:.2e} → val={val:.5f}")

        best_i = losses.index(min(losses))
        if best_i == 0:       # lo wins → next bracket is [lo, mid]
            hi = mid
        elif best_i == 2:     # hi wins → next bracket is [mid, hi]
            lo = mid
        else:                 # mid wins → narrow symmetrically
            lo = math.exp((math.log(lo) + math.log(mid)) / 2)
            hi = math.exp((math.log(mid) + math.log(hi)) / 2)

        print(f"  → bracket [{lo:.2e}, {hi:.2e}]")

    # Clean up temporary checkpoints
    for i in range(n_iter):
        ckpt = f"lr_search_{i}.pt"
        if os.path.exists(ckpt):
            os.remove(ckpt)

    best_lr = math.exp((math.log(lo) + math.log(hi)) / 2)
    result = copy.deepcopy(base_cfg)
    result["lr"] = best_lr
    print(f"\n[LR Search] Best LR: {best_lr:.2e}")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_transformer_hyperparams(
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    n_features: int,
    output_dim: int,
    horizon: int,
    seq_len: int,
    space: TransformerSearchSpace | None = None,
) -> dict:
    """Full search: Successive Halving → ternary LR refinement.

    Returns a config dict ready to be passed to TransformerModel:
        {d_model, nhead, num_layers, patch_size, dropout, lr}

    Example
    -------
    >>> cfg = search_transformer_hyperparams(
    ...     train_loader, test_loader,
    ...     n_features=13, output_dim=11, horizon=3, seq_len=30,
    ... )
    >>> model = TransformerModel(input_dim=13, **cfg, output_dim=11, horizon=3)
    """
    if space is None:
        space = TransformerSearchSpace()

    def build_fn(cfg: dict) -> nn.Module:
        return _build_model(cfg, n_features, output_dim, horizon, seq_len)

    configs = _make_configs(space)
    print(f"[Search] {len(configs)} candidate configs, SHA eta={space.eta}, "
          f"min_epochs={space.min_epochs}, max_epochs={space.max_epochs}")

    # Phase 1: Successive Halving over architecture × initial LR grid
    best_cfg = successive_halving(
        configs, train_loader, test_loader, build_fn,
        min_epochs=space.min_epochs,
        max_epochs=space.max_epochs,
        eta=space.eta,
    )

    # Phase 2: Ternary search on log-LR around the winning config's LR
    best_cfg = ternary_search_lr(
        best_cfg, train_loader, test_loader, build_fn,
        n_iter=space.lr_search_iters,
        epochs_per_trial=space.lr_search_epochs,
    )

    print(f"\n[Search] Final config: {best_cfg}")
    return best_cfg


# ---------------------------------------------------------------------------
# LSTM search space + builder
# ---------------------------------------------------------------------------

@dataclass
class LSTMSearchSpace:
    """Hyperparameter axes to search over for the LSTM model."""
    hidden_dims: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    num_layers_list: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    dropouts: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])

    # Initial LR grid (log-spaced) fed into SHA
    lr_lo: float = 5e-4
    lr_hi: float = 3e-3
    lr_points: int = 2

    # SHA settings
    min_epochs: int = 10
    max_epochs: int = 1000
    eta: int = 3

    # Ternary LR search settings (after SHA)
    lr_search_iters: int = 4
    lr_search_epochs: int = 80


def _make_lstm_configs(space: LSTMSearchSpace) -> list[dict]:
    """Cartesian product of discrete LSTM axes × initial LR grid."""
    lrs = [
        math.exp(
            math.log(space.lr_lo)
            + i * (math.log(space.lr_hi) - math.log(space.lr_lo))
            / max(space.lr_points - 1, 1)
        )
        for i in range(space.lr_points)
    ]
    configs = []
    for hidden_dim, num_layers, dropout, lr in itertools.product(
        space.hidden_dims, space.num_layers_list, space.dropouts, lrs
    ):
        configs.append({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
        })
    return configs


def _build_lstm(cfg: dict, n_features: int, output_dim: int, horizon: int) -> nn.Module:
    from weather_LSTM import LSTMModel
    return LSTMModel(
        input_dim=n_features,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        output_dim=output_dim,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# LSTM public API
# ---------------------------------------------------------------------------

def search_lstm_hyperparams(
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    n_features: int,
    output_dim: int,
    horizon: int,
    space: LSTMSearchSpace | None = None,
) -> dict:
    """Full LSTM search: Successive Halving → ternary LR refinement.

    Replaces the 360-config × 1000-epoch full grid with SHA, giving a
    typical speedup of 30-100× for the same search space.

    Returns a config dict with keys: {hidden_dim, num_layers, dropout, lr}

    Example
    -------
    >>> cfg = search_lstm_hyperparams(
    ...     train_loader, test_loader,
    ...     n_features=13, output_dim=11, horizon=3,
    ... )
    >>> model = LSTMModel(input_dim=13, output_dim=11, horizon=3, **cfg)
    """
    if space is None:
        space = LSTMSearchSpace()

    def build_fn(cfg: dict) -> nn.Module:
        return _build_lstm(cfg, n_features, output_dim, horizon)

    configs = _make_lstm_configs(space)
    print(f"[LSTM Search] {len(configs)} candidate configs, SHA eta={space.eta}, "
          f"min_epochs={space.min_epochs}, max_epochs={space.max_epochs}")

    # Phase 1: Successive Halving
    best_cfg = successive_halving(
        configs, train_loader, test_loader, build_fn,
        min_epochs=space.min_epochs,
        max_epochs=space.max_epochs,
        eta=space.eta,
    )

    # Phase 2: Ternary LR search
    best_cfg = ternary_search_lr(
        best_cfg, train_loader, test_loader, build_fn,
        n_iter=space.lr_search_iters,
        epochs_per_trial=space.lr_search_epochs,
    )

    print(f"\n[LSTM Search] Final config: {best_cfg}")
    return best_cfg
