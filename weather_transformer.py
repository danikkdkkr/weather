"""Transformer weather forecasting model — modern architecture.

Key improvements over vanilla transformer:
- RevIN (Reversible Instance Normalization): handles per-sequence distribution shift
- PatchEmbedding: splits 30-day window into temporal patches → fewer attention tokens,
  better local semantics, quadratically less attention compute
- Pre-LayerNorm: more stable gradients than post-norm
- SwiGLU FFN: gated activation (SiLU ⊙ linear) — outperforms ReLU/GELU in practice
- Learned positional encoding: simpler and often better than sinusoidal for short seqs
- Mean pooling over patches: uses all patch representations, not just the last token
"""

from __future__ import annotations

from datetime import datetime

import torch
import torch.nn as nn

from data_pipeline import TARGET_COLS, prepare_data
from dataset import Params, build_loaders
from training import evaluate_full_test, predict_and_compare, run_training


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    Normalises each feature independently per sequence (removes instance-level
    mean and std) before the model, then restores them after.  Handles
    distribution shift across different weather regimes without re-fitting the
    MinMaxScaler.

    Args:
        num_features: number of features to normalise (= output_dim = 11).
        eps: stability epsilon for std.
        affine: learnable per-feature scale/shift after normalisation.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, T, C) → normalised x, (mean, std) needed for denormalisation."""
        mean = x.mean(dim=1, keepdim=True)               # (B, 1, C)
        std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps  # (B, 1, C)
        x = (x - mean) / std
        weight, bias = self.weight, self.bias
        if weight is not None and bias is not None:
            x = x * weight + bias
        return x, (mean, std)

    def denormalize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, H, C) → denormalised back to MinMaxScaler space."""
        weight, bias = self.weight, self.bias
        if weight is not None and bias is not None:
            x = (x - bias) / (weight + self.eps)
        return x * std + mean


class PatchEmbedding(nn.Module):
    """Temporal patch embedding (PatchTST-style, Nie et al., ICLR 2023).

    Splits the time axis into non-overlapping patches and projects each flattened
    patch (patch_size × input_dim values) into d_model via a linear layer.
    Benefits:
      - Attention complexity drops from O(T²) to O((T/P)²)
      - Each token carries local temporal context (P days of all features)
      - Longer effective receptive field per token
    """

    def __init__(self, input_dim: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(input_dim * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, n_patches, d_model)."""
        B, T, C = x.shape
        n_patches = T // self.patch_size
        x = x[:, : n_patches * self.patch_size, :]       # trim to exact multiple
        x = x.reshape(B, n_patches, self.patch_size * C)  # (B, n_patches, P*C)
        return self.proj(x)                               # (B, n_patches, d_model)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020).

    FFN(x) = (SiLU(x·W1) ⊙ (x·W2)) · W3

    Uses ~2/3 of the nominal feedforward width so total parameter count matches
    a standard FFN.  Outperforms ReLU and GELU in most transformer benchmarks
    (PaLM, LLaMA, etc.).  No biases in the gate layers, following convention.
    """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        # 2/3 width keeps param count similar to a vanilla FFN at dim_feedforward
        hidden = int(dim_feedforward * 2 / 3)
        hidden = (hidden + 7) // 8 * 8   # round to multiple of 8 for efficiency
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate (SiLU branch)
        self.w2 = nn.Linear(d_model, hidden, bias=False)  # value branch
        self.w3 = nn.Linear(hidden, d_model, bias=False)  # output projection
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(self.act(self.w1(x)) * self.w2(x)))


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with SwiGLU FFN.

    Pre-LayerNorm (norm before attention/FFN) is more stable than post-norm
    (the original "Attention is All You Need" layout), especially with deeper
    networks or smaller datasets.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = SwiGLUFFN(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        # Pre-norm SwiGLU FFN with residual
        h = self.norm2(x)
        h = self.ffn(h)
        return x + h


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """Modern weather forecasting transformer.

    Pipeline:
        input (B, T, input_dim)
        → RevIN normalise target features (first output_dim cols)
        → PatchEmbedding  →  (B, n_patches, d_model)
        → learned positional encoding
        → N × EncoderLayer (pre-norm + SwiGLU)
        → final LayerNorm
        → mean-pool over patches  →  (B, d_model)
        → linear head  →  (B, horizon, output_dim)
        → RevIN denormalise
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
        horizon: int,
        patch_size: int = 5,
        seq_len: int = 30,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.horizon = horizon
        self.patch_size = patch_size

        # RevIN on target features (first output_dim cols — always the 11 TARGET_COLS)
        self.revin = RevIN(output_dim)

        # Patch embedding: (B, T, input_dim) → (B, n_patches, d_model)
        self.patch_embed = PatchEmbedding(input_dim, patch_size, d_model)

        # Learned positional encoding; large table to handle variable seq_len
        self.pos_embed = nn.Embedding(512, d_model)

        # Transformer encoder
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, d_model * 4, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Output head: d_model → horizon × output_dim
        self.fc = nn.Linear(d_model, horizon * output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers, small normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, horizon, output_dim)."""
        B = x.size(0)

        # RevIN: normalise target features, keep aux features unchanged
        x_target = x[:, :, : self.output_dim]
        x_target_norm, (mean, std) = self.revin.normalize(x_target)
        if x.size(-1) > self.output_dim:
            x = torch.cat([x_target_norm, x[:, :, self.output_dim :]], dim=-1)
        else:
            x = x_target_norm

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, d_model)

        # Positional encoding
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embed(positions)

        # Transformer
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Mean pooling over all patch representations
        x = x.mean(dim=1)  # (B, d_model)

        # Project to output and reshape
        out = self.fc(x).view(B, self.horizon, self.output_dim)

        # RevIN: denormalise back to MinMaxScaler space
        out = self.revin.denormalize(out, mean, std)
        return out


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_transformer(
    target_lat: float = 48.78,
    target_lon: float = 9.18,
    max_radius_km: float = 50,
    n_rings: int = 0,
    n_segments: int = 4,
    sanity_test: bool = True,
    search_hyperparams: bool = True,
) -> tuple:
    """Full Transformer pipeline: data fetch → [hyperparameter search] → train → evaluate.

    Args:
        sanity_test: if True, uses tiny model + few epochs for a quick smoke test.
        search_hyperparams: if True, runs Successive Halving to find the best config
            before final training.  Ignored when sanity_test=True.
    """
    start = datetime(1970, 1, 1)
    end = datetime(2024, 12, 31)

    df_scaled, scaler, primary, aux = prepare_data(
        target_lat, target_lon, start, end,
        max_radius_km=max_radius_km,
        n_rings=n_rings,
        n_segments=n_segments,
    )

    n_features = df_scaled.shape[1]
    output_dim = len(TARGET_COLS)
    seq_len = 30
    horizon = 3

    if sanity_test:
        # Fast smoke-test: tiny model, very few epochs
        cfg = dict(d_model=64, nhead=8, num_layers=2, dropout=0.1,
                   patch_size=5, lr=1e-3)
        max_epochs = 5
        patience = 3
    elif search_hyperparams:
        from hyperparameter_search import search_transformer_hyperparams
        params_tmp = Params(seq_len=seq_len, horizon=horizon,
                            hidden_dim=64, num_layers=2, lr=1e-3, dropout=0.1)
        train_loader, test_loader = build_loaders(df_scaled, params_tmp)
        cfg = search_transformer_hyperparams(
            train_loader, test_loader,
            n_features=n_features,
            output_dim=output_dim,
            horizon=horizon,
            seq_len=seq_len,
        )
        max_epochs = 500
        patience = 15
    else:
        cfg = dict(d_model=128, nhead=8, num_layers=4, dropout=0.1,
                   patch_size=5, lr=5e-4)
        max_epochs = 500
        patience = 15

    params = Params(
        seq_len=seq_len, horizon=horizon,
        hidden_dim=cfg["d_model"],
        num_layers=cfg["num_layers"],
        lr=cfg["lr"],
        dropout=cfg["dropout"],
    )
    train_loader, test_loader = build_loaders(df_scaled, params)

    model = TransformerModel(
        input_dim=n_features,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dropout=cfg["dropout"],
        output_dim=output_dim,
        horizon=horizon,
        patch_size=int(cfg["patch_size"]),
        seq_len=seq_len,
    )

    best_val = run_training(
        model, train_loader, test_loader,
        lr=cfg["lr"], max_epochs=max_epochs, patience=patience,
        grad_clip=1.0,
    )
    print(f"Best val_loss={best_val:.5f}")

    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    model.eval()

    predict_and_compare(model, scaler, test_loader, params)
    maes = evaluate_full_test(model, scaler, test_loader)
    print("\n--- Full test MAE ---")
    for col, mae in maes.items():
        print(f"  {col:<10s}: {mae:.3f}")

    return model, scaler, params, maes


if __name__ == "__main__":
    run_transformer(sanity_test=True)
