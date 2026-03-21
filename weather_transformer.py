"""Transformer weather forecasting model."""

from datetime import datetime

import torch
import torch.nn as nn
import math

from weather_common import (
    TARGET_COLS,
    Params,
    build_loaders,
    evaluate_full_test,
    predict_and_compare,
    prepare_data,
    run_training,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, output_dim, horizon):
        super().__init__()
        self.output_dim = output_dim
        self.horizon = horizon

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon * output_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        out = x[:, -1, :]  # last time step
        out = self.fc(out)
        return out.view(B, self.horizon, self.output_dim)


def run_transformer(
    target_lat: float = 48.78,
    target_lon: float = 9.18,
    max_radius_km: float = 50,
    n_rings: int = 0,
    n_segments: int = 4,
    sanity_test: bool = True,
):
    """Full Transformer pipeline: data fetch → train → evaluate."""

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

    params = Params(
        seq_len=30, horizon=3,
        hidden_dim=64,  # used as d_model
        num_layers=2 if sanity_test else 4,
        lr=1e-3,
        dropout=0.1,
    )
    max_epochs = 5 if sanity_test else 500
    patience = 3 if sanity_test else 15

    train_loader, test_loader = build_loaders(df_scaled, params)

    # nhead must divide d_model; pick largest power-of-2 ≤ 8 that divides
    d_model = params.hidden_dim
    nhead = min(8, d_model)
    while d_model % nhead != 0:
        nhead //= 2

    model = TransformerModel(
        input_dim=n_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=params.num_layers,
        dropout=params.dropout,
        output_dim=output_dim,
        horizon=params.horizon,
    )

    best_val = run_training(
        model, train_loader, test_loader,
        lr=params.lr, max_epochs=max_epochs, patience=patience,
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
