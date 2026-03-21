"""LSTM weather forecasting model with hyperparameter grid search."""

from datetime import datetime

import torch
import torch.nn as nn

from weather_common import (
    TARGET_COLS,
    Params,
    build_loaders,
    evaluate_full_test,
    hyperparameter_grid,
    predict_and_compare,
    prepare_data,
    run_training,
)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim, horizon):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, horizon * output_dim)

    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(B, self.horizon, self.output_dim)


def run_lstm(
    target_lat: float = 48.78,
    target_lon: float = 9.18,
    max_radius_km: float = 50,
    n_rings: int = 0,
    n_segments: int = 4,
    sanity_test: bool = True,
):
    """Full LSTM pipeline: data fetch → grid search → best model evaluation."""

    start = datetime(1970, 1, 1)
    end = datetime(2024, 12, 31)

    df_scaled, scaler, primary, aux = prepare_data(
        target_lat, target_lon, start, end,
        max_radius_km=max_radius_km,
        n_rings=n_rings,
        n_segments=n_segments,
    )

    if sanity_test:
        grid = hyperparameter_grid(hidden_dims=(32,), num_layers_list=(1,),
                                   lrs=(1e-3,), dropouts=(0.0, 0.1))
    else:
        grid = hyperparameter_grid(
            hidden_dims=(20, 32, 48, 64, 84, 128),
            num_layers_list=(1, 2, 3, 4, 5),
            lrs=(5e-4, 1e-3, 2e-3),
            dropouts=(0.0, 0.1, 0.2, 0.3),
        )

    n_features = df_scaled.shape[1]
    output_dim = len(TARGET_COLS)
    max_epochs = 5 if sanity_test else 1000
    patience = 3 if sanity_test else 30

    results = []
    for params in grid:
        train_loader, test_loader = build_loaders(df_scaled, params)

        model = LSTMModel(
            input_dim=n_features,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            dropout=params.dropout,
            output_dim=output_dim,
            horizon=params.horizon,
        )

        loss = run_training(
            model, train_loader, test_loader,
            lr=params.lr, max_epochs=max_epochs, patience=patience,
        )

        results.append((loss, params))
        print(f"  val_loss={loss:.5f}  hid={params.hidden_dim} "
              f"layers={params.num_layers} lr={params.lr} drop={params.dropout}")

    # Reload best
    best_loss, best_params = min(results, key=lambda t: t[0])
    print(f"\nBest val_loss={best_loss:.5f}  params: hid={best_params.hidden_dim} "
          f"layers={best_params.num_layers} lr={best_params.lr} drop={best_params.dropout}")

    train_loader, test_loader = build_loaders(df_scaled, best_params)
    best_model = LSTMModel(
        input_dim=n_features,
        hidden_dim=best_params.hidden_dim,
        num_layers=best_params.num_layers,
        dropout=best_params.dropout,
        output_dim=output_dim,
        horizon=best_params.horizon,
    )
    best_model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    best_model.eval()

    predict_and_compare(best_model, scaler, test_loader, best_params)
    maes = evaluate_full_test(best_model, scaler, test_loader)
    print("\n--- Full test MAE ---")
    for col, mae in maes.items():
        print(f"  {col:<10s}: {mae:.3f}")

    return best_model, scaler, best_params, maes


if __name__ == "__main__":
    run_lstm(sanity_test=True)
