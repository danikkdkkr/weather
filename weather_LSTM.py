"""LSTM weather forecasting model with efficient hyperparameter search."""

from datetime import datetime

import torch
import torch.nn as nn

from data_pipeline import TARGET_COLS, prepare_data
from dataset import Params, build_loaders
from training import evaluate_full_test, predict_and_compare, run_training


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
    search_hyperparams: bool = True,
):
    """Full LSTM pipeline: data fetch → [hyperparameter search] → train → evaluate.

    Args:
        sanity_test: if True, uses a tiny fixed config for a quick smoke test.
        search_hyperparams: if True, runs Successive Halving + ternary LR search
            to find the best config.  Ignored when sanity_test=True.
    """
    start = datetime(1970, 1, 1)
    end = datetime(2024, 12, 31)

    df_scaled, scaler, *_ = prepare_data(
        target_lat, target_lon, start, end,
        max_radius_km=max_radius_km,
        n_rings=n_rings,
        n_segments=n_segments,
    )

    n_features = df_scaled.shape[1]
    output_dim = len(TARGET_COLS)
    seq_len, horizon = 30, 3

    if sanity_test:
        cfg = dict(hidden_dim=32, num_layers=1, dropout=0.0, lr=1e-3)
        max_epochs, patience = 5, 3
    elif search_hyperparams:
        from hyperparameter_search import search_lstm_hyperparams
        _params_tmp = Params(seq_len=seq_len, horizon=horizon,
                             hidden_dim=32, num_layers=1, lr=1e-3, dropout=0.0)
        train_loader, test_loader = build_loaders(df_scaled, _params_tmp)
        cfg = search_lstm_hyperparams(
            train_loader, test_loader,
            n_features=n_features,
            output_dim=output_dim,
            horizon=horizon,
        )
        max_epochs, patience = 1000, 30
    else:
        cfg = dict(hidden_dim=128, num_layers=2, dropout=0.1, lr=1e-3)
        max_epochs, patience = 1000, 30

    best_params = Params(
        seq_len=seq_len, horizon=horizon,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        lr=cfg["lr"],
        dropout=cfg["dropout"],
    )
    train_loader, test_loader = build_loaders(df_scaled, best_params)

    best_model = LSTMModel(
        input_dim=n_features,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        output_dim=output_dim,
        horizon=horizon,
    )

    best_loss = run_training(
        best_model, train_loader, test_loader,
        lr=cfg["lr"], max_epochs=max_epochs, patience=patience,
    )
    print(f"\nBest val_loss={best_loss:.5f}  "
          f"hid={cfg['hidden_dim']} layers={cfg['num_layers']} "
          f"lr={cfg['lr']:.2e} drop={cfg['dropout']}")

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
