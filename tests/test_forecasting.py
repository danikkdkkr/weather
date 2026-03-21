"""Integration tests — train models on real Meteostat data for different target locations.

These tests hit the Meteostat API and run actual (short) training loops.
Mark with -m integration and skip with: pytest -m "not integration"
"""

import pytest
import torch

from weather_common import (
    TARGET_COLS,
    Params,
    build_loaders,
    evaluate_full_test,
    prepare_data,
    run_training,
)
from weather_LSTM import LSTMModel
from weather_transformer import TransformerModel

from datetime import datetime

# Shorter date range + tiny training for CI speed
_START = datetime(2015, 1, 1)
_END = datetime(2023, 12, 31)
_MAX_EPOCHS = 5
_PATIENCE = 3

# Cities to test — spread across Europe for diversity
_TARGETS = {
    "stuttgart": (48.78, 9.18),
    "munich": (48.14, 11.58),
    "berlin": (52.52, 13.40),
    "zurich": (47.38, 8.54),
    "vienna": (48.21, 16.37),
}

# Acceptable MAE thresholds per variable (in original units).
# These are generous — just checking the model produces something reasonable, not SOTA.
_MAE_THRESHOLDS = {
    "tavg": 8.0,   # °C
    "tmin": 8.0,
    "tmax": 8.0,
    "prcp": 6.0,   # mm
    "snow": 10.0,   # mm (noisy)
    "wspd": 15.0,  # km/h
    "wpgt": 30.0,  # km/h (peak gusts are noisy)
    "pres": 15.0,  # hPa
    "tsun": 200.0, # minutes (very noisy)
    # sin/cos are bounded [-1,1]; MAE should be well under 1
    "wdir_sin": 1.0,
    "wdir_cos": 1.0,
}


def _train_lstm(lat, lon, n_rings=0, n_segments=4, max_radius_km=30):
    """Helper: prepare data, train a tiny LSTM, return (model, scaler, maes)."""
    df_scaled, scaler, primary, aux = prepare_data(
        lat, lon, _START, _END,
        max_radius_km=max_radius_km,
        n_rings=n_rings,
        n_segments=n_segments,
    )

    params = Params(seq_len=30, horizon=3, hidden_dim=32, num_layers=1, lr=1e-3, dropout=0.0)
    train_loader, test_loader = build_loaders(df_scaled, params)
    n_features = df_scaled.shape[1]

    model = LSTMModel(
        input_dim=n_features,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        dropout=params.dropout,
        output_dim=len(TARGET_COLS),
        horizon=params.horizon,
    )

    run_training(
        model, train_loader, test_loader,
        lr=params.lr, max_epochs=_MAX_EPOCHS, patience=_PATIENCE,
    )

    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    model.eval()

    maes = evaluate_full_test(model, scaler, test_loader)
    return model, scaler, maes


# ---------------------------------------------------------------------------
# LSTM Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLSTMPrimaryOnly:
    """Train LSTM with no auxiliary stations (primary only)."""

    @pytest.mark.parametrize("city,coords", list(_TARGETS.items()))
    def test_mae_within_threshold(self, city, coords):
        lat, lon = coords
        _, _, maes = _train_lstm(lat, lon, n_rings=0)

        print(f"\n--- {city} (LSTM primary-only) MAEs ---")
        for col, mae in maes.items():
            threshold = _MAE_THRESHOLDS[col]
            print(f"  {col:<10s}: {mae:.3f}  (threshold {threshold})")
            assert mae < threshold, (
                f"{city}: {col} MAE {mae:.3f} exceeds threshold {threshold}"
            )


@pytest.mark.integration
class TestLSTMWithRings:
    """Train LSTM with ring+segment auxiliary stations."""

    def test_single_ring_stuttgart(self):
        lat, lon = _TARGETS["stuttgart"]
        _, _, maes = _train_lstm(lat, lon, n_rings=1, n_segments=4, max_radius_km=30)

        print("\n--- Stuttgart (LSTM 1 ring, 4 segments) MAEs ---")
        for col, mae in maes.items():
            threshold = _MAE_THRESHOLDS[col]
            print(f"  {col:<10s}: {mae:.3f}  (threshold {threshold})")
            assert mae < threshold

    def test_multi_ring_stuttgart(self):
        lat, lon = _TARGETS["stuttgart"]
        _, _, maes = _train_lstm(lat, lon, n_rings=3, n_segments=4, max_radius_km=50)

        print("\n--- Stuttgart (LSTM 3 rings, 4 segments) MAEs ---")
        for col, mae in maes.items():
            threshold = _MAE_THRESHOLDS[col]
            print(f"  {col:<10s}: {mae:.3f}  (threshold {threshold})")
            assert mae < threshold


# ---------------------------------------------------------------------------
# Transformer Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTransformerPrimaryOnly:
    """Train Transformer with no auxiliary stations."""

    def test_stuttgart(self):
        lat, lon = _TARGETS["stuttgart"]
        df_scaled, scaler, primary, aux = prepare_data(
            lat, lon, _START, _END, n_rings=0,
        )
        params = Params(seq_len=30, horizon=3, hidden_dim=32, num_layers=2, lr=1e-3, dropout=0.1)
        train_loader, test_loader = build_loaders(df_scaled, params)

        model = TransformerModel(
            input_dim=df_scaled.shape[1],
            d_model=params.hidden_dim,
            nhead=4,
            num_layers=params.num_layers,
            dropout=params.dropout,
            output_dim=len(TARGET_COLS),
            horizon=params.horizon,
        )

        run_training(model, train_loader, test_loader,
                     lr=params.lr, max_epochs=_MAX_EPOCHS, patience=_PATIENCE)
        model.load_state_dict(torch.load("best_model.pt", weights_only=True))
        model.eval()

        maes = evaluate_full_test(model, scaler, test_loader)
        print("\n--- Stuttgart (Transformer primary-only) MAEs ---")
        for col, mae in maes.items():
            threshold = _MAE_THRESHOLDS[col]
            print(f"  {col:<10s}: {mae:.3f}  (threshold {threshold})")
            assert mae < threshold


# ---------------------------------------------------------------------------
# Data Pipeline Integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDataPipelineIntegration:
    """Verify the data pipeline produces valid shapes for different configs."""

    @pytest.mark.parametrize("n_rings,n_segments", [(0, 4), (1, 4), (2, 6), (3, 8)])
    def test_data_shape_varies_with_config(self, n_rings, n_segments):
        lat, lon = _TARGETS["zurich"]
        df_scaled, scaler, primary, aux = prepare_data(
            lat, lon, _START, _END,
            max_radius_km=40,
            n_rings=n_rings,
            n_segments=n_segments,
        )
        # Primary has 11 target cols + day_sin + day_cos = 13 minimum
        assert df_scaled.shape[1] >= 13
        # Should have enough data to form at least one sequence
        assert len(df_scaled) > 33  # seq_len(30) + horizon(3)
        # First 11 columns must be target cols
        assert list(df_scaled.columns[:11]) == TARGET_COLS
