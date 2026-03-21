"""Modal cloud training for weather forecasting models.

Usage:
    # Train LSTM (full training with hyperparameter search):
    modal run modal_train.py --model lstm

    # Train Transformer:
    modal run modal_train.py --model transformer

    # Quick sanity test on Modal:
    modal run modal_train.py --model lstm --sanity-test

    # Custom location + skip hyperparameter search:
    modal run modal_train.py --model transformer --lat 52.52 --lon 13.41 --no-search

    # With auxiliary stations:
    modal run modal_train.py --model lstm --n-rings 3 --n-segments 4 --max-radius 90

    # Download results after training:
    modal volume get weather-results /lstm/ ./results/lstm/
    modal volume get weather-results /transformer/ ./results/transformer/

    # List all results:
    modal volume ls weather-results
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App("weather-forecasting")

# Persistent volume for models, scalers, and plots
volume = modal.Volume.from_name("weather-results", create_if_missing=True)
RESULTS_DIR = "/results"

# Image with all dependencies + project source code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "meteostat<2.0",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "joblib",
    )
    .add_local_file("stations.py", "/root/stations.py")
    .add_local_file("data_pipeline.py", "/root/data_pipeline.py")
    .add_local_file("dataset.py", "/root/dataset.py")
    .add_local_file("training.py", "/root/training.py")
    .add_local_file("hyperparameter_search.py", "/root/hyperparameter_search.py")
    .add_local_file("weather_LSTM.py", "/root/weather_LSTM.py")
    .add_local_file("weather_transformer.py", "/root/weather_transformer.py")
    .add_local_file("plotting.py", "/root/plotting.py")
)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={RESULTS_DIR: volume},
    gpu="any",
    timeout=7200,
)
def train_lstm(
    target_lat: float = 48.78,
    target_lon: float = 9.18,
    max_radius_km: float = 50,
    n_rings: int = 0,
    n_segments: int = 4,
    sanity_test: bool = False,
    search_hyperparams: bool = True,
) -> dict:
    """Train LSTM model on Modal, save artifacts and plots to volume."""
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    os.chdir("/root")

    import torch
    from data_pipeline import TARGET_COLS, prepare_data
    from dataset import Params, build_loaders
    from plotting import generate_all_plots
    from training import evaluate_full_test, run_training
    from weather_LSTM import LSTMModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from datetime import datetime
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
    seq_len, horizon = 30, 3

    if sanity_test:
        cfg = dict(hidden_dim=32, num_layers=1, dropout=0.0, lr=1e-3)
        max_epochs, patience = 5, 3
    elif search_hyperparams:
        from hyperparameter_search import search_lstm_hyperparams
        params_tmp = Params(seq_len=seq_len, horizon=horizon,
                            hidden_dim=32, num_layers=1, lr=1e-3, dropout=0.0)
        train_loader, test_loader = build_loaders(df_scaled, params_tmp)
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

    model = LSTMModel(
        input_dim=n_features,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        output_dim=output_dim,
        horizon=horizon,
    )

    best_loss = run_training(
        model, train_loader, test_loader,
        lr=cfg["lr"], max_epochs=max_epochs, patience=patience,
    )
    print(f"\nBest val_loss={best_loss:.5f}  "
          f"hid={cfg['hidden_dim']} layers={cfg['num_layers']} "
          f"lr={cfg['lr']:.2e} drop={cfg['dropout']}")

    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    model.eval()

    maes = evaluate_full_test(model, scaler, test_loader)
    print("\n--- Full test MAE ---")
    for col, mae in maes.items():
        print(f"  {col:<10s}: {mae:.3f}")

    # --- Save artifacts to volume ---
    out_dir = f"{RESULTS_DIR}/lstm"
    os.makedirs(out_dir, exist_ok=True)

    import shutil
    import joblib
    torch.save(model.state_dict(), f"{out_dir}/best_model.pt")
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")

    results = {
        "model": "lstm",
        "config": cfg,
        "best_val_loss": best_loss,
        "maes": maes,
        "location": {"lat": target_lat, "lon": target_lon},
        "n_features": n_features,
        "n_stations": 1 + len(aux),
        "dataset_days": len(df_scaled),
    }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Generate plots ---
    plot_dir = f"{out_dir}/plots"
    generate_all_plots(model, scaler, test_loader, plot_dir, model_name="LSTM")

    volume.commit()
    print(f"\nArtifacts saved to volume 'weather-results' at /lstm/")
    print("Download with: modal volume get weather-results /lstm/ ./results/lstm/")
    return results


@app.function(
    image=image,
    volumes={RESULTS_DIR: volume},
    gpu="any",
    timeout=7200,
)
def train_transformer(
    target_lat: float = 48.78,
    target_lon: float = 9.18,
    max_radius_km: float = 50,
    n_rings: int = 0,
    n_segments: int = 4,
    sanity_test: bool = False,
    search_hyperparams: bool = True,
) -> dict:
    """Train Transformer model on Modal, save artifacts and plots to volume."""
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    os.chdir("/root")

    import torch
    from data_pipeline import TARGET_COLS, prepare_data
    from dataset import Params, build_loaders
    from plotting import generate_all_plots
    from training import evaluate_full_test, run_training
    from weather_transformer import TransformerModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from datetime import datetime
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
    seq_len, horizon = 30, 3

    if sanity_test:
        cfg = dict(d_model=64, nhead=8, num_layers=2, dropout=0.1,
                   patch_size=5, lr=1e-3)
        max_epochs, patience = 5, 3
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
        max_epochs, patience = 500, 15
    else:
        cfg = dict(d_model=128, nhead=8, num_layers=4, dropout=0.1,
                   patch_size=5, lr=5e-4)
        max_epochs, patience = 500, 15

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
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        output_dim=output_dim,
        horizon=horizon,
        patch_size=cfg["patch_size"],
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

    maes = evaluate_full_test(model, scaler, test_loader)
    print("\n--- Full test MAE ---")
    for col, mae in maes.items():
        print(f"  {col:<10s}: {mae:.3f}")

    # --- Save artifacts to volume ---
    out_dir = f"{RESULTS_DIR}/transformer"
    os.makedirs(out_dir, exist_ok=True)

    import joblib
    torch.save(model.state_dict(), f"{out_dir}/best_model.pt")
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")

    results = {
        "model": "transformer",
        "config": cfg,
        "best_val_loss": best_val,
        "maes": maes,
        "location": {"lat": target_lat, "lon": target_lon},
        "n_features": n_features,
        "n_stations": 1 + len(aux),
        "dataset_days": len(df_scaled),
    }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Generate plots ---
    plot_dir = f"{out_dir}/plots"
    generate_all_plots(model, scaler, test_loader, plot_dir, model_name="Transformer")

    volume.commit()
    print(f"\nArtifacts saved to volume 'weather-results' at /transformer/")
    print("Download with: modal volume get weather-results /transformer/ ./results/transformer/")
    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str = "lstm",
    lat: float = 48.78,
    lon: float = 9.18,
    max_radius: float = 50,
    n_rings: int = 0,
    n_segments: int = 4,
    sanity_test: bool = False,
    no_search: bool = False,
):
    """Train a weather model on Modal.

    Args:
        model: "lstm" or "transformer" (or "both").
        lat/lon: target location coordinates.
        max_radius: radius in km for auxiliary station search.
        n_rings: number of concentric rings for station search.
        n_segments: angular segments per ring.
        sanity_test: quick smoke test with tiny model.
        no_search: skip hyperparameter search, use defaults.
    """
    import json

    search = not no_search

    if model in ("lstm", "both"):
        print("=" * 60)
        print("  Training LSTM on Modal")
        print("=" * 60)
        result = train_lstm.remote(
            target_lat=lat, target_lon=lon,
            max_radius_km=max_radius,
            n_rings=n_rings, n_segments=n_segments,
            sanity_test=sanity_test,
            search_hyperparams=search,
        )
        print("\n--- LSTM Results ---")
        print(json.dumps(result, indent=2))

    if model in ("transformer", "both"):
        print("=" * 60)
        print("  Training Transformer on Modal")
        print("=" * 60)
        result = train_transformer.remote(
            target_lat=lat, target_lon=lon,
            max_radius_km=max_radius,
            n_rings=n_rings, n_segments=n_segments,
            sanity_test=sanity_test,
            search_hyperparams=search,
        )
        print("\n--- Transformer Results ---")
        print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("  Done! Download results with:")
    print("    modal volume get weather-results / ./results/")
    print("=" * 60)
