# Weather Forecasting — Any Location

## Project Overview

PyTorch deep learning for daily weather forecasting at any lat/lon coordinate.
Models predict **11 meteorological variables 3 days ahead** using a 30-day input window.
Data: Meteostat daily station observations, 1970-2024.

## Repository Layout

```
stations.py             — WeatherStation class, haversine, ring+segment coordinate generation.
data_pipeline.py        — Data fetching, feature engineering (wdir→sin/cos, snow NaN→0),
                          normalization (MinMaxScaler), inverse transform, train/test split,
                          prepare_data() convenience function.
dataset.py              — SequenceDataset (sliding window), Params, build_loaders,
                          hyperparameter_grid.
training.py             — run_training (early stopping), predict_and_compare,
                          evaluate_full_test.
weather_common.py       — Re-export facade: imports & re-exports everything from the four
                          modules above for backward compatibility.
weather_LSTM.py         — LSTM model class + run_lstm(). Imports from individual modules.
weather_transformer.py  — Transformer model class + run_transformer(). Imports from individual modules.
tests/test_common.py    — Unit tests (synthetic data, no API calls).
tests/test_forecasting.py — Integration tests (Meteostat API, real training).
```

## Multi-Station Ring+Segment Search

Auxiliary stations are selected by generating coordinate sample points on concentric
circles around the target location:

- `n_rings` (0–1000): number of equally-spaced concentric circles from centre to `max_radius_km`.
  0 = use only the target coordinate's nearest station (no auxiliaries).
- `n_segments` (≥1): number of equally-spaced angular positions per ring.

Total sample points = `n_rings × n_segments`. For each point, the nearest Meteostat
station is selected (duplicates are deduplicated).

Example: `n_rings=3, n_segments=4, max_radius_km=90` → 3 circles at 30/60/90 km,
4 compass points each → up to 12 auxiliary stations.

## Data Pipeline (canonical order)

```python
primary = WeatherStation(lat, lon)
aux = primary.find_nearby(max_radius_km, n_rings, n_segments)
df = fetch_multi_station_data(primary, aux, start, end)
df_scaled, scaler = normalize(df)          # MinMaxScaler fit on train; saves scaler.pkl
train_loader, test_loader = build_loaders(df_scaled, params)
```

## Target Variables (fixed order — do not reorder)

```python
["tavg", "tmin", "tmax", "prcp", "snow", "wdir_sin", "wdir_cos", "wspd", "wpgt", "pres", "tsun"]
```

Positional order matters: inverse-transform logic relies on it.

## Key Design Decisions

- **L1Loss (MAE)** — more robust than MSE for precipitation and wind outliers.
- **Early stopping** (patience 30 for LSTM, 15 for transformer).
- **Scaler fit on train split only** — never re-fit or apply to full dataset before splitting.
- **Sin/cos columns skipped during scaling** — already in [-1, 1].
- **Snow NaN → 0**, all other NaN rows dropped.
- **Batch size 64**, `shuffle=True` on train, `shuffle=False` on test.

## Coding Conventions

- Model classes: `nn.Module` subclasses, `forward()` takes `(B, seq_len, n_features)` → `(B, horizon, output_dim)`.
- Dataset: `SequenceDataset(Dataset)` — sliding window over a scaled DataFrame/ndarray.
- Preprocessing in standalone functions (not inside classes) — keeps them reusable across model files.
- Model files import from individual modules (`data_pipeline`, `dataset`, `training`), not from `weather_common.py`.
- `weather_common.py` exists only as a backward-compatible re-export facade.
- Serialise scalers with `joblib`; save model checkpoints with `torch.save(model.state_dict(), path)`.
- Do not use `sklearn.model_selection.train_test_split` — use the custom time-ordered split to prevent leakage.

## Running

```bash
python weather_LSTM.py          # Sanity test by default (fast)
python weather_transformer.py   # Sanity test by default (fast)
pytest tests/test_common.py     # Unit tests (no network)
pytest -m integration tests/    # Integration tests (Meteostat API, slow)
```

## Dependencies

```
torch  meteostat (<2.0)  pandas  numpy  scikit-learn  matplotlib  joblib  pytest
```

## Do Not Commit

`scaler.pkl`, `best_model.pt`, any `*.pt` or `*.pkl` model artefacts (in `.gitignore`).
