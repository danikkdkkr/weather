# AGENTS.md — Weather Forecasting Project

Instructions for AI coding agents working in this repository.

## What This Repo Does

PyTorch deep learning for daily weather forecasting at any lat/lon coordinate.
Models predict 11 meteorological variables 3 days ahead using a 30-day sliding window of daily station observations from Meteostat (1970-2024).

## Repository Layout

```
weather_common.py       — Shared pipeline (ALL non-model code lives here).
weather_LSTM.py         — LSTM model class + run_lstm(). Imports from weather_common.
weather_transformer.py  — Transformer model class + run_transformer(). Imports from weather_common.
tests/test_common.py    — Unit tests (synthetic data, no API calls).
tests/test_forecasting.py — Integration tests (Meteostat API, real training).
```

## Multi-Station Ring+Segment Search

The `WeatherStation.find_nearby(max_radius_km, n_rings, n_segments)` method generates
coordinate sample points on `n_rings` concentric circles with `n_segments` angular
positions each, then finds the nearest Meteostat station to each point.

- `n_rings=0` → no auxiliaries (primary station only)
- `max_radius_km` clamped to 0–1000

## Coding Standards

- Python 3.12+, PyTorch >= 2.0
- Model classes inherit `nn.Module`; `forward(x)`: `(B, seq_len, n_features)` → `(B, horizon, output_dim)`
- All shared code (pipeline, training, evaluation) lives in `weather_common.py`
- Model files only contain the model class + a `run_<name>()` function
- Use `joblib` for scaler serialisation; `torch.save(model.state_dict(), path)` for checkpoints

## Critical Constraints

| Constraint | Reason |
|---|---|
| Train/test split is **chronological** (80/20) | Prevents temporal leakage |
| Scaler fit **on train only** | Prevents test-set leakage |
| Target column order **fixed** | `inverse_transform` relies on positional index |
| Sin/cos excluded from scaling | Already in [-1, 1] |
| No `sklearn.model_selection.train_test_split` | Use project's custom time-ordered split |

## Testing

```bash
pytest tests/test_common.py              # Unit tests (fast, no network)
pytest -m integration tests/             # Integration tests (Meteostat API, slow)
```

- Unit tests use synthetic DataFrames — never call Meteostat API
- Integration tests use short date ranges (2015-2023) and tiny models (5 epochs) for speed
- MAE thresholds are generous (sanity checks, not SOTA benchmarks)

## Pull Requests

- One logical change per PR
- If affecting training, include val loss before/after
- Never commit `*.pt` or `*.pkl` artefacts
