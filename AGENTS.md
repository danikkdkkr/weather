# AGENTS.md — Weather Forecasting Project

Instructions for AI coding agents working in this repository.

## What This Repo Does

PyTorch deep learning for daily weather forecasting at any lat/lon coordinate.
Models predict 11 meteorological variables 3 days ahead using a 30-day sliding window of daily station observations from Meteostat (1970-2024).

## Repository Layout

```
stations.py             — WeatherStation class + geo helpers (haversine, ring coords).
data_pipeline.py        — Data fetch, feature engineering, normalization, splitting.
dataset.py              — SequenceDataset, Params, build_loaders, hyperparameter_grid.
training.py             — Training loop, prediction, evaluation.
weather_common.py       — Re-export facade (backward compat only — do not add new code here).
weather_LSTM.py         — LSTM model class + run_lstm().
weather_transformer.py  — Transformer model class + run_transformer().
tests/test_common.py    — Unit tests (synthetic data, no API calls).
tests/test_forecasting.py — Integration tests (Meteostat API, real training).
```

## Where to Put New Code

| Code type | File |
|---|---|
| Station discovery, geo math | `stations.py` |
| Data fetching, feature engineering, normalization | `data_pipeline.py` |
| Dataset, loaders, hyperparameter utilities | `dataset.py` |
| Training loop, evaluation, prediction | `training.py` |
| New model architecture | `weather_<name>.py` (imports from the modules above) |
| Backward-compat re-exports only | `weather_common.py` |

## Multi-Station Ring+Segment Search

`WeatherStation.find_nearby(max_radius_km, n_rings, n_segments)` generates coordinate
sample points on `n_rings` concentric circles with `n_segments` angular positions each,
then finds the nearest Meteostat station to each point.

- `n_rings=0` → no auxiliaries (primary station only)
- `max_radius_km` clamped to 0–1000

## Coding Standards

- Python 3.12+, PyTorch >= 2.0, meteostat < 2.0
- Model classes inherit `nn.Module`; `forward(x)`: `(B, seq_len, n_features)` → `(B, horizon, output_dim)`
- Import from individual modules (`stations`, `data_pipeline`, `dataset`, `training`), not from `weather_common`
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
