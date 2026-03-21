Scaffold a new weather forecasting model file for this project.

Ask the user:
1. Model architecture name (e.g. "TCN", "Informer", "PatchTST")
2. Any architecture-specific parameters (e.g. kernel size for TCN, attention heads for Informer)

Then create a new file `weather_<name>.py` that:
- Imports the shared data pipeline from `weather_common.py` (TARGET_COLS, Params, build_loaders, prepare_data, run_training, evaluate_full_test, predict_and_compare)
- Defines the model class as `nn.Module` with `forward(x)`: `(B, seq_len, n_features)` → `(B, horizon, output_dim)`
- Defines a `run_<name>()` function following the same pattern as `run_lstm()` and `run_transformer()`
- Accepts `target_lat`, `target_lon`, `max_radius_km`, `n_rings`, `n_segments`, `sanity_test` parameters
- Includes `if __name__ == "__main__": run_<name>(sanity_test=True)`

Follow the conventions in CLAUDE.md.
