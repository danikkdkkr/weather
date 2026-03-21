"""Shared utilities for weather forecasting models.

Re-exports from the refactored modules so existing imports continue to work:
  stations       — WeatherStation, geo helpers
  data_pipeline  — data fetching, processing, normalization
  dataset        — SequenceDataset, Params, build_loaders
  training       — training loop, prediction, evaluation
"""

# Stations & geo helpers
from stations import (  # noqa: F401
    WeatherStation,
    destination_point as _destination_point,
    generate_ring_coords as _generate_ring_coords,
    haversine as _haversine,
)

# Data pipeline
from data_pipeline import (  # noqa: F401
    TARGET_COLS,
    fetch_multi_station_data,
    inverse_transform_cols,
    normalize,
    prepare_data,
    process_station_df,
    train_test_split,
)

# Dataset & loaders
from dataset import (  # noqa: F401
    Params,
    SequenceDataset,
    build_loaders,
    hyperparameter_grid,
)

# Training & evaluation
from training import (  # noqa: F401
    evaluate_full_test,
    predict_and_compare,
    run_training,
)
