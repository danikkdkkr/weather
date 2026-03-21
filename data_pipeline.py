"""Data fetching, feature engineering, normalization, and splitting.

Contains: process_station_df, fetch_multi_station_data, normalize,
inverse_transform_cols, train_test_split, prepare_data.
"""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from stations import WeatherStation

TARGET_COLS = [
    "tavg", "tmin", "tmax", "prcp", "snow",
    "wdir_sin", "wdir_cos", "wspd", "wpgt", "pres", "tsun",
]


def process_station_df(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station feature engineering: wdir -> sin/cos, snow NaN -> 0, drop NaN rows."""
    df = df.copy()
    df["snow"] = df["snow"].fillna(0)
    df["wdir_sin"] = np.sin(np.deg2rad(df["wdir"]))
    df["wdir_cos"] = np.cos(np.deg2rad(df["wdir"]))
    df = df.drop(columns=["wdir"])
    df = df.dropna()
    return df


def fetch_multi_station_data(
    primary: WeatherStation,
    aux_stations: list[WeatherStation],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch and merge data from primary + auxiliary stations.

    Primary station columns are unprefixed and ordered per TARGET_COLS so that
    SequenceDataset's [:11] target slice remains correct.
    Auxiliary columns are prefixed s1_, s2_, ...
    day_sin / day_cos are appended once after the join.
    """
    primary_df = primary.fetch(start, end)
    primary_df.index = pd.to_datetime(primary_df.index)
    primary_df = process_station_df(primary_df)[TARGET_COLS]

    combined = primary_df.copy()

    for i, station in enumerate(aux_stations, start=1):
        raw = station.fetch(start, end)
        if raw.empty:
            continue
        raw.index = pd.to_datetime(raw.index)
        processed = process_station_df(raw).add_prefix(f"s{i}_")
        combined = combined.join(processed, how="inner")

    doy = combined.index.dayofyear
    combined["day_sin"] = np.sin(2 * np.pi * doy / 365)
    combined["day_cos"] = np.cos(2 * np.pi * doy / 365)

    return combined


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """MinMaxScaler([-1,1]) on non-sin/cos columns.  Saves scaler.pkl."""
    skip_cols = [c for c in df.columns if c.endswith("_sin") or c.endswith("_cos")]
    cols_to_scale = [c for c in df.columns if c not in skip_cols]
    df_scaled = df.copy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    joblib.dump(scaler, "scaler.pkl")
    return df_scaled, scaler


def inverse_transform_cols(scaler: MinMaxScaler, df_subset: pd.DataFrame) -> pd.DataFrame:
    """Inverse-transform a column subset using per-column min_/scale_ attributes."""
    fitted_cols = list(scaler.feature_names_in_)
    result = df_subset.copy()
    for col in df_subset.columns:
        if col not in fitted_cols:
            continue
        idx = fitted_cols.index(col)
        result[col] = (df_subset[col].to_numpy() - scaler.min_[idx]) / scaler.scale_[idx]
    return result


def train_test_split(split: float, df_scaled: pd.DataFrame):
    """Chronological train/test split — no shuffling."""
    train_size = int(len(df_scaled) * split)
    return df_scaled.iloc[:train_size], df_scaled.iloc[train_size:]


def prepare_data(
    target_lat: float,
    target_lon: float,
    start: datetime,
    end: datetime,
    max_radius_km: float = 50,
    n_rings: int = 0,
    n_segments: int = 4,
) -> tuple[pd.DataFrame, MinMaxScaler, WeatherStation, list[WeatherStation]]:
    """Fetch, engineer, merge, and scale data for an arbitrary target location.

    Returns (df_scaled, scaler, primary_station, aux_stations).
    """
    primary = WeatherStation(lat=target_lat, lon=target_lon)
    aux = primary.find_nearby(max_radius_km, n_rings=n_rings, n_segments=n_segments)
    df = fetch_multi_station_data(primary, aux, start, end)
    print(f"Dataset: {len(df)} days, {df.shape[1]} features "
          f"({1 + len(aux)} stations, target={primary})")
    df_scaled, scaler = normalize(df)
    return df_scaled, scaler, primary, aux
