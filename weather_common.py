"""Shared utilities for weather forecasting models.

Contains: WeatherStation, SequenceDataset, Params, data pipeline,
training loop, prediction, and multi-radius station discovery.
"""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from meteostat import Daily, Point, Stations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COLS = [
    "tavg", "tmin", "tmax", "prcp", "snow",
    "wdir_sin", "wdir_cos", "wspd", "wpgt", "pres", "tsun",
]


# ---------------------------------------------------------------------------
# WeatherStation
# ---------------------------------------------------------------------------

class WeatherStation:
    """A weather station defined by lat/lon, optionally backed by a Meteostat station ID."""

    def __init__(self, lat: float, lon: float, station_id: str | None = None):
        self.lat = lat
        self.lon = lon
        self.station_id = station_id

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        loc = self.station_id if self.station_id is not None else Point(self.lat, self.lon)
        return pd.DataFrame(Daily(loc, start, end).fetch())

    def find_nearby(
        self,
        max_radius_km: float,
        n_rings: int = 0,
        n_segments: int = 4,
    ) -> list["WeatherStation"]:
        """Return auxiliary stations on concentric rings with angular segments.

        Generates sample coordinates on *n_rings* equally-spaced circles (from
        centre to *max_radius_km*) with *n_segments* equally-spaced angular
        positions on each circle, then finds the nearest Meteostat station to
        each sample point.

        Parameters
        ----------
        max_radius_km : float
            Outer radius in km (0–1000).
        n_rings : int
            Number of concentric circles.  0 → no auxiliary stations (use only
            the primary coordinate).  Max 1000.
        n_segments : int
            Number of equally-spaced angular positions per ring.  Min 1.

        Returns
        -------
        list[WeatherStation]
            Flat list of unique auxiliary stations (excludes this station).
        """
        max_radius_km = min(max(max_radius_km, 0), 1000)
        n_rings = min(max(n_rings, 0), 1000)
        if n_rings == 0 or n_segments < 1:
            return []

        sample_coords = _generate_ring_coords(
            self.lat, self.lon, max_radius_km, n_rings, n_segments,
        )

        seen_ids: set[str] = set()
        result: list[WeatherStation] = []

        for lat, lon in sample_coords:
            nearest = Stations().nearby(lat, lon).fetch()
            if nearest.empty:
                continue
            sid = nearest.index[0]
            if sid in seen_ids:
                continue
            row = nearest.iloc[0]
            # Skip if it's essentially the same location as primary
            if _haversine(self.lat, self.lon, row.latitude, row.longitude) < 0.5:
                continue
            seen_ids.add(sid)
            result.append(
                WeatherStation(lat=row.latitude, lon=row.longitude, station_id=sid)
            )

        return result

    def __repr__(self) -> str:
        return f"WeatherStation(id={self.station_id!r}, lat={self.lat:.4f}, lon={self.lon:.4f})"


def _generate_ring_coords(
    center_lat: float,
    center_lon: float,
    max_radius_km: float,
    n_rings: int,
    n_segments: int,
) -> list[tuple[float, float]]:
    """Generate (lat, lon) sample points on concentric circles.

    Returns one coordinate per (ring, segment) intersection.
    """
    coords: list[tuple[float, float]] = []
    for ring in range(1, n_rings + 1):
        radius_km = max_radius_km * ring / n_rings
        for seg in range(n_segments):
            bearing_rad = 2 * np.pi * seg / n_segments
            lat, lon = _destination_point(center_lat, center_lon, radius_km, bearing_rad)
            coords.append((lat, lon))
    return coords


def _destination_point(
    lat: float, lon: float, distance_km: float, bearing_rad: float,
) -> tuple[float, float]:
    """Compute destination lat/lon given start, distance (km), and bearing (radians)."""
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    d = distance_km / R

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(bearing_rad)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float(np.degrees(lon2))


def _haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def process_station_df(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station feature engineering: wdir → sin/cos, snow NaN → 0, drop NaN rows."""
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
    Auxiliary columns are prefixed s1_, s2_, …
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


# ---------------------------------------------------------------------------
# Dataset & DataLoaders
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Sliding-window dataset.  First 11 columns of each target window are the targets."""

    def __init__(self, data, seq_len: int = 30, horizon: int = 3, step: int = 1):
        self.X = data.values if hasattr(data, "values") else np.asarray(data)
        self.seq_len = seq_len
        self.horizon = horizon
        self.step = step
        self.n = len(self.X)
        self.max_start = self.n - seq_len - horizon + 1

    def __len__(self):
        return max(0, (self.max_start + self.step - 1) // self.step)

    def __getitem__(self, idx):
        i = idx * self.step
        x = self.X[i : i + self.seq_len]
        y = self.X[i + self.seq_len : i + self.seq_len + self.horizon, :11]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class Params:
    """Hyper-parameter bundle."""

    def __init__(self, seq_len, horizon, hidden_dim, num_layers, lr, dropout):
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.dropout = dropout


def build_loaders(
    df_scaled: pd.DataFrame,
    params: Params,
    split: float = 0.8,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Split + wrap into DataLoaders."""
    train_df, test_df = train_test_split(split, df_scaled)
    train_ds = SequenceDataset(train_df, seq_len=params.seq_len, horizon=params.horizon)
    test_ds = SequenceDataset(test_df, seq_len=params.seq_len, horizon=params.horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    lr: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 30,
    checkpoint_path: str = "best_model.pt",
) -> float:
    """Train with early stopping.  Returns best validation loss."""
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    trigger = 0

    for epoch in range(max_epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        n = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= n

        # --- validate ---
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for X, y in test_loader:
                val_loss += criterion(model(X), y).item()
                n += 1
        val_loss /= n

        # --- early stopping ---
        if val_loss < best_val:
            best_val = val_loss
            trigger = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            trigger += 1
            if trigger >= patience:
                break

    return best_val


# ---------------------------------------------------------------------------
# Prediction & evaluation
# ---------------------------------------------------------------------------

def predict_and_compare(
    model: nn.Module,
    scaler: MinMaxScaler,
    test_loader: DataLoader,
    params: Params,
    target_cols: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Run inference on the last batch and return per-variable MAE dict."""
    if target_cols is None:
        target_cols = TARGET_COLS

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X)

    y_hat_np = y_hat[-1].cpu().numpy()
    y_true_np = y[-1].cpu().numpy()

    pred_df = pd.DataFrame(y_hat_np, columns=target_cols)
    true_df = pd.DataFrame(y_true_np, columns=target_cols)

    scaled_cols = list(scaler.feature_names_in_)
    pred_scaled = pred_df[[c for c in target_cols if c in scaled_cols]].copy()
    true_scaled = true_df[[c for c in target_cols if c in scaled_cols]].copy()

    inv_pred_scaled = inverse_transform_cols(scaler, pred_scaled)
    inv_true_scaled = inverse_transform_cols(scaler, true_scaled)

    inv_pred = pred_df.copy()
    inv_true = true_df.copy()
    for c in inv_pred_scaled.columns:
        inv_pred[c] = inv_pred_scaled[c]
        inv_true[c] = inv_true_scaled[c]

    # Post-processing: clip precipitation/snow
    for col in ("prcp", "snow"):
        if col in inv_pred.columns:
            inv_pred[col] = np.clip(inv_pred[col], 0, None)
            inv_pred.loc[inv_pred[col] < 0.3, col] = 0.0

    # Per-variable MAE
    maes: dict[str, float] = {}
    for col in target_cols:
        maes[col] = float(np.abs(inv_pred[col].values - inv_true[col].values).mean())

    if verbose:
        print("\n--- Prediction vs True (last sample) ---")
        for i, name in enumerate(target_cols):
            for h in range(params.horizon):
                p = inv_pred.iloc[h, i]
                t = inv_true.iloc[h, i]
                print(f"Day +{h+1:>2}: {name:<10s} | Pred: {p:8.3f} | True: {t:8.3f} | Diff: {p-t:+8.3f}")

    return maes


def evaluate_full_test(
    model: nn.Module,
    scaler: MinMaxScaler,
    test_loader: DataLoader,
    target_cols: list[str] | None = None,
) -> dict[str, float]:
    """Compute per-variable MAE over the *entire* test set (all batches, all samples)."""
    if target_cols is None:
        target_cols = TARGET_COLS

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X)
            all_preds.append(y_hat.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # (N, horizon, 11)
    trues = np.concatenate(all_trues, axis=0)

    # Flatten horizon dimension for inverse transform
    N, H, D = preds.shape
    preds_flat = preds.reshape(-1, D)
    trues_flat = trues.reshape(-1, D)

    pred_df = pd.DataFrame(preds_flat, columns=target_cols)
    true_df = pd.DataFrame(trues_flat, columns=target_cols)

    scaled_cols = list(scaler.feature_names_in_)
    for df in (pred_df, true_df):
        sub = df[[c for c in target_cols if c in scaled_cols]].copy()
        inv = inverse_transform_cols(scaler, sub)
        for c in inv.columns:
            df[c] = inv[c]

    for col in ("prcp", "snow"):
        if col in pred_df.columns:
            pred_df[col] = np.clip(pred_df[col], 0, None)
            pred_df.loc[pred_df[col] < 0.3, col] = 0.0

    maes = {}
    for col in target_cols:
        maes[col] = float(np.abs(pred_df[col].values - true_df[col].values).mean())

    return maes


# ---------------------------------------------------------------------------
# Convenience: full pipeline for any target location
# ---------------------------------------------------------------------------

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


def hyperparameter_grid(*,
    seq_lens=(30,),
    horizons=(3,),
    hidden_dims=(32,),
    num_layers_list=(1,),
    lrs=(1e-3,),
    dropouts=(0.0,),
) -> list[Params]:
    """Cartesian product of hyper-parameters."""
    result = []
    for sl in seq_lens:
        for h in horizons:
            for hd in hidden_dims:
                for nl in num_layers_list:
                    for lr in lrs:
                        for do in dropouts:
                            result.append(Params(sl, h, hd, nl, lr, do))
    return result
