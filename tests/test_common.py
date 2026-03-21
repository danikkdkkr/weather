"""Unit tests for the shared pipeline — no network calls, uses synthetic data."""

import numpy as np
import pandas as pd
import pytest

from data_pipeline import (
    TARGET_COLS,
    inverse_transform_cols,
    normalize,
    process_station_df,
    train_test_split,
)
from dataset import SequenceDataset
from stations import destination_point, generate_ring_coords, haversine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_df(n_days: int = 200) -> pd.DataFrame:
    """Synthetic DataFrame mimicking raw Meteostat daily output."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "tavg": rng.uniform(0, 30, n_days),
        "tmin": rng.uniform(-5, 20, n_days),
        "tmax": rng.uniform(10, 40, n_days),
        "prcp": rng.uniform(0, 20, n_days),
        "snow": rng.choice([0, np.nan, 5], n_days),
        "wdir": rng.uniform(0, 360, n_days),
        "wspd": rng.uniform(0, 50, n_days),
        "wpgt": rng.uniform(0, 100, n_days),
        "pres": rng.uniform(990, 1030, n_days),
        "tsun": rng.uniform(0, 600, n_days),
    }, index=dates)
    return df


# ---------------------------------------------------------------------------
# process_station_df
# ---------------------------------------------------------------------------

class TestProcessStationDf:
    def test_drops_wdir_adds_sin_cos(self):
        df = _make_raw_df()
        result = process_station_df(df)
        assert "wdir" not in result.columns
        assert "wdir_sin" in result.columns
        assert "wdir_cos" in result.columns

    def test_snow_nan_filled(self):
        df = _make_raw_df()
        result = process_station_df(df)
        assert result["snow"].isna().sum() == 0

    def test_no_nans_remain(self):
        df = _make_raw_df()
        result = process_station_df(df)
        assert result.isna().sum().sum() == 0

    def test_sin_cos_range(self):
        df = _make_raw_df()
        result = process_station_df(df)
        assert result["wdir_sin"].between(-1, 1).all()
        assert result["wdir_cos"].between(-1, 1).all()


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_scaled_range(self):
        df = process_station_df(_make_raw_df())
        cols = TARGET_COLS
        df = df[cols]
        df_scaled, scaler = normalize(df)
        skip = [c for c in df_scaled.columns if c.endswith("_sin") or c.endswith("_cos")]
        non_skip = [c for c in df_scaled.columns if c not in skip]
        for c in non_skip:
            assert df_scaled[c].min() >= -1.0 - 1e-9
            assert df_scaled[c].max() <= 1.0 + 1e-9

    def test_sin_cos_untouched(self):
        df = process_station_df(_make_raw_df())[TARGET_COLS]
        original_sin = df["wdir_sin"].copy()
        df_scaled, _ = normalize(df)
        pd.testing.assert_series_equal(df_scaled["wdir_sin"], original_sin)


# ---------------------------------------------------------------------------
# inverse_transform_cols
# ---------------------------------------------------------------------------

class TestInverseTransform:
    def test_roundtrip(self):
        df = process_station_df(_make_raw_df())[TARGET_COLS]
        df_scaled, scaler = normalize(df)
        scaled_cols = [c for c in df_scaled.columns
                       if not c.endswith("_sin") and not c.endswith("_cos")]
        subset_scaled = df_scaled[scaled_cols].copy()
        recovered = inverse_transform_cols(scaler, subset_scaled)
        original = df[scaled_cols]
        for c in scaled_cols:
            np.testing.assert_allclose(recovered[c].values, original[c].values, atol=1e-5)


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_split_sizes(self):
        df = process_station_df(_make_raw_df())[TARGET_COLS]
        df_scaled, _ = normalize(df)
        train, test = train_test_split(0.8, df_scaled)
        assert len(train) + len(test) == len(df_scaled)
        assert len(train) == int(len(df_scaled) * 0.8)

    def test_chronological_order(self):
        df = process_station_df(_make_raw_df())[TARGET_COLS]
        df_scaled, _ = normalize(df)
        train, test = train_test_split(0.8, df_scaled)
        assert train.index[-1] < test.index[0]


# ---------------------------------------------------------------------------
# SequenceDataset
# ---------------------------------------------------------------------------

class TestSequenceDataset:
    def test_shapes(self):
        df = process_station_df(_make_raw_df())[TARGET_COLS]
        df_scaled, _ = normalize(df)
        ds = SequenceDataset(df_scaled, seq_len=30, horizon=3)
        x, y = ds[0]
        assert x.shape == (30, len(TARGET_COLS))
        assert y.shape == (3, 11)

    def test_length(self):
        n = 100
        ds = SequenceDataset(np.random.randn(n, 11), seq_len=30, horizon=3)
        expected = n - 30 - 3 + 1
        assert len(ds) == expected

    def test_no_overlap_with_step(self):
        ds = SequenceDataset(np.random.randn(100, 11), seq_len=10, horizon=1, step=10)
        assert len(ds) > 0
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.shape[0] == 10

    def test_empty_dataset(self):
        ds = SequenceDataset(np.random.randn(5, 11), seq_len=10, horizon=3)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# haversine
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_zero_distance(self):
        assert haversine(48.78, 9.18, 48.78, 9.18) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # Stuttgart to Munich ~ 190 km
        dist = haversine(48.78, 9.18, 48.14, 11.58)
        assert 170 < dist < 220

    def test_vectorised(self):
        lats = np.array([48.78, 48.14])
        lons = np.array([9.18, 11.58])
        dists = haversine(48.78, 9.18, lats, lons)
        assert dists[0] == pytest.approx(0.0, abs=1e-6)
        assert 170 < dists[1] < 220


# ---------------------------------------------------------------------------
# destination_point
# ---------------------------------------------------------------------------

class TestDestinationPoint:
    def test_north(self):
        """Moving 100 km north from Stuttgart should increase latitude."""
        lat, lon = destination_point(48.78, 9.18, 100.0, 0.0)
        assert lat > 48.78
        assert lon == pytest.approx(9.18, abs=0.01)

    def test_east(self):
        """Moving east should increase longitude."""
        lat, lon = destination_point(48.78, 9.18, 100.0, np.pi / 2)
        assert lon > 9.18
        assert lat == pytest.approx(48.78, abs=0.2)

    def test_zero_distance(self):
        lat, lon = destination_point(48.78, 9.18, 0.0, 0.0)
        assert lat == pytest.approx(48.78, abs=1e-6)
        assert lon == pytest.approx(9.18, abs=1e-6)

    def test_roundtrip_distance(self):
        """Destination at distance d should be ~d km away from origin."""
        d_km = 50.0
        lat2, lon2 = destination_point(48.78, 9.18, d_km, np.pi / 4)
        actual = haversine(48.78, 9.18, lat2, lon2)
        assert actual == pytest.approx(d_km, rel=0.01)


# ---------------------------------------------------------------------------
# generate_ring_coords
# ---------------------------------------------------------------------------

class TestGenerateRingCoords:
    def test_count(self):
        """n_rings * n_segments coordinates should be generated."""
        coords = generate_ring_coords(48.78, 9.18, 100, n_rings=3, n_segments=4)
        assert len(coords) == 3 * 4

    def test_zero_rings(self):
        coords = generate_ring_coords(48.78, 9.18, 100, n_rings=0, n_segments=4)
        assert len(coords) == 0

    def test_distances_increase_per_ring(self):
        """Points on outer rings should be farther from centre."""
        coords = generate_ring_coords(48.78, 9.18, 90, n_rings=3, n_segments=4)
        for ring in range(3):
            ring_coords = coords[ring * 4 : (ring + 1) * 4]
            for lat, lon in ring_coords:
                dist = haversine(48.78, 9.18, lat, lon)
                expected = 90 * (ring + 1) / 3
                assert dist == pytest.approx(expected, rel=0.02)

    def test_segments_evenly_spaced(self):
        """4 segments should produce points at ~90 deg apart."""
        coords = generate_ring_coords(48.78, 9.18, 50, n_rings=1, n_segments=4)
        for lat, lon in coords:
            dist = haversine(48.78, 9.18, lat, lon)
            assert dist == pytest.approx(50, rel=0.02)
        dists_between = []
        for i in range(4):
            j = (i + 1) % 4
            d = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            dists_between.append(d)
        assert max(dists_between) / min(dists_between) < 1.15
