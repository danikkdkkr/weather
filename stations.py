"""Weather station discovery and geospatial helpers.

Contains: WeatherStation class, haversine distance, destination point
calculation, and concentric ring coordinate generation.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from meteostat import Daily, Point, Stations


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
            Outer radius in km (0-1000).
        n_rings : int
            Number of concentric circles.  0 = no auxiliary stations (use only
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

        sample_coords = generate_ring_coords(
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
            if haversine(self.lat, self.lon, row.latitude, row.longitude) < 0.5:
                continue
            seen_ids.add(sid)
            result.append(
                WeatherStation(lat=row.latitude, lon=row.longitude, station_id=sid)
            )

        return result

    def __repr__(self) -> str:
        return f"WeatherStation(id={self.station_id!r}, lat={self.lat:.4f}, lon={self.lon:.4f})"


# ---------------------------------------------------------------------------
# Geospatial helpers
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def destination_point(
    lat: float, lon: float, distance_km: float, bearing_rad: float,
) -> tuple[float, float]:
    """Compute destination lat/lon given start, distance (km), and bearing (radians)."""
    R = 6371.0
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


def generate_ring_coords(
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
            lat, lon = destination_point(center_lat, center_lon, radius_km, bearing_rad)
            coords.append((lat, lon))
    return coords
