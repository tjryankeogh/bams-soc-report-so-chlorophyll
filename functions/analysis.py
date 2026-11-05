"""Anomalies, grid-cell area and sector (basin) means."""

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6371000.0

# Calendar months in July-June seasonal order
SEASON_MONTHS = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]

# Southern Ocean band used for all sector means
SO_LAT_MAX = -40
SO_LAT_MIN = -65

# Sector boundaries, matching marcelduplessis/bams-soc-report-so-heat-fluxes
#   Atlantic         70°W - 20°E
#   Indian           20°E - 170°E
#   West Pacific    170°E - 135°W  (crosses the dateline)
#   East Pacific    135°W - 70°W
# Boundaries are half-open so no longitude column is counted twice.
SECTOR_BOUNDARIES = [-70, 20, 170, -135]  # for the dashed lines on the map


def sector_masks(lon):
    """Boolean masks for each sector, given a -180..180 longitude coordinate."""
    return {
        "Southern Ocean": xr.ones_like(lon, dtype=bool),
        "West Pacific": (lon >= 170) | (lon < -135),
        "East Pacific": (lon >= -135) & (lon < -70),
        "Atlantic": (lon >= -70) & (lon < 20),
        "Indian": (lon >= 20) & (lon < 170),
    }


def cell_area(lat, lon):
    """
    Exact area (m^2) of each cell on a regular lat/lon grid,
    A = R^2 * dlon * |sin(lat_north) - sin(lat_south)|.
    Proportional to cos(lat) weighting, so sector means are identical to the
    heat-flux analysis.
    """
    lat_vals = np.asarray(lat, dtype=float)
    edges = np.empty(lat_vals.size + 1)
    edges[1:-1] = 0.5 * (lat_vals[:-1] + lat_vals[1:])
    edges[0] = lat_vals[0] - 0.5 * (lat_vals[1] - lat_vals[0])
    edges[-1] = lat_vals[-1] + 0.5 * (lat_vals[-1] - lat_vals[-2])
    edges = np.deg2rad(edges)

    dlon = np.abs(np.deg2rad(np.median(np.diff(np.asarray(lon, dtype=float)))))
    dphi = np.abs(np.sin(edges[1:]) - np.sin(edges[:-1]))

    area_1d = xr.DataArray(EARTH_RADIUS_M**2 * dlon * dphi, dims=["lat"], coords={"lat": lat})
    return area_1d * xr.ones_like(lon, dtype=float)


def monthly_in_season_order(path):
    """
    Open a seasonal monthly file (Jul..Jun) and index it by calendar month
    (1-12) so it aligns with the monthly climatology. Using the true calendar
    month avoids subtracting e.g. the January climatology from July.
    """
    ds = xr.open_dataset(path)
    if ds.sizes["time"] != 12:
        raise ValueError(f"{path} has {ds.sizes['time']} months, expected 12")
    return ds.groupby("time.month").mean("time")


def sector_means(da, area, lat_max=SO_LAT_MAX, lat_min=SO_LAT_MIN):
    """
    Area-weighted mean over each sector within lat_max..lat_min.
    NaNs (no ocean-colour data) are excluded, with weights renormalised.
    Returns {sector name: DataArray}.
    """
    da = da.sel(lat=slice(lat_max, lat_min))
    w = area.sel(lat=slice(lat_max, lat_min))
    out = {}
    for name, mask in sector_masks(da["lon"]).items():
        out[name] = da.where(mask).weighted(w.where(mask, 0)).mean(["lat", "lon"])
    return out
