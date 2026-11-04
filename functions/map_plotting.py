"""
Map and line-plot helpers. Styling follows
marcelduplessis/bams-soc-report-so-heat-fluxes so the chlorophyll figures
match the heat-flux figures in the report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
from cartopy.util import add_cyclic_point

from .analysis import SO_LAT_MAX, SO_LAT_MIN, SECTOR_BOUNDARIES

# line colours, as in the heat-flux annual-cycle figure
SECTOR_COLOURS = {
    "Southern Ocean": "k",
    "West Pacific": "r",
    "East Pacific": "g",
    "Atlantic": "b",
    "Indian": "c",
}

SEASON_MONTH_LABELS = ["J", "A", "S", "O", "N", "D", "J", "F", "M", "A", "M", "J"]


def gridlines(ax):
    """
    Gridlines (code from Marcel du Plessis) with the four longitude labels
    placed just outside the circular boundary on the 45° diagonals.

    The labels are placed by hand because cartopy >= 0.26 positions and
    hides auto-labels on a circular boundary differently from the version
    used for the heat-flux figures (0.25), dropping 45°W/45°E. Call after
    ax.set_boundary(); positions are in axes coordinates of a circle of
    radius 0.5 centred on (0.5, 0.5), with 0° longitude at the top.
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color="gray", alpha=0.5, linestyle="--", zorder=110)
    gl.xlocator = mticker.FixedLocator([-45, -135, 45, 135])

    r = 0.53
    for lon in [-45, 45, -135, 135]:
        ang = np.deg2rad(lon)  # clockwise from the top
        x, y = 0.5 + r * np.sin(ang), 0.5 + r * np.cos(ang)
        ax.text(x, y, LONGITUDE_FORMATTER(lon), transform=ax.transAxes, fontsize=12,
                ha="left" if lon > 0 else "right", va="bottom" if abs(lon) < 90 else "top")
    return gl


def load_orsi_front(path, nrows):
    """
    Read an Orsi et al. (1995) front file (e.g. saf.txt, saccf.txt).
    Only the first `nrows` rows are used (the remainder are separate
    segments delimited by 'X' rows), as in the heat-flux analysis.
    """
    df = pd.read_csv(path, sep="\t").iloc[:nrows]
    parts = df.iloc[:, 0].astype(str).str.split(expand=True)
    return parts[0].astype(float).values, parts[1].astype(float).values


def southern_ocean_anomaly_map(da, levels, cbar_label, fronts=(), figsize=(8, 5)):
    """
    Polar stereographic contourf map of an annual-mean anomaly with the
    40°S/65°S rings, sector boundaries and fronts, in the heat-flux style.

    da     : 2-D (lat, lon) DataArray, lon in -180..180
    fronts : iterable of (lon, lat) arrays
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, zorder=90, edgecolor="black", color="0.55")

    gridlines(ax)

    # add a cyclic column so contourf has no seam at the dateline
    data, lon = add_cyclic_point(da.values, coord=da["lon"].values)
    img = ax.contourf(lon, da["lat"].values, data, levels=levels, cmap="RdBu_r",
                      extend="both", transform=ccrs.PlateCarree())

    cb = plt.colorbar(img, ax=ax, orientation="vertical", shrink=0.9, pad=0.1)
    cb.ax.tick_params(length=0, labelsize=12)
    cb.set_label(cbar_label, rotation=-90, labelpad=15, fontsize=14)
    for label in cb.ax.get_yticklabels():
        label.set_weight("bold")

    ring_lon = np.linspace(0, 360, 360)
    line = dict(c="k", ls="--", transform=ccrs.PlateCarree(), lw=1, zorder=100)
    for b in SECTOR_BOUNDARIES:
        ax.plot([b, b], [SO_LAT_MIN, SO_LAT_MAX], **line)
    ax.plot(ring_lon, np.full(360, SO_LAT_MIN), **line)
    ax.plot(ring_lon, np.full(360, SO_LAT_MAX), **line)

    for flon, flat in fronts:
        ax.plot(flon, flat, c="purple", lw=1.5, transform=ccrs.PlateCarree(), zorder=10)

    return fig, ax


def annual_cycle_plot(means, stderr, ylabel, ylim, yticks, shade_months=None, figsize=(6, 6.5)):
    """
    Monthly sector-mean anomalies with standard-error bars on a July-June
    axis, in the heat-flux style.

    means, stderr : {sector: 1-D array of 12 values in Jul..Jun order}
    shade_months  : optional (start, end) x-positions (1 = Jul) to shade
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, 13)

    for name, colour in SECTOR_COLOURS.items():
        ax.errorbar(x=x, y=means[name], yerr=stderr[name], c=colour, lw=1.5, label=name)

    if shade_months is not None:
        ax.axvspan(*shade_months, color="lightgray", alpha=0.4, lw=0, zorder=0)

    ax.legend(loc=2, ncol=1, fontsize=12, markerscale=1.0)
    ax.grid(ls="--")
    ax.set_ylim(*ylim)
    ax.xaxis.set_ticks(x)
    ax.xaxis.set_ticklabels(SEASON_MONTH_LABELS)
    ax.axhline(lw=0.5, c="k")
    ax.set_ylabel(ylabel, fontsize=17, labelpad=10)
    ax.yaxis.set_ticks(yticks)
    ax.tick_params(axis="both", which="major", labelsize=15, length=6)
    fig.tight_layout()
    return fig, ax
