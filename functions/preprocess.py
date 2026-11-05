"""
Download and pre-processing of ESA OC-CCI 8-day chlorophyll-a for the
BAMS State of the Climate Southern Ocean section.

Seasonal-year convention
------------------------
Chlorophyll is analysed on a July-June year so that the austral growing
season is not split. The year label is the year in which the season ENDS:

    season 2025 = 1 July 2024 -> 30 June 2025
    (Jul-Dec from 2024, Jan-Jun from 2025)
"""

import os
import re
import glob
from collections import defaultdict

import xarray as xr
from tqdm import tqdm

OCCCI_URL = "https://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-8DAY"
COMPOSITES_PER_SEASON = 46

ENCODING = {"chlor_a": {"dtype": "float32", "zlib": True, "complevel": 4, "shuffle": True}}

# matches the trailing 8 digits (YYYYMMDD) before ".nc"
_DATE_RE = re.compile(r"-(\d{8})\.nc$")


def get_yyyymmdd(path):
    """Parse the YYYYMMDD date stamp from an 8-day file name."""
    m = _DATE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse date from filename: {path}")
    return int(m.group(1))


def season_year(yyyymmdd):
    """July-June season label: Jul-Dec belong to the following year's season."""
    year, month = yyyymmdd // 10000, (yyyymmdd // 100) % 100
    return year + 1 if month >= 7 else year


def download_8day(outdir, url=OCCCI_URL, lat_north=-30, lat_south=-65):
    """
    Download OC-CCI L3 8-day chlor_a via OPeNDAP, one composite per file,
    subset to lat_north..lat_south. Existing files are skipped, so re-running
    only fetches new composites.
    """
    os.makedirs(outdir, exist_ok=True)
    ds_8d = xr.open_dataset(url, decode_times=True)

    for i in tqdm(range(ds_8d.sizes["time"])):
        ds_out = ds_8d.isel(time=slice(i, i + 1))
        t = ds_out.time.dt
        fname = (f"{outdir}/ESACCI-OC-L3S-CHLOR_A-MERGED-8D_DAILY_4km_GEO_PML_OCx-"
                 f"{t.year.item()}{t.month.item():02d}{t.day.item():02d}.nc")
        if os.path.exists(fname):
            continue

        # OC-CCI latitude is descending (90 -> -90)
        ds_chl = ds_out.sel(lat=slice(lat_north, lat_south))[["chlor_a"]]
        ds_chl["chlor_a"] = ds_chl["chlor_a"].astype("float32")
        ds_chl.to_netcdf(fname, engine="netcdf4", encoding=ENCODING)


def group_by_season(indir, first_season=1999):
    """
    Return {season: [files]} for complete July-June seasons only
    (46 composites). Incomplete seasons (e.g. the current one) are dropped.
    """
    groups = defaultdict(list)
    for f in sorted(glob.glob(f"{indir}/*.nc")):
        s = season_year(get_yyyymmdd(f))
        if s >= first_season:
            groups[s].append(f)

    complete = {s: fl for s, fl in groups.items() if len(fl) == COMPOSITES_PER_SEASON}
    skipped = sorted(set(groups) - set(complete))
    if skipped:
        print(f"Skipping incomplete seasons: {skipped}")
    return dict(sorted(complete.items()))


def build_seasonal_files(indir, outdir, first_season=1999):
    """
    Combine each July-June season of 8-day composites into one file, following
    Thomalla et al. (2023): fill short gaps (<=2 consecutive missing values)
    by linear interpolation along lon, lat then time, then apply a centred
    3-point (24-day) running mean in time.
    """
    os.makedirs(outdir, exist_ok=True)

    for season, files in group_by_season(indir, first_season).items():
        outpath = f"{outdir}/BAMS_chlorophyll_a_southern_ocean_{season}.nc"
        if os.path.exists(outpath):
            print(f"{season}: already exists, skipping")
            continue
        print(f"{season}: processing {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")

        xds = xr.open_mfdataset(files, concat_dim="time", combine="nested")

        # interpolate_na needs the interpolated dimension in a single chunk
        for dim in ["lon", "lat", "time"]:
            xds = xds.chunk({dim: -1})
            xds = xds.interpolate_na(dim=dim, method="linear", limit=2, use_coordinate=False)

        xds = xds.rolling(time=3, min_periods=1, center=True).mean()
        xds.to_netcdf(outpath, encoding=ENCODING, engine="netcdf4")


def build_monthly_files(indir, outdir):
    """Monthly means (calendar months) of each seasonal file."""
    os.makedirs(outdir, exist_ok=True)

    for file in tqdm(sorted(glob.glob(f"{indir}/BAMS_chlorophyll_a_southern_ocean_*.nc"))):
        name = os.path.basename(file).replace(".nc", "")
        outpath = f"{outdir}/{name}_monthly.nc"
        if os.path.exists(outpath):
            continue
        xr.open_dataset(file).resample(time="ME").mean().to_netcdf(outpath, encoding=ENCODING)


def season_files(indir, first, last, suffix=""):
    """Files for seasons first..last (inclusive)."""
    return [f"{indir}/BAMS_chlorophyll_a_southern_ocean_{s}{suffix}.nc"
            for s in range(first, last + 1)]


def build_climatologies(annual_dir, monthly_dir, outdir, clim_start, clim_end, variable="chlor_a"):
    """
    Annual mean, monthly mean and monthly standard error climatologies over
    seasons clim_start..clim_end.

    Missing values are skipped (NOT filled with zero), so pixels with cloud or
    ice gaps in some years are not biased low.

    The standard error follows the heat-flux analysis (du Plessis): the
    interannual standard deviation for each calendar month and pixel divided
    by sqrt(N - 1), N = number of seasons.
    """
    os.makedirs(outdir, exist_ok=True)
    tag = f"{clim_start}_{clim_end}"
    n_years = clim_end - clim_start + 1

    annual = xr.open_mfdataset(season_files(annual_dir, clim_start, clim_end))[[variable]]
    annual.mean("time").to_netcdf(f"{outdir}/oc-cci_annual_climatology_{tag}.nc")

    monthly = xr.open_mfdataset(season_files(monthly_dir, clim_start, clim_end, "_monthly"))[[variable]]
    monthly.groupby("time.month").mean("time").to_netcdf(f"{outdir}/oc-cci_monthly_climatology_{tag}.nc")

    stderr = monthly.groupby("time.month").std("time") / (n_years - 1) ** 0.5
    stderr.to_netcdf(f"{outdir}/oc-cci_monthly_stderr_{tag}.nc")
