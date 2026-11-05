"""
Download and pre-processing of ESA OC-CCI 8-day chlorophyll-a for the
BAMS State of the Climate Southern Ocean section.

Seasonal-year convention
------------------------
Chlorophyll is analysed on a July-June year so that the austral growing
season is not split. The year label is the year in which the season ENDS:

    season 2025 = 1 July 2024 -> 30 June 2025
    (Jul-Dec from 2024, Jan-Jun from 2025)

Data layout (under DATA/)
-------------------------
    8day/           raw OC-CCI composites. The permanent record: everything
                    else can be rebuilt from these.
    seasonal_mean/  one file per season: mean of the gap-filled, smoothed
                    8-day composites (used for the map and annual climatology)
    monthly/        one file per season: calendar-month means (used for the
                    annual cycle, monthly climatology and error bars)
    output/         climatologies and standard errors
"""

import os
import re
import glob
from collections import defaultdict

import dask
import xarray as xr
from tqdm import tqdm

OCCCI_URL = "https://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-8DAY"
COMPOSITES_PER_SEASON = 46
PREFIX = "BAMS_chlorophyll_a_southern_ocean"

ENCODING = {"chlor_a": {"dtype": "float32", "zlib": True, "complevel": 4, "shuffle": True}}

# matches the trailing 8 digits (YYYYMMDD) before ".nc"
_DATE_RE = re.compile(r"-(\d{8})\.nc$")


# ---------------------------------------------------------------- file names

def seasonal_mean_path(mean_dir, season):
    return f"{mean_dir}/{PREFIX}_{season}_mean.nc"


def monthly_path(monthly_dir, season):
    return f"{monthly_dir}/{PREFIX}_{season}_monthly.nc"


def climatology_paths(outdir, clim_start, clim_end):
    tag = f"{clim_start}_{clim_end}"
    return {
        "annual": f"{outdir}/chlor_a_annual_climatology_{tag}.nc",
        "monthly": f"{outdir}/chlor_a_monthly_climatology_{tag}.nc",
        "stderr": f"{outdir}/chlor_a_monthly_stderr_{tag}.nc",
    }


def _write_atomic(delayed_writes):
    """Compute several delayed to_netcdf writes together, then move each .tmp
    into place, so an interrupted run never leaves a half-written file that
    later runs would mistake for a finished one."""
    dask.compute(*[d for d, _ in delayed_writes])
    for _, path in delayed_writes:
        os.replace(path + ".tmp", path)


# ------------------------------------------------------------------- seasons

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


def files_by_season(indir):
    """{season: [8-day files]} for every season with any files."""
    groups = defaultdict(list)
    for f in sorted(glob.glob(f"{indir}/*.nc")):
        groups[season_year(get_yyyymmdd(f))].append(f)
    return dict(sorted(groups.items()))


# ------------------------------------------------------------------ download

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


# ------------------------------------------------------------ pre-processing

def process_seasons(indir, mean_dir, monthly_dir, first_season=1999):
    """
    Process every complete July-June season (46 composites) that does not yet
    have both a seasonal-mean and a monthly file. Incomplete seasons (e.g. the
    one in progress) are reported and skipped.

    Pre-processing follows Thomalla et al. (2023): short gaps (<=2 consecutive
    missing values) are filled by linear interpolation along lon, lat, then
    time, and a centred 3-point (24-day) running mean is applied in time.
    Only the seasonal mean and the calendar-month means of the result are
    saved; the full 46-step series is not kept.
    """
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(monthly_dir, exist_ok=True)

    for season, files in files_by_season(indir).items():
        if season < first_season:
            continue
        mpath, opath = seasonal_mean_path(mean_dir, season), monthly_path(monthly_dir, season)
        if os.path.exists(mpath) and os.path.exists(opath):
            continue
        if len(files) != COMPOSITES_PER_SEASON:
            print(f"{season}: incomplete ({len(files)} of {COMPOSITES_PER_SEASON} composites), skipping")
            continue

        print(f"{season}: processing {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")
        xds = xr.open_mfdataset(files, concat_dim="time", combine="nested")

        # interpolate_na needs the interpolated dimension in a single chunk
        for dim in ["lon", "lat", "time"]:
            xds = xds.chunk({dim: -1})
            xds = xds.interpolate_na(dim=dim, method="linear", limit=2, use_coordinate=False)
        xds = xds.rolling(time=3, min_periods=1, center=True).mean()

        seasonal_mean = xds.mean("time").assign_attrs(season=season)
        monthly = xds.resample(time="ME").mean().assign_attrs(season=season)

        # computed together so the pre-processing runs only once
        _write_atomic([
            (seasonal_mean.to_netcdf(mpath + ".tmp", encoding=ENCODING, compute=False), mpath),
            (monthly.to_netcdf(opath + ".tmp", encoding=ENCODING, compute=False), opath),
        ])


def migrate_old_seasonal_files(old_dir, mean_dir):
    """
    One-off: create seasonal-mean files from the old full 46-step files in
    DATA/annual/ (pre-2026 layout) so those seasons need not be re-processed.
    Safe to run repeatedly; does nothing if old_dir does not exist.
    """
    if not os.path.isdir(old_dir):
        return
    os.makedirs(mean_dir, exist_ok=True)
    for f in tqdm(sorted(glob.glob(f"{old_dir}/{PREFIX}_*.nc"))):
        season = int(re.search(r"_(\d{4})\.nc$", f).group(1))
        out = seasonal_mean_path(mean_dir, season)
        if not os.path.exists(out):
            ds = xr.open_dataset(f).mean("time").assign_attrs(season=season)
            ds.to_netcdf(out + ".tmp", encoding=ENCODING)
            os.replace(out + ".tmp", out)
    print(f"Seasonal means written. Once checked, {old_dir} can be deleted.")


# -------------------------------------------------------------- climatology

def build_climatologies(mean_dir, monthly_dir, outdir, clim_start, clim_end,
                        variable="chlor_a", overwrite=False):
    """
    Annual-mean, monthly-mean and monthly standard-error climatologies over
    seasons clim_start..clim_end. Skipped if they already exist, unless
    overwrite=True (e.g. after re-processing).

    - Annual climatology: mean of the seasonal means (each season weighted equally).
    - Missing values are skipped, NOT filled with zero, so pixels with cloud
      or ice gaps in some years are not biased low.
    - Standard error, as in the heat-flux analysis (du Plessis): interannual
      standard deviation for each calendar month and pixel / sqrt(N - 1).
    """
    paths = climatology_paths(outdir, clim_start, clim_end)
    if not overwrite and all(os.path.exists(p) for p in paths.values()):
        print("Climatology exists, skipping (overwrite=True to rebuild)")
        return

    seasons = range(clim_start, clim_end + 1)
    missing = [s for s in seasons
               if not (os.path.exists(seasonal_mean_path(mean_dir, s))
                       and os.path.exists(monthly_path(monthly_dir, s)))]
    if missing:
        raise FileNotFoundError(f"Climatology seasons not processed yet: {missing}")

    os.makedirs(outdir, exist_ok=True)
    n_years = len(seasons)

    means = xr.open_mfdataset([seasonal_mean_path(mean_dir, s) for s in seasons],
                              combine="nested", concat_dim="season")[[variable]]
    monthly = xr.open_mfdataset([monthly_path(monthly_dir, s) for s in seasons])[[variable]]
    by_month = monthly.groupby("time.month")

    _write_atomic([
        (means.mean("season").to_netcdf(paths["annual"] + ".tmp", compute=False), paths["annual"]),
        (by_month.mean("time").to_netcdf(paths["monthly"] + ".tmp", compute=False), paths["monthly"]),
        ((by_month.std("time") / (n_years - 1) ** 0.5).to_netcdf(paths["stderr"] + ".tmp", compute=False),
         paths["stderr"]),
    ])


def check_season(mean_dir, monthly_dir, indir, season):
    """Clear error if the requested season has not been processed."""
    if os.path.exists(seasonal_mean_path(mean_dir, season)) and os.path.exists(monthly_path(monthly_dir, season)):
        return
    n = len(files_by_season(indir).get(season, []))
    raise FileNotFoundError(
        f"Season {season} (Jul {season - 1} - Jun {season}) is not available: "
        f"{n} of {COMPOSITES_PER_SEASON} 8-day composites downloaded. OC-CCI may not "
        f"have published the rest yet.")
