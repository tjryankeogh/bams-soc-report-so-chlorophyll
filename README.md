# bams-soc-report-so-chlorophyll

Analysis for the chlorophyll-a part of the Southern Ocean section of the BAMS *State of the Climate* report.

Plotting style, sector definitions and the standard-error method match the heat-flux analysis for the same section: [marcelduplessis/bams-soc-report-so-heat-fluxes](https://github.com/marcelduplessis/bams-soc-report-so-heat-fluxes).

All the analysis is in `BAMS_Chlorophyll_Anomaly.ipynb`.

## Data

ESA Ocean Colour Climate Change Initiative (OC-CCI) v6.0, Level-3 merged chlorophyll-a, 8-day composites, 4 km. Accessed through THREDDS OPeNDAP: `https://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-8DAY`

Sathyendranath, S., et al. (2019). An ocean-colour time series for use in climate studies: the experience of the Ocean-Colour Climate Change Initiative (OC-CCI). *Sensors*, 19(19), 4285.

The data are too large to include here. The notebook downloads them, and re-running it only fetches new composites. It expects this layout **next to** the repository:

```
DATA/
  8day/           raw 8-day composites, 30–65°S (~0.65 GB per year). The permanent
                  record: everything else is rebuilt from these.
  seasonal_mean/  one small file per July–June season (seasonal mean)
  monthly/        one small file per season (calendar-month means)
  output/         climatologies and standard errors
  fronts/         saf.txt, saccf.txt (Orsi et al., 1995)
FIGURES/
bams-soc-report-so-chlorophyll/   <- this repo
```

The front files are the same Orsi et al. (1995) files used in the heat-flux analysis. Place them in `DATA/fronts/` yourself; they are not downloaded.

Orsi, A. H., Whitworth III, T., & Nowlin Jr, W. D. (1995). On the meridional extent and fronts of the Antarctic Circumpolar Current. *Deep Sea Research I*, 42(5), 641–673.

## Method

- **Seasonal year:** July–June, labelled by the year it ends. **2025 = 1 July 2024 – 30 June 2025.**
- **Climatology:** 1999–2020 seasons. The heat-flux analysis uses 1991–2020 (the WMO standard period), but OC-CCI starts in September 1997, so 1999 is the first complete July–June season.
- **Pre-processing** (Thomalla et al., 2023): gaps of up to two consecutive missing values are filled by linear interpolation along longitude, latitude, then time; a centred three-point (24-day) running mean is then applied in time.
- **Sectors**, all 40–65°S:

  | Sector | Longitudes |
  |---|---|
  | Atlantic | 70°W–20°E |
  | Indian | 20°E–170°E |
  | West Pacific | 170°E–135°W |
  | East Pacific | 135°W–70°W |

- **Sector means** are area-weighted. Missing (cloud or ice) pixels are excluded rather than set to zero.
- **Error bars** are the interannual standard deviation of each calendar month at each pixel over the climatology, divided by √(N−1), then area-averaged over the sector. This is the same method as the heat-flux analysis.
- **Map:** annual-mean anomaly, averaged to 0.25° (`map_coarsen = 6`) before contouring. Set `map_coarsen = 1` to contour the full 4 km field.

## Updating for a new year

1. In GitHub Desktop, tag the current commit with last year's report (e.g. `soc-2025`) so its figures stay reproducible.
2. Set `anomyear` in the notebook's settings cell (e.g. `2026` = July 2025–June 2026).
3. *Run All*. Only new composites are downloaded and only new seasons are processed. The climatology is reused. If the season isn't complete on OC-CCI yet, the notebook stops with a message saying how many composites are available.
4. Commit and push.

To rebuild everything (for example after changing the pre-processing, or for a new OC-CCI version), delete `seasonal_mean/`, `monthly/` and `output/` and set `rebuild_climatology = True`. Keep the climatology and the report year on the same OC-CCI version.

## Environment

```
conda env create -f environment.yml
conda activate soc-chlorophyll
```
