# data/pvgis_series/

PVGIS 5.2 `seriescalc` hourly output for 2005-2020 at the four study sites, radiation database PVGIS-SARAH2. These files drive validation layer 1 (`src/validation.py`), the sixteen-year reliability check and the final designs (`src/multiyear.py`), and the tests in `tests/test_validation_multiyear.py`. They are read by `src/pvgis_series.py` (`load_all()`, `load_edinburgh_horizon()`); nothing in the pipeline calls the network.

## Files

| File | Site | Latitude, longitude (request) | Nameplate requested (kWp) | Header prints (kWp) | Slope (header) | Elevation (header, m) | Horizon | Mean annual PVGIS yield 2005-2020 (kWh/kWp) |
|---|---|---|---|---|---|---|---|---|
| `southampton_seriescalc_2005_2020.csv` | Southampton | 50.9097, -1.4044 | 0.50 | 0.5 | 51° | 19 | off | 1,222 |
| `birmingham_seriescalc_2005_2020.csv` | Birmingham | 52.4862, -1.8904 | 0.55 | 0.6 | 52° | 130 | off | 1,090 |
| `liverpool_seriescalc_2005_2020.csv` | Liverpool | 53.4084, -2.9916 | 0.65 | 0.7 | 53° | 31 | off | 1,088 |
| `edinburgh_seriescalc_2005_2020.csv` | Edinburgh | 55.9533, -3.1883 | 0.65 | 0.7 | 56° | 65 | off | 1,031 |
| `edinburgh_seriescalc_2005_2020_horizon.csv` | Edinburgh | 55.9533, -3.1883 | 0.65 | 0.7 | 56° | 65 | on | 1,028 |

Each file has 140,256 hourly rows (16 years including leap days), from 2005-01-01 to 2020-12-31, about 7 MB. No row is flagged as reconstructed (`Int` = 1). The elevation in the header is the PVGIS grid-cell value and differs from the site altitudes in `src/sites.py`.

## Request parameters

Endpoint `https://re.jrc.ec.europa.eu/api/v5_2/seriescalc`, parameters as recorded in `src/fetch_pvgis_series.py`:

```
lat=<site latitude>, lon=<site longitude>, raddatabase=PVGIS-SARAH2,
startyear=2005, endyear=2020, pvcalculation=1, peakpower=<kWp above>,
loss=0, trackingtype=0, angle=<site latitude, i.e. 50.91, 52.49, 53.41, 55.95>,
aspect=0 (south), pvtechchoice=crystSi, mountingplace=free,
usehorizon=0 (1 for the second Edinburgh file), components=1, outputformat=csv
```

That is: fixed mount at latitude tilt facing south, crystalline silicon, free-standing, PVGIS system loss set to zero, plane-of-array irradiance components included, terrain horizon off (a second Edinburgh request with the horizon on quantifies terrain shading: 0.24% of annual yield, 1.5% of December yield, `results/validation.json` -> `edinburgh_horizon`).

## Columns

PVGIS names in the file, and the names given by `src/pvgis_series.py`:

| File column | Loader name | Unit | Meaning |
|---|---|---|---|
| `time` | index (UTC) | `YYYYMMDD:HHMM` | time stamp |
| `P` | `p_pvgis_w` | W | PV system power from PVGIS's own model (Huld power model, Faiman module temperature, Martin-Ruiz reflectance) at zero system loss |
| `Gb(i)` | `poa_direct` | W/m² | beam irradiance in the plane of the array |
| `Gd(i)` | `poa_sky_diffuse` | W/m² | sky-diffuse irradiance in the plane of the array |
| `Gr(i)` | `poa_ground_diffuse` | W/m² | ground-reflected irradiance in the plane of the array |
| `H_sun` | `sun_elevation` | degrees | sun elevation |
| `T2m` | `temp_air` | °C | 2 m air temperature |
| `WS10m` | `wind_speed` | m/s | 10 m wind speed |
| `Int` | `reconstructed` | 0/1 | 1 = solar radiation value reconstructed (satellite gap) |

The loader also adds `poa_global` = `Gb(i)` + `Gd(i)` + `Gr(i)`. Header lines above `time,...` and the footer (column legend and copyright) are skipped.

## Time stamps

SARAH2 hourly values are stamped at 10 minutes past the hour at Southampton (`20050101:0010`) and 11 minutes past the hour at the other three sites (`20050101:0011`), UTC. The loader parses them with `format='%Y%m%d:%H%M'` and keeps them as UTC-aware timestamps; the multi-year simulation treats each row as one hour.

## Nameplate note

The header line `Nominal power of the PV system (c-Si) (kWp)` prints the nameplate to one decimal place, so 0.55 kWp shows as 0.6 and 0.65 kWp as 0.7. The values PVGIS used are the requested 0.50, 0.55, 0.65 and 0.65 kWp (`REQUESTED_KWP` in `src/pvgis_series.py`); `tests/test_validation_multiyear.py::test_nameplate_used_by_pvgis_matches_request` verifies this by reproducing `P` from the irradiance columns with PVGIS's own Huld and Faiman models at high irradiance. The analysis divides `P` by this nameplate to obtain W per kWp, so the requested value only sets the scale of the `P` column.

## Reuse terms

PVGIS © European Union, 2001-2026 (copyright line in the footer of each file). The data are free to reuse with attribution to PVGIS, European Commission Joint Research Centre.

## To re-fetch

```bash
pip install requests
python -m src.fetch_pvgis_series
```

Downloads the five files into this folder (internet access to `re.jrc.ec.europa.eu` required; each request returns about 7 MB and the script retries up to five times). The committed files are the reproducibility record; re-fetching is only needed to refresh them.
