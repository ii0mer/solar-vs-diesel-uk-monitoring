# data/pvgis/

Real PVGIS-SARAH2 Typical Meteorological Year (TMY) data for the four study sites.

## Files

| File | Site | GHI (kWh/m²/yr) | Rows |
|---|---|---|---|
| `southampton_tmy.csv` | Southampton (50.91°N, -1.40°E) | 1133 | 8,760 |
| `birmingham_tmy.csv`  | Birmingham  (52.49°N, -1.89°E) | 1028 | 8,760 |
| `liverpool_tmy.csv`   | Liverpool   (53.41°N, -2.99°E) | 1018 | 8,760 |
| `edinburgh_tmy.csv`   | Edinburgh   (55.95°N, -3.19°E) |  945 | 8,760 |

## Data source

**PVGIS-SARAH2** — Photovoltaic Geographical Information System, European Commission Joint Research Centre.  
Satellite-derived irradiance record: **2005–2020**.  
Retrieved May 2026 via `pvlib.iotools.get_pvgis_tmy()`.

Reference year in timestamps: 1990 (standard PVGIS TMY reference year — not a real calendar year; the data represents a synthetic composite of the most typical months from the 2005–2020 long-term record).

## Columns

| Column | Unit | Description |
|---|---|---|
| `ghi` | W/m² | Global horizontal irradiance |
| `dni` | W/m² | Direct normal irradiance |
| `dhi` | W/m² | Diffuse horizontal irradiance |
| `temp_air` | °C | Ambient air temperature at 2 m |
| `wind_speed` | m/s | Wind speed at 10 m |

## To re-fetch

```bash
python -m src.fetch_pvgis
```

Or to load a manually-downloaded CSV from the PVGIS web interface:

```bash
python -m src.convert_manual_pvgis
```

Visit https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY to download manually.
