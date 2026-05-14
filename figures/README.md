# figures/

Publication-quality dissertation figures, generated programmatically from real PVGIS-SARAH2 data.

## Figures

| File | Description |
|---|---|
| `fig01_architecture.png` | System architecture block diagrams: (a) solar PV–battery, (b) diesel + lead-acid battery (Architecture 2 realistic) |
| `fig02_monthly_ghi.png` | Monthly global horizontal irradiance at the four study sites — real PVGIS-SARAH2 TMY (2005–2020) |
| `fig03_load_profile.png` | Load profile composition: 14.6 W continuous, 128.2 kWh/year |
| `fig04_edinburgh_winter_week.png` | Edinburgh winter-week stress case (14–21 January): PV generation, load, battery SoC, unmet demand |
| `fig05_sizing_landscape.png` | System sizing landscape — LOLP across PV × battery combinations for the four sites |
| `fig06_lcoe_comparison.png` | 25-year LCOE comparison: solar vs diesel across four sites, linear and logarithmic scales |
| `fig07_tornado_sensitivity.png` | Sensitivity tornado: parameters ranked by impact on diesel-to-solar LCOE ratio |

## To regenerate

```bash
python -m src.figures
```

All figures use real PVGIS-SARAH2 data by default (`synthetic=False`). Ensure `data/pvgis/*_tmy.csv` files are present before running.

## Optimal sizing used in figures

| Site | PV capacity | Battery | LOLP (real data) |
|---|---|---|---|
| Southampton | 400 Wp | 1.0 kWh | 0.23% |
| Birmingham  | 500 Wp | 1.0 kWh | 0.61% |
| Liverpool   | 500 Wp | 1.0 kWh | 0.89% |
| Edinburgh   | 600 Wp | 1.0 kWh | 0.79% |
