# results/

Simulation output files. All values based on real PVGIS-SARAH2 TMY data (2005–2020).

## Files

| File | Description |
|---|---|
| `sizing_summary.txt` | Optimal PV+battery sizing for each site at LOLP < 1% |
| `sizing_sweep_southampton.csv` | Full LOLP grid search: PV × battery for Southampton |
| `sizing_sweep_birmingham.csv`  | Full LOLP grid search: PV × battery for Birmingham |
| `sizing_sweep_liverpool.csv`   | Full LOLP grid search: PV × battery for Liverpool |
| `sizing_sweep_edinburgh.csv`   | Full LOLP grid search: PV × battery for Edinburgh |
| `lcoe_comparison.csv` | LCOE by site and system, central assumptions |

## To regenerate

```bash
python -m src.sizing    # regenerates sizing_summary.txt and sizing_sweep_*.csv
python -m src.figures   # regenerates lcoe_comparison.csv
```
