# Solar PV–Battery vs Diesel for UK Remote Monitoring Stations

**MSc Renewable Energy Dissertation, Liverpool John Moores University**

**Student:** Omar Farooq Mahmood Al Obaidi (1181101) · **Module:** 7400MENR · **Supervisor:** M. Sooi
**Submission target:** 12 September 2026

[![Tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

## Summary

Hourly techno-economic simulation comparing standalone solar PV–LFP battery systems against diesel generators for low-power (14.6 W continuous) remote industrial monitoring stations at four UK sites (Southampton, Birmingham, Liverpool, Edinburgh). The framework outputs:

- Loss-of-load probability (LOLP) and Expected Energy Not Served (EENS) reliability metrics
- Levelised Cost of Electricity (LCOE) per IEA/NEA (2020) DCF methodology
- Net Present Value (NPV), simple payback, and lifetime CO₂e
- Sensitivity analysis across seven parameters with tornado-chart visualisation
- Two diesel architectures: continuous 24/7 (worst case) and intermittent diesel + battery (realistic case)
- Dual-discount-rate framing: 5% real (HM Treasury Green Book + technology premium) and 8% real (commercial WACC)

## Headline result (preliminary, on synthetic TMY calibrated to PVGIS-SARAH2 means)

| System | LCOE @ 5% Green Book | LCOE @ 8% commercial WACC |
|---|---|---|
| Solar PV–battery (Southampton) | £6.04/kWh | £6.54/kWh |
| Solar PV–battery (Birmingham)  | £6.04/kWh | £6.54/kWh |
| Solar PV–battery (Liverpool)   | £6.04/kWh | £6.54/kWh |
| Solar PV–battery (Edinburgh)   | £6.30/kWh | £6.88/kWh |
| Diesel — Architecture 2 (intermittent, realistic) | £25.27/kWh | £25.83/kWh |
| Diesel — Architecture 1 (continuous 24/7, worst) | £74.49/kWh | £75.00/kWh |

**Solar-battery is 4.0–4.2× cheaper than the realistic diesel case under the central assumptions, and 2.6× cheaper even under combined worst-case parameters for solar.**

## Quick start

```bash
# Install dependencies
pip install pvlib pandas numpy matplotlib scipy requests pytest

# Pull real PVGIS-SARAH2 weather data (requires internet)
python -m src.fetch_pvgis

# Run baseline simulation and sizing
python -m src.simulation
python -m src.sizing

# Generate all 7 dissertation figures
python -m src.figures

# Run sensitivity sweeps
python -m src.sensitivity

# Run the test suite
python -m pytest tests/ -v
```

## Repository layout

```
src/                    Core modelling code (Python modules)
  ├ sites.py            Three UK study sites with coordinates
  ├ load_profile.py     14.6 W continuous monitoring station load
  ├ weather.py          PVGIS fetch + synthetic fallback
  ├ pv_model.py         pvlib PVWatts + Hay-Davies + SAPM cell temperature
  ├ battery.py          LFP battery state-of-charge tracker
  ├ simulation.py       8,760-hour energy-balance loop
  ├ sizing.py           Auto-sizer grid search (PV × battery)
  ├ diesel.py           Both diesel architectures with full lifecycle
  ├ economics.py        IEA/NEA (2020) DCF, LCOE, NPV at 5% and 8%
  ├ sensitivity.py      One-at-a-time tornado + combined worst-case
  ├ figures.py          Generator for all 7 dissertation figures
  └ fetch_pvgis.py      Standalone PVGIS retrieval script

tests/                  Pytest sanity checks (14 tests, all passing)
data/pvgis/             TMY hourly weather files per site
data/costs/             Cost assumption modules with citations
figures/                Output PNGs used in the dissertation
results/                CSV outputs from simulation runs
docs/                   Methodology notes, assumption log
```

## Data sources (all primary, all DOI-verified or government-issued)

| Dataset | Source | Citation |
|---|---|---|
| Hourly irradiance (UK) | PVGIS-SARAH2, European Commission JRC | [12] in dissertation |
| Equipment power draws | Campbell Scientific CR1000X, Digi IX20 datasheets | Manufacturer |
| Battery costs ($/kWh pack) | BloombergNEF Lithium-Ion Battery Survey 2025 | [9] |
| UK red diesel prices | AHDB Fuel Prices, 14-year series 2012–2025 | [16] |
| Carbon factors | DESNZ GHG Conversion Factors 2025 | [25] |
| Solar PV cost data | DESNZ Solar PV Cost Data, May 2025 | [15] |
| LCOE methodology | IEA/NEA (2020) Projected Costs of Generating Electricity | [14] |
| Discount rate guidance | HM Treasury Green Book 2022 | [24] |

Full reference list with DOIs in the dissertation report.

## Methodology highlights

1. **Hourly resolution.** Each of 8,760 hours simulated in sequence, updating battery state of charge using an energy-balance algorithm with round-trip efficiency (0.92 LFP), depth-of-discharge constraint (0.80), and self-discharge (2%/month).

2. **Two diesel architectures.** Architecture 1 models continuous 24/7 operation as a worst-case bound. Architecture 2 models intermittent diesel + small lead-acid battery as the realistic engineering deployment (~4 hr/day genset runtime at 30% load).

3. **Dual-discount-rate framing.** LCOE reported at both 5% real (HM Treasury Green Book Social Time Preference Rate plus technology premium, for public-sector decisions) and 8% real (typical UK commercial WACC for private renewable-energy investment). Demonstrates conclusion robustness across financing perspectives.

4. **Combined worst-case sensitivity.** In addition to one-at-a-time parameter sweeps, a combined stress test pushes every parameter to its diesel-favourable extreme (PV cost +30%, battery cost +30%, fuel price 44.96 ppl historical minimum, PV degradation 0.8%/yr, halved visit cost, 3% discount rate). Solar still beats diesel by 2.6× under this combination.

5. **Validated load profile.** 14.6 W continuous = 1.2 W datalogger + 4×0.5 W sensors + 6.0 W modem + 3.0 W ancillary, all sourced from manufacturer datasheets, then uplifted by 20% for losses, derating, and future expansion. Annual energy 128.2 kWh.

## Reproducibility

- All assumptions logged in `docs/assumptions.md` with source citations and version history.
- Simulation runs are deterministic given the same input TMY file.
- 16 pytest sanity tests verify the report's headline numbers can be reproduced from the code (`python -m pytest tests/`).
- Synthetic TMY fallback enables identical reproduction even without PVGIS API access.

## Status

- [x] Repository structure and module skeleton
- [x] Site configuration and load profile (1.2 + 2.0 + 6.0 + 3.0 = 12.2 W base, +20% margin = 14.64 W)
- [x] PV model with pvlib (PVWatts DC + Hay-Davies transposition + SAPM cell temperature)
- [x] LFP battery state-of-charge tracker with manufacturer-spec defaults
- [x] Hourly energy-balance simulation (8,760 timesteps)
- [x] Auto-sizing grid search (LOLP < 1% target)
- [x] Diesel model with both architectures (continuous, intermittent)
- [x] DCF / LCOE / NPV per IEA/NEA (2020) at 5% and 8%
- [x] Sensitivity sweeps (7 parameters one-at-a-time + combined worst case)
- [x] All 7 dissertation figures generated programmatically
- [x] Sanity test suite (14 pytest tests)
- [ ] Real PVGIS hourly data integration (runs on local machine; pending execution)
- [ ] Validation against PVGIS annual yield benchmarks (gating criterion before final dissertation)

## Citation

If using this code or methodology, please cite:

```
Al Obaidi, O. F. M. (2026). Techno-Economic Comparison of Off-Grid Solar
Photovoltaic–Battery and Diesel Generator Systems for Remote Environmental
Monitoring Stations Across Three UK Climate Zones. MSc Dissertation,
Liverpool John Moores University, Module 7400MENR.
```

## License

Released under the MIT License — see [LICENSE](LICENSE). The code is open-source; the dissertation document is © 2026 Omar Farooq Mahmood Al Obaidi.
