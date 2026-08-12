"""Automated system sizing.

For each site, find the smallest (PV_size_W, battery_kWh) combination that
achieves a target Loss-of-Load Probability. We grid-search across a coarse
mesh first, then refine near the boundary.

"Smallest" is defined by a cost surrogate (PV_W * 1.5 + battery_kWh * 700)
so we don't pick big-PV/tiny-battery solutions that would be uneconomical.
This is just a sizing heuristic — the proper economic comparison is in
economics.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .pv_model import PVDesign
from .battery import BatteryDesign
from .load_profile import LoadProfile
from .simulation import run_simulation, SimulationResult
from .sites import Site


DEFAULT_DEGRADATION_PCT_YR = 0.5
DEFAULT_PROJECT_YEARS = 25


def eol_ageing_factor(degradation_pct_yr: float = DEFAULT_DEGRADATION_PCT_YR,
                      project_years: int = DEFAULT_PROJECT_YEARS) -> float:
    """PV output multiplier in the final service year (year N output at
    0.5%/yr linear-compound fade: (1-0.005)**(N-1))."""
    return (1.0 - degradation_pct_yr / 100.0) ** (project_years - 1)


@dataclass
class SizingResult:
    site_name: str
    pv_w: float
    battery_kwh: float
    lolp: float                    # END-OF-LIFE (design-governing) LOLP
    lolp_year1: float              # year-1 LOLP at the same sizing
    annual_curtailed_kwh: float
    cost_index: float
    full_result: SimulationResult


def _cost_index(pv_w: float, battery_kwh: float) -> float:
    """Ranking metric = discounted 25-year lifetime cost (NPV at 5%) of
    the design, using the same cash-flow builder as the economics module
    (capex + lumpy battery/inverter replacements with salvage + O&M +
    visits). A year-0-capex surrogate is NOT sufficient: battery
    replacements at years ~12/24 make a smaller battery with more PV
    cheaper over life even when its day-one cost is higher.
    """
    from .economics import build_pv_battery_cashflows
    cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50, pv_size_wp=pv_w,
        battery_capex_gbp_per_kwh=700.0, battery_kwh=battery_kwh,
        annual_energy_delivered_kwh=128.2,
    )
    return cf.total_npv(0.05)


def size_system(weather: pd.DataFrame,
                site: Site,
                load: LoadProfile,
                target_lolp: float = 0.01,
                pv_grid: Optional[np.ndarray] = None,
                battery_grid: Optional[np.ndarray] = None,
                degradation_pct_yr: float = DEFAULT_DEGRADATION_PCT_YR,
                verbose: bool = False) -> Optional[SizingResult]:
    """Grid-search PV × battery space for the cheapest config whose
    END-OF-LIFE (year-25) LOLP meets the target.

    Sizing at end-of-life makes the reliability criterion hold across the
    entire project, not only in year 1 — module degradation means a
    year-1-sized system silently breaches its LOLP target in later life.
    """
    if pv_grid is None:
        pv_grid = np.arange(100, 1250, 50)
    if battery_grid is None:
        battery_grid = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 7, 10])
    eol = eol_ageing_factor(degradation_pct_yr)

    # PVWatts DC is linear in nameplate: compute the chain once per site
    # at 1 kWp and scale, instead of re-running pvlib per grid cell.
    from .pv_model import simulate_pv_dc
    ref = simulate_pv_dc(weather, site, PVDesign(nameplate_w=1000.0))

    best: Optional[SizingResult] = None
    rows = []
    for pv_w in pv_grid:
        pv_series = ref * (pv_w / 1000.0)
        for bat_k in battery_grid:
            pv = PVDesign(nameplate_w=float(pv_w), n_modules=1)
            bat = BatteryDesign(capacity_kwh=float(bat_k))
            r1 = run_simulation(weather, site, pv, bat, load,
                                pv_ageing_factor=eol, pv_series_w=pv_series)
            r2 = run_simulation(weather, site, pv, bat, load,
                                initial_soc_frac=r1.final_soc_kwh / bat.capacity_kwh,
                                pv_ageing_factor=eol, pv_series_w=pv_series)
            ci = _cost_index(pv_w, bat_k)
            rows.append((pv_w, bat_k, r2.lolp, r2.annual_curtailed_kwh, ci))
            if r2.lolp <= target_lolp:
                if best is None or ci < best.cost_index:
                    y1a = run_simulation(weather, site, pv, bat, load,
                                         pv_series_w=pv_series)
                    y1 = run_simulation(
                        weather, site, pv, bat, load,
                        initial_soc_frac=y1a.final_soc_kwh / bat.capacity_kwh,
                        pv_series_w=pv_series)
                    best = SizingResult(
                        site_name=site.name,
                        pv_w=float(pv_w),
                        battery_kwh=float(bat_k),
                        lolp=r2.lolp,
                        lolp_year1=y1.lolp,
                        annual_curtailed_kwh=r2.annual_curtailed_kwh,
                        cost_index=ci,
                        full_result=r2,
                    )
                    if verbose:
                        print(f"  [{site.name}] PV={pv_w} W, Bat={bat_k} kWh → "
                              f"EoL LOLP={r2.lolp*100:.3f}% "
                              f"(yr-1 {y1.lolp*100:.3f}%) "
                              f"cost_idx={ci:.0f} *new best*")
    return best, pd.DataFrame(rows, columns=['pv_w', 'battery_kwh', 'lolp',
                                             'curtailed_kwh', 'cost_index'])


if __name__ == '__main__':
    from .weather import get_or_create_tmy
    from .sites import SITES

    load = LoadProfile()
    print(f"Sizing for LOLP target ≤ 1.0% at all sites\n")

    for key, site in SITES.items():
        w = get_or_create_tmy(site, prefer='real')
        best, sweep = size_system(w, site, load, target_lolp=0.01)
        if best is None:
            print(f"{site.name:12s}  NO solution under LOLP=1% in search grid")
        else:
            print(f"{site.name:12s}  {best.pv_w:.0f} Wp + {best.battery_kwh:.1f} kWh "
                  f"→ LOLP={best.lolp*100:.3f}%, "
                  f"curtailed {best.annual_curtailed_kwh:.0f} kWh/yr "
                  f"(cost idx {best.cost_index:.0f})")
        # Save the sweep for later analysis
        sweep.to_csv(f'results/sizing_sweep_{site.name.lower()}.csv', index=False)
