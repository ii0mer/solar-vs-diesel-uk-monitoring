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


@dataclass
class SizingResult:
    site_name: str
    pv_w: float
    battery_kwh: float
    lolp: float
    annual_curtailed_kwh: float
    cost_index: float
    full_result: SimulationResult


def _cost_index(pv_w: float, battery_kwh: float) -> float:
    """Surrogate cost for ranking. NOT the real economic LCOE."""
    # rough: ~£1.5/Wp PV, ~£700/kWh installed battery (UK retail 2025)
    return pv_w * 1.5 + battery_kwh * 700.0


def size_system(weather: pd.DataFrame,
                site: Site,
                load: LoadProfile,
                target_lolp: float = 0.01,
                pv_grid: Optional[np.ndarray] = None,
                battery_grid: Optional[np.ndarray] = None,
                verbose: bool = False) -> Optional[SizingResult]:
    """Grid-search PV × battery space for the cheapest config under LOLP target."""
    if pv_grid is None:
        pv_grid = np.array([100, 150, 200, 300, 400, 500, 600, 800, 1000])
    if battery_grid is None:
        battery_grid = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30])

    best: Optional[SizingResult] = None
    rows = []
    for pv_w in pv_grid:
        for bat_k in battery_grid:
            pv = PVDesign(nameplate_w=float(pv_w), n_modules=1)
            bat = BatteryDesign(capacity_kwh=float(bat_k))
            r1 = run_simulation(weather, site, pv, bat, load)
            r2 = run_simulation(weather, site, pv, bat, load,
                                initial_soc_frac=r1.final_soc_kwh / bat.capacity_kwh)
            ci = _cost_index(pv_w, bat_k)
            rows.append((pv_w, bat_k, r2.lolp, r2.annual_curtailed_kwh, ci))
            if r2.lolp <= target_lolp:
                if best is None or ci < best.cost_index:
                    best = SizingResult(
                        site_name=site.name,
                        pv_w=float(pv_w),
                        battery_kwh=float(bat_k),
                        lolp=r2.lolp,
                        annual_curtailed_kwh=r2.annual_curtailed_kwh,
                        cost_index=ci,
                        full_result=r2,
                    )
                    if verbose:
                        print(f"  [{site.name}] PV={pv_w} W, Bat={bat_k} kWh → "
                              f"LOLP={r2.lolp*100:.3f}% cost_idx={ci:.0f} *new best*")
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
