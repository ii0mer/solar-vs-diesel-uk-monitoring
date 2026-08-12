"""Two-dimensional sensitivity sweeps: discount rate x battery cost.

Phase-E completion item. The tornado ranks parameters one at a time; the
2-D sweep answers the interaction question examiners ask: does a high
discount rate COMBINED with expensive batteries ever erode the solar
advantage to parity?

Grid
----
  discount rate     3% .. 10%  (15 points)  — spans Green Book STPR to a
                                              punitive commercial hurdle
  battery capex     0.5x .. 2.0x of GBP 700/kWh (25 points)
                                            — GBP 350 .. 1,400 per kWh

For each grid cell and site, the diesel-to-solar LCOE ratio is computed
with every other parameter at central. Ratio = 1 is the breakeven
contour; the analysis also solves the breakeven battery cost analytically
per (site, rate): solar LCOE is affine in the battery capex multiplier,
so  m* = (LCOE_diesel - a) / b  where a, b are recovered from two
evaluations.

Outputs
-------
  results/sweep2d_ratio_<site>.csv     ratio matrix (rows=rate, cols=mult)
  results/sweep2d_breakeven.csv        breakeven battery cost per site/rate

Run:  python -m src.sweeps_2d
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .economics import build_pv_battery_cashflows, build_diesel_cashflows
from .diesel import DieselArchitecture2
from .sensitivity import CENTRAL
from .monte_carlo import SITE_PV_WP

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

RATES = np.linspace(0.03, 0.10, 15)
BATT_MULTS = np.linspace(0.5, 2.0, 25)


def solar_lcoe(pv_size_wp: float, rate: float, batt_mult: float) -> float:
    cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=CENTRAL['pv_capex_gbp_per_wp'],
        pv_size_wp=pv_size_wp,
        battery_capex_gbp_per_kwh=(CENTRAL['battery_capex_gbp_per_kwh']
                                   * batt_mult),
        battery_kwh=CENTRAL['battery_kwh'],
        annual_energy_delivered_kwh=CENTRAL['annual_energy_delivered_kwh'],
        pv_degradation_pct_per_year=CENTRAL['pv_degradation_pct_per_year'],
        battery_lifetime_years=CENTRAL['battery_lifetime_years'],
        visit_cost_gbp=CENTRAL['visit_cost_gbp'],
        annual_site_visits=CENTRAL['site_visits_per_year_pv'],
        project_years=CENTRAL['project_years'],
    )
    return cf.lcoe_gbp_per_kwh(rate)


def diesel_lcoe(rate: float) -> float:
    cf = build_diesel_cashflows(
        DieselArchitecture2(),
        fuel_price_ppl=CENTRAL['fuel_price_ppl'],
        annual_energy_delivered_kwh=CENTRAL['annual_energy_delivered_kwh'],
        project_years=CENTRAL['project_years'],
    )
    return cf.lcoe_gbp_per_kwh(rate)


def ratio_matrix(site: str) -> pd.DataFrame:
    """Diesel/solar LCOE ratio over the (rate, battery-mult) grid."""
    wp = SITE_PV_WP[site]
    di = {r: diesel_lcoe(r) for r in RATES}
    mat = np.empty((len(RATES), len(BATT_MULTS)))
    for i, r in enumerate(RATES):
        for j, m in enumerate(BATT_MULTS):
            mat[i, j] = di[r] / solar_lcoe(wp, r, m)
    return pd.DataFrame(
        mat,
        index=pd.Index([f'{r:.4f}' for r in RATES], name='discount_rate'),
        columns=pd.Index([f'{m:.3f}' for m in BATT_MULTS],
                         name='battery_cost_mult'),
    )


def breakeven_battery_cost() -> pd.DataFrame:
    """Battery capex at which solar LCOE equals diesel LCOE (ratio = 1).

    Solar LCOE is affine in the battery-cost multiplier m:
        LCOE(m) = a + b m,   a = LCOE(0), b = LCOE(1) - LCOE(0)
    so   m* = (LCOE_diesel - a) / b.
    """
    rows = []
    for site, wp in SITE_PV_WP.items():
        for r in (0.05, 0.08, 0.10):
            a = solar_lcoe(wp, r, 0.0)
            b = solar_lcoe(wp, r, 1.0) - a
            m_star = (diesel_lcoe(r) - a) / b
            rows.append({
                'site': site,
                'discount_rate': r,
                'breakeven_batt_mult': m_star,
                'breakeven_batt_gbp_per_kwh':
                    m_star * CENTRAL['battery_capex_gbp_per_kwh'],
            })
    return pd.DataFrame(rows)


def main() -> None:
    for site in SITE_PV_WP:
        df = ratio_matrix(site)
        out = RESULTS_DIR / f'sweep2d_ratio_{site.lower()}.csv'
        df.to_csv(out, float_format='%.4f')
        print(f"Wrote {out}  (min ratio in grid: {df.values.min():.2f}x "
              f"at rate={RATES[np.where(df.values == df.values.min())[0][0]]:.1%}, "
              f"mult={BATT_MULTS[np.where(df.values == df.values.min())[1][0]]:.2f})")
    be = breakeven_battery_cost()
    out = RESULTS_DIR / 'sweep2d_breakeven.csv'
    be.to_csv(out, index=False, float_format='%.2f')
    print(f"Wrote {out}")
    print(be.to_string(index=False,
                       float_format=lambda x: f'{x:,.2f}'))


if __name__ == '__main__':
    main()
