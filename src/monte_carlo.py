"""Monte Carlo LCOE uncertainty analysis (Upgrade 3 + Upgrade 4B).

The one-at-a-time tornado (sensitivity.py) attributes impact to individual
parameters but cannot answer the aggregate question: given simultaneous
uncertainty in all parameters, what is the probability that solar-battery
beats diesel? Deterministic and probabilistic LCOE estimates are known to
diverge by 11-30% when input uncertainty is material, so the dissertation
reports both.

Method
------
Plain Monte Carlo: N independent joint draws of the
uncertain parameters, each draw evaluated through the SAME cash-flow
builders used for the deterministic results (economics.py). Both systems
are evaluated on the same draw, so parameters that affect both (load,
site-visit cost) are properly correlated across the comparison.

PV degradation is NOT a Monte-Carlo variable: with end-of-life sizing it
acts through the array size, and exact re-sizing (sensitivity.py) bounds
its effect on the LCOE ratio at +0.08 / -0.00 across 0.3-0.8 %/yr, so it
is reported one-at-a-time rather than sampled.

Distributions (triangular unless stated; min, mode, max)
--------------------------------------------------------
  1. Diesel fuel price     tri(44.96, 76.02, 117.56) ppl
                           14-year AHDB UK red diesel range: 2016 minimum,
                           2025 mean (central), 2026 Iran-strait spike.
  2. Battery capex         tri(0.70, 1.00, 1.30) x GBP 700/kWh
                           BNEF 2025 baseline +/-30% (survey dispersion).
  3. PV capex              tri(0.70, 1.00, 1.30) x GBP 4.50/Wp
                           DESNZ 2025 small off-grid estimate +/-30%.
  4. Load magnitude        tri(0.80, 1.00, 1.20) x 128.2 kWh/yr
                           Mandelli et al. (2016) +/-20% load uncertainty;
                           applied to BOTH systems (same station).
  5. Site-visit cost       tri(0.5, 1.0, 2.0) x central visit costs
                           Same multiplier applied to solar visits AND
                           diesel delivery+inspection visits: one labour
                           market serves both (correlated logistics).
  6. Solar battery life    discrete uniform {8, 10, 12, 15} years
                           (Upgrade 4B) stationary LFP calendar-life range
                           at low C-rate; replacement year re-enters the
                           cash flow as year-(L, 2L, ...) spikes.

Discount rate is a DECISION variable, not a stochastic one: the full MC is
run at fixed 5% (Green Book + technology premium) and fixed 8% (commercial
WACC), mirroring the dissertation's dual framing. A third exploratory run
treats r ~ uniform(3%, 10%) for the appendix.

Outputs
-------
  results/monte_carlo_lcoe.csv          summary percentiles per site x rate
  results/monte_carlo_draws_<rate>.npz  raw draws (reproducible, seed=42)
  figures/fig08_monte_carlo_lcoe.png    distributions + P10/P50/P90

Run:  python -m src.monte_carlo
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .economics import build_pv_battery_cashflows, build_diesel_cashflows
from .diesel import DieselArchitecture2
from .sensitivity import CENTRAL

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
N_DRAWS_DEFAULT = 5000

# Per-site (PV Wp, battery kWh) from the governing-year LOLP <= 1% grid
# search (PV at year-24 output, battery at 80 % SoH; explicit loss chain
# incl. AOI; lifetime-NPV ranking). Provenance: results/sizing_summary.txt
SITE_DESIGN = {
    'Southampton': (500.0, 1.0),
    'Birmingham': (550.0, 1.25),
    'Liverpool': (650.0, 1.25),
    'Edinburgh': (650.0, 1.5),
}
# Exact re-optimised designs at 0.8 %/yr module degradation (used by the
# combined worst case). Provenance: results/degradation_resize_check.txt
SITE_DESIGN_DEG08 = {
    'Southampton': (450.0, 1.25),
    'Birmingham': (600.0, 1.25),
    'Liverpool': (700.0, 1.25),
    'Edinburgh': (600.0, 1.75),
}
# Back-compat alias (PV only) for modules/tests that iterate site names
SITE_PV_WP = {k: v[0] for k, v in SITE_DESIGN.items()}

BATTERY_LIFETIME_CHOICES = (8, 10, 12, 15)   # years, Upgrade 4B


@dataclass
class McDraws:
    """Joint parameter draws, one row per Monte Carlo trial."""
    fuel_price_ppl: np.ndarray
    battery_cost_mult: np.ndarray
    pv_cost_mult: np.ndarray
    load_mult: np.ndarray
    visit_cost_mult: np.ndarray
    battery_life_years: np.ndarray
    discount_rate: np.ndarray      # constant array for fixed-rate runs

    @property
    def n(self) -> int:
        return len(self.fuel_price_ppl)


def sample_draws(n: int = N_DRAWS_DEFAULT,
                 discount_rate: Optional[float] = 0.05,
                 seed: int = SEED) -> McDraws:
    """Draw N joint samples. discount_rate=None -> uniform(0.03, 0.10)."""
    rng = np.random.default_rng(seed)
    if discount_rate is None:
        r = rng.uniform(0.03, 0.10, n)
    else:
        r = np.full(n, float(discount_rate))
    return McDraws(
        fuel_price_ppl=rng.triangular(44.96, 76.02, 117.56, n),
        battery_cost_mult=rng.triangular(0.70, 1.00, 1.30, n),
        pv_cost_mult=rng.triangular(0.70, 1.00, 1.30, n),
        load_mult=rng.triangular(0.80, 1.00, 1.20, n),
        visit_cost_mult=rng.triangular(0.5, 1.0, 2.0, n),
        battery_life_years=rng.choice(BATTERY_LIFETIME_CHOICES, n),
        discount_rate=r,
    )


def _solar_lcoe_one(design: tuple, d: McDraws, i: int) -> float:
    pv_size_wp, battery_kwh = design
    cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=CENTRAL['pv_capex_gbp_per_wp'] * d.pv_cost_mult[i],
        pv_size_wp=pv_size_wp,
        battery_capex_gbp_per_kwh=(CENTRAL['battery_capex_gbp_per_kwh']
                                   * d.battery_cost_mult[i]),
        battery_kwh=battery_kwh,
        annual_energy_delivered_kwh=(CENTRAL['annual_energy_delivered_kwh']
                                     * d.load_mult[i]),
        battery_lifetime_years=int(d.battery_life_years[i]),
        visit_cost_gbp=CENTRAL['visit_cost_gbp'] * d.visit_cost_mult[i],
        annual_site_visits=CENTRAL['site_visits_per_year_pv'],
        project_years=CENTRAL['project_years'],
    )
    return cf.lcoe_gbp_per_kwh(d.discount_rate[i])


def _diesel_lcoe_one(d: McDraws, i: int) -> float:
    arch = DieselArchitecture2()
    arch.fuel_delivery_cost_per_visit_gbp *= d.visit_cost_mult[i]
    arch.inspection_cost_per_visit_gbp *= d.visit_cost_mult[i]
    arch.daily_load_wh *= d.load_mult[i]      # runtime/fuel follow the load
    cf = build_diesel_cashflows(
        arch,
        fuel_price_ppl=d.fuel_price_ppl[i],
        annual_energy_delivered_kwh=(CENTRAL['annual_energy_delivered_kwh']
                                     * d.load_mult[i]),
        project_years=CENTRAL['project_years'],
    )
    return cf.lcoe_gbp_per_kwh(d.discount_rate[i])


def run_monte_carlo(n: int = N_DRAWS_DEFAULT,
                    discount_rate: Optional[float] = 0.05,
                    seed: int = SEED) -> Dict[str, np.ndarray]:
    """Run the full MC for all four sites at one (fixed or varying) rate.

    Returns dict with per-site solar LCOE arrays, one shared diesel LCOE
    array (diesel economics are site-independent), and the draws.
    """
    d = sample_draws(n=n, discount_rate=discount_rate, seed=seed)
    diesel = np.array([_diesel_lcoe_one(d, i) for i in range(d.n)])
    out: Dict[str, np.ndarray] = {'diesel_lcoe': diesel}
    for site, design in SITE_DESIGN.items():
        out[f'solar_lcoe_{site}'] = np.array(
            [_solar_lcoe_one(design, d, i) for i in range(d.n)])
    out['_draws'] = d  # type: ignore[assignment]
    return out


def summarise(mc: Dict[str, np.ndarray], rate_label: str) -> pd.DataFrame:
    """P10/P50/P90 per site + P(solar LCOE < diesel LCOE)."""
    diesel = mc['diesel_lcoe']
    rows = []
    for site in SITE_PV_WP:
        solar = mc[f'solar_lcoe_{site}']
        ratio = diesel / solar
        rows.append({
            'site': site,
            'rate': rate_label,
            'solar_p10': np.percentile(solar, 10),
            'solar_p50': np.percentile(solar, 50),
            'solar_p90': np.percentile(solar, 90),
            'diesel_p10': np.percentile(diesel, 10),
            'diesel_p50': np.percentile(diesel, 50),
            'diesel_p90': np.percentile(diesel, 90),
            'ratio_p10': np.percentile(ratio, 10),
            'ratio_p50': np.percentile(ratio, 50),
            'ratio_p90': np.percentile(ratio, 90),
            'p_solar_beats_diesel': float(np.mean(solar < diesel)),
        })
    return pd.DataFrame(rows)


def main(n: int = N_DRAWS_DEFAULT) -> pd.DataFrame:
    frames = []
    for rate, label in [(0.05, '5%'), (0.08, '8%'), (None, 'r~U(3,10)%')]:
        mc = run_monte_carlo(n=n, discount_rate=rate)
        frames.append(summarise(mc, label))
        # Persist raw arrays for reproducibility / later figures
        tag = label.replace('%', 'pct').replace('~', '').replace('(', '') \
                   .replace(')', '').replace(',', '_')
        np.savez_compressed(
            RESULTS_DIR / f'monte_carlo_draws_{tag}.npz',
            diesel_lcoe=mc['diesel_lcoe'],
            **{k: v for k, v in mc.items()
               if k.startswith('solar_lcoe_')})
    summary = pd.concat(frames, ignore_index=True)
    out_csv = RESULTS_DIR / 'monte_carlo_lcoe.csv'
    summary.to_csv(out_csv, index=False, float_format='%.3f')
    print(f"Wrote {out_csv}")
    print(summary.to_string(index=False,
                            float_format=lambda x: f'{x:.3f}'))
    return summary


if __name__ == '__main__':
    main()
