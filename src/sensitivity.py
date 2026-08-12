"""Sensitivity analysis — vary key parameters and compute LCOE / NPV impact.

The dissertation needs to demonstrate that conclusions are ROBUST to plausible
parameter uncertainty. The sensitivity tornado chart is the figure that
distinguishes a Distinction-grade dissertation from a 2:1.

Parameters varied:
  1. Diesel fuel price (44.96 → 117.56 ppl, 14-year UK historical range)
  2. Battery cost (-30% to +30% from $70/kWh BNEF 2025 baseline)
  3. PV cost (-30% to +30% from £4.50/Wp baseline)
  4. Discount rate (3%, 5%, 7%, 10%)
  5. Load magnitude (-20% to +20%)
  6. PV degradation rate (0.3, 0.5, 0.8 %/yr)
  7. PV system lifetime (20, 25, 30 yr)
  8. Site visit cost (50%, 100%, 200% of central case)
  9. Solar LFP battery replacement interval (8, 10, 12, 15 yr) — Upgrade 4B;
     stationary LFP calendar-life range at ultra-low C-rate (~0.015C)

Output: ranked impact on PV-battery LCOE and on PV/Diesel LCOE ratio.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .economics import (
    build_pv_battery_cashflows,
    build_diesel_cashflows,
    compare_systems,
)
from .diesel import DieselArchitecture1, DieselArchitecture2


# Central case parameters (the dissertation baseline)
# Southampton reference design from the corrected sizing: explicit loss
# chain, year-25 (end-of-life) LOLP criterion, ranked by discounted
# 25-year lifetime cost (NOT year-0 capex): 450 Wp + 1.0 kWh.
CENTRAL = {
    'pv_capex_gbp_per_wp': 4.50,
    'pv_size_wp': 450.0,
    'battery_capex_gbp_per_kwh': 700.0,
    'battery_kwh': 1.0,
    'annual_energy_delivered_kwh': 128.2,
    'pv_degradation_pct_per_year': 0.5,
    'battery_lifetime_years': 12,
    'inverter_lifetime_years': 15,
    'discount_rate': 0.05,
    'project_years': 25,
    'fuel_price_ppl': 76.02,
    'site_visits_per_year_pv': 2,
    'visit_cost_gbp': 200.0,
}


def central_pv_lcoe(p: dict) -> float:
    cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=p['pv_capex_gbp_per_wp'],
        pv_size_wp=p['pv_size_wp'],
        battery_capex_gbp_per_kwh=p['battery_capex_gbp_per_kwh'],
        battery_kwh=p['battery_kwh'],
        annual_energy_delivered_kwh=p['annual_energy_delivered_kwh'],
        pv_degradation_pct_per_year=p['pv_degradation_pct_per_year'],
        battery_lifetime_years=p['battery_lifetime_years'],
        inverter_lifetime_years=p['inverter_lifetime_years'],
        annual_site_visits=p['site_visits_per_year_pv'],
        visit_cost_gbp=p['visit_cost_gbp'],
        project_years=p['project_years'],
    )
    return cf.lcoe_gbp_per_kwh(p['discount_rate'])


def central_diesel_lcoe(p: dict, architecture: int = 2) -> float:
    arch = DieselArchitecture2() if architecture == 2 else DieselArchitecture1()
    cf = build_diesel_cashflows(
        arch,
        fuel_price_ppl=p['fuel_price_ppl'],
        annual_energy_delivered_kwh=p['annual_energy_delivered_kwh'],
        project_years=p['project_years'],
    )
    return cf.lcoe_gbp_per_kwh(p['discount_rate'])


def one_at_a_time_sensitivity(
    architecture: int = 2
) -> pd.DataFrame:
    """Vary one parameter at a time, hold others at central. Returns LCOE table."""
    rows = []
    base_pv = central_pv_lcoe(CENTRAL)
    base_di = central_diesel_lcoe(CENTRAL, architecture=architecture)

    rows.append(('CENTRAL', 'baseline', '', base_pv, base_di, base_di / base_pv))

    # Diesel fuel price (UK historical range)
    for label, price in [('Fuel: 2016 minimum', 44.96),
                         ('Fuel: 2020 COVID low', 52.47),
                         ('Fuel: 2025 mean (CENTRAL)', 76.02),
                         ('Fuel: 2024 mean', 80.84),
                         ('Fuel: 2022 spike', 104.27),
                         ('Fuel: 2026 Iran spike', 117.56)]:
        p = dict(CENTRAL); p['fuel_price_ppl'] = price
        di = central_diesel_lcoe(p, architecture=architecture)
        rows.append(('Fuel price', label, f'{price} ppl', base_pv, di, di / base_pv))

    # Battery cost
    for label, mult in [('Battery: -30%', 0.7), ('Battery: -15%', 0.85),
                        ('Battery: +15%', 1.15), ('Battery: +30%', 1.3)]:
        p = dict(CENTRAL); p['battery_capex_gbp_per_kwh'] = CENTRAL['battery_capex_gbp_per_kwh'] * mult
        pv = central_pv_lcoe(p)
        rows.append(('Battery cost', label, f"{mult*100:.0f}%",
                     pv, base_di, base_di / pv))

    # PV cost
    for label, mult in [('PV: -30%', 0.7), ('PV: -15%', 0.85),
                        ('PV: +15%', 1.15), ('PV: +30%', 1.3)]:
        p = dict(CENTRAL); p['pv_capex_gbp_per_wp'] = CENTRAL['pv_capex_gbp_per_wp'] * mult
        pv = central_pv_lcoe(p)
        rows.append(('PV cost', label, f"{mult*100:.0f}%", pv, base_di, base_di / pv))

    # Discount rate
    for label, r in [('Discount: 3%', 0.03), ('Discount: 5% (CENTRAL)', 0.05),
                     ('Discount: 7%', 0.07), ('Discount: 10%', 0.10)]:
        p = dict(CENTRAL); p['discount_rate'] = r
        pv = central_pv_lcoe(p)
        di = central_diesel_lcoe(p, architecture=architecture)
        rows.append(('Discount rate', label, f"{r*100:.1f}%", pv, di, di / pv))

    # Load magnitude (changes annual energy denominator)
    for label, mult in [('Load: -20%', 0.8), ('Load: +20%', 1.2)]:
        p = dict(CENTRAL); p['annual_energy_delivered_kwh'] = CENTRAL['annual_energy_delivered_kwh'] * mult
        pv = central_pv_lcoe(p)
        di = central_diesel_lcoe(p, architecture=architecture)
        rows.append(('Load magnitude', label, f'{mult*100:.0f}%', pv, di, di / pv))

    # PV degradation — SIZING-COUPLED: for a fixed load served at an
    # end-of-life LOLP criterion, degradation costs capex (a different
    # optimal design), not delivered energy. Designs below are EXACT
    # re-optimisations at Southampton from the sizing grid search
    # (provenance: python -m src.make_results →
    # results/degradation_resize_check.txt). Note the optimizer moves
    # along the PV–battery frontier: slower fade favours more PV and
    # less storage, not a simple array rescale.
    # 0.5%/yr central → 450 Wp + 1.0 kWh; 0.3 → 400+1.0; 0.8 →
    # unchanged 450+1.0 (the central design already tolerates 0.8%/yr
    # at EoL LOLP 0.833% — itself a reportable robustness result).
    for label, deg, wp, kwh in [('PV degr: 0.3%', 0.3, 400.0, 1.0),
                                ('PV degr: 0.8%', 0.8, 450.0, 1.0)]:
        p = dict(CENTRAL)
        p['pv_size_wp'] = wp
        p['battery_kwh'] = kwh
        pv = central_pv_lcoe(p)
        rows.append(('PV degradation', label,
                     f'{deg}%/yr (re-sized {wp:.0f} Wp+{kwh:.1f} kWh)',
                     pv, base_di, base_di / pv))

    # Diesel visit cadence — THE dominant assumption, tested explicitly:
    # monthly (central) down to quarterly-minus. PV visits stay at 2/yr
    # (no fuel, no oil; annual inspection + one weather/vegetation visit).
    for label, visits in [('Diesel visits: 12/yr (central)', 12),
                          ('Diesel visits: 6/yr', 6),
                          ('Diesel visits: 4/yr', 4),
                          ('Diesel visits: 2/yr', 2)]:
        arch = DieselArchitecture2() if architecture == 2 \
            else DieselArchitecture1()
        arch.annual_site_visits = visits
        cf = build_diesel_cashflows(
            arch, fuel_price_ppl=CENTRAL['fuel_price_ppl'],
            annual_energy_delivered_kwh=CENTRAL['annual_energy_delivered_kwh'],
            project_years=CENTRAL['project_years'])
        di = cf.lcoe_gbp_per_kwh(CENTRAL['discount_rate'])
        rows.append(('Diesel visit cadence', label, f'{visits}/yr',
                     base_pv, di, di / base_pv))

    # Solar battery replacement interval (Upgrade 4B)
    for label, yrs in [('Batt life: 8 yr', 8), ('Batt life: 10 yr', 10),
                       ('Batt life: 15 yr', 15)]:
        p = dict(CENTRAL); p['battery_lifetime_years'] = yrs
        pv = central_pv_lcoe(p)
        rows.append(('Solar battery life', label, f'{yrs} yr',
                     pv, base_di, base_di / pv))

    # Site visit cost
    for label, mult in [('Visits: -50%', 0.5), ('Visits: +100%', 2.0)]:
        p = dict(CENTRAL)
        # Same multiplier on PV-side (other costs are mostly fuel/capex)
        p_pv = dict(p); p_pv['visit_cost_gbp'] = CENTRAL['visit_cost_gbp'] * mult
        pv = central_pv_lcoe(p_pv)
        # On diesel side, the same labour-market multiplier applies to
        # its per-visit costs (module-level imports; a local import here
        # would shadow the name for the whole function scope).
        arch = DieselArchitecture2() if architecture == 2 \
            else DieselArchitecture1()
        arch.fuel_delivery_cost_per_visit_gbp *= mult
        arch.inspection_cost_per_visit_gbp *= mult
        cf = build_diesel_cashflows(
            arch, fuel_price_ppl=p['fuel_price_ppl'],
            annual_energy_delivered_kwh=p['annual_energy_delivered_kwh'],
            project_years=p['project_years'],
        )
        di = cf.lcoe_gbp_per_kwh(p['discount_rate'])
        rows.append(('Site visit cost', label, f'{mult*100:.0f}%', pv, di, di / pv))

    df = pd.DataFrame(rows, columns=[
        'parameter', 'label', 'value',
        'pv_lcoe_gbp_per_kwh', 'diesel_lcoe_gbp_per_kwh', 'diesel_pv_ratio'
    ])
    return df


def tornado_data(architecture: int = 2) -> pd.DataFrame:
    """Compute tornado-chart deltas: how much does each parameter shift the
    PV-vs-diesel LCOE ratio? Sorted by magnitude of impact.
    """
    df = one_at_a_time_sensitivity(architecture=architecture)
    central = df[df['parameter'] == 'CENTRAL'].iloc[0]
    central_ratio = central['diesel_pv_ratio']

    rows = []
    for param in df['parameter'].unique():
        if param == 'CENTRAL':
            continue
        sub = df[df['parameter'] == param]
        ratios = sub['diesel_pv_ratio'].values
        delta_low = ratios.min() - central_ratio
        delta_high = ratios.max() - central_ratio
        rows.append({
            'parameter': param,
            'low_ratio': ratios.min(),
            'central_ratio': central_ratio,
            'high_ratio': ratios.max(),
            'delta_low': delta_low,
            'delta_high': delta_high,
            'range': abs(delta_high - delta_low),
        })
    return pd.DataFrame(rows).sort_values('range', ascending=False)


if __name__ == '__main__':
    print("=== One-at-a-time sensitivity (Architecture 2) ===\n")
    df = one_at_a_time_sensitivity(architecture=2)
    print(df.to_string(index=False))

    print("\n=== Tornado data — parameters ranked by impact ===\n")
    td = tornado_data(architecture=2)
    print(td.to_string(index=False))
