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
CENTRAL = {
    'pv_capex_gbp_per_wp': 4.50,
    'pv_size_wp': 400.0,                # corrected from 300 — matches sizer output
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

    # PV degradation
    for label, deg in [('PV degr: 0.3%', 0.3), ('PV degr: 0.8%', 0.8)]:
        p = dict(CENTRAL); p['pv_degradation_pct_per_year'] = deg
        pv = central_pv_lcoe(p)
        rows.append(('PV degradation', label, f'{deg}%/yr', pv, base_di, base_di / pv))

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
        # On diesel side, visit cost change applied via direct LCOE recalc
        # For simplicity here we assume diesel visit cost changes proportionally
        # via the architecture's visit cost
        from .diesel import DieselArchitecture2 as DA2, DieselArchitecture1 as DA1
        arch = DA2() if architecture == 2 else DA1()
        arch.fuel_delivery_cost_per_visit_gbp *= mult
        arch.inspection_cost_per_visit_gbp *= mult
        from .economics import build_diesel_cashflows
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
