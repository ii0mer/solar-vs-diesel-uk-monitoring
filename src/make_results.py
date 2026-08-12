"""Single-command results pipeline — regenerates every results artifact
from the corrected model. This is the reproducibility entry point cited
in the dissertation:

    python -m src.make_results

Writes:
    results/sizing_summary.txt      corrected designs, EoL + year-1 LOLP
    results/lcoe_comparison.csv     per-site LCOE at 5% and 8%, both diesels
    results/diesel_ops_summary.txt  runtimes, fuel, replacement cadence
    results/cold_charge_bound.txt   LFP cold-charge LOLP bound (Edinburgh)
    results/degradation_resize_check.txt  exact vs linear-scaled sizing
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .sites import SITES
from .weather import get_or_create_tmy
from .load_profile import LoadProfile
from .pv_model import PVDesign, LossChain
from .battery import BatteryDesign
from .simulation import run_simulation
from .sizing import size_system, eol_ageing_factor
from .economics import build_pv_battery_cashflows, build_diesel_cashflows
from .diesel import (DieselArchitecture1, DieselArchitecture2,
                     lifetime_summary, annual_co2e_kg)
from .monte_carlo import SITE_DESIGN

RESULTS = Path(__file__).resolve().parents[1] / 'results'
RESULTS.mkdir(exist_ok=True)


def sizing_and_reliability() -> pd.DataFrame:
    load = LoadProfile()
    eol = eol_ageing_factor()
    rows = []
    lines = [
        "CORRECTED sizing for LOLP <= 1.0% at END OF LIFE (year 25),",
        f"explicit loss chain (total {LossChain().total_loss_pct:.1f}% incl. "
        "MPPT controller), real PVGIS-SARAH2 TMY (2005-2020).",
        "",
    ]
    for key, site in SITES.items():
        w = get_or_create_tmy(site, prefer='real')
        best, _ = size_system(w, site, load, target_lolp=0.01)
        assert best is not None, f"no feasible design at {site.name}"
        rows.append({
            'site': site.name,
            'pv_wp': best.pv_w,
            'battery_kwh': best.battery_kwh,
            'lolp_eol_pct': best.lolp * 100,
            'lolp_year1_pct': best.lolp_year1 * 100,
            'curtailed_eol_kwh': best.annual_curtailed_kwh,
        })
        lines.append(
            f"{site.name:12s} {best.pv_w:.0f} Wp + {best.battery_kwh:.1f} kWh"
            f"  EoL LOLP={best.lolp*100:.3f}%  year-1 LOLP="
            f"{best.lolp_year1*100:.3f}%  curtailed {best.annual_curtailed_kwh:.0f} kWh/yr")
    lines += [
        "",
        "Design criterion: the system must meet its reliability target in",
        "its WORST year (year 25 output = (1-0.005)^24 = "
        f"{eol:.4f} of year-1), not merely on day one.",
    ]
    (RESULTS / 'sizing_summary.txt').write_text('\n'.join(lines) + '\n')
    df = pd.DataFrame(rows)
    # cross-check the module-level constants
    for _, r in df.iterrows():
        wp, kwh = SITE_DESIGN[r['site']]
        assert (wp, kwh) == (r['pv_wp'], r['battery_kwh']), \
            f"SITE_DESIGN out of date for {r['site']}: {(r['pv_wp'], r['battery_kwh'])}"
    return df


def lcoe_comparison() -> pd.DataFrame:
    rows = []
    for rate in (0.05, 0.08):
        di1 = build_diesel_cashflows(DieselArchitecture1(),
                                     fuel_price_ppl=76.02
                                     ).lcoe_gbp_per_kwh(rate)
        di2 = build_diesel_cashflows(DieselArchitecture2(),
                                     fuel_price_ppl=76.02
                                     ).lcoe_gbp_per_kwh(rate)
        for site, (wp, kwh) in SITE_DESIGN.items():
            s = build_pv_battery_cashflows(
                pv_capex_gbp_per_wp=4.50, pv_size_wp=wp,
                battery_capex_gbp_per_kwh=700.0, battery_kwh=kwh,
                annual_energy_delivered_kwh=128.2,
            ).lcoe_gbp_per_kwh(rate)
            rows.append({
                'site': site, 'rate': f'{rate:.0%}',
                'pv_wp': wp, 'battery_kwh': kwh,
                'solar_lcoe': round(s, 3),
                'diesel_a2_lcoe': round(di2, 3),
                'diesel_a1_lcoe': round(di1, 3),
                'ratio_a2_solar': round(di2 / s, 2),
                'ratio_a1_solar': round(di1 / s, 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / 'lcoe_comparison.csv', index=False)
    return df


def diesel_ops() -> None:
    lines = []
    for name, arch in [('A1 (continuous 24/7)', DieselArchitecture1()),
                       ('A2 (battery-buffered)', DieselArchitecture2())]:
        s = lifetime_summary(arch, fuel_price_ppl=76.02)
        rt = getattr(arch, 'daily_runtime_hours', 24.0)
        life_yr = arch.genset_lifetime_hours / arch.annual_runtime_hours
        lines += [
            f"{name}:",
            f"  runtime {rt if isinstance(rt, float) else 24.0:.2f} h/day "
            f"({arch.annual_runtime_hours} h/yr)",
            f"  fuel {arch.fuel_litres_per_hour_at_op_point:.4f} L/h "
            f"(Skarstein-Uhlen) -> {arch.annual_fuel_litres:.0f} L/yr "
            f"= £{s['annual_fuel_cost_gbp']:.0f}/yr",
            f"  oil £{s['annual_oil_cost_gbp']:.0f}/yr; visits "
            f"£{s['annual_visit_cost_gbp']:.0f}/yr",
            f"  genset life {life_yr:.1f} yr at this duty",
            f"  CO2e {annual_co2e_kg(arch):.0f} kg/yr",
            "",
        ]
    (RESULTS / 'diesel_ops_summary.txt').write_text('\n'.join(lines))


def cold_charge_bound() -> None:
    site = SITES['edinburgh']
    df = get_or_create_tmy(site, prefer='real')
    wp, kwh = SITE_DESIGN['Edinburgh']
    pv, bat, load = PVDesign(nameplate_w=wp), BatteryDesign(capacity_kwh=kwh), LoadProfile()
    eol = eol_ageing_factor()
    out = []
    for label, age in [('year-1', 1.0), ('end-of-life', eol)]:
        base = run_simulation(df, site, pv, bat, load, pv_ageing_factor=age)
        cold = run_simulation(df, site, pv, bat, load, pv_ageing_factor=age,
                              charge_temp_limit_c=0.0)
        sub_zero_h = int((df['temp_air'] <= 0).sum())
        out.append(
            f"Edinburgh {label}: LOLP {base.lolp*100:.3f}% -> "
            f"{cold.lolp*100:.3f}% with charging blocked at <=0 degC "
            f"(ambient proxy; {sub_zero_h} sub-zero h/yr in TMY); "
            f"EENS {base.annual_unmet_kwh:.2f} -> {cold.annual_unmet_kwh:.2f} kWh/yr")
    (RESULTS / 'cold_charge_bound.txt').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


def degradation_resize_check() -> None:
    """Verify the linear sizing approximation used in the tornado."""
    site = SITES['southampton']
    w = get_or_create_tmy(site, prefer='real')
    load = LoadProfile()
    lines = []
    central_wp = SITE_DESIGN['Southampton'][0]
    for deg in (0.3, 0.8):
        best, _ = size_system(w, site, load, target_lolp=0.01,
                              degradation_pct_yr=deg)
        lines.append(
            f"deg {deg}%/yr: exact re-size {best.pv_w:.0f} Wp + "
            f"{best.battery_kwh:.1f} kWh (EoL LOLP {best.lolp*100:.3f}%). "
            f"NOTE: the optimum moves along the PV-battery frontier, so a "
            f"linear array rescale is NOT valid; the tornado hard-codes "
            f"these exact designs (central {central_wp:.0f} Wp + 1.5 kWh).")
    (RESULTS / 'degradation_resize_check.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    print("1/5 sizing + reliability ...")
    print(sizing_and_reliability().to_string(index=False))
    print("2/5 LCOE comparison ...")
    print(lcoe_comparison().to_string(index=False))
    print("3/5 diesel ops ...")
    diesel_ops()
    print("4/5 cold-charge bound ...")
    cold_charge_bound()
    print("5/5 degradation resize check ...")
    degradation_resize_check()
    print("done.")
