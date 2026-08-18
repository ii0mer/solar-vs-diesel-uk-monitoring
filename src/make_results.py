"""Single-command results pipeline — regenerates every results artifact
from the corrected model. This is the reproducibility entry point cited
in the dissertation:

    python -m src.make_results

Writes:
    results/sizing_summary.txt      step-1 TMY designs, governing-year + year-1 LOLP
    results/lcoe_comparison.csv     per-site LCOE at 5% and 8%, both diesels
                                    (FINAL multi-year designs)
    results/diesel_ops_summary.txt  runtimes, fuel, replacement cadence
    results/resize_checks.txt       exact multi-year re-optimisations used by
                                    the sizing-coupled sensitivity rows and
                                    the combined worst case

Run AFTER python -m src.multiyear (which fixes the final designs).
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
from .monte_carlo import SITE_DESIGN, SITE_DESIGN_TMY, SITE_DESIGN_DEG08
from .sensitivity import LOAD_DESIGNS, DEGRADATION_DESIGNS

RESULTS = Path(__file__).resolve().parents[1] / 'results'
RESULTS.mkdir(exist_ok=True)


def sizing_and_reliability() -> pd.DataFrame:
    """Step 1: TMY governing-year sizing (the multi-year step 2 is in
    src/multiyear.py)."""
    load = LoadProfile()
    eol = eol_ageing_factor()
    rows = []
    lines = [
        "STEP 1 — sizing for LOLP <= 1.0% in the design-governing year",
        "(year 24: PV at (1-d)^23, battery at 80% SoH) on the PVGIS-SARAH2",
        f"typical meteorological year; explicit loss chain (total "
        f"{LossChain().total_loss_pct:.1f}% incl. MPPT controller); ranked by",
        "25-year lifetime NPV at 5%. These are SITE_DESIGN_TMY in",
        "src/monte_carlo.py. The FINAL designs (SITE_DESIGN) come from the",
        "sixteen-year check in src/multiyear.py (results/multiyear_summary.txt).",
        "",
    ]
    for key, site in SITES.items():
        w = get_or_create_tmy(site, prefer='real')
        best, sweep = size_system(w, site, load, target_lolp=0.01)
        assert best is not None, f"no feasible design at {site.name}"
        sweep.to_csv(RESULTS / f'sizing_sweep_{key}.csv', index=False)
        rows.append({
            'site': site.name,
            'pv_wp': best.pv_w,
            'battery_kwh': best.battery_kwh,
            'lolp_eol_pct': best.lolp * 100,
            'lolp_year1_pct': best.lolp_year1 * 100,
            'curtailed_eol_kwh': best.annual_curtailed_kwh,
        })
        lines.append(
            f"{site.name:12s} {best.pv_w:.0f} Wp + {best.battery_kwh:.2f} kWh"
            f"  governing-year LOLP={best.lolp*100:.3f}%  year-1 LOLP="
            f"{best.lolp_year1*100:.3f}%  curtailed {best.annual_curtailed_kwh:.0f} kWh/yr")
    lines += [
        "",
        f"Governing-year PV factor (1-0.005)^23 = {eol:.4f}; battery SoH 0.80.",
    ]
    (RESULTS / 'sizing_summary.txt').write_text('\n'.join(lines) + '\n')
    df = pd.DataFrame(rows)
    # cross-check the module-level constants
    for _, r in df.iterrows():
        wp, kwh = SITE_DESIGN_TMY[r['site']]
        assert (wp, kwh) == (r['pv_wp'], r['battery_kwh']), \
            f"SITE_DESIGN_TMY out of date for {r['site']}: {(r['pv_wp'], r['battery_kwh'])}"
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


def resize_checks() -> None:
    """Exact multi-year re-optimisations (final criterion) used by the
    sizing-coupled sensitivity rows (load, degradation; Southampton) and by
    the combined worst case (0.8 %/yr, all sites). Asserts the constants in
    sensitivity.py / monte_carlo.py match."""
    from .multiyear import robust_design
    lines = ["Exact re-optimisations under the FINAL multi-year criterion",
             "(annual LOLP <= 1% in >= 15 of 16 years, both PV chains).", ""]
    for mult, (wp, kwh) in LOAD_DESIGNS.items():
        got = robust_design('southampton', load_mult=mult)
        lines.append(f"Southampton load x{mult}: {got[0]:.0f} Wp + {got[1]:.2f} kWh "
                     f"(constant {wp:.0f}/{kwh})")
        assert (got[0], got[1]) == (wp, kwh), f"LOAD_DESIGNS stale at x{mult}: {got[:2]}"
    for deg, (wp, kwh) in DEGRADATION_DESIGNS.items():
        got = robust_design('southampton', degradation_pct_yr=deg)
        lines.append(f"Southampton degradation {deg}%/yr: {got[0]:.0f} Wp + {got[1]:.2f} kWh "
                     f"(constant {wp:.0f}/{kwh})")
        assert (got[0], got[1]) == (wp, kwh), f"DEGRADATION_DESIGNS stale at {deg}: {got[:2]}"
    for key, site in SITES.items():
        got = robust_design(key, degradation_pct_yr=0.8)
        wp, kwh = SITE_DESIGN_DEG08[site.name]
        lines.append(f"{site.name} at 0.8%/yr: {got[0]:.0f} Wp + {got[1]:.2f} kWh (constant {wp:.0f}/{kwh})")
        assert (got[0], got[1]) == (wp, kwh), f"SITE_DESIGN_DEG08 stale at {site.name}: {got[:2]}"
    (RESULTS / 'resize_checks.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    print("1/4 step-1 TMY sizing + reliability ...")
    print(sizing_and_reliability().to_string(index=False))
    print("2/4 LCOE comparison (final designs) ...")
    print(lcoe_comparison().to_string(index=False))
    print("3/4 diesel ops ...")
    diesel_ops()
    print("4/4 multi-year re-size checks ...")
    resize_checks()
    print("done.")
