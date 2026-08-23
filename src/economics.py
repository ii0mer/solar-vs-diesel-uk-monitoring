"""Discounted-cash-flow economic model.

Implements the IEA/NEA (2020) "Projected Costs of Generating Electricity"
LCOE methodology:

    LCOE = Σ_t [ C_t / (1+r)^t ] / Σ_t [ E_t / (1+r)^t ]

where:
    C_t = total cost in year t (CapEx, OpEx, replacements, fuel)
    E_t = useful electrical energy delivered in year t (kWh)
    r   = real discount rate (5% for UK public infrastructure)
    t   = year, 0..N
    N   = project lifetime (25 years)

Reference:
- IEA/NEA (2020) "Projected Costs of Generating Electricity 2020", OECD
  Publishing, Paris. https://doi.org/10.1787/a6002f3b-en
- HM Treasury (2022) "The Green Book: Central Government Guidance on
  Appraisal and Evaluation". Social Time Preference Rate of 3.5% real,
  declining over very long horizons. We use 5% as central case (slightly
  above STPR to reflect technology risk).

All monetary values in GBP, real terms (inflation-adjusted).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


# -- Default economic parameters --
# Two discount rates are reported throughout the dissertation.
DEFAULT_DISCOUNT_RATE = 0.05      # 5% real, public-sector / Green Book + technology premium (central case)
COMMERCIAL_WACC       = 0.08      # 8% real, typical UK private-sector renewable-energy WACC
DEFAULT_PROJECT_YEARS = 25         # PV system lifetime
DEFAULT_GBP_PER_USD = 1 / 1.32     # 2025 average exchange rate (AHDB)


@dataclass
class CashFlow:
    """A single cash-flow stream (annual, real GBP)."""
    label: str
    annual_amounts_gbp: np.ndarray   # length = project_years + 1 (year 0 is CapEx)

    def npv(self, discount_rate: float) -> float:
        years = np.arange(len(self.annual_amounts_gbp))
        discount_factors = 1.0 / (1.0 + discount_rate) ** years
        return float(np.sum(self.annual_amounts_gbp * discount_factors))


@dataclass
class SystemCashFlows:
    """Container for all cash flows of a system over project lifetime."""
    label: str
    project_years: int
    flows: List[CashFlow] = field(default_factory=list)
    annual_energy_kwh: np.ndarray = None   # energy delivered per year, kWh

    def total_npv(self, discount_rate: float) -> float:
        return sum(f.npv(discount_rate) for f in self.flows)

    def lcoe_gbp_per_kwh(self, discount_rate: float) -> float:
        """Compute LCOE per IEA/NEA (2020) DCF methodology."""
        cost_npv = self.total_npv(discount_rate)
        if self.annual_energy_kwh is None:
            raise ValueError("annual_energy_kwh must be set to compute LCOE")
        years = np.arange(len(self.annual_energy_kwh))
        discount_factors = 1.0 / (1.0 + discount_rate) ** years
        energy_npv = float(np.sum(self.annual_energy_kwh * discount_factors))
        if energy_npv <= 0:
            return float('inf')
        return cost_npv / energy_npv


# -- Replacement convention (applies to BOTH systems) --
#
# Components with service life >= 2 years are modelled as LUMPY
# replacements in their actual replacement years, with a straight-line
# SALVAGE CREDIT at end of project for remaining life (this removes the
# distortion of a replacement bought in year 24 being costed in full).
# Components with sub-2-year life (A1 genset at 8,760 h/yr) are
# annualised — replacement occurs multiple times per year, so lumpy
# treatment degenerates to the same thing.

def lumpy_replacement_flows(capex_gbp: float, lifetime_years: float,
                            project_years: int) -> np.ndarray:
    """Return length-(N+1) array: replacement costs in replacement years
    and a negative salvage credit in the final year for remaining life."""
    n = project_years + 1
    flow = np.zeros(n)
    if lifetime_years < 2.0:
        # Annualised continuous-renewal stream. The year-0 purchase already
        # covers the first unit's life, so one unit is credited in year 1;
        # otherwise the initial unit would be paid for twice.
        flow[1:] = capex_gbp / lifetime_years
        flow[1] -= capex_gbp
        return flow
    year = int(round(lifetime_years))
    last_install = 0
    y = year
    while y <= project_years:
        flow[y] = capex_gbp
        last_install = y
        y += year
    age_at_end = project_years - last_install
    remaining_frac = max(0.0, (year - age_at_end) / year)
    if last_install > 0 or age_at_end < year:
        flow[project_years] -= capex_gbp * remaining_frac * (
            1.0 if last_install > 0 else 0.0)
    # salvage for the ORIGINAL unit if never replaced but outlives project
    if last_install == 0 and lifetime_years > project_years:
        flow[project_years] -= capex_gbp * (
            (lifetime_years - project_years) / lifetime_years)
    return flow


# -- PV-Battery system cash flow builder --

def build_pv_battery_cashflows(
    pv_capex_gbp_per_wp: float,
    pv_size_wp: float,
    battery_capex_gbp_per_kwh: float,
    battery_kwh: float,
    annual_energy_delivered_kwh: float,
    pv_degradation_pct_per_year: float = 0.5,
    battery_lifetime_years: int = 12,
    inverter_capex_gbp: float = 150.0,
    inverter_lifetime_years: int = 15,
    site_install_one_off_gbp: float = 600.0,
    annual_om_gbp: float = 60.0,
    annual_site_visits: int = 2,
    visit_cost_gbp: float = 200.0,
    project_years: int = DEFAULT_PROJECT_YEARS,
    label: str = "Solar-Battery"
) -> SystemCashFlows:
    """Build cash flows for a PV-battery system.

    Notes
    -----
    - CapEx is in year 0.
    - Battery and inverter replacements are lumpy with end-of-project
      salvage credit (see lumpy_replacement_flows).
    - ENERGY DENOMINATOR: energy SERVED to the fixed load, flat across
      years. Module degradation does NOT shrink served energy — the
      array is sized so that even at end-of-life the load is met (LOLP
      criterion at year 25); degradation is therefore a SIZING cost,
      already embedded in capex, not a delivered-energy loss. The
      pv_degradation_pct_per_year argument is retained for API
      compatibility and for sizing-coupled sensitivity, where it scales
      the required array (handled by the caller), but it no longer
      distorts the LCOE denominator. Both systems use the same
      served-energy convention, making the LCOE ratio equal to the
      cost ratio of serving the same load.
    - O&M and site visits are flat real annual costs.
    """
    n = project_years + 1   # year 0 through year N

    # Year-0 CapEx
    capex_pv = pv_capex_gbp_per_wp * pv_size_wp
    capex_battery = battery_capex_gbp_per_kwh * battery_kwh
    capex_year_0 = capex_pv + capex_battery + inverter_capex_gbp + site_install_one_off_gbp

    capex_flow = np.zeros(n)
    capex_flow[0] = capex_year_0

    # Battery replacements — lumpy + salvage
    battery_repl_flow = lumpy_replacement_flows(
        capex_battery, float(battery_lifetime_years), project_years)

    # Inverter replacements — lumpy + salvage
    inverter_repl_flow = lumpy_replacement_flows(
        inverter_capex_gbp, float(inverter_lifetime_years), project_years)

    # Annual O&M and site visits
    om_flow = np.zeros(n)
    om_flow[1:] = annual_om_gbp
    visits_flow = np.zeros(n)
    visits_flow[1:] = annual_site_visits * visit_cost_gbp

    # Energy served each year — flat (see docstring)
    energy_flow = np.zeros(n)
    energy_flow[1:] = annual_energy_delivered_kwh

    flows = [
        CashFlow("CapEx (PV + battery + inverter + install)", capex_flow),
        CashFlow("Battery replacement", battery_repl_flow),
        CashFlow("Inverter replacement", inverter_repl_flow),
        CashFlow("Annual O&M", om_flow),
        CashFlow("Site visits", visits_flow),
    ]
    return SystemCashFlows(
        label=label,
        project_years=project_years,
        flows=flows,
        annual_energy_kwh=energy_flow,
    )


# -- Diesel system cash flow builder --

def build_diesel_cashflows(
    diesel_arch,                   # DieselArchitecture1 or DieselArchitecture2
    fuel_price_ppl: float = 76.02,
    annual_energy_delivered_kwh: float = 128.2,
    project_years: int = DEFAULT_PROJECT_YEARS,
    label: str = "Diesel"
) -> SystemCashFlows:
    """Build cash flows for a diesel architecture.

    Treats every annual cost as flat in real terms (we add fuel-price
    sensitivity separately). The cash flows are:
      - Year 0: CapEx
      - Years 1..N: fuel + oil + visits + (annualised replacements)

    Replacements use the SAME convention as the solar system (see
    lumpy_replacement_flows): lumpy years + end-of-project salvage for
    lives >= 2 years (A2 genset ~24 yr at 334 h/yr at derived runtime, A2 lead-acid
    4 yr), annualised only where life is sub-2-years (A1 genset, whose
    5,000 h life at 8,760 h/yr means replacement every ~7 months).
    """
    n = project_years + 1

    # CapEx
    capex_flow = np.zeros(n)
    capex_flow[0] = diesel_arch.total_capex_gbp

    # Annual costs (year 1 onwards)
    fuel_flow = np.zeros(n)
    fuel_flow[1:] = diesel_arch.annual_fuel_cost_gbp(fuel_price_ppl)

    oil_flow = np.zeros(n)
    oil_flow[1:] = diesel_arch.annual_oil_cost_gbp

    visits_flow = np.zeros(n)
    visits_flow[1:] = diesel_arch.annual_visit_cost_gbp

    genset_life_years = (diesel_arch.genset_lifetime_hours
                         / max(diesel_arch.annual_runtime_hours, 1))
    cal = getattr(diesel_arch, 'genset_calendar_life_years', None)
    if cal is not None:
        genset_life_years = min(genset_life_years, float(cal))
    genset_repl_flow = lumpy_replacement_flows(
        diesel_arch.capex_genset_gbp, genset_life_years, project_years)

    flows = [
        CashFlow("CapEx (genset + tank + enclosure + install)", capex_flow),
        CashFlow("Fuel", fuel_flow),
        CashFlow("Oil & maintenance", oil_flow),
        CashFlow("Site visits", visits_flow),
        CashFlow("Genset replacement (lumpy + salvage)", genset_repl_flow),
    ]

    # Battery replacement for Architecture 2 — same lumpy convention
    if hasattr(diesel_arch, 'battery_lifetime_years'):
        battery_repl_flow = lumpy_replacement_flows(
            diesel_arch.capex_battery_gbp,
            float(diesel_arch.battery_lifetime_years), project_years)
        flows.append(CashFlow("Battery replacement (lumpy + salvage)",
                              battery_repl_flow))

    # Energy delivered = annual load (assume 100% reliability if PV-battery
    # is the comparator's reliability target; diesel is by definition reliable)
    energy_flow = np.zeros(n)
    energy_flow[1:] = annual_energy_delivered_kwh

    return SystemCashFlows(
        label=label,
        project_years=project_years,
        flows=flows,
        annual_energy_kwh=energy_flow,
    )


# -- Comparison helpers --

def compare_systems(
    pv_battery_cf: SystemCashFlows,
    diesel_cf: SystemCashFlows,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> Dict:
    """Side-by-side NPV, LCOE, and payback."""
    pv_npv = pv_battery_cf.total_npv(discount_rate)
    di_npv = diesel_cf.total_npv(discount_rate)
    pv_lcoe = pv_battery_cf.lcoe_gbp_per_kwh(discount_rate)
    di_lcoe = diesel_cf.lcoe_gbp_per_kwh(discount_rate)

    # Simple payback: how many years for cumulative diesel OpEx to exceed
    # PV-battery total cost (if installing PV instead of diesel).
    # = Year when (diesel cost up to t) > (pv-battery cost up to t)
    pv_cum = np.cumsum([sum(f.annual_amounts_gbp[t] for f in pv_battery_cf.flows)
                        for t in range(len(pv_battery_cf.flows[0].annual_amounts_gbp))])
    di_cum = np.cumsum([sum(f.annual_amounts_gbp[t] for f in diesel_cf.flows)
                        for t in range(len(diesel_cf.flows[0].annual_amounts_gbp))])
    payback_years = None
    for t in range(len(pv_cum)):
        if di_cum[t] >= pv_cum[t]:
            payback_years = t
            break

    return {
        'pv_battery_npv_gbp': pv_npv,
        'diesel_npv_gbp': di_npv,
        'avoided_cost_gbp': di_npv - pv_npv,
        'pv_battery_lcoe_gbp_per_kwh': pv_lcoe,
        'diesel_lcoe_gbp_per_kwh': di_lcoe,
        'lcoe_ratio_diesel_over_pv': di_lcoe / pv_lcoe if pv_lcoe > 0 else float('inf'),
        'payback_years_simple': payback_years,
        'discount_rate': discount_rate,
    }


# -- CLI smoke test --

if __name__ == '__main__':
    from .diesel import DieselArchitecture1, DieselArchitecture2

    # PV-battery: Southampton sized example (300 Wp + 1 kWh, 128 kWh/yr load)
    pv_cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50,
        pv_size_wp=300,
        battery_capex_gbp_per_kwh=700,
        battery_kwh=1.0,
        annual_energy_delivered_kwh=128.2,
        label="Solar-Battery (Southampton, 300Wp+1kWh)",
    )

    # Diesel arch 2 (realistic)
    arch2 = DieselArchitecture2()
    di2_cf = build_diesel_cashflows(arch2, label="Diesel (Architecture 2 — realistic)")

    # Diesel arch 1 (worst case)
    arch1 = DieselArchitecture1()
    di1_cf = build_diesel_cashflows(arch1, label="Diesel (Architecture 1 — continuous)")

    print(f"PV-Battery: total CapEx + 25-yr OpEx = "
          f"£{sum(f.npv(0.0) for f in pv_cf.flows):,.0f} (undiscounted)")
    print(f"PV-Battery: NPV at 5% = £{pv_cf.total_npv(0.05):,.0f}")
    print(f"PV-Battery: LCOE at 5% = £{pv_cf.lcoe_gbp_per_kwh(0.05):.2f}/kWh")
    print()
    print(f"Diesel A2 : NPV at 5% = £{di2_cf.total_npv(0.05):,.0f}")
    print(f"Diesel A2 : LCOE at 5% = £{di2_cf.lcoe_gbp_per_kwh(0.05):.2f}/kWh")
    print()
    print(f"Diesel A1 : NPV at 5% = £{di1_cf.total_npv(0.05):,.0f}")
    print(f"Diesel A1 : LCOE at 5% = £{di1_cf.lcoe_gbp_per_kwh(0.05):.2f}/kWh")
    print()
    cmp = compare_systems(pv_cf, di2_cf, discount_rate=0.05)
    print(f"PV-battery vs Diesel A2:")
    print(f"  LCOE ratio (diesel/PV): {cmp['lcoe_ratio_diesel_over_pv']:.1f}x")
    print(f"  Avoided cost (NPV)    : £{cmp['avoided_cost_gbp']:,.0f}")
    print(f"  Simple payback        : {cmp['payback_years_simple']} years")
