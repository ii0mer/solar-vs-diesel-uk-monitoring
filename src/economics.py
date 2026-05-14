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
    - Battery and inverter replacements occur at fixed intervals.
    - PV output degrades linearly at `pv_degradation_pct_per_year`.
    - O&M and site visits are flat real annual costs.
    """
    n = project_years + 1   # year 0 through year N

    # Year-0 CapEx
    capex_pv = pv_capex_gbp_per_wp * pv_size_wp
    capex_battery = battery_capex_gbp_per_kwh * battery_kwh
    capex_year_0 = capex_pv + capex_battery + inverter_capex_gbp + site_install_one_off_gbp

    capex_flow = np.zeros(n)
    capex_flow[0] = capex_year_0

    # Battery replacements at year 12, 24 (if before end)
    battery_repl_flow = np.zeros(n)
    repl_year = battery_lifetime_years
    while repl_year < n:
        battery_repl_flow[repl_year] = capex_battery
        repl_year += battery_lifetime_years

    # Inverter replacements at year 15
    inverter_repl_flow = np.zeros(n)
    repl_year = inverter_lifetime_years
    while repl_year < n:
        inverter_repl_flow[repl_year] = inverter_capex_gbp
        repl_year += inverter_lifetime_years

    # Annual O&M and site visits
    om_flow = np.zeros(n)
    om_flow[1:] = annual_om_gbp
    visits_flow = np.zeros(n)
    visits_flow[1:] = annual_site_visits * visit_cost_gbp

    # Energy delivered each year — accounting for degradation
    energy_flow = np.zeros(n)
    energy_flow[0] = 0  # year 0 = CapEx, no energy yet
    for year in range(1, n):
        years_operated = year
        derate = (1.0 - pv_degradation_pct_per_year / 100.0) ** (years_operated - 1)
        energy_flow[year] = annual_energy_delivered_kwh * derate

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

    Note on replacements: rather than spike costs in specific years, we
    spread genset/battery replacements as their annualised cost. This is
    standard practice for sub-5-year-life equipment in DCF analysis where
    the replacement frequency is 4-44 times in 25 years.
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

    genset_repl_flow = np.zeros(n)
    genset_repl_flow[1:] = diesel_arch.annualised_genset_replacement_gbp

    flows = [
        CashFlow("CapEx (genset + tank + enclosure + install)", capex_flow),
        CashFlow("Fuel", fuel_flow),
        CashFlow("Oil & maintenance", oil_flow),
        CashFlow("Site visits", visits_flow),
        CashFlow("Genset replacement (annualised)", genset_repl_flow),
    ]

    # Battery replacement for Architecture 2
    if hasattr(diesel_arch, 'annualised_battery_replacement_gbp'):
        battery_repl_flow = np.zeros(n)
        battery_repl_flow[1:] = diesel_arch.annualised_battery_replacement_gbp
        flows.append(CashFlow("Battery replacement (annualised)", battery_repl_flow))

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
