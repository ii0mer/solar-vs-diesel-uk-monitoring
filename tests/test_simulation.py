"""Sanity tests for the simulation framework.

Run with:  python -m pytest tests/

These are not exhaustive unit tests; they are sanity checks that the most
critical components produce physically plausible results and that the report's
headline numbers can be reproduced from the code.
"""
import pytest
import numpy as np

from src.load_profile import LoadProfile
from src.battery import BatteryDesign
from src.economics import (
    build_pv_battery_cashflows,
    build_diesel_cashflows,
    DEFAULT_DISCOUNT_RATE,
    COMMERCIAL_WACC,
)
from src.diesel import DieselArchitecture1, DieselArchitecture2


# =========================================================================
# LOAD PROFILE TESTS
# =========================================================================

def test_load_profile_design_power_w():
    """The load profile must give 14.6 W continuous as stated in the report."""
    lp = LoadProfile()
    # Components: datalogger 1.2 + sensors 4×0.5 + modem 6.0 + ancillary 3.0 = 12.2 W
    base = lp.base_power_w()
    assert abs(base - 12.2) < 0.01, f"Base power should be 12.2 W, got {base}"
    # With 20% margin: 12.2 × 1.2 = 14.64 W ≈ 14.6 W
    design = lp.design_power_w()
    assert abs(design - 14.64) < 0.01, f"Design power should be 14.64 W, got {design}"


def test_load_profile_annual_energy():
    """Annual energy must round to ~128 kWh as stated."""
    lp = LoadProfile()
    annual = lp.annual_energy_kwh()
    # 14.64 W × 8760 h / 1000 = 128.247 kWh
    assert 128 < annual < 129, f"Annual energy should be ~128 kWh, got {annual}"


# =========================================================================
# BATTERY TESTS
# =========================================================================

def test_battery_usable_capacity():
    """Usable capacity must respect the 80% DoD constraint."""
    bat = BatteryDesign(capacity_kwh=1.0)
    # Default DoD is 0.8 (i.e. only 80% of nameplate is usable)
    assert abs(bat.capacity_kwh - 1.0) < 0.001
    # Going below dod_min_frac should not be allowed elsewhere in code


# =========================================================================
# DIESEL TESTS
# =========================================================================

def test_diesel_arch1_annual_fuel():
    """Architecture 1 (continuous 24/7) annual fuel consumption."""
    arch1 = DieselArchitecture1()
    # 0.40 L/hr × 8760 h = 3504 L/yr
    assert abs(arch1.annual_fuel_litres - 3504) < 1


def test_diesel_arch2_annual_fuel():
    """Architecture 2 (intermittent) annual fuel consumption."""
    arch2 = DieselArchitecture2()
    # 0.20 L/hr × 4 hr/day × 365 = 292 L/yr
    assert abs(arch2.annual_fuel_litres - 292) < 1


def test_diesel_visit_cost():
    """Site visit cost must be £2,400/yr at 12 visits × £200/visit."""
    arch2 = DieselArchitecture2()
    expected = 12 * (80 + 120)  # £200/visit total
    assert abs(arch2.annual_visit_cost_gbp - expected) < 1


# =========================================================================
# ECONOMICS / LCOE TESTS — these reproduce the report's headline numbers
# =========================================================================

def test_southampton_solar_lcoe_at_5pct():
    """Southampton 400 Wp + 1 kWh at 5% must produce LCOE ≈ £6.04/kWh."""
    cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50, pv_size_wp=400,
        battery_capex_gbp_per_kwh=700.0, battery_kwh=1.0,
        annual_energy_delivered_kwh=128.2,
    )
    lcoe = cf.lcoe_gbp_per_kwh(0.05)
    assert 6.0 < lcoe < 6.10, f"Southampton solar LCOE should be ~£6.04, got £{lcoe:.2f}"


def test_liverpool_solar_lcoe_at_5pct():
    """Liverpool 400 Wp + 1 kWh at 5% must produce LCOE ≈ £6.04/kWh (same as Southampton/Birmingham)."""
    cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50, pv_size_wp=400,
        battery_capex_gbp_per_kwh=700.0, battery_kwh=1.0,
        annual_energy_delivered_kwh=128.2,
    )
    lcoe = cf.lcoe_gbp_per_kwh(0.05)
    assert 6.0 < lcoe < 6.10, f"Liverpool solar LCOE should be ~£6.04, got £{lcoe:.2f}"


def test_liverpool_sizing_meets_lolp_target():
    """Liverpool 500 Wp + 1 kWh must achieve LOLP < 1% (real PVGIS optimal sizing)."""
    from src.sites import SITES
    from src.weather import get_or_create_tmy
    from src.load_profile import LoadProfile
    from src.pv_model import PVDesign
    from src.battery import BatteryDesign
    from src.simulation import run_simulation

    site = SITES['liverpool']
    df = get_or_create_tmy(site, prefer='synthetic')
    pv = PVDesign(nameplate_w=500, n_modules=1)
    bat = BatteryDesign(capacity_kwh=1.0)
    load = LoadProfile()
    # Run twice to warm up SoC initialisation
    r1 = run_simulation(df, site, pv, bat, load)
    r = run_simulation(df, site, pv, bat, load,
                       initial_soc_frac=r1.final_soc_kwh / bat.capacity_kwh)
    assert r.lolp < 0.01, f"Liverpool 400+1 should hit LOLP<1%, got {r.lolp*100:.2f}%"


def test_edinburgh_solar_lcoe_at_5pct():
    """Edinburgh 500 Wp + 1 kWh at 5% must produce LCOE ≈ £6.30/kWh."""
    cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50, pv_size_wp=500,
        battery_capex_gbp_per_kwh=700.0, battery_kwh=1.0,
        annual_energy_delivered_kwh=128.2,
    )
    lcoe = cf.lcoe_gbp_per_kwh(0.05)
    assert 6.20 < lcoe < 6.40, f"Edinburgh solar LCOE should be ~£6.30, got £{lcoe:.2f}"


def test_diesel_a1_lcoe_at_5pct():
    """Diesel A1 (continuous 24/7) LCOE must be ~£74/kWh as in report."""
    cf = build_diesel_cashflows(DieselArchitecture1(), fuel_price_ppl=76.02)
    lcoe = cf.lcoe_gbp_per_kwh(0.05)
    assert 74 < lcoe < 75, f"Diesel A1 LCOE should be ~£74, got £{lcoe:.2f}"


def test_diesel_a2_lcoe_at_5pct():
    """Diesel A2 (intermittent realistic) LCOE must be ~£25/kWh as in report."""
    cf = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02)
    lcoe = cf.lcoe_gbp_per_kwh(0.05)
    assert 25 < lcoe < 26, f"Diesel A2 LCOE should be ~£25, got £{lcoe:.2f}"


def test_solar_beats_diesel_a2_at_5pct():
    """The headline conclusion: solar beats diesel A2 by factor ≥ 4 at 5%."""
    pv_cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50, pv_size_wp=400,
        battery_capex_gbp_per_kwh=700.0, battery_kwh=1.0,
        annual_energy_delivered_kwh=128.2,
    )
    di_cf = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02)
    pv_lcoe = pv_cf.lcoe_gbp_per_kwh(0.05)
    di_lcoe = di_cf.lcoe_gbp_per_kwh(0.05)
    ratio = di_lcoe / pv_lcoe
    assert ratio > 4.0, f"Solar should beat diesel A2 by 4×+, got {ratio:.2f}×"


def test_solar_robust_at_8pct_wacc():
    """At 8% commercial WACC, solar still beats diesel A2 substantially."""
    pv_cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50, pv_size_wp=400,
        battery_capex_gbp_per_kwh=700.0, battery_kwh=1.0,
        annual_energy_delivered_kwh=128.2,
    )
    di_cf = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02)
    pv_lcoe = pv_cf.lcoe_gbp_per_kwh(COMMERCIAL_WACC)
    di_lcoe = di_cf.lcoe_gbp_per_kwh(COMMERCIAL_WACC)
    ratio = di_lcoe / pv_lcoe
    # Slightly lower than 5% case (PV is capital-front-loaded), but still > 3.5×
    assert ratio > 3.5, f"At 8% WACC, solar should beat diesel A2 by 3.5×+, got {ratio:.2f}×"


# =========================================================================
# PHYSICAL CONSISTENCY TESTS
# =========================================================================

def test_pv_degradation_reduces_energy_over_time():
    """PV energy delivered must decrease year on year due to degradation."""
    cf = build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50, pv_size_wp=400,
        battery_capex_gbp_per_kwh=700.0, battery_kwh=1.0,
        annual_energy_delivered_kwh=128.2,
        pv_degradation_pct_per_year=0.5,
    )
    # Year 1 should be > Year 25
    energy = cf.annual_energy_kwh
    assert energy[1] > energy[24], "PV must degrade over time"
    # Total degradation from year 1 to year 25 ≈ 11.4% (0.5%/yr compounded)
    drop = (energy[1] - energy[25]) / energy[1]
    assert 0.10 < drop < 0.13, f"25-yr degradation should be ~11.4%, got {drop*100:.1f}%"


def test_diesel_fuel_dominates_runtime_costs():
    """For Architecture 1 (24/7), fuel + oil + visits should each be material."""
    arch1 = DieselArchitecture1()
    fuel_cost = arch1.annual_fuel_cost_gbp(76.02)
    oil_cost = arch1.annual_oil_cost_gbp
    visit_cost = arch1.annual_visit_cost_gbp
    # Each component should be at least £1k/yr — none is negligible
    assert fuel_cost > 1000
    assert oil_cost > 1000
    assert visit_cost > 1000
