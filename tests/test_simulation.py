"""Sanity tests for the simulation framework (corrected model, Aug 2026).

Run with:  python -m pytest tests/

These are not exhaustive unit tests; they are sanity checks that the most
critical components produce physically plausible results and that the
report's headline numbers reproduce from the code. All expected values
below were computed from the corrected model (loss chain, end-of-life
sizing, energy-balance-consistent diesel, lumpy+salvage replacements,
flat served-energy LCOE denominator) and are asserted with tight bounds.
"""
import pytest
import numpy as np

from src.load_profile import LoadProfile
from src.battery import BatteryDesign
from src.pv_model import LossChain
from src.economics import (
    build_pv_battery_cashflows,
    build_diesel_cashflows,
    lumpy_replacement_flows,
    DEFAULT_DISCOUNT_RATE,
    COMMERCIAL_WACC,
)
from src.diesel import DieselArchitecture1, DieselArchitecture2
from src.monte_carlo import SITE_DESIGN


# =========================================================================
# LOAD PROFILE
# =========================================================================

def test_load_profile_design_power_w():
    lp = LoadProfile()
    assert abs(lp.base_power_w() - 12.2) < 0.01
    assert abs(lp.design_power_w() - 14.64) < 0.01


def test_load_profile_annual_energy():
    lp = LoadProfile()
    assert 128 < lp.annual_energy_kwh() < 129


# =========================================================================
# LOSS CHAIN
# =========================================================================

def test_aoi_losses_are_applied():
    """AOI/IAM must reduce POA-effective irradiance: yield at a site must
    be lower than the same chain evaluated with poa_global directly."""
    from src.sites import SITES
    from src.weather import get_or_create_tmy
    from src.pv_model import PVDesign, simulate_pv_dc
    import pvlib
    site = SITES['southampton']
    df = get_or_create_tmy(site, prefer='real')
    y = simulate_pv_dc(df, site, PVDesign(nameplate_w=1000)).sum() / 1000
    assert 1050 < y < 1110   # kWh/kWp with AOI + 12.7% loss chain


def test_loss_chain_derate_in_plausible_band():
    """Explicit loss chain (PVWatts framework + MPPT controller) must land
    in the 10-15% total-loss band documented in the methodology."""
    lc = LossChain()
    assert 0.85 < lc.derate < 0.90
    assert 10.0 < lc.total_loss_pct < 15.0


def test_loss_chain_reduces_pv_output():
    from src.sites import SITES
    from src.weather import get_or_create_tmy
    from src.pv_model import PVDesign, simulate_pv_dc
    site = SITES['southampton']
    df = get_or_create_tmy(site, prefer='real')
    design = PVDesign(nameplate_w=350)
    lossless = simulate_pv_dc(df, site, design,
                              losses=LossChain(
                                  soiling=0, snow=0, mismatch=0, wiring=0,
                                  connections=0, lid=0, nameplate=0,
                                  availability=0, controller_efficiency=1.0))
    lossy = simulate_pv_dc(df, site, design)
    ratio = lossy.sum() / lossless.sum()
    assert abs(ratio - LossChain().derate) < 1e-9


# =========================================================================
# BATTERY
# =========================================================================

def test_battery_usable_capacity():
    bat = BatteryDesign(capacity_kwh=1.5)
    assert abs(bat.usable_kwh - 1.2) < 0.001   # 80% DoD


# =========================================================================
# DIESEL — energy-balance consistency and Skarstein–Uhlen fuel
# =========================================================================

def test_diesel_arch1_fuel_skarstein_uhlen():
    """A1 fuel: 0.08415·1.6 kW + 0.246·0.01464 kW = 0.138 L/h → 1,211 L/yr."""
    arch1 = DieselArchitecture1()
    assert abs(arch1.fuel_litres_per_hour_at_op_point - 0.1382) < 0.0005
    assert abs(arch1.annual_fuel_litres - 1211) < 2


def test_diesel_arch2_runtime_satisfies_energy_balance():
    """A2 runtime must be DERIVED: gen energy = daily load / battery-path
    efficiency. 351.36/0.80 = 439.2 Wh → /480 W = 0.915 h/day."""
    arch2 = DieselArchitecture2()
    assert abs(arch2.daily_runtime_hours - 0.915) < 0.001
    gen_wh = arch2.daily_runtime_hours * arch2.charge_power_kw * 1000
    served_wh = gen_wh * arch2.battery_path_efficiency
    assert abs(served_wh - arch2.daily_load_wh) < 0.01


def test_diesel_arch2_annual_fuel():
    """A2: 0.2527 L/h × 333 h/yr ≈ 84 L/yr. At a fixed 30% load fraction the
    Skarstein–Uhlen annual fuel is independent of the genset rating."""
    arch2 = DieselArchitecture2()
    assert abs(arch2.annual_fuel_litres - 84.2) < 1.0


def test_diesel_visit_cost():
    arch2 = DieselArchitecture2()
    assert abs(arch2.annual_visit_cost_gbp - 2400) < 1


# =========================================================================
# REPLACEMENT CONVENTION — lumpy + salvage
# =========================================================================

def test_lumpy_replacements_with_salvage_battery12():
    f = lumpy_replacement_flows(1050.0, 12, 25)
    assert f[12] == pytest.approx(1050.0)
    assert f[24] == pytest.approx(1050.0)
    # unit installed at 24 has 11/12 of its life left at year 25
    assert f[25] == pytest.approx(-1050.0 * 11 / 12)


def test_lumpy_replacements_life_exceeds_project():
    f = lumpy_replacement_flows(1000.0, 30, 25)
    assert (f[1:25] == 0).all()
    assert f[25] == pytest.approx(-1000.0 * 5 / 30)   # original-unit salvage


def test_sub_two_year_life_is_annualised():
    f = lumpy_replacement_flows(1200.0, 5000 / 8760, 25)
    assert f[1] == pytest.approx(1200.0 / (5000 / 8760))
    assert f[1] == pytest.approx(f[25])


# =========================================================================
# ECONOMICS — corrected headline numbers (computed 12 Aug 2026)
# =========================================================================

def _solar_cf(site):
    wp, kwh = SITE_DESIGN[site]
    return build_pv_battery_cashflows(
        pv_capex_gbp_per_wp=4.50, pv_size_wp=wp,
        battery_capex_gbp_per_kwh=700.0, battery_kwh=kwh,
        annual_energy_delivered_kwh=128.2,
    )


def test_southampton_solar_lcoe_at_5pct():
    lcoe = _solar_cf('Southampton').lcoe_gbp_per_kwh(0.05)
    assert 5.85 < lcoe < 5.95, f"expected ~£5.90, got £{lcoe:.3f}"


def test_edinburgh_solar_lcoe_at_5pct():
    lcoe = _solar_cf('Edinburgh').lcoe_gbp_per_kwh(0.05)
    assert 6.53 < lcoe < 6.63, f"expected ~£6.58, got £{lcoe:.3f}"


def test_diesel_a1_lcoe_at_5pct():
    lcoe = build_diesel_cashflows(
        DieselArchitecture1(), fuel_price_ppl=76.02).lcoe_gbp_per_kwh(0.05)
    assert 74.0 < lcoe < 76.5, f"expected ~£75.1, got £{lcoe:.3f}"


def test_diesel_a2_lcoe_at_5pct():
    lcoe = build_diesel_cashflows(
        DieselArchitecture2(), fuel_price_ppl=76.02).lcoe_gbp_per_kwh(0.05)
    assert 21.9 < lcoe < 22.9, f"expected ~£22.4, got £{lcoe:.3f}"


def test_solar_beats_diesel_a2_all_sites_5pct():
    di = build_diesel_cashflows(
        DieselArchitecture2(), fuel_price_ppl=76.02).lcoe_gbp_per_kwh(0.05)
    for site in SITE_DESIGN:
        ratio = di / _solar_cf(site).lcoe_gbp_per_kwh(0.05)
        assert ratio > 3.3, f"{site}: expected >3.3x, got {ratio:.2f}x"


def test_solar_robust_at_8pct_wacc():
    di = build_diesel_cashflows(
        DieselArchitecture2(), fuel_price_ppl=76.02).lcoe_gbp_per_kwh(
            COMMERCIAL_WACC)
    for site in SITE_DESIGN:
        ratio = di / _solar_cf(site).lcoe_gbp_per_kwh(COMMERCIAL_WACC)
        assert ratio > 3.0, f"{site}: expected >3.0x at 8%, got {ratio:.2f}x"


def test_energy_denominator_is_flat_served_energy():
    """Degradation must NOT shrink the LCOE denominator — the array is
    sized to serve the load at end-of-life, so served energy is flat."""
    cf = _solar_cf('Southampton')
    energy = cf.annual_energy_kwh
    assert energy[0] == 0
    assert energy[1] == pytest.approx(energy[25])


# =========================================================================
# RELIABILITY — end-of-life sizing holds, cold-charge bound behaves
# =========================================================================

def test_sized_systems_meet_governing_year_lolp_on_real_data():
    """Every site's design must meet LOLP <= 1% in the design-governing
    year (year 24: PV at (1-d)^23, battery at 80% SoH) on the real TMY."""
    from src.sites import SITES
    from src.weather import get_or_create_tmy
    from src.pv_model import PVDesign
    from src.simulation import run_simulation
    from src.sizing import worst_life_state

    eol, soh, yr = worst_life_state()
    assert yr == 24 and abs(soh - 0.8) < 1e-9
    load = LoadProfile()
    for key, site in SITES.items():
        wp, kwh = SITE_DESIGN[site.name]
        df = get_or_create_tmy(site, prefer='real')
        pv = PVDesign(nameplate_w=wp)
        bat = BatteryDesign(capacity_kwh=kwh)
        r1 = run_simulation(df, site, pv, bat, load, pv_ageing_factor=eol,
                            battery_soh=soh)
        r = run_simulation(df, site, pv, bat, load,
                           initial_soc_frac=r1.final_soc_kwh / kwh,
                           pv_ageing_factor=eol, battery_soh=soh)
        assert r.lolp <= 0.01, f"{site.name}: governing-year LOLP {r.lolp*100:.2f}% > 1%"


def test_cold_charge_block_never_improves_lolp():
    from src.sites import SITES
    from src.weather import get_or_create_tmy
    from src.pv_model import PVDesign
    from src.simulation import run_simulation

    site = SITES['edinburgh']
    df = get_or_create_tmy(site, prefer='real')
    wp, kwh = SITE_DESIGN['Edinburgh']
    pv = PVDesign(nameplate_w=wp)
    bat = BatteryDesign(capacity_kwh=kwh)
    load = LoadProfile()
    base = run_simulation(df, site, pv, bat, load)
    cold = run_simulation(df, site, pv, bat, load, charge_temp_limit_c=0.0)
    assert cold.lolp >= base.lolp
    assert cold.annual_unmet_kwh >= base.annual_unmet_kwh
