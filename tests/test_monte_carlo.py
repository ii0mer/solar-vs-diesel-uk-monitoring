"""Tests for the Monte Carlo LCOE module (Upgrade 3 + 4B)."""
import numpy as np
import pytest

from src.monte_carlo import (
    sample_draws, run_monte_carlo, summarise,
    SITE_PV_WP, BATTERY_LIFETIME_CHOICES,
)
from src.sensitivity import CENTRAL, central_pv_lcoe, central_diesel_lcoe

N_TEST = 400  # small but statistically adequate for the assertions below


def test_draws_reproducible_with_seed():
    a = sample_draws(n=50, seed=123)
    b = sample_draws(n=50, seed=123)
    assert np.array_equal(a.fuel_price_ppl, b.fuel_price_ppl)
    assert np.array_equal(a.battery_life_years, b.battery_life_years)


def test_draws_respect_distribution_bounds():
    d = sample_draws(n=N_TEST)
    assert d.fuel_price_ppl.min() >= 44.96 and d.fuel_price_ppl.max() <= 117.56
    assert d.battery_cost_mult.min() >= 0.70 and d.battery_cost_mult.max() <= 1.30
    assert d.pv_cost_mult.min() >= 0.70 and d.pv_cost_mult.max() <= 1.30
    assert d.load_mult.min() >= 0.80 and d.load_mult.max() <= 1.20
    assert d.visit_cost_mult.min() >= 0.5 and d.visit_cost_mult.max() <= 2.0
    assert set(np.unique(d.battery_life_years)) <= set(BATTERY_LIFETIME_CHOICES)


def test_degenerate_draws_reproduce_deterministic_exactly():
    """Collapse every distribution to its central value: the MC plumbing
    must reproduce the deterministic LCOE to machine precision. This
    isolates wiring errors from distribution-shape effects."""
    from src.monte_carlo import McDraws, _solar_lcoe_one, _diesel_lcoe_one
    one = np.ones(1)
    d = McDraws(
        fuel_price_ppl=np.array([76.02]),
        battery_cost_mult=one.copy(),
        pv_cost_mult=one.copy(),
        load_mult=one.copy(),
        visit_cost_mult=one.copy(),
        battery_life_years=np.array([12]),
        diesel_capex_mult=one.copy(),
        discount_rate=np.array([0.05]),
    )
    assert _solar_lcoe_one((500.0, 2.0), d, 0) == pytest.approx(
        central_pv_lcoe(CENTRAL), rel=1e-9)
    assert _diesel_lcoe_one(d, 0) == pytest.approx(
        central_diesel_lcoe(CENTRAL, architecture=2), rel=1e-9)


def test_median_within_expected_band_of_deterministic():
    """The MC median sits ABOVE the deterministic point by design: the
    visit-cost triangle (0.5, 1.0, 2.0) has mean 7/6 of its mode, the
    fuel triangle mean exceeds its mode, and mean battery life (11.25 yr)
    is below the deterministic 12 yr. Assert the median lands in the
    band that asymmetry predicts, not outside it."""
    mc = run_monte_carlo(n=1500, discount_rate=0.05)
    det_solar = central_pv_lcoe(CENTRAL)          # Southampton final design (500 Wp + 2.0 kWh) central
    p50 = np.percentile(mc['solar_lcoe_Southampton'], 50)
    assert 0.98 < p50 / det_solar < 1.20
    det_diesel = central_diesel_lcoe(CENTRAL, architecture=2)
    d50 = np.percentile(mc['diesel_lcoe'], 50)
    assert 0.98 < d50 / det_diesel < 1.20


def test_solar_beats_diesel_probability_is_one():
    """Across joint parameter uncertainty the solar advantage never inverts.

    The deterministic combined worst case leaves solar 2.2x ahead, so no
    plausible joint draw should invert the ranking.
    """
    mc = run_monte_carlo(n=N_TEST, discount_rate=0.05)
    s = summarise(mc, '5%')
    assert (s['p_solar_beats_diesel'] == 1.0).all()


def test_ratio_p10_above_two():
    """Even the pessimistic tail (P10 of the ratio) stays above 2x."""
    mc = run_monte_carlo(n=N_TEST, discount_rate=0.05)
    s = summarise(mc, '5%')
    assert (s['ratio_p10'] > 2.0).all()


def test_higher_rate_raises_solar_lcoe():
    """Capital-heavy solar is penalised more by 8% than diesel."""
    mc5 = run_monte_carlo(n=N_TEST, discount_rate=0.05)
    mc8 = run_monte_carlo(n=N_TEST, discount_rate=0.08)
    for site in SITE_PV_WP:
        assert (np.percentile(mc8[f'solar_lcoe_{site}'], 50)
                > np.percentile(mc5[f'solar_lcoe_{site}'], 50))
    # and the ratio narrows at the higher rate
    r5 = np.percentile(mc5['diesel_lcoe'] / mc5['solar_lcoe_Edinburgh'], 50)
    r8 = np.percentile(mc8['diesel_lcoe'] / mc8['solar_lcoe_Edinburgh'], 50)
    assert r8 < r5
