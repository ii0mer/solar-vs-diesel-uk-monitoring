"""Tests for Upgrade 1 (validation against PVGIS) and Upgrade 2 (sixteen-
year reliability and final designs). They read the committed PVGIS series
in data/pvgis_series and the results files written by

    python -m src.validation
    python -m src.multiyear
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.sites import SITES
from src.pvgis_series import (load_all, REQUESTED_KWP,
                              huld_relative_efficiency,
                              faiman_module_temperature)
from src.validation import chain_from_poa, nmbe_nrmse
from src.multiyear import (simulate_multiyear, annual_table, pv_per_kwp,
                           select, grid_search)
from src.monte_carlo import SITE_DESIGN, SITE_DESIGN_TMY, SITE_DESIGN_DEG08
from src.sensitivity import LOAD_DESIGNS, DEGRADATION_DESIGNS, CENTRAL
from src.pv_model import LossChain
from src.load_profile import LoadProfile
from src.sizing import worst_life_state

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'


@pytest.fixture(scope='module')
def series():
    return load_all()


# =========================================================================
# PVGIS series files
# =========================================================================

def test_series_cover_sixteen_full_years(series):
    for k, s in series.items():
        assert s.years == list(range(2005, 2021)), k
        assert len(s.data) == 16 * 8766, k          # incl. leap days
        assert s.data['poa_global'].max() > 700, k
        assert int(s.data['reconstructed'].sum()) == 0, k


def test_nameplate_used_by_pvgis_matches_request(series):
    """The CSV header prints kWp to one decimal; verify the value actually
    used by reproducing P with PVGIS's own power model at high irradiance
    (residual = reflectance, a consistent ~2-3% across sites)."""
    for k, s in series.items():
        d = s.data
        g = d['poa_global'].values
        tm = faiman_module_temperature(g, d['temp_air'].values, d['wind_speed'].values)
        eta = huld_relative_efficiency(g, tm)
        m = g > 700
        p_est = REQUESTED_KWP[k] * 1000 * g[m] / 1000 * eta[m]
        ratio = np.median(d['p_pvgis_w'].values[m] / p_est)
        assert 0.96 < ratio < 0.99, f"{k}: median ratio {ratio:.3f}"
        assert s.kwp == REQUESTED_KWP[k]


# =========================================================================
# Validation (Layer 1: same inputs; Layer 2: TMY chain vs climatology)
# =========================================================================

def test_conversion_chain_within_5pct_of_pvgis_on_same_inputs(series):
    for k, s in series.items():
        ref = s.data['p_pvgis_w'] / s.kwp
        study = chain_from_poa(s, SITES[k])
        day = (ref > 0) | (study > 0)
        nmbe, nrmse = nmbe_nrmse(study[day].values, ref[day].values)
        assert 0 < nmbe < 5.0, f"{k}: NMBE {nmbe:.2f}%"      # PVWatts has no low-light loss
        assert nrmse < 8.0, f"{k}: NRMSE {nrmse:.2f}%"


def test_validation_results_file_consistent():
    V = json.loads((RES / 'validation.json').read_text())
    for site, r in V['sites'].items():
        l1, l2, pr = r['layer1_same_inputs'], r['layer2_tmy_chain'], r['pvgis_reference']
        assert abs(l1['annual_nmbe_pct']) < 5
        assert pr['annual_min_kwh_per_kwp'] <= l2['tmy_annual_kwh_per_kwp_noloss'] <= pr['annual_max_kwh_per_kwp'] * 1.03
        # low-irradiance classes carry the bias; high-irradiance classes agree
        assert l1['by_irradiance_class']['>800']['study_over_pvgis'] == pytest.approx(1.0, abs=0.02)
        assert l1['by_irradiance_class']['0-100']['study_over_pvgis'] > 1.2
    assert 0 < V['edinburgh_horizon']['annual_loss_pct'] < 2


# =========================================================================
# Multi-year loop and final designs
# =========================================================================

def test_fast_bucket_loop_matches_run_simulation():
    from src.weather import get_or_create_tmy
    from src.pv_model import PVDesign, simulate_pv_dc
    from src.battery import BatteryDesign
    from src.simulation import run_simulation
    site = SITES['edinburgh']
    df = get_or_create_tmy(site, prefer='real')
    wp, kwh = SITE_DESIGN_TMY[site.name]
    eol, soh, _ = worst_life_state()
    pv = simulate_pv_dc(df, site, PVDesign(nameplate_w=wp))
    load = LoadProfile(); bat = BatteryDesign(capacity_kwh=kwh)
    r1 = run_simulation(df, site, PVDesign(nameplate_w=wp), bat, load,
                        pv_ageing_factor=eol, battery_soh=soh, pv_series_w=pv)
    r2 = run_simulation(df, site, PVDesign(nameplate_w=wp), bat, load,
                        initial_soc_frac=r1.final_soc_kwh / kwh,
                        pv_ageing_factor=eol, battery_soh=soh, pv_series_w=pv)
    unmet, curtail = simulate_multiyear(pv.values * eol, load.design_power_w(),
                                        kwh, battery_soh=soh)
    assert np.abs(unmet - r2.hourly['unmet_wh'].values).max() < 1e-9
    assert curtail.sum() / 1000 == pytest.approx(r2.annual_curtailed_kwh, abs=1e-6)


def test_final_designs_meet_p90_under_both_chains(series):
    """Every final design must hold annual LOLP <= 1% in >= 15 of the 16
    years under BOTH PV conversion models (governing-year conditions)."""
    eol, soh, _ = worst_life_state()
    load_w = LoadProfile().design_power_w()
    losses = LossChain()
    for k, s in series.items():
        site = SITES[k]
        wp, kwh = SITE_DESIGN[site.name]
        for model in ('study', 'pvgis'):
            pvk = pv_per_kwp(s, site, model, losses)
            unmet, curtail = simulate_multiyear(pvk * wp / 1000 * eol, load_w, kwh,
                                                battery_soh=soh)
            yr = annual_table(unmet, curtail, s.data.index)
            ok = int((yr['lolp_pct'] <= 1.0).sum())
            assert ok >= 15, f"{site.name} {wp}/{kwh} under {model}: {ok}/16 years"


def test_tmy_designs_breach_in_real_years(series):
    """The step-1 TMY designs breach the 1% criterion in several real years
    (the reason the final designs exist)."""
    eol, soh, _ = worst_life_state()
    load_w = LoadProfile().design_power_w()
    losses = LossChain()
    for k, s in series.items():
        site = SITES[k]
        wp, kwh = SITE_DESIGN_TMY[site.name]
        pvk = pv_per_kwp(s, site, 'study', losses)
        unmet, curtail = simulate_multiyear(pvk * wp / 1000 * eol, load_w, kwh,
                                            battery_soh=soh)
        yr = annual_table(unmet, curtail, s.data.index)
        assert int((yr['lolp_pct'] > 1.0).sum()) >= 3, site.name


def test_constants_match_multiyear_results():
    M = json.loads((RES / 'multiyear.json').read_text())
    for site, (wp, kwh) in SITE_DESIGN.items():
        f = M['sites'][site]['final_design']
        assert (f['pv_wp'], f['battery_kwh']) == (wp, kwh), site
        assert f['years_ok_study'] >= 15 and f['years_ok_pvgis'] >= 15
    for site, (wp, kwh) in SITE_DESIGN_TMY.items():
        t = M['sites'][site]['tmy_design']
        assert (t['pv_wp'], t['battery_kwh']) == (wp, kwh), site
    assert (CENTRAL['pv_size_wp'], CENTRAL['battery_kwh']) == SITE_DESIGN['Southampton']
    assert LOAD_DESIGNS[1.0] == SITE_DESIGN['Southampton']
    assert DEGRADATION_DESIGNS[0.5] == SITE_DESIGN['Southampton']
    assert SITE_DESIGN_DEG08['Southampton'] == DEGRADATION_DESIGNS[0.8]


def test_select_criteria_logic():
    import pandas as pd
    g = pd.DataFrame({'pv_wp': [100, 200, 300], 'battery_kwh': [1, 1, 1],
                      'cost_index': [1, 2, 3],
                      'years_ok_study': [14, 15, 16], 'years_ok_pvgis': [16, 14, 16],
                      'mean_study': [1.5, 0.9, 0.2], 'mean_pvgis': [0.5, 1.2, 0.3],
                      'max_study': [3, 2, 1], 'max_pvgis': [1, 3, 1]})
    assert select(g, 'p90', ('study',))['pv_wp'] == 200
    assert select(g, 'p90', ('study', 'pvgis'))['pv_wp'] == 300
    assert select(g, 'all', ('study', 'pvgis'))['pv_wp'] == 300
    assert select(g, 'mean', ('study', 'pvgis'))['pv_wp'] == 300
    assert select(g, 'mean', ('study',))['pv_wp'] == 200
