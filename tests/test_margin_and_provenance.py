"""Tests for the reliability-margin diagnostics added after the round-3
review (cross-year worst window, July-June seasons, compliance margin,
PV perturbation and start-state conventions), for the weather-data
provenance guard, and for the costing-horizon block in numbers.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.multiyear import (worst_window, season_table, compliance_margin,
                           annual_table, COMPLIANT_MAX_H, YEARS)
from src.monte_carlo import SITE_DESIGN

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'


def _index(y0=2005, y1=2021):
    """Hourly UTC index over calendar years [y0, y1), PVGIS-style (HH:10)."""
    return pd.date_range(f'{y0}-01-01 00:10', f'{y1 - 1}-12-31 23:10', freq='h', tz='UTC')


def _unmet(index, *spans):
    """Unmet-energy series (Wh) that is positive over the given [start, end]
    date spans (inclusive, whole days) and zero elsewhere."""
    u = np.zeros(len(index))
    for a, b in spans:
        m = (index >= pd.Timestamp(a, tz='UTC')) & (index < pd.Timestamp(b, tz='UTC') + pd.Timedelta(days=1))
        u[m] = 1.0
    return u


# -------------------------------------------------------------------------
# worst_window: a New-Year event must be found whole
# -------------------------------------------------------------------------

def test_worst_window_straddles_year_boundary():
    idx = _index(2006, 2008)
    u = _unmet(idx, ('2006-12-26', '2007-01-04'))          # 10 days = 240 h
    start, end, h = worst_window(u, idx, 2006, days=10)
    assert (start, end, h) == ('2006-12-26', '2007-01-04', 240)
    # the same event is the worst window overlapping 2007 as well
    assert worst_window(u, idx, 2007, days=10)[2] == 240


def test_worst_window_ignores_events_outside_the_year():
    idx = _index(2006, 2008)
    u = _unmet(idx, ('2006-03-01', '2006-03-02'), ('2007-06-01', '2007-06-10'))
    start, end, h = worst_window(u, idx, 2006, days=10)
    assert h == 48 and start <= '2006-03-01' <= end
    assert worst_window(u, idx, 2007, days=10)[2] == 240
    assert worst_window(np.zeros(len(idx)), idx, 2006) == (None, None, 0)


# -------------------------------------------------------------------------
# season_table / compliance_margin
# -------------------------------------------------------------------------

def test_season_table_keeps_winter_whole():
    idx = _index()
    # 48 unmet hours in late Dec 2006 and 48 in early Jan 2007: two compliant
    # calendar years (0.55% each) but one season at 1.10%
    u = _unmet(idx, ('2006-12-29', '2006-12-30'), ('2007-01-01', '2007-01-02'))
    yr = annual_table(u, np.zeros_like(u), idx)
    assert yr.loc[2006, 'unmet_hours'] == 48 and yr.loc[2007, 'unmet_hours'] == 48
    assert (yr['lolp_pct'] <= 1.0).all()
    sea = season_table(u, idx)
    assert list(sea.index) == list(range(YEARS[0], YEARS[-1]))      # 2005/06 ... 2019/20
    assert len(sea) == 15
    assert sea.loc[2006, 'unmet_hours'] == 96
    assert sea.loc[2006, 'lolp_pct'] > 1.0
    assert sea.loc[2005, 'unmet_hours'] == 0 and sea.loc[2007, 'unmet_hours'] == 0


def test_compliance_margin_counts_hours_and_seasons():
    idx = _index()
    u = np.zeros(len(idx))
    y2010 = np.where(idx.year == 2010)[0]
    u[y2010[:COMPLIANT_MAX_H]] = 1.0            # 87 h: compliant, inside the last 5 h
    m = compliance_margin(u, idx)
    assert m['max_unmet_h_in_compliant_years'] == COMPLIANT_MAX_H
    assert m['years_within_5h_of_limit'] == 1
    assert m['seasons_within_1pct'] == 15 and m['seasons_total'] == 15
    assert m['seasons_over_1pct'] == []
    u[y2010[COMPLIANT_MAX_H]] = 1.0             # 88 h: the year now fails
    yr = annual_table(u, np.zeros_like(u), idx)
    assert yr.loc[2010, 'lolp_pct'] > 1.0
    m = compliance_margin(u, idx)
    assert m['max_unmet_h_in_compliant_years'] == 0
    assert m['years_within_5h_of_limit'] == 0


def test_compliant_limit_is_87_hours():
    assert COMPLIANT_MAX_H == 87
    assert COMPLIANT_MAX_H / 8760 * 100 <= 1.0 < (COMPLIANT_MAX_H + 1) / 8760 * 100


# -------------------------------------------------------------------------
# weather provenance: no silent synthetic fallback
# -------------------------------------------------------------------------

def test_synthetic_weather_requires_explicit_opt_in(tmp_path, monkeypatch):
    import src.weather as weather
    from src.sites import SITES
    site = SITES['southampton']
    monkeypatch.setattr(weather, 'DATA_DIR', tmp_path)

    def boom(_site):
        raise ConnectionError('PVGIS unreachable')
    monkeypatch.setattr(weather, 'fetch_pvgis_tmy', boom)
    monkeypatch.delenv('ALLOW_SYNTHETIC_WEATHER', raising=False)
    with pytest.raises(RuntimeError, match='ALLOW_SYNTHETIC_WEATHER'):
        weather.get_or_create_tmy(site, prefer='real')
    assert not (tmp_path / 'southampton_tmy.csv').exists()

    monkeypatch.setenv('ALLOW_SYNTHETIC_WEATHER', '1')
    df = weather.get_or_create_tmy(site, prefer='real')
    assert len(df) == 8760
    # the synthetic series is never cached under the real-data name
    assert not (tmp_path / 'southampton_tmy.csv').exists()
    assert (tmp_path / 'southampton_tmy_synthetic.csv').exists()


# -------------------------------------------------------------------------
# results files carry the new blocks and are internally consistent
# -------------------------------------------------------------------------

def test_multiyear_results_carry_margin_blocks():
    M = json.loads((RES / 'multiyear.json').read_text())
    for site in SITE_DESIGN:
        for model in ('study', 'pvgis'):
            b = M['sites'][site]['final_design']['by_model'][model]
            for key in ('margin', 'pv_perturbation', 'start_state', 'worst_window'):
                assert key in b, (site, model, key)
            mg = b['margin']
            assert mg['seasons_total'] == 15
            assert 0 <= mg['max_unmet_h_in_compliant_years'] <= COMPLIANT_MAX_H
            assert set(b['pv_perturbation']) == {'0.95', '0.98', '1.02'}
            assert set(b['start_state']) == {'two_pass', 'full', 'half', 'floor'}
            # compliance cannot improve when the array is scaled down
            p = b['pv_perturbation']
            assert p['0.95']['years_within_1pct'] <= p['0.98']['years_within_1pct'] <= p['1.02']['years_within_1pct']
            # the two-pass start equals the full-start count and is never
            # better than what the report claims (15 of 16)
            assert b['start_state']['two_pass']['years_within_1pct'] >= 15
            w = b['worst_window']
            # a 240-h window spans nine or ten calendar dates depending on
            # the hour at which it starts
            span = pd.Timestamp(w['end']) - pd.Timestamp(w['start'])
            assert span in (pd.Timedelta(days=9), pd.Timedelta(days=10))
            assert 0 < w['unmet_h'] <= 240


def test_liverpool_worst_window_crosses_new_year():
    """The Liverpool worst event runs into 1 January 2007; before the fix it
    was truncated at 31 December 2006 and under-counted by ten hours."""
    M = json.loads((RES / 'multiyear.json').read_text())
    w = M['sites']['Liverpool']['final_design']['by_model']['study']['worst_window']
    assert w['start'] == '2006-12-22' and w['end'] == '2007-01-01'
    assert w['unmet_h'] == 157


def test_numbers_horizon_block_matches_baseline_at_25_years():
    N = json.loads((RES / 'numbers.json').read_text())
    H = N['horizon']
    assert set(H) == {'10', '15', '25'}
    for years in H:
        assert set(H[years]) == {'0.05', '0.08'}
    base = N['econ']['0.05']
    h25 = H['25']['0.05']
    assert h25['diesel_a2_lcoe'] == pytest.approx(base['diesel_a2_lcoe'], rel=1e-9)
    for site in SITE_DESIGN:
        assert h25['sites'][site]['solar_lcoe'] == pytest.approx(base['sites'][site]['solar_lcoe'], rel=1e-9)
        # a shorter horizon raises the solar LCOE more than the diesel LCOE,
        # so the ratio falls monotonically as the horizon shortens
        r10, r15, r25 = (H[y]['0.05']['sites'][site]['ratio_a2'] for y in ('10', '15', '25'))
        assert r10 < r15 < r25
        assert r10 > 2.0           # solar still cheaper at every horizon tested
    fd = N['fuel_doubled']
    assert fd['fuel_price_ppl'] == pytest.approx(2 * 76.02)
    assert 0 < fd['a2_lcoe_delta_pct'] < 5
