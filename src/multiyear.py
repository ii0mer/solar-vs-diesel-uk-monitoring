"""Upgrade 2 — sixteen-winter reliability and the final (robust) designs.

Step 1 of the sizing (sizing.py) finds the cheapest design that holds
LOLP <= 1% in the design-governing year on a typical meteorological year.
Step 2, here, runs candidate designs continuously through the sixteen
real years 2005-2020 (PVGIS-SARAH2 hourly series, ``pvgis_series.py``)
under the same governing-year conditions (PV at the governing-year ageing
factor, battery at end-of-life state of health) and reports the annual
LOLP year by year.

Two PV conversion models drive the battery simulation, both carrying this
study's loss chain:

    'study'  — this study's chain (physical IAM, SAPM cell temperature,
               PVWatts DC) fed with the PVGIS plane-of-array components;
    'pvgis'  — PVGIS's own hourly output (Huld power model, Faiman
               temperature, Martin-Ruiz reflectance), scaled to the design
               nameplate. Section IV-A shows it runs ~3% below 'study'.

Multi-year criteria (annual LOLP, governing-year conditions):

    'mean'  — sixteen-year mean <= 1%
    'p90'   — <= 1% in at least fifteen of the sixteen years
    'all'   — <= 1% in every year

Each is evaluated for the 'study' chain alone and for BOTH chains
('robust'). The FINAL design adopted for the economics is the cheapest
(lifetime-cost index, as in sizing.py) design meeting 'p90' under both
chains. Provenance of the constants in monte_carlo.py / sensitivity.py:
results/multiyear_summary.txt.

    python -m src.multiyear   → results/multiyear.json,
                                results/multiyear_lolp_matrix.csv,
                                results/multiyear_summary.txt,
                                results/multiyear_grid_<site>.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sites import SITES, Site
from .load_profile import LoadProfile
from .battery import BatteryDesign
from .pv_model import LossChain
from .sizing import worst_life_state, _cost_index, DEFAULT_DEGRADATION_PCT_YR
from .pvgis_series import load_all, PvgisSeries
from .validation import chain_from_poa
from .economics import build_pv_battery_cashflows

RESULTS = Path(__file__).resolve().parents[1] / 'results'
YEARS = list(range(2005, 2021))
PV_GRID = np.arange(100, 1300, 50)
BATTERY_GRID = np.array([0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 5, 7, 10])
CRITERIA = {
    'mean': 'sixteen-year mean annual LOLP <= target',
    'p90': 'annual LOLP <= target in at least 15 of the 16 years',
    'all': 'annual LOLP <= target in all 16 years',
}
FINAL_CRITERION = ('p90', 'robust')      # criterion, model set

try:                                    # optional acceleration
    from numba import njit
except Exception:                       # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=True)
def _bucket_loop(pv_wh, load_wh, cap_wh, soc_min_wh, eta, self_dis, soc0,
                 charge_blocked):
    """Energy-bucket logic identical to simulation.run_simulation
    (including the optional cold-charge block: charging disabled in hours
    where ``charge_blocked`` is 1). Returns hourly unmet Wh, hourly
    curtailed Wh and the final SoC (Wh)."""
    n = pv_wh.shape[0]
    unmet = np.empty(n)
    curtail = np.empty(n)
    soc_out = np.empty(n)
    soc = soc0
    for i in range(n):
        soc *= (1.0 - self_dis)
        net = pv_wh[i] - load_wh[i]
        if net >= 0.0:
            headroom = 0.0 if charge_blocked[i] == 1 else (cap_wh - soc)
            e = net if net < headroom / eta else headroom / eta
            soc += e * eta
            unmet[i] = 0.0
            curtail[i] = net - e
        else:
            deficit = -net
            avail = (soc - soc_min_wh) * eta
            if avail < 0.0:
                avail = 0.0
            e = deficit if deficit < avail else avail
            soc -= e / eta
            unmet[i] = deficit - e
            curtail[i] = 0.0
        soc_out[i] = soc
    return unmet, curtail, soc_out, soc


def simulate_multiyear(pv_w: np.ndarray, load_w: float, battery_kwh: float,
                       battery_soh: float = 1.0,
                       battery: BatteryDesign | None = None,
                       charge_blocked: np.ndarray | None = None,
                       return_soc: bool = False,
                       start: str = 'two_pass'):
    """Continuous simulation over the whole series. ``start`` sets the
    1 January 2005 state of charge: 'two_pass' (default) starts the scored
    pass from the final SoC of a first pass, i.e. the weather-settled state;
    'full', 'half' and 'floor' start from 100%, 50% and the 20% floor of the
    SoH-reduced capacity. Returns (unmet_wh, curtail_wh) hourly arrays, plus
    the SoC fraction of the SoH-reduced capacity when ``return_soc`` is True."""
    bat = battery or BatteryDesign(capacity_kwh=battery_kwh)
    cap = battery_kwh * 1000.0 * battery_soh
    soc_min = cap * (1.0 - bat.max_dod)
    eta = float(np.sqrt(bat.round_trip_eff))
    sd = bat.hourly_self_discharge_frac
    load = np.full(pv_w.shape[0], load_w, dtype=float)
    pv = np.ascontiguousarray(pv_w, dtype=float)
    blk = (np.zeros(pv.shape[0], dtype=np.uint8) if charge_blocked is None
           else np.ascontiguousarray(charge_blocked, dtype=np.uint8))
    if start == 'two_pass':
        _, _, _, soc0 = _bucket_loop(pv, load, cap, soc_min, eta, sd, cap * 0.5, blk)
    elif start == 'full':
        soc0 = cap
    elif start == 'half':
        soc0 = cap * 0.5
    elif start == 'floor':
        soc0 = soc_min
    else:
        raise ValueError(start)
    unmet, curtail, soc_series, _ = _bucket_loop(pv, load, cap, soc_min, eta, sd,
                                                 soc0, blk)
    if return_soc:
        return unmet, curtail, soc_series / cap
    return unmet, curtail


def annual_table(unmet_wh, curtail_wh, index: pd.DatetimeIndex) -> pd.DataFrame:
    s = pd.Series(unmet_wh, index=index)
    c = pd.Series(curtail_wh, index=index)
    hours = s.resample('YS').size()
    unmet_h = (s > 0).resample('YS').sum()
    return pd.DataFrame({
        'lolp_pct': (unmet_h / hours * 100.0).values,
        'unmet_hours': unmet_h.values.astype(int),
        'eens_kwh': (s.resample('YS').sum() / 1000.0).values,
        'curtailed_kwh': (c.resample('YS').sum() / 1000.0).values,
    }, index=hours.index.year)


def worst_window(unmet_wh, index: pd.DatetimeIndex, year: int, days: int = 10):
    """Worst ``days``-day window of unmet hours among the windows that overlap
    calendar ``year``. Windows may straddle 31 December, so a New-Year event
    is found whole rather than truncated at the year boundary."""
    s = pd.Series(unmet_wh, index=index)
    y0 = pd.Timestamp(f'{year}-01-01', tz=index.tz)
    y1 = pd.Timestamp(f'{year + 1}-01-01', tz=index.tz)
    pad = pd.Timedelta(days=days)
    s = s[(s.index >= y0 - pad) & (s.index < y1 + pad)]
    u = (s > 0).astype(int)
    roll = u.rolling(days * 24).sum()
    starts = roll.index - pd.Timedelta(hours=days * 24 - 1)
    overlap = (roll.index >= y0) & (starts < y1)
    roll = roll[overlap]
    if len(roll) and roll.max() > 0:
        end = roll.idxmax(); start = end - pd.Timedelta(hours=days * 24 - 1)
        return str(start.date()), str(end.date()), int(roll.max())
    return None, None, 0


def season_table(unmet_wh, index: pd.DatetimeIndex) -> pd.DataFrame:
    """LOLP per July-June season (winter kept whole). Only the fifteen
    complete seasons 2005/06 ... 2019/20 are returned."""
    s = pd.Series(unmet_wh, index=index)
    season = np.where(s.index.month >= 7, s.index.year, s.index.year - 1)
    g = s.groupby(season)
    hours = g.size(); unmet_h = g.apply(lambda x: int((x > 0).sum()))
    df = pd.DataFrame({'lolp_pct': unmet_h / hours * 100.0,
                       'unmet_hours': unmet_h.astype(int), 'hours': hours})
    return df[(df['hours'] >= 8700) & (df.index >= YEARS[0]) & (df.index < YEARS[-1])]


COMPLIANT_MAX_H = 87   # 87 h = 0.99% of 8,760 h; 88 h exceeds 1% in any year


def compliance_margin(unmet_wh, index) -> dict:
    """Unmet hours in the closest compliant year and in the worst year, the
    number of years within the last 5 h of the 87 h limit, and the seasonal
    (July-June) compliance count."""
    yr = annual_table(unmet_wh, np.zeros_like(unmet_wh), index)
    ok = yr[yr['lolp_pct'] <= 1.0]['unmet_hours']
    sea = season_table(unmet_wh, index)
    return {'max_unmet_h_in_compliant_years': int(ok.max()) if len(ok) else 0,
            'years_within_5h_of_limit': int((ok >= COMPLIANT_MAX_H - 5).sum()),
            'seasons_within_1pct': int((sea['lolp_pct'] <= 1.0).sum()),
            'seasons_total': int(len(sea)),
            'season_max_lolp_pct': float(sea['lolp_pct'].max()),
            'seasons_over_1pct': [f'{int(y)}/{str(int(y) + 1)[-2:]}' for y in sea[sea['lolp_pct'] > 1.0].index]}


def pv_perturbation(pv_kwp: np.ndarray, index, wp, kwh, load_w, eol, soh,
                    factors=(0.95, 0.98, 1.02)) -> dict:
    """Compliance of a design when the whole PV series is scaled by a factor
    (a proxy for a shared irradiance error in the satellite record)."""
    out = {}
    for f in factors:
        u, _ = simulate_multiyear(pv_kwp * wp / 1000.0 * eol * f, load_w, kwh, battery_soh=soh)
        t = annual_table(u, np.zeros_like(u), index)
        out[f'{f:.2f}'] = {'years_within_1pct': int((t['lolp_pct'] <= 1.0).sum()),
                           'mean_lolp_pct': float(t['lolp_pct'].mean()),
                           'max_lolp_pct': float(t['lolp_pct'].max())}
    return out


def start_state_sensitivity(pv_kwp: np.ndarray, index, wp, kwh, load_w, eol, soh) -> dict:
    """Compliance of a design for each 1 January 2005 start convention."""
    out = {}
    for st in ('two_pass', 'full', 'half', 'floor'):
        u, _ = simulate_multiyear(pv_kwp * wp / 1000.0 * eol, load_w, kwh,
                                  battery_soh=soh, start=st)
        t = annual_table(u, np.zeros_like(u), index)
        out[st] = {'years_within_1pct': int((t['lolp_pct'] <= 1.0).sum()),
                   'unmet_h_2005': int(t['unmet_hours'].iloc[0]),
                   'max_lolp_pct': float(t['lolp_pct'].max())}
    return out


def pv_per_kwp(series: PvgisSeries, site: Site, model: str,
               losses: LossChain) -> np.ndarray:
    """Hourly PV output (W per kWp of nameplate) at the battery bus."""
    if model == 'study':
        return chain_from_poa(series, site, pdc0_w=1000.0, losses=losses).values
    if model == 'pvgis':
        return (series.data['p_pvgis_w'].values / series.kwp) * losses.derate
    raise ValueError(model)


def evaluate(pv_kwp: np.ndarray, index, wp: float, kwh: float, load_w: float,
             eol: float, soh: float):
    unmet, curtail = simulate_multiyear(pv_kwp * wp / 1000.0 * eol, load_w,
                                        kwh, battery_soh=soh)
    return annual_table(unmet, curtail, index), unmet


def grid_search(pv_kwp_by_model: dict, index, load_w: float, eol: float,
                soh: float, pv_grid=PV_GRID, battery_grid=BATTERY_GRID,
                target_pct: float = 1.0) -> pd.DataFrame:
    """Annual-LOLP statistics of every grid design under each PV model."""
    rows = []
    for wp in pv_grid:
        for kwh in battery_grid:
            rec = {'pv_wp': float(wp), 'battery_kwh': float(kwh),
                   'cost_index': float(_cost_index(float(wp), float(kwh)))}
            for model, pvk in pv_kwp_by_model.items():
                yr, _ = evaluate(pvk, index, float(wp), float(kwh), load_w,
                                 eol, soh)
                l = yr['lolp_pct'].values
                rec[f'years_ok_{model}'] = int((l <= target_pct).sum())
                rec[f'mean_{model}'] = float(l.mean())
                rec[f'max_{model}'] = float(l.max())
            rows.append(rec)
    return pd.DataFrame(rows)


def select(grid: pd.DataFrame, criterion: str, models: tuple,
           target_pct: float = 1.0):
    """Cheapest grid design meeting ``criterion`` under every model in
    ``models``. Returns the grid row (Series) or None."""
    ok = np.ones(len(grid), dtype=bool)
    for m in models:
        if criterion == 'mean':
            ok &= grid[f'mean_{m}'].values <= target_pct
        elif criterion == 'p90':
            ok &= grid[f'years_ok_{m}'].values >= 15
        elif criterion == 'all':
            ok &= grid[f'years_ok_{m}'].values == 16
        else:
            raise ValueError(criterion)
    feas = grid[ok]
    if feas.empty:
        return None
    return feas.sort_values('cost_index').iloc[0]


def _lcoe_npv(wp, kwh, rate=0.05):
    cf = build_pv_battery_cashflows(pv_capex_gbp_per_wp=4.50, pv_size_wp=wp,
                                    battery_capex_gbp_per_kwh=700.0,
                                    battery_kwh=kwh,
                                    annual_energy_delivered_kwh=128.2)
    return float(cf.lcoe_gbp_per_kwh(rate)), float(cf.total_npv(rate))


# ---------------------------------------------------------------------------
# Re-sizing helper for the sensitivity analysis (load, degradation)
# ---------------------------------------------------------------------------

_SERIES_CACHE: dict = {}


def _series():
    if 'all' not in _SERIES_CACHE:
        _SERIES_CACHE['all'] = load_all()
    return _SERIES_CACHE['all']


def _pv_kwp_cache(site_key: str, losses: LossChain) -> dict:
    k = ('pvk', site_key)
    if k not in _SERIES_CACHE:
        s = _series()[site_key]
        site = SITES[site_key]
        _SERIES_CACHE[k] = {m: pv_per_kwp(s, site, m, losses)
                            for m in ('study', 'pvgis')}
    return _SERIES_CACHE[k]


def robust_design(site_key: str, load_mult: float = 1.0,
                  degradation_pct_yr: float = DEFAULT_DEGRADATION_PCT_YR,
                  criterion: str = FINAL_CRITERION[0],
                  models: tuple = ('study', 'pvgis'),
                  target_pct: float = 1.0):
    """Cheapest design meeting the multi-year criterion under the given
    models, for a load multiplier and degradation rate. Returns
    (pv_wp, battery_kwh, grid_row)."""
    losses = LossChain()
    eol, soh, _ = worst_life_state(degradation_pct_yr)
    load_w = LoadProfile().design_power_w() * load_mult
    s = _series()[site_key]
    grid = grid_search(_pv_kwp_cache(site_key, losses), s.data.index, load_w,
                       eol, soh, target_pct=target_pct)
    row = select(grid, criterion, models, target_pct)
    if row is None:
        return None
    return float(row['pv_wp']), float(row['battery_kwh']), row



# ---------------------------------------------------------------------------
# Out-of-sample check (leave-one-year-out) and life-average LOLP
# ---------------------------------------------------------------------------

def lolp_matrix(pv_kwp_by_model: dict, index, load_w: float, eol: float,
                soh: float, pv_grid=PV_GRID, battery_grid=BATTERY_GRID):
    """Annual LOLP (%) of every grid design under each model:
    dict model -> array [n_designs, n_years]; plus the design list and cost."""
    designs, cost = [], []
    for wp in pv_grid:
        for kwh in battery_grid:
            designs.append((float(wp), float(kwh)))
            cost.append(float(_cost_index(float(wp), float(kwh))))
    out = {}
    for model, pvk in pv_kwp_by_model.items():
        rows = []
        for wp, kwh in designs:
            yr, _ = evaluate(pvk, index, wp, kwh, load_w, eol, soh)
            rows.append(yr['lolp_pct'].values)
        out[model] = np.array(rows)
    return designs, np.array(cost), out


def leave_one_year_out(designs, cost, mats: dict, models=('study', 'pvgis'),
                       target_pct: float = 1.0, allowance: int = 1) -> dict:
    """For each held-out year: select the cheapest design with at most
    ``allowance`` exceedances among the other years under every model in
    ``models`` (the final criterion applied to fifteen years), then test the
    held-out year. Returns per-model held-out exceedance counts and the
    list of (year, design, LOLP) exceedances."""
    n_years = next(iter(mats.values())).shape[1]
    years = list(range(2005, 2005 + n_years))
    exceed = {m: [] for m in models}
    chosen = []
    for h in range(n_years):
        keep = [j for j in range(n_years) if j != h]
        ok = np.ones(len(designs), dtype=bool)
        for m in models:
            ok &= (mats[m][:, keep] > target_pct).sum(axis=1) <= allowance
        idx = np.where(ok)[0]
        if len(idx) == 0:
            chosen.append(None); continue
        best = idx[np.argmin(cost[idx])]
        chosen.append((years[h], designs[best]))
        for m in models:
            v = float(mats[m][best, h])
            if v > target_pct:
                exceed[m].append({'held_out_year': years[h], 'pv_wp': designs[best][0],
                                  'battery_kwh': designs[best][1], 'lolp_pct': v})
    return {'held_out_exceedances': {m: len(exceed[m]) for m in models},
            'held_out_rate_pct': {m: len(exceed[m]) / n_years * 100 for m in models},
            'details': exceed,
            'in_sample_allowance': allowance, 'n_years': n_years}


def life_average_lolp(pv_kwp: np.ndarray, index, wp: float, kwh: float,
                      load_w: float, degradation_pct_yr: float = 0.5,
                      battery_life_years: int = 12, project_years: int = 25,
                      eol_soh: float = 0.8) -> dict:
    """Expected LOLP along the actual ageing trajectory: for each project
    year t the PV factor is (1-d)^(t-1) and the battery SoH falls linearly
    from 1.0 at installation to ``eol_soh`` at the end of each battery
    life; each year state is run through the sixteen weather years and the
    annual LOLPs averaged, then averaged over the project years."""
    per_year = []
    for t in range(1, project_years + 1):
        pv_f = (1.0 - degradation_pct_yr / 100.0) ** (t - 1)
        age = ((t - 1) % battery_life_years) + 1          # 1..L
        soh_t = 1.0 - (1.0 - eol_soh) * age / battery_life_years
        yr, _ = evaluate(pv_kwp, index, wp, kwh, load_w, pv_f, soh_t)
        per_year.append(float(yr['lolp_pct'].mean()))
    return {'life_average_lolp_pct': float(np.mean(per_year)),
            'by_project_year': per_year,
            'max_project_year_mean_pct': float(np.max(per_year))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _design_block(row, base_lcoe, base_npv, wp0, kwh0):
    if row is None:
        return None
    lcoe, npv = _lcoe_npv(row['pv_wp'], row['battery_kwh'])
    d = {'pv_wp': float(row['pv_wp']), 'battery_kwh': float(row['battery_kwh']),
         'cost_index': float(row['cost_index']),
         'delta_pv_wp': float(row['pv_wp'] - wp0),
         'delta_battery_kwh': float(row['battery_kwh'] - kwh0),
         'lcoe_5pct': lcoe, 'lcoe_delta_pct': (lcoe / base_lcoe - 1) * 100,
         'npv_5pct': npv, 'npv_delta_pct': (npv / base_npv - 1) * 100,
         'capex_gbp': float(row['pv_wp'] * 4.5 + row['battery_kwh'] * 700 + 150 + 600)}
    for m in ('study', 'pvgis'):
        d[f'years_ok_{m}'] = int(row[f'years_ok_{m}'])
        d[f'mean_lolp_pct_{m}'] = float(row[f'mean_{m}'])
        d[f'max_lolp_pct_{m}'] = float(row[f'max_{m}'])
    return d


def _series_stats(pv_kwp: np.ndarray, index, wp, kwh, load_w, eol, soh) -> dict:
    yr, unmet = evaluate(pv_kwp, index, wp, kwh, load_w, eol, soh)
    worst_year = int(yr['lolp_pct'].idxmax())
    ww = worst_window(unmet, index, worst_year)
    return {
        'lolp_pct_by_year': {int(y): float(v) for y, v in yr['lolp_pct'].items()},
        'unmet_hours_by_year': {int(y): int(v) for y, v in yr['unmet_hours'].items()},
        'eens_kwh_by_year': {int(y): float(v) for y, v in yr['eens_kwh'].items()},
        'curtailed_kwh_by_year': {int(y): float(v) for y, v in yr['curtailed_kwh'].items()},
        'mean_lolp_pct': float(yr['lolp_pct'].mean()),
        'median_lolp_pct': float(yr['lolp_pct'].median()),
        'max_lolp_pct': float(yr['lolp_pct'].max()),
        'min_lolp_pct': float(yr['lolp_pct'].min()),
        'worst_year': worst_year,
        'worst_window': {'start': ww[0], 'end': ww[1], 'unmet_h': ww[2]},
        'years_within_1pct': int((yr['lolp_pct'] <= 1.0).sum()),
        'years_over_1pct': int((yr['lolp_pct'] > 1.0).sum()),
        'p90_lolp_pct': float(np.percentile(yr['lolp_pct'].values, 90)),
        'total_unmet_hours_16yr': int(yr['unmet_hours'].sum()),
        'mean_eens_kwh': float(yr['eens_kwh'].mean()),
        'max_eens_kwh': float(yr['eens_kwh'].max()),
        'mean_curtailed_kwh': float(yr['curtailed_kwh'].mean()),
    }


def main(verbose: bool = True) -> dict:
    from .monte_carlo import SITE_DESIGN_TMY
    load = LoadProfile()
    load_w = load.design_power_w()
    losses = LossChain()
    eol, soh, gov_year = worst_life_state()
    series = _series()
    out = {'years': YEARS, 'governing_year': gov_year, 'pv_ageing_factor': eol,
           'battery_soh': soh, 'criteria': CRITERIA,
           'final_criterion': {'criterion': FINAL_CRITERION[0],
                               'models': ['study', 'pvgis']},
           'pv_grid_wp': [int(v) for v in PV_GRID],
           'battery_grid_kwh': [float(v) for v in BATTERY_GRID],
           'sites': {}}
    matrix_rows, summary = [], []
    for key, s in series.items():
        site = SITES[key]
        idx = s.data.index
        pvk = _pv_kwp_cache(key, losses)
        wp0, kwh0 = SITE_DESIGN_TMY[site.name]
        base_lcoe, base_npv = _lcoe_npv(wp0, kwh0)
        rec = {'tmy_design': {'pv_wp': wp0, 'battery_kwh': kwh0,
                              'lcoe_5pct': base_lcoe, 'npv_5pct': base_npv,
                              'capex_gbp': wp0 * 4.5 + kwh0 * 700 + 150 + 600,
                              'by_model': {}}}
        for m in ('study', 'pvgis'):
            rec['tmy_design']['by_model'][m] = _series_stats(pvk[m], idx, wp0, kwh0,
                                                             load_w, eol, soh)
        grid = grid_search(pvk, idx, load_w, eol, soh)
        grid.to_csv(RESULTS / f'multiyear_grid_{site.name.lower()}.csv', index=False)
        designs = {}
        for crit in CRITERIA:
            designs[crit] = {
                'study': _design_block(select(grid, crit, ('study',)), base_lcoe, base_npv, wp0, kwh0),
                'robust': _design_block(select(grid, crit, ('study', 'pvgis')), base_lcoe, base_npv, wp0, kwh0),
            }
        rec['designs'] = designs
        fin = designs[FINAL_CRITERION[0]][FINAL_CRITERION[1]]
        rec['final_design'] = dict(fin)
        rec['final_design']['by_model'] = {}
        for m in ('study', 'pvgis'):
            rec['final_design']['by_model'][m] = _series_stats(
                pvk[m], idx, fin['pv_wp'], fin['battery_kwh'], load_w, eol, soh)
        # year-1 conditions for context
        y1 = _series_stats(pvk['study'], idx, fin['pv_wp'], fin['battery_kwh'],
                           load_w, 1.0, 1.0)
        rec['final_design']['year1_conditions_study'] = {
            'mean_lolp_pct': y1['mean_lolp_pct'], 'max_lolp_pct': y1['max_lolp_pct'],
            'years_within_1pct': y1['years_within_1pct']}
        # LFP cold-charge bounds for the final design over the sixteen years
        # (ambient <= 0 degC blocks charging; 15% winter capacity derate)
        subzero = (s.data['temp_air'].values <= 0.0).astype(np.uint8)
        pv_fin = pvk['study'] * fin['pv_wp'] / 1000.0 * eol
        u_base, _ = simulate_multiyear(pv_fin, load_w, fin['battery_kwh'], battery_soh=soh)
        u_cold, _ = simulate_multiyear(pv_fin, load_w, fin['battery_kwh'], battery_soh=soh,
                                       charge_blocked=subzero)
        u_cap, _ = simulate_multiyear(pv_fin, load_w, fin['battery_kwh'], battery_soh=soh * 0.85)
        def _yr(u):
            t = annual_table(u, np.zeros_like(u), idx)
            return {'mean_lolp_pct': float(t['lolp_pct'].mean()),
                    'max_lolp_pct': float(t['lolp_pct'].max()),
                    'years_within_1pct': int((t['lolp_pct'] <= 1.0).sum())}
        # does extra array restore the criterion under the charge block?
        incr = {}
        for dwp in (50, 100, 150, 200):
            incr[str(dwp)] = {}
            for m in ('study', 'pvgis'):
                pv_i = pvk[m] * (fin['pv_wp'] + dwp) / 1000.0 * eol
                u_i, _ = simulate_multiyear(pv_i, load_w, fin['battery_kwh'],
                                            battery_soh=soh, charge_blocked=subzero)
                incr[str(dwp)][m] = _yr(u_i)
        rec['final_design']['cold_bounds'] = {
            'base': _yr(u_base), 'charge_blocked_below_0c': _yr(u_cold),
            'capacity_derate_15pct': _yr(u_cap),
            'charge_blocked_with_extra_array_wp': incr,
            'subzero_hours_per_year_mean': float(subzero.sum() / 16.0)}
        # margin to the 87 h limit, July-June seasons, PV perturbation and
        # start-state conventions, for both chains
        for m in ('study', 'pvgis'):
            u_m, _ = simulate_multiyear(pvk[m] * fin['pv_wp'] / 1000.0 * eol, load_w,
                                        fin['battery_kwh'], battery_soh=soh)
            rec['final_design']['by_model'][m]['margin'] = compliance_margin(u_m, idx)
            rec['final_design']['by_model'][m]['pv_perturbation'] = pv_perturbation(
                pvk[m], idx, fin['pv_wp'], fin['battery_kwh'], load_w, eol, soh)
            rec['final_design']['by_model'][m]['start_state'] = start_state_sensitivity(
                pvk[m], idx, fin['pv_wp'], fin['battery_kwh'], load_w, eol, soh)
        # life-average LOLP along the ageing trajectory (both chains)
        rec['final_design']['life_average'] = {
            m: life_average_lolp(pvk[m], idx, fin['pv_wp'], fin['battery_kwh'], load_w)
            for m in ('study', 'pvgis')}
        # energy-based loss-of-load (EENS / demand) for the record
        for m in ('study', 'pvgis'):
            b = rec['final_design']['by_model'][m]
            b['llp_energy_pct_mean'] = b['mean_eens_kwh'] / (load_w * 8.766) * 100.0
            b['llp_energy_pct_max'] = b['max_eens_kwh'] / (load_w * 8.766) * 100.0
        # leave-one-year-out check of the selection rule
        designs_l, cost_l, mats = lolp_matrix(pvk, idx, load_w, eol, soh)
        rec['final_design']['leave_one_year_out'] = leave_one_year_out(
            designs_l, cost_l, mats)
        # 0.1 % target under the final criterion (price of reliability)
        g01 = grid_search(pvk, idx, load_w, eol, soh, target_pct=0.1)
        r01 = select(g01, FINAL_CRITERION[0], ('study', 'pvgis'), target_pct=0.1)
        rec['final_design_lolp_0p1'] = _design_block(r01, fin['lcoe_5pct'], fin['npv_5pct'],
                                                     fin['pv_wp'], fin['battery_kwh'])
        out['sites'][site.name] = rec
        for dsg_name, dsg in (('tmy', rec['tmy_design']), ('final', rec['final_design'])):
            for m in ('study', 'pvgis'):
                for y, v in dsg['by_model'][m]['lolp_pct_by_year'].items():
                    matrix_rows.append({'site': site.name, 'design': dsg_name,
                                        'model': m, 'year': int(y),
                                        'lolp_pct': float(v),
                                        'unmet_hours': dsg['by_model'][m]['unmet_hours_by_year'][y]})
        st, pg = rec['tmy_design']['by_model']['study'], rec['tmy_design']['by_model']['pvgis']
        summary += [
            f"{site.name}: TMY design {wp0:.0f} Wp / {kwh0} kWh — sixteen-year LOLP "
            f"(governing-year conditions): study chain mean {st['mean_lolp_pct']:.2f}%, "
            f"max {st['max_lolp_pct']:.2f}% ({st['worst_year']}), years <= 1%: "
            f"{st['years_within_1pct']}/16; PVGIS chain mean {pg['mean_lolp_pct']:.2f}%, "
            f"max {pg['max_lolp_pct']:.2f}% ({pg['worst_year']}), years <= 1%: {pg['years_within_1pct']}/16",
        ]
        for crit in CRITERIA:
            for ms in ('study', 'robust'):
                d = designs[crit][ms]
                if d:
                    summary.append(f"    {crit:4s}/{ms:6s}: {d['pv_wp']:.0f} Wp / {d['battery_kwh']} kWh "
                                   f"(dPV {d['delta_pv_wp']:+.0f} Wp, dBatt {d['delta_battery_kwh']:+.2f} kWh, "
                                   f"LCOE {d['lcoe_delta_pct']:+.1f}%) years ok study/pvgis "
                                   f"{d['years_ok_study']}/{d['years_ok_pvgis']}")
        f = rec['final_design']
        summary.append(f"    FINAL ({FINAL_CRITERION[0]}, both chains): {f['pv_wp']:.0f} Wp / {f['battery_kwh']} kWh; "
                       f"LCOE £{f['lcoe_5pct']:.2f}/kWh (+{f['lcoe_delta_pct']:.1f}% vs TMY design); "
                       f"study chain mean {f['by_model']['study']['mean_lolp_pct']:.2f}% max {f['by_model']['study']['max_lolp_pct']:.2f}% "
                       f"({f['by_model']['study']['worst_year']}); pvgis chain mean {f['by_model']['pvgis']['mean_lolp_pct']:.2f}% "
                       f"max {f['by_model']['pvgis']['max_lolp_pct']:.2f}%")
        if rec['final_design_lolp_0p1']:
            q = rec['final_design_lolp_0p1']
            summary.append(f"    0.1% target (same criterion): {q['pv_wp']:.0f} Wp / {q['battery_kwh']} kWh, LCOE {q['lcoe_delta_pct']:+.1f}% vs final")
        if verbose:
            print('\n'.join(summary[-8:]))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'multiyear.json').write_text(json.dumps(out, indent=1))
    pd.DataFrame(matrix_rows).to_csv(RESULTS / 'multiyear_lolp_matrix.csv', index=False)
    header = [
        "Sixteen-year (2005-2020) reliability of the TMY-sized designs and the",
        "final designs, PVGIS-SARAH2 hourly series, governing-year conditions",
        f"(PV x {eol:.4f}, battery SoH {soh}). Two PV conversion models: 'study'",
        "(this study's chain on PVGIS plane-of-array components) and 'pvgis'",
        "(PVGIS's own P). Final design = cheapest lifetime-cost design with",
        "annual LOLP <= 1% in at least 15 of 16 years under BOTH models.",
        "These are the SITE_DESIGN constants in src/monte_carlo.py.", "",
    ]
    (RESULTS / 'multiyear_summary.txt').write_text('\n'.join(header + summary) + '\n')
    return out


if __name__ == '__main__':
    main()
