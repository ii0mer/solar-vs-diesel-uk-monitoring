"""Dump every number the dissertation text quotes into results/numbers.json.

The Word build script reads this file, so a figure in the text can never
drift from the model. Run after any model change:

    python -m src.dump_numbers
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sites import SITES
from .weather import get_or_create_tmy, annual_ghi_kwh_per_m2
from .load_profile import LoadProfile
from .pv_model import PVDesign, LossChain, simulate_pv_dc
from .battery import BatteryDesign
from .simulation import run_simulation
from .sizing import eol_ageing_factor, worst_life_state, size_system
from .economics import (build_pv_battery_cashflows, build_diesel_cashflows,
                        DEFAULT_PROJECT_YEARS)
from .diesel import (DieselArchitecture1, DieselArchitecture2,
                     annual_co2e_kg, DEFRA_2025_GAS_OIL_KGCO2E_PER_LITRE)
from .monte_carlo import (SITE_DESIGN, SITE_DESIGN_TMY, SITE_DESIGN_DEG08,
                          run_monte_carlo, summarise)
from .sensitivity import (CENTRAL, one_at_a_time_sensitivity, tornado_data,
                          combined_worst_case, LOAD_DESIGNS, DEGRADATION_DESIGNS)
from .sweeps_2d import breakeven_battery_cost, ratio_matrix, RATES, BATT_MULTS

RESULTS = Path(__file__).resolve().parents[1] / 'results'


def two_pass(df, site, pv, bat, load, **kw):
    r1 = run_simulation(df, site, pv, bat, load, **kw)
    return run_simulation(df, site, pv, bat, load,
                          initial_soc_frac=r1.final_soc_kwh / bat.capacity_kwh,
                          **kw)


def outage_stats(hourly):
    """Longest consecutive unmet run (h), days with any unmet, and the
    10-day window holding the most unmet hours (start, end, hours)."""
    u = (hourly['unmet_wh'] > 0).astype(int)
    # longest run
    best = cur = 0
    for v in u.values:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    days = int((u.resample('D').sum() > 0).sum())
    roll = u.rolling(240).sum()
    if roll.max() > 0:
        end = roll.idxmax(); start = end - pd.Timedelta(hours=239)
        win = (str(start.date()), str(end.date()), int(roll.max()))
    else:
        win = (None, None, 0)
    return best, days, win


def main():
    N = {}
    load = LoadProfile()
    lc = LossChain()
    eol, soh, gov_year = worst_life_state()
    N['governing_year'] = gov_year
    N['battery_eol_soh'] = soh
    N['load'] = {
        'base_w': load.base_power_w(), 'design_w': load.design_power_w(),
        'daily_wh': load.daily_energy_wh(), 'annual_kwh': load.annual_energy_kwh(),
    }
    N['loss_chain'] = {
        'soiling': lc.soiling, 'shading': lc.shading, 'snow': lc.snow,
        'mismatch': lc.mismatch, 'wiring': lc.wiring,
        'connections': lc.connections, 'lid': lc.lid,
        'nameplate': lc.nameplate, 'availability': lc.availability,
        'controller_eff': lc.controller_efficiency,
        'derate': lc.derate, 'total_loss_pct': lc.total_loss_pct,
    }
    N['eol_factor'] = eol

    # ---- sites, resource, sizing, reliability, yield -------------------
    def _design_metrics(df, site, wp, kwh, with_windows=True):
        pv = PVDesign(nameplate_w=wp)
        bat = BatteryDesign(capacity_kwh=kwh)
        y1 = two_pass(df, site, pv, bat, load)
        ye = two_pass(df, site, pv, bat, load, pv_ageing_factor=eol,
                      battery_soh=soh)
        m = {
            'pv_wp': wp, 'battery_kwh': kwh,
            'pv_annual_kwh_y1': y1.annual_pv_kwh,
            'lolp_y1_pct': y1.lolp * 100, 'eens_y1_kwh': y1.annual_unmet_kwh,
            'lolp_eol_pct': ye.lolp * 100, 'eens_eol_kwh': ye.annual_unmet_kwh,
            'curtailed_y1_kwh': y1.annual_curtailed_kwh,
            'curtailed_eol_kwh': ye.annual_curtailed_kwh,
            'unmet_hours_eol': int(round(ye.lolp * 8760)),
            'pv_capex_gbp': wp * 4.5, 'batt_capex_gbp': kwh * 700.0,
            'capex_gbp': wp * 4.5 + kwh * 700.0 + 150.0 + 600.0,
        }
        if with_windows:
            longest, days_hit, gov_win = outage_stats(ye.hourly)
            m.update({'longest_unmet_run_h': longest, 'days_with_unmet': days_hit,
                      'gov_window_start': gov_win[0], 'gov_window_end': gov_win[1],
                      'gov_window_unmet_h': gov_win[2]})
        return m

    sites = {}
    for key, site in SITES.items():
        df = get_or_create_tmy(site, prefer='real')
        monthly_ghi = df['ghi'].resample('MS').sum() / 1000.0
        pv1kw = simulate_pv_dc(df, site, PVDesign(nameplate_w=1000.0))
        wp, kwh = SITE_DESIGN[site.name]            # FINAL design
        rec = {
            'lat': site.latitude, 'lon': site.longitude,
            'ghi_annual': annual_ghi_kwh_per_m2(df),
            'ghi_dec': float(monthly_ghi.iloc[11]),
            'ghi_jun': float(monthly_ghi.iloc[5]),
            'ghi_min_month': float(monthly_ghi.min()),
            'ghi_max_month': float(monthly_ghi.max()),
            'specific_yield_kwh_per_kwp': float(pv1kw.sum() / 1000.0),
            'subzero_hours': int((df['temp_air'] <= 0).sum()),
        }
        # final design evaluated on the TMY (context only; the reliability
        # claim for the final design is the sixteen-year record)
        rec.update(_design_metrics(df, site, wp, kwh, with_windows=False))
        # step-1 TMY design and its TMY reliability metrics (Table VIII)
        wpt, kwht = SITE_DESIGN_TMY[site.name]
        rec['tmy_design'] = _design_metrics(df, site, wpt, kwht, with_windows=True)
        sites[site.name] = rec
    N['sites'] = sites
    g = [sites[s]['ghi_annual'] for s in sites]
    N['ghi_gradient_pct'] = (max(g) - min(g)) / min(g) * 100

    # ---- diesel ---------------------------------------------------------
    d = {}
    for name, arch in [('A1', DieselArchitecture1()), ('A2', DieselArchitecture2())]:
        d[name] = {
            'capex': arch.total_capex_gbp,
            'runtime_h_day': getattr(arch, 'daily_runtime_hours', 24.0),
            'runtime_h_yr': arch.annual_runtime_hours,
            'fuel_l_per_h': arch.fuel_litres_per_hour_at_op_point,
            'fuel_l_yr': arch.annual_fuel_litres,
            'fuel_gbp_yr': arch.annual_fuel_cost_gbp(76.02),
            'oil_changes_yr': arch.annual_oil_changes,
            'oil_gbp_yr': arch.annual_oil_cost_gbp,
            'visits_yr': arch.annual_site_visits,
            'visit_gbp_yr': arch.annual_visit_cost_gbp,
            'genset_life_yr': arch.genset_lifetime_hours / arch.annual_runtime_hours,
            'co2e_kg_yr': annual_co2e_kg(arch),
            'co2e_t_25yr': annual_co2e_kg(arch) * 25 / 1000.0,
        }
    d['A2']['pba_life_yr'] = DieselArchitecture2().battery_lifetime_years
    d['A2']['charge_power_w'] = DieselArchitecture2().charge_power_kw * 1000
    d['A2']['pba_eff'] = DieselArchitecture2().battery_path_efficiency
    d['ef_kgco2e_per_l'] = DEFRA_2025_GAS_OIL_KGCO2E_PER_LITRE
    N['diesel'] = d

    # ---- LCOE / NPV -----------------------------------------------------
    econ = {}
    for rate in (0.05, 0.08):
        k = f'{rate:.2f}'
        a1 = build_diesel_cashflows(DieselArchitecture1(), fuel_price_ppl=76.02)
        a2 = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02)
        e = {'diesel_a1_lcoe': a1.lcoe_gbp_per_kwh(rate),
             'diesel_a2_lcoe': a2.lcoe_gbp_per_kwh(rate),
             'diesel_a1_npv': a1.total_npv(rate),
             'diesel_a2_npv': a2.total_npv(rate),
             'diesel_a2_flows_npv': {f.label: f.npv(rate) for f in a2.flows},
             'sites': {}}
        for site, (wp, kwh) in SITE_DESIGN.items():
            cf = build_pv_battery_cashflows(
                pv_capex_gbp_per_wp=4.50, pv_size_wp=wp,
                battery_capex_gbp_per_kwh=700.0, battery_kwh=kwh,
                annual_energy_delivered_kwh=128.2)
            s = cf.lcoe_gbp_per_kwh(rate)
            e['sites'][site] = {
                'solar_lcoe': s, 'solar_npv': cf.total_npv(rate),
                'solar_capex_y0': float(cf.flows[0].annual_amounts_gbp[0]),
                'solar_flows_npv': {f.label: f.npv(rate) for f in cf.flows},
                'ratio_a2': e['diesel_a2_lcoe'] / s,
                'ratio_a1': e['diesel_a1_lcoe'] / s,
                'avoided_npv_vs_a2': e['diesel_a2_npv'] - cf.total_npv(rate),
            }
        econ[k] = e
    N['econ'] = econ

    # ---- tornado / OAT / visit cadence / worst case ------------------------
    oat = one_at_a_time_sensitivity(architecture=2)
    N['oat'] = oat.to_dict(orient='records')
    td = tornado_data(architecture=2)
    N['tornado'] = td.to_dict(orient='records')
    N['central_ratio'] = float(oat[oat['parameter'] == 'CENTRAL']['diesel_pv_ratio'].iloc[0])
    cad = oat[oat['parameter'] == 'Diesel visit cadence']
    N['visit_cadence'] = {int(v.split('/')[0]): r for v, r in
                          zip(cad['value'], cad['diesel_pv_ratio'])}
    N['worst_case'] = combined_worst_case(SITE_DESIGN_DEG08).to_dict(orient='records')
    N['worst_case_cadence'] = {
        str(v): combined_worst_case(SITE_DESIGN_DEG08, diesel_visits=v).to_dict(orient='records')
        for v in (6, 4, 2)}

    # ---- diesel visit cadence per site x rate + analytic break-even cadence
    cad = {}
    for rate in (0.05, 0.08):
        k = f'{rate:.2f}'
        cad[k] = {}
        # diesel LCOE is affine in visits: L(v) = a + b v
        a2_0 = DieselArchitecture2(); a2_0.annual_site_visits = 0
        a2_1 = DieselArchitecture2(); a2_1.annual_site_visits = 1
        L0 = build_diesel_cashflows(a2_0, fuel_price_ppl=76.02).lcoe_gbp_per_kwh(rate)
        L1 = build_diesel_cashflows(a2_1, fuel_price_ppl=76.02).lcoe_gbp_per_kwh(rate)
        b = L1 - L0
        for site, (wp, kwh) in SITE_DESIGN.items():
            s_l = build_pv_battery_cashflows(
                pv_capex_gbp_per_wp=4.50, pv_size_wp=wp,
                battery_capex_gbp_per_kwh=700.0, battery_kwh=kwh,
                annual_energy_delivered_kwh=128.2).lcoe_gbp_per_kwh(rate)
            row = {}
            for v in (12, 6, 4, 2):
                row[str(v)] = (L0 + b * v) / s_l
            row['breakeven_visits'] = (s_l - L0) / b
            cad[k][site] = row
    N['cadence'] = cad

    # ---- year-1-sized TMY design evaluated in the governing year -----------
    y1_breach = {}
    from .sizing import _cost_index
    from .pv_model import simulate_pv_dc as _sim
    for key, site in SITES.items():
        df = get_or_create_tmy(site, prefer='real')
        ref = _sim(df, site, PVDesign(nameplate_w=1000.0))
        best = None
        for pv_w in np.arange(100, 1300, 50):
            pvs = ref * (pv_w / 1000.0)
            for bk in [0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 5, 7, 10]:
                bat_ = BatteryDesign(capacity_kwh=bk); pv_ = PVDesign(nameplate_w=float(pv_w))
                r1 = run_simulation(df, site, pv_, bat_, load, pv_series_w=pvs)
                r = run_simulation(df, site, pv_, bat_, load,
                                   initial_soc_frac=r1.final_soc_kwh / bk, pv_series_w=pvs)
                if r.lolp <= 0.01:
                    ci = _cost_index(pv_w, bk)
                    if best is None or ci < best[2]:
                        best = (float(pv_w), float(bk), ci)
        pv_w, bk, _ = best
        pvs = ref * (pv_w / 1000.0)
        bat_ = BatteryDesign(capacity_kwh=bk); pv_ = PVDesign(nameplate_w=pv_w)
        r1 = run_simulation(df, site, pv_, bat_, load, pv_series_w=pvs,
                            pv_ageing_factor=eol, battery_soh=soh)
        r = run_simulation(df, site, pv_, bat_, load, initial_soc_frac=r1.final_soc_kwh / bk,
                           pv_series_w=pvs, pv_ageing_factor=eol, battery_soh=soh)
        y1_breach[site.name] = {'pv_wp': pv_w, 'battery_kwh': bk,
                                'lolp_gov_pct': r.lolp * 100}
    N['year1_sized'] = y1_breach

    # ---- validation (Upgrade 1) and sixteen-year reliability (Upgrade 2) --
    N['validation'] = json.loads((RESULTS / 'validation.json').read_text())
    N['multiyear'] = json.loads((RESULTS / 'multiyear.json').read_text())
    # price of reliability: 0.1 % target under the final criterion
    N['price_of_reliability'] = {
        s_: N['multiyear']['sites'][s_]['final_design_lolp_0p1']
        for s_ in N['multiyear']['sites']}

    # ---- tilt check (all sites): latitude vs latitude+15, yield and step-1 design
    tilt = {}
    for key, site in SITES.items():
        df = get_or_create_tmy(site, prefer='real')
        base = simulate_pv_dc(df, site, PVDesign(nameplate_w=1000.0))
        steep = simulate_pv_dc(df, site, PVDesign(nameplate_w=1000.0,
                                                  tilt_deg=site.latitude + 15))
        dec_b = base[base.index.month == 12].sum(); dec_s = steep[steep.index.month == 12].sum()
        best_s, _ = size_system(df, site, load, target_lolp=0.01,
                                tilt_deg=site.latitude + 15)
        wpt, kwht = SITE_DESIGN_TMY[site.name]
        tilt[site.name] = {'annual_delta_pct': (steep.sum() / base.sum() - 1) * 100,
                           'december_delta_pct': (dec_s / dec_b - 1) * 100,
                           'design_lat': [wpt, kwht],
                           'design_lat_plus15': [best_s.pv_w, best_s.battery_kwh],
                           'lolp_gov_pct_lat_plus15': best_s.lolp * 100,
                           'design_changed': (best_s.pv_w, best_s.battery_kwh) != (wpt, kwht)}
    N['tilt_plus15'] = tilt

    # ---- solar-side visits: OAT rows, break-even solar visits, common visits
    sv = oat[oat['parameter'] == 'Solar visit count']
    N['solar_visits'] = {int(v.split('/')[0]): r for v, r in
                         zip(sv['value'], sv['diesel_pv_ratio'])}
    common = {}
    sbe = {}
    for rate in (0.05, 0.08):
        k = f'{rate:.2f}'
        common[k] = {}; sbe[k] = {}
        a2_cf = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02)
        di = a2_cf.lcoe_gbp_per_kwh(rate)
        for site, (wp, kwh) in SITE_DESIGN.items():
            def _solar(v):
                return build_pv_battery_cashflows(
                    pv_capex_gbp_per_wp=4.50, pv_size_wp=wp,
                    battery_capex_gbp_per_kwh=700.0, battery_kwh=kwh,
                    annual_energy_delivered_kwh=128.2,
                    annual_site_visits=v).lcoe_gbp_per_kwh(rate)
            s0, s1 = _solar(0), _solar(1)
            b_solar = s1 - s0                        # £/kWh per solar visit
            sbe[k][site] = {'breakeven_solar_visits': (di - s0) / b_solar,
                            'lcoe_per_solar_visit': b_solar}
            # N common visits added to both systems (station visits that
            # occur under either power system)
            row = {}
            for ncom in (0, 2, 4, 12):
                a2n = DieselArchitecture2(); a2n.annual_site_visits = 12 + ncom
                dn = build_diesel_cashflows(a2n, fuel_price_ppl=76.02).lcoe_gbp_per_kwh(rate)
                row[str(ncom)] = dn / _solar(2 + ncom)
            common[k][site] = row
    N['solar_breakeven_visits'] = sbe
    N['common_visits'] = common

    # ---- diesel genset calendar-life sensitivity (A2, 10-yr calendar cap)
    a2c = DieselArchitecture2(); a2c.genset_calendar_life_years = 10.0
    base_npv = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02).total_npv(0.05)
    cal_npv = build_diesel_cashflows(a2c, fuel_price_ppl=76.02).total_npv(0.05)
    N['diesel_calendar_life'] = {'years': 10, 'npv_delta_gbp': cal_npv - base_npv,
                                 'npv_delta_pct': (cal_npv / base_npv - 1) * 100}
    # A2 with the 5,000 h light-load life instead of 8,000 h
    a2l = DieselArchitecture2(); a2l.genset_lifetime_hours = 5000
    l_npv = build_diesel_cashflows(a2l, fuel_price_ppl=76.02).total_npv(0.05)
    N['diesel_life_5000h'] = {'npv_delta_gbp': l_npv - base_npv,
                              'npv_delta_pct': (l_npv / base_npv - 1) * 100}

    # ---- Monte Carlo ------------------------------------------------------
    mc = {}
    for rate, label in [(0.05, '5%'), (0.08, '8%')]:
        m = run_monte_carlo(n=5000, discount_rate=rate)
        s = summarise(m, label)
        mc[label] = s.to_dict(orient='records')
        # rule of three: no parity in n draws → P < 3/n at 95%
        mc[label + '_min_ratio'] = float(min(
            (m['diesel_lcoe'] / m[f'solar_lcoe_{site}']).min() for site in SITE_DESIGN))
    mc['n_draws'] = 5000
    mc['rule_of_three_pct'] = 3 / 5000 * 100
    N['mc'] = mc

    # ---- 2-D sweeps -------------------------------------------------------
    be = breakeven_battery_cost()
    N['breakeven'] = be.to_dict(orient='records')
    grid_min = {}
    for site in SITE_DESIGN:
        m = ratio_matrix(site).values
        i, j = np.unravel_index(m.argmin(), m.shape)
        grid_min[site] = {'ratio': float(m.min()), 'rate': float(RATES[i]),
                          'batt_mult': float(BATT_MULTS[j])}
    N['grid_min'] = grid_min

    # ---- sizing-coupled sensitivity designs (single source: sensitivity.py)
    N['degradation_designs'] = {str(k): list(v) for k, v in DEGRADATION_DESIGNS.items()}
    N['load_designs'] = {str(k): list(v) for k, v in LOAD_DESIGNS.items()}
    N['site_design_tmy'] = {k: list(v) for k, v in SITE_DESIGN_TMY.items()}
    N['site_design'] = {k: list(v) for k, v in SITE_DESIGN.items()}
    N['site_design_deg08'] = {k: list(v) for k, v in SITE_DESIGN_DEG08.items()}

    # ---- carbon (DESNZ central values, 2020 prices) -----------------------
    N['carbon'] = {'central_2026_gbp_per_t': 264, 'central_2030_gbp_per_t': 280,
                   'price_base': '2020 prices'}

    # ---- costing-horizon sensitivity (station refresh cycles are shorter
    # than 25 years): diesel A2 to solar LCOE ratio at 10, 15 and 25 years ---
    horizon = {}
    for years in (10, 15, 25):
        a2h = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02,
                                     project_years=years)
        horizon[str(years)] = {}
        for rate in (0.05, 0.08):
            row = {'diesel_a2_lcoe': a2h.lcoe_gbp_per_kwh(rate), 'sites': {}}
            for site, (wp, kwh) in SITE_DESIGN.items():
                cf = build_pv_battery_cashflows(
                    pv_capex_gbp_per_wp=4.50, pv_size_wp=wp,
                    battery_capex_gbp_per_kwh=700.0, battery_kwh=kwh,
                    annual_energy_delivered_kwh=128.2, project_years=years)
                sl = cf.lcoe_gbp_per_kwh(rate)
                row['sites'][site] = {'solar_lcoe': sl, 'ratio_a2': row['diesel_a2_lcoe'] / sl}
            horizon[str(years)][f'{rate:.2f}'] = row
    N['horizon'] = horizon

    # ---- fuel price doubled (white diesel, 2 x 76.02 p/L) at monthly
    # attendance: effect on the A2 LCOE and on the Southampton ratio -------
    a2_base = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02)
    a2_dbl = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=2 * 76.02)
    so = SITE_DESIGN['Southampton']
    cf_so = build_pv_battery_cashflows(pv_capex_gbp_per_wp=4.50, pv_size_wp=so[0],
                                       battery_capex_gbp_per_kwh=700.0, battery_kwh=so[1],
                                       annual_energy_delivered_kwh=128.2)
    N['fuel_doubled'] = {
        'fuel_price_ppl': 2 * 76.02,
        'a2_lcoe_5pct': a2_dbl.lcoe_gbp_per_kwh(0.05),
        'a2_lcoe_delta_pct': (a2_dbl.lcoe_gbp_per_kwh(0.05) / a2_base.lcoe_gbp_per_kwh(0.05) - 1) * 100,
        'southampton_ratio_5pct': a2_dbl.lcoe_gbp_per_kwh(0.05) / cf_so.lcoe_gbp_per_kwh(0.05),
        'southampton_ratio_delta': (a2_dbl.lcoe_gbp_per_kwh(0.05) - a2_base.lcoe_gbp_per_kwh(0.05)) / cf_so.lcoe_gbp_per_kwh(0.05)}

    out = RESULTS / 'numbers.json'
    out.write_text(json.dumps(N, indent=1, default=float))
    print(f'wrote {out}')
    # brief echo of the headline
    e5 = econ['0.05']
    print('5%: A2', round(e5['diesel_a2_lcoe'], 3),
          {s: round(v['solar_lcoe'], 3) for s, v in e5['sites'].items()})
    print('visit cadence ratios:', {k: round(v, 2) for k, v in N['visit_cadence'].items()})
    print('worst case:', [(r['site'], round(r['ratio_worst'], 2)) for r in N['worst_case']])
    print('MC 5% P50 ratio:', [(r['site'], round(r['ratio_p50'], 2)) for r in mc['5%']])


if __name__ == '__main__':
    main()
