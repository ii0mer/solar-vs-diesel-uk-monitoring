"""Upgrade 1 — validation of the PV chain against the PVGIS reference model.

Two layers, both DC-to-DC with PVGIS system losses set to zero and this
study's loss chain (Table IV) switched off, so that only the physics is
compared:

Layer 1 — conversion chain on identical inputs (2005-2020, hourly).
    PVGIS's plane-of-array components (Gb(i), Gd(i), Gr(i)), air
    temperature and wind speed are fed into this study's incidence-angle,
    cell-temperature and PVWatts power steps. The result is compared hour
    by hour with PVGIS's own P (Huld power model, Faiman temperature,
    Martin-Ruiz reflectance). This isolates the conversion model from the
    transposition step and from the choice of weather year.

Layer 2 — full chain climatology.
    The TMY-driven chain (Hay-Davies transposition from GHI/DNI/DHI, as
    used for sizing) is compared with the sixteen-year PVGIS mean by
    month and by year. This tests transposition and TMY representativeness
    together.

Metrics follow the convention of Deville et al. (2024): normalised mean
bias error (NMBE) and normalised root-mean-square error (NRMSE), each
normalised by the mean of the reference series over the compared samples.

    python -m src.validation      → results/validation.json,
                                    results/validation_monthly.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from .sites import SITES, Site
from .weather import get_or_create_tmy
from .pv_model import PVDesign, LossChain, simulate_pv_dc
from .pvgis_series import (load_all, load_edinburgh_horizon, PvgisSeries,
                           REQUESTED_KWP)

RESULTS = Path(__file__).resolve().parents[1] / 'results'
NO_LOSS = LossChain(soiling=0, shading=0, snow=0, mismatch=0, wiring=0,
                    connections=0, lid=0, nameplate=0, availability=0,
                    controller_efficiency=1.0)


# ---------------------------------------------------------------------------
# This study's conversion chain, starting from plane-of-array components
# ---------------------------------------------------------------------------

def chain_from_poa(series: PvgisSeries, site: Site,
                   tilt_deg: float | None = None,
                   pdc0_w: float = 1000.0,
                   losses: LossChain | None = None,
                   gamma_pdc: float = -0.0035) -> pd.Series:
    """Hourly DC output (W) of this study's chain driven by PVGIS
    plane-of-array components, temperature and wind — the same
    incidence-angle, cell-temperature and PVWatts steps as
    :func:`pv_model.simulate_pv_dc`, without the transposition step.
    ``losses=None`` means no loss chain (physics only)."""
    d = series.data
    tilt = site.latitude if tilt_deg is None else tilt_deg
    loc = pvlib.location.Location(site.latitude, site.longitude,
                                  altitude=site.altitude, tz='UTC')
    sp = loc.get_solarposition(d.index)
    aoi = pvlib.irradiance.aoi(tilt, 180.0, sp['apparent_zenith'],
                               sp['azimuth'])
    iam_beam = pvlib.iam.physical(aoi).fillna(0.0)
    iam_diff = pvlib.iam.marion_diffuse('physical', surface_tilt=tilt)
    poa_eff = (d['poa_direct'] * iam_beam
               + d['poa_sky_diffuse'] * iam_diff['sky']
               + d['poa_ground_diffuse'] * iam_diff['ground'])
    cell_t = pvlib.temperature.sapm_cell(
        poa_global=d['poa_global'], temp_air=d['temp_air'],
        wind_speed=d['wind_speed'], a=-3.47, b=-0.0594, deltaT=3)
    p = pvlib.pvsystem.pvwatts_dc(effective_irradiance=poa_eff,
                                  temp_cell=cell_t, pdc0=pdc0_w,
                                  gamma_pdc=gamma_pdc)
    p = p.fillna(0.0).clip(lower=0.0)
    if losses is not None:
        p = p * losses.derate
    p.name = 'p_study_w'
    return p


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nmbe_nrmse(model: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(NMBE %, NRMSE %) normalised by mean(ref)."""
    model = np.asarray(model, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mref = ref.mean()
    nmbe = (model - ref).mean() / mref * 100.0
    nrmse = np.sqrt(((model - ref) ** 2).mean()) / mref * 100.0
    return float(nmbe), float(nrmse)


def _monthly_kwh(series_w: pd.Series) -> pd.Series:
    return series_w.resample('MS').sum() / 1000.0


def _annual_kwh(series_w: pd.Series) -> pd.Series:
    return series_w.resample('YS').sum() / 1000.0


def _monthly_climatology(series_w: pd.Series) -> pd.DataFrame:
    """Mean/min/max monthly energy (kWh per unit) across the years."""
    m = _monthly_kwh(series_w)
    g = m.groupby(m.index.month)
    return pd.DataFrame({'mean': g.mean(), 'min': g.min(), 'max': g.max()})


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_site(key: str, series: PvgisSeries, site: Site) -> dict:
    d = series.data
    kwp = series.kwp
    ref_w_per_kwp = d['p_pvgis_w'] / kwp            # W per kWp, PVGIS model
    study_w_per_kwp = chain_from_poa(series, site)  # W per kWp, this chain

    # --- Layer 1: same inputs, hourly / monthly / annual --------------------
    day = (ref_w_per_kwp > 0) | (study_w_per_kwp > 0)
    h_nmbe, h_nrmse = nmbe_nrmse(study_w_per_kwp[day], ref_w_per_kwp[day])
    m_ref = _monthly_kwh(ref_w_per_kwp)
    m_study = _monthly_kwh(study_w_per_kwp)
    m_nmbe, m_nrmse = nmbe_nrmse(m_study.values, m_ref.values)
    a_ref = _annual_kwh(ref_w_per_kwp)
    a_study = _annual_kwh(study_w_per_kwp)
    a_nmbe, a_nrmse = nmbe_nrmse(a_study.values, a_ref.values)

    # energy-weighted ratio by irradiance class (explains the residual)
    bins = [0, 100, 200, 400, 600, 800, 2000]
    labels = ['0-100', '100-200', '200-400', '400-600', '600-800', '>800']
    cls = pd.cut(d['poa_global'], bins=bins, labels=labels, right=False)
    by_class = {}
    for lab in labels:
        m = (cls == lab) & day
        if m.sum() == 0:
            continue
        e_ref = ref_w_per_kwp[m].sum()
        e_study = study_w_per_kwp[m].sum()
        by_class[lab] = {
            'hours': int(m.sum()),
            'share_of_pvgis_energy_pct': float(e_ref / ref_w_per_kwp.sum() * 100),
            'study_over_pvgis': float(e_study / e_ref) if e_ref > 0 else None,
        }
    # winter subset (Nov-Feb) — the sizing-relevant part of the year
    win = d.index.month.isin([11, 12, 1, 2])
    w_ref = _monthly_kwh(ref_w_per_kwp[win])
    w_study = _monthly_kwh(study_w_per_kwp[win])
    w_ref = w_ref[w_ref > 0]; w_study = w_study.reindex(w_ref.index)
    w_nmbe, w_nrmse = nmbe_nrmse(w_study.values, w_ref.values)

    # --- Layer 2: full TMY chain vs sixteen-year PVGIS climatology ---------
    tmy = get_or_create_tmy(site, prefer='real')
    p_tmy = simulate_pv_dc(tmy, site, PVDesign(nameplate_w=1000.0),
                           losses=NO_LOSS)              # W per kWp, no losses
    tmy_month = p_tmy.groupby(p_tmy.index.month).sum() / 1000.0   # kWh/kWp
    clim = _monthly_climatology(ref_w_per_kwp)
    c_nmbe, c_nrmse = nmbe_nrmse(tmy_month.values, clim['mean'].values)
    tmy_annual = float(p_tmy.sum() / 1000.0)
    pvgis_annual_mean = float(a_ref.mean())
    tmy_within_band = bool(a_ref.min() <= tmy_annual <= a_ref.max())
    # winter months only
    wm = [11, 12, 1, 2]
    cw_nmbe, cw_nrmse = nmbe_nrmse(tmy_month.loc[wm].values,
                                   clim['mean'].loc[wm].values)
    # POA climatology (transposition + TMY representativeness only)
    loc = pvlib.location.Location(site.latitude, site.longitude,
                                  altitude=site.altitude, tz='UTC')
    sp = loc.get_solarposition(tmy.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=site.latitude, surface_azimuth=180.0,
        solar_zenith=sp['apparent_zenith'], solar_azimuth=sp['azimuth'],
        dni=tmy['dni'], ghi=tmy['ghi'], dhi=tmy['dhi'],
        dni_extra=pvlib.irradiance.get_extra_radiation(tmy.index),
        model='haydavies')['poa_global'].fillna(0.0)
    poa_tmy_month = poa.groupby(poa.index.month).sum() / 1000.0
    poa_clim = _monthly_climatology(d['poa_global'])
    poa_nmbe, poa_nrmse = nmbe_nrmse(poa_tmy_month.values,
                                     poa_clim['mean'].values)

    # study chain on PVGIS POA, sixteen-year monthly climatology (for figure)
    study_clim = _monthly_climatology(study_w_per_kwp)

    return {
        'kwp_pvgis': kwp,
        'slope_deg_pvgis': series.slope_deg,
        'n_hours': int(len(d)),
        'years': [int(y) for y in series.years],
        'layer1_same_inputs': {
            'hourly_nmbe_pct': h_nmbe, 'hourly_nrmse_pct': h_nrmse,
            'hourly_n_daylight': int(day.sum()),
            'monthly_nmbe_pct': m_nmbe, 'monthly_nrmse_pct': m_nrmse,
            'monthly_n': int(len(m_ref)),
            'annual_nmbe_pct': a_nmbe, 'annual_nrmse_pct': a_nrmse,
            'winter_monthly_nmbe_pct': w_nmbe, 'winter_monthly_nrmse_pct': w_nrmse,
            'by_irradiance_class': by_class,
            'study_annual_mean_kwh_per_kwp': float(a_study.mean()),
        },
        'pvgis_reference': {
            'annual_mean_kwh_per_kwp': pvgis_annual_mean,
            'annual_min_kwh_per_kwp': float(a_ref.min()),
            'annual_min_year': int(a_ref.idxmin().year),
            'annual_max_kwh_per_kwp': float(a_ref.max()),
            'annual_max_year': int(a_ref.idxmax().year),
            'annual_by_year': {int(t.year): float(v) for t, v in a_ref.items()},
            'poa_annual_mean_kwh_m2': float(d['poa_global'].resample('YS').sum().mean() / 1000.0),
            'monthly_mean_kwh_per_kwp': {int(m): float(v) for m, v in clim['mean'].items()},
            'monthly_min_kwh_per_kwp': {int(m): float(v) for m, v in clim['min'].items()},
            'monthly_max_kwh_per_kwp': {int(m): float(v) for m, v in clim['max'].items()},
            'study_on_pvgis_poa_monthly_mean_kwh_per_kwp': {int(m): float(v) for m, v in study_clim['mean'].items()},
        },
        'layer2_tmy_chain': {
            'tmy_annual_kwh_per_kwp_noloss': tmy_annual,
            'annual_bias_vs_pvgis_mean_pct': (tmy_annual / pvgis_annual_mean - 1) * 100.0,
            'tmy_within_16yr_band': tmy_within_band,
            'monthly_nmbe_pct': c_nmbe, 'monthly_nrmse_pct': c_nrmse,
            'winter_monthly_nmbe_pct': cw_nmbe, 'winter_monthly_nrmse_pct': cw_nrmse,
            'tmy_monthly_kwh_per_kwp': {int(m): float(v) for m, v in tmy_month.items()},
            'poa_monthly_nmbe_pct': poa_nmbe, 'poa_monthly_nrmse_pct': poa_nrmse,
            'poa_tmy_annual_kwh_m2': float(poa.sum() / 1000.0),
        },
    }


def horizon_effect(edinburgh_no_horizon: PvgisSeries,
                   edinburgh_horizon: PvgisSeries) -> dict:
    a = edinburgh_no_horizon.data['p_pvgis_w']
    b = edinburgh_horizon.data['p_pvgis_w']
    win = a.index.month.isin([11, 12, 1, 2])
    dec = a.index.month == 12
    monthly = {int(m): float(b[b.index.month == m].sum() / a[a.index.month == m].sum() * 100 - 100)
               for m in range(1, 13)}
    return {
        'annual_loss_pct': float((1 - b.sum() / a.sum()) * 100),
        'nov_feb_loss_pct': float((1 - b[win].sum() / a[win].sum()) * 100),
        'december_loss_pct': float((1 - b[dec].sum() / a[dec].sum()) * 100),
        'monthly_delta_pct': monthly,
        'annual_yield_no_horizon_kwh_per_kwp': float(a.sum() / 1000 / edinburgh_no_horizon.kwp / 16),
        'annual_yield_horizon_kwh_per_kwp': float(b.sum() / 1000 / edinburgh_horizon.kwp / 16),
    }


def main(verbose: bool = True) -> dict:
    series = load_all()
    out = {'sites': {}, 'method': {
        'layer1': 'study chain (physical IAM, Marion diffuse IAM, SAPM open-rack '
                  'cell temperature, PVWatts DC gamma=-0.35%/K, no loss chain) '
                  'driven by PVGIS Gb(i)/Gd(i)/Gr(i)/T2m/WS10m 2005-2020 vs '
                  'PVGIS P with system loss 0',
        'layer2': 'study TMY chain (Hay-Davies from GHI/DNI/DHI, no loss chain) '
                  'vs PVGIS 2005-2020 monthly and annual means',
        'normalisation': 'NMBE and NRMSE normalised by the mean of the PVGIS '
                         'series over the compared samples (daylight hours '
                         'for hourly, all months for monthly)',
    }}
    monthly_rows = []
    for key, s in series.items():
        site = SITES[key]
        r = validate_site(key, s, site)
        out['sites'][site.name] = r
        for m in range(1, 13):
            monthly_rows.append({
                'site': site.name, 'month': m,
                'pvgis_mean_kwh_per_kwp': r['pvgis_reference']['monthly_mean_kwh_per_kwp'][m],
                'pvgis_min_kwh_per_kwp': r['pvgis_reference']['monthly_min_kwh_per_kwp'][m],
                'pvgis_max_kwh_per_kwp': r['pvgis_reference']['monthly_max_kwh_per_kwp'][m],
                'study_on_pvgis_poa_kwh_per_kwp': r['pvgis_reference']['study_on_pvgis_poa_monthly_mean_kwh_per_kwp'][m],
                'study_tmy_kwh_per_kwp': r['layer2_tmy_chain']['tmy_monthly_kwh_per_kwp'][m],
            })
        if verbose:
            l1 = r['layer1_same_inputs']; l2 = r['layer2_tmy_chain']; pr = r['pvgis_reference']
            print(f"{site.name:12s} L1 hourly NMBE {l1['hourly_nmbe_pct']:+.2f}% NRMSE {l1['hourly_nrmse_pct']:.1f}% | "
                  f"monthly NMBE {l1['monthly_nmbe_pct']:+.2f}% NRMSE {l1['monthly_nrmse_pct']:.1f}% | "
                  f"annual NMBE {l1['annual_nmbe_pct']:+.2f}%  winter NMBE {l1['winter_monthly_nmbe_pct']:+.2f}%")
            print(f"{'':12s} L2 TMY {l2['tmy_annual_kwh_per_kwp_noloss']:.0f} vs PVGIS mean {pr['annual_mean_kwh_per_kwp']:.0f} "
                  f"({pr['annual_min_kwh_per_kwp']:.0f}-{pr['annual_max_kwh_per_kwp']:.0f}) bias {l2['annual_bias_vs_pvgis_mean_pct']:+.2f}% "
                  f"| monthly NMBE {l2['monthly_nmbe_pct']:+.2f}% NRMSE {l2['monthly_nrmse_pct']:.1f}% "
                  f"| POA monthly NMBE {l2['poa_monthly_nmbe_pct']:+.2f}% NRMSE {l2['poa_monthly_nrmse_pct']:.1f}%")
            print(f"{'':12s}    by class: " + ' '.join(f"{k}:{v['study_over_pvgis']:.3f}({v['share_of_pvgis_energy_pct']:.0f}%)"
                                                     for k, v in l1['by_irradiance_class'].items()))
    out['edinburgh_horizon'] = horizon_effect(series['edinburgh'], load_edinburgh_horizon())
    if verbose:
        h = out['edinburgh_horizon']
        print(f"Edinburgh horizon: annual -{h['annual_loss_pct']:.2f}%, Nov-Feb -{h['nov_feb_loss_pct']:.2f}%, "
              f"Dec -{h['december_loss_pct']:.2f}%")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'validation.json').write_text(json.dumps(out, indent=1))
    pd.DataFrame(monthly_rows).to_csv(RESULTS / 'validation_monthly.csv', index=False)
    return out


if __name__ == '__main__':
    main()
