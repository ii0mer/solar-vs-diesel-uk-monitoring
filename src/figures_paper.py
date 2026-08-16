"""Paper figure suite — one visual system for the IEEE-format dissertation.

Design rules (applied everywhere):
  * Liberation Sans (Arial-metric; template-approved family), 8 pt body,
    7 pt ticks/annotations, no in-figure titles (captions carry titles).
  * Okabe–Ito colour-blind-safe palette. Sites run warm→cool south→north.
  * Top/right spines removed; light y-grid only where it aids reading.
  * Column-width figures 3.45 in wide; page-width 7.1 in; 600 dpi PNG.
  * Every number plotted comes from the model (SITE_DESIGN, results/).

    python -m src.figures_paper
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

from .sites import SITES
from .weather import get_or_create_tmy, annual_ghi_kwh_per_m2
from .load_profile import LoadProfile
from .pv_model import PVDesign
from .battery import BatteryDesign
from .simulation import run_simulation
from .sizing import eol_ageing_factor
from .economics import build_pv_battery_cashflows, build_diesel_cashflows
from .diesel import DieselArchitecture1, DieselArchitecture2
from .monte_carlo import SITE_DESIGN, run_monte_carlo
from .sensitivity import tornado_data, CENTRAL
from .sweeps_2d import RATES, BATT_MULTS, ratio_matrix

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures' / 'paper'
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / 'results'

# ---------------------------------------------------------------- style
COL_W, PAGE_W = 3.50, 7.16
SITE_C = {'Southampton': '#D55E00', 'Birmingham': '#E69F00',
          'Liverpool': '#009E73', 'Edinburgh': '#0072B2'}
SOLAR, DIESEL_A2, DIESEL_A1 = '#009E73', '#D55E00', '#7A0000'
GREY, INK = '#6c757d', '#212529'

mpl.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 8,
    'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.grid': False, 'grid.color': '#dee2e6', 'grid.linewidth': 0.5,
    'legend.frameon': False, 'figure.dpi': 120, 'savefig.dpi': 600,
    'savefig.bbox': None, 'savefig.pad_inches': 0.02,
    'figure.constrained_layout.use': True,
    'axes.unicode_minus': False,
})
SERIF = 'Liberation Serif'


def _panel_below(ax, s, y=-0.30):
    """Sub-figure label centred below the axes, 8-pt serif (template IV-I-2)."""
    ax.text(0.5, y, s, transform=ax.transAxes, ha='center', va='top',
            fontsize=8, family=SERIF)


def _save(fig, name, tight=False):
    out = FIG / name
    fig.savefig(out, bbox_inches='tight' if tight else None)
    plt.close(fig)
    print('  wrote', out.name)
    return out


def _panel_label(ax, s, x=-0.18, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='left')


# ================================================================ Fig 1
def fig_ghi():
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    months = np.arange(1, 13)
    for key, site in SITES.items():
        df = get_or_create_tmy(site, prefer='real')
        m = df['ghi'].resample('MS').sum().values / 1000.0
        ax.plot(months, m, ls=['-', '--', '-.', ':'][list(SITES).index(key)],
                marker=['o', 's', '^', 'D'][list(SITES).index(key)],
                color=SITE_C[site.name], lw=1.3, ms=3, mec='white', mew=0.4,
                label=f'{site.name} ({annual_ghi_kwh_per_m2(df):.0f})')
    ax.set_xticks(months)
    ax.set_xticklabels(list('JFMAMJJASOND'))
    ax.set_xlabel('Month')
    ax.set_ylabel('Global horizontal irradiation (kWh/m²)')
    ax.set_ylim(0, 200)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(title='Site (annual GHI, kWh/m²)', loc='upper center',
              bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.8,
              title_fontsize=7, columnspacing=1.2)
    return _save(fig, 'fig01_monthly_ghi.png')


# ================================================================ Fig 2
def fig_architecture():
    fig, axs = plt.subplots(1, 2, figsize=(PAGE_W, 2.45), layout=None)
    for ax in axs:
        ax.set_xlim(0, 10); ax.set_ylim(-0.4, 4.6); ax.axis('off')

    def box(ax, x, y, w, h, text, fc='#f8f9fa', ec=INK, fs=7, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.02,rounding_size=0.12',
                                    lw=0.7, ec=ec, fc=fc))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal',
                color=INK, linespacing=1.15)

    def arrow(ax, x0, y0, x1, y1, text=None, ty=0.16):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle='-|>', mutation_scale=7,
                                     lw=0.8, color=INK))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ty, text, ha='center',
                    va='bottom', fontsize=7, color=GREY, style='italic')

    # (a) Solar PV–battery
    ax = axs[0]
    box(ax, 0.3, 2.6, 2.4, 1.3, 'PV array\n500–650 Wp\nlatitude tilt, south',
        fc='#e6f4ee', ec=SOLAR, bold=True)
    box(ax, 3.7, 2.6, 2.4, 1.3, 'MPPT charge\ncontroller\n(η ≈ 97%)')
    box(ax, 7.1, 2.6, 2.6, 1.3, 'DC load 14.64 W\nlogger · sensors ·\ncellular modem',
        fc='#f1f3f5')
    box(ax, 3.7, 0.5, 2.4, 1.3, 'LFP battery\n1.0–1.5 kWh\n80% DoD',
        fc='#e6f4ee', ec=SOLAR)
    arrow(ax, 2.7, 3.25, 3.7, 3.25, 'DC')
    arrow(ax, 6.1, 3.25, 7.1, 3.25, 'DC bus')
    arrow(ax, 4.9, 2.6, 4.9, 1.8)
    arrow(ax, 5.3, 1.8, 5.3, 2.6)
    ax.text(0.3, 0.55, '2 site visits/yr\nno fuel, no oil', fontsize=7,
            color=SOLAR, va='bottom')
    ax.text(5, -0.35, '(a)', fontsize=8, family=SERIF, ha='center', va='center')

    # (b) Diesel A2
    ax = axs[1]
    box(ax, 0.3, 2.6, 2.4, 1.3, '1 kW diesel\ngenset\n≈1.5 h/day at 30% load',
        fc='#fdecec', ec=DIESEL_A2, bold=True)
    box(ax, 3.7, 2.6, 2.4, 1.3, 'Rectifier +\ncharge controller')
    box(ax, 7.1, 2.6, 2.6, 1.3, 'DC load 14.64 W\nlogger · sensors ·\ncellular modem',
        fc='#f1f3f5')
    box(ax, 3.7, 0.5, 2.4, 1.3, 'Lead-acid buffer\n≈1.2 kWh, 4-yr life',
        fc='#fdecec', ec=DIESEL_A2)
    box(ax, 0.3, 0.5, 2.4, 1.3, 'Fuel tank 200 L\n≈84 L/yr used')
    arrow(ax, 2.7, 3.25, 3.7, 3.25, 'AC')
    arrow(ax, 6.1, 3.25, 7.1, 3.25, 'DC bus')
    arrow(ax, 4.9, 2.6, 4.9, 1.8)
    arrow(ax, 5.3, 1.8, 5.3, 2.6)
    arrow(ax, 1.5, 1.8, 1.5, 2.6)
    ax.text(7.1, 0.55, '12 site visits/yr\ninspection, oil, fuel',
            fontsize=7, color=DIESEL_A2, va='bottom')
    ax.text(5, -0.35, '(b)', fontsize=8, family=SERIF, ha='center', va='center')
    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.99,
                        bottom=0.01)
    return _save(fig, 'fig02_architecture.png', tight=True)


# ================================================================ Fig 3
def fig_winter_week(site_key='edinburgh', days=10):
    site = SITES[site_key]
    df = get_or_create_tmy(site, prefer='real')
    wp, kwh = SITE_DESIGN[site.name]
    pv, bat, load = PVDesign(nameplate_w=wp), BatteryDesign(capacity_kwh=kwh), LoadProfile()
    from .sizing import worst_life_state
    eol, soh, _ = worst_life_state()
    r1 = run_simulation(df, site, pv, bat, load, pv_ageing_factor=eol,
                        battery_soh=soh)
    r = run_simulation(df, site, pv, bat, load, pv_ageing_factor=eol,
                       battery_soh=soh, initial_soc_frac=r1.final_soc_kwh / kwh)
    h = r.hourly
    # window: centred on the hour of maximum cumulative unmet in any 10-day run
    roll = h['unmet_wh'].rolling(days * 24).sum()
    end = roll.idxmax()
    start = end - pd.Timedelta(days=days)
    w = h.loc[start:end]

    fig, axs = plt.subplots(3, 1, figsize=(COL_W, 3.5), sharex=True,
                            gridspec_kw={'height_ratios': [1.3, 1.1, 0.6]})
    axs[0].fill_between(w.index, 0, w['pv_wh'], color=SITE_C[site.name],
                        alpha=0.35, lw=0, label='PV output')
    axs[0].plot(w.index, w['load_wh'], color=INK, lw=0.9, label='Load')
    axs[0].set_ylabel('Power (W)')
    axs[0].legend(loc='upper right', ncol=2)
    axs[0].text(0.01, 0.97, '(a)', transform=axs[0].transAxes, va='top', fontsize=8, family=SERIF)
    axs[1].text(0.01, 0.97, '(b)', transform=axs[1].transAxes, va='top', fontsize=8, family=SERIF)
    axs[2].text(0.01, 0.95, '(c)', transform=axs[2].transAxes, va='top', fontsize=8, family=SERIF)
    axs[1].fill_between(w.index, 0, w['soc_frac'] * 100, color=SOLAR,
                        alpha=0.5, lw=0)
    axs[1].axhline((1 - bat.max_dod) * 100, color=DIESEL_A2, ls='--', lw=0.8)
    axs[1].text(w.index[30], (1 - bat.max_dod) * 100 + 4, 'DoD floor (20%)',
                fontsize=7, color=DIESEL_A2)
    axs[1].set_ylabel('State of charge (%)'); axs[1].set_ylim(0, 105)
    axs[2].bar(w.index, w['unmet_wh'], width=1 / 24, color=DIESEL_A2, lw=0)
    axs[2].set_ylabel('Unmet load\n(Wh)')
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axs[2].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    for ax in axs:
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    axs[2].set_xlabel('Date (typical meteorological year)')
    return _save(fig, 'fig03_winter_window.png'), (start, end)


# ================================================================ Fig 4
def fig_sizing_landscape():
    fig, axs = plt.subplots(1, 4, figsize=(PAGE_W, 2.3), sharey=True)
    mesh = None
    for i, (ax, (key, site)) in enumerate(zip(axs, SITES.items())):
        sw = pd.read_csv(RES / f'sizing_sweep_{key}.csv')
        piv = sw.pivot(index='battery_kwh', columns='pv_w', values='lolp')
        X, Y = np.meshgrid(piv.columns.values, piv.index.values)
        Z = np.log10(np.clip(piv.values, 1e-4, 1))
        mesh = ax.pcolormesh(X, Y, Z, cmap='viridis_r', vmin=-4, vmax=0,
                             shading='nearest')
        cs = ax.contour(X, Y, piv.values, levels=[0.01], colors='white',
                        linewidths=1.0)
        ax.clabel(cs, fmt={0.01: 'LOLP 1%'}, fontsize=7, inline=True)
        wp, kwh = SITE_DESIGN[site.name]
        ax.plot(wp, kwh, marker='*', ms=9, color='white', mec=INK, mew=0.6)
        ax.set_title(site.name, fontsize=8, pad=3)
        ax.set_xlabel('PV array rating (Wp)')
        _panel_below(ax, f'({chr(97 + i)})', y=-0.42)
        ax.set_xticks([200, 600, 1000])
        ax.set_yscale('log'); ax.set_yticks([1, 2, 5, 10])
        ax.set_yticklabels(['1', '2', '5', '10'])
        ax.set_ylim(0.9, 11)
    axs[0].set_ylabel('Battery capacity (kWh)')
    cb = fig.colorbar(mesh, ax=axs, pad=0.015, fraction=0.03)
    cb.set_label('LOLP in governing year', fontsize=7)
    cb.set_ticks([-4, -3, -2, -1, 0])
    cb.set_ticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
    cb.ax.tick_params(labelsize=7)
    return _save(fig, 'fig04_sizing_landscape.png')


# ================================================================ Fig 5
def fig_lcoe_and_breakdown():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(PAGE_W, 2.6), layout=None,
                                  gridspec_kw={'width_ratios': [1.35, 1]})
    sites = list(SITE_DESIGN)
    a1 = build_diesel_cashflows(DieselArchitecture1(), fuel_price_ppl=76.02)
    a2 = build_diesel_cashflows(DieselArchitecture2(), fuel_price_ppl=76.02)
    solar = {}
    for s, (wp, kwh) in SITE_DESIGN.items():
        solar[s] = build_pv_battery_cashflows(
            pv_capex_gbp_per_wp=4.5, pv_size_wp=wp,
            battery_capex_gbp_per_kwh=700, battery_kwh=kwh,
            annual_energy_delivered_kwh=128.2)
    x = np.arange(len(sites)); w = 0.2
    v5 = [solar[s].lcoe_gbp_per_kwh(0.05) for s in sites]
    v8 = [solar[s].lcoe_gbp_per_kwh(0.08) for s in sites]
    d2 = a2.lcoe_gbp_per_kwh(0.05); d1 = a1.lcoe_gbp_per_kwh(0.05)
    ax.bar(x - 1.5 * w, v5, w * 0.92, color=SOLAR)
    ax.bar(x - 0.5 * w, v8, w * 0.92, color=SOLAR, alpha=0.5)
    ax.bar(x + 0.5 * w, [d2] * 4, w * 0.92, color=DIESEL_A2)
    ax.bar(x + 1.5 * w, [d1] * 4, w * 0.92, color=DIESEL_A1)
    for xi, a, b in zip(x, v5, v8):
        ax.text(xi - w, max(a, b) + 0.6, f'{a:.2f} | {b:.2f}', ha='center',
                fontsize=6.5)
        ax.text(xi + 0.5 * w, d2 + 0.6, f'{d2:.1f}', ha='center', fontsize=6.5)
        ax.text(xi + 1.5 * w, 27.4, f'{d1:.1f}', ha='center', fontsize=6.5,
                color='white', rotation=90, va='top')
        # break marks on the clipped A1 bar
        ax.plot([xi + 1.5 * w - 0.1, xi + 1.5 * w + 0.1], [29.0, 30.0],
                color='white', lw=1.4)
        ax.plot([xi + 1.5 * w - 0.1, xi + 1.5 * w + 0.1], [28.3, 29.3],
                color='white', lw=1.4)
    ax.set_ylim(0, 31)
    ax.set_xticks(x); ax.set_xticklabels(sites)
    ax.set_ylabel('LCOE (£/kWh)')
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    handles = [mpl.patches.Patch(color=SOLAR, label='Solar–battery, 5%'),
               mpl.patches.Patch(color=SOLAR, alpha=0.5, label='Solar–battery, 8%'),
               mpl.patches.Patch(color=DIESEL_A2, label='Diesel A2, 5%'),
               mpl.patches.Patch(color=DIESEL_A1, label='Diesel A1, 5% (bar clipped)')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=2, columnspacing=1.0, handlelength=1.2)
    ax.set_ylabel('Levelised cost of electricity (£/kWh)')
    _panel_below(ax, '(a)', y=-0.46)

    # (b) discounted 25-yr cost breakdown, Southampton solar vs A2, 5%
    cats = ['Capital', 'Replacements', 'O&M / oil', 'Fuel', 'Site visits']
    def parts(cf):
        d = {f.label: f.npv(0.05) for f in cf.flows}
        cap = sum(v for k, v in d.items() if k.startswith('CapEx'))
        rep = sum(v for k, v in d.items() if 'replacement' in k.lower())
        om = sum(v for k, v in d.items() if 'O&M' in k or 'Oil' in k)
        fuel = d.get('Fuel', 0.0)
        vis = d.get('Site visits', 0.0)
        return [cap, rep, om, fuel, vis]
    ps = parts(solar['Southampton']); pd2 = parts(a2)
    colours = ['#495057', '#adb5bd', '#ced4da', '#E69F00', '#D55E00']
    for j, (label, vals) in enumerate([('Solar–battery\n(Southampton)', ps),
                                       ('Diesel A2', pd2)]):
        bottom = 0
        for c, v, name in zip(colours, vals, cats):
            ax2.barh(j, v / 1000, left=bottom / 1000, color=c, height=0.55,
                     label=name if j == 0 else None)
            bottom += v
        ax2.text(bottom / 1000 + 0.6, j, f'£{bottom/1000:.1f}k', va='center',
                 fontsize=7)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(['Solar–battery\n(Southampton)', 'Diesel A2'])
    ax2.set_xlabel('Discounted 25-year cost at 5% (£ thousand)')
    ax2.set_xlim(0, 47)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=3,
               handlelength=1.0, columnspacing=0.8)
    ax2.xaxis.grid(True); ax2.set_axisbelow(True)
    ax2.invert_yaxis()
    _panel_below(ax2, '(b)', y=-0.62)
    fig.subplots_adjust(wspace=0.55, bottom=0.34, left=0.08, right=0.99, top=0.97)
    return _save(fig, 'fig05_lcoe_breakdown.png', tight=True)


# ================================================================ Fig 6
def fig_tornado():
    td = tornado_data(architecture=2)
    order = {'Diesel visit cadence': 'Diesel visits (12→2/yr)',
             'Site visit cost': 'Visit cost (×0.5–×2)',
             'Discount rate': 'Discount rate (3–10%)',
             'PV cost': 'PV capital (±30%)',
             'Load magnitude': 'Load (±20%, re-sized)',
             'Battery cost': 'Battery capital (±30%)',
             'Solar battery life': 'LFP life (8–15 yr)',
             'PV degradation': 'Degradation (0.3–1.0%/yr)',
             'Fuel price': 'Diesel price (45–118 p/L)'}
    td = td.set_index('parameter').loc[list(order)].reset_index()
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    y = np.arange(len(td))[::-1]
    c = td['central_ratio'].iloc[0]
    for yi, (_, r) in zip(y, td.iterrows()):
        ax.barh(yi, r['low_ratio'] - c, left=c, height=0.62, color=DIESEL_A2)
        ax.barh(yi, r['high_ratio'] - c, left=c, height=0.62, color=SOLAR)
        ax.text(r['low_ratio'] - 0.05, yi, f"{r['low_ratio']:.2f}", va='center',
                ha='right', fontsize=7)
        if r['high_ratio'] - c > 0.02:
            ax.text(r['high_ratio'] + 0.05, yi, f"{r['high_ratio']:.2f}",
                    va='center', ha='left', fontsize=7)
    ax.axvline(c, color=INK, lw=0.7, ls='--')
    ax.axvline(1.0, color=GREY, lw=0.7, ls=':')
    ax.text(1.0, len(td) - 0.3, 'parity', fontsize=7, color=GREY, ha='center')
    ax.text(c, len(td) - 0.3, f'central {c:.2f}', fontsize=7, ha='center')
    ax.set_yticks(y); ax.set_yticklabels([order[p] for p in td['parameter']],
                                         fontsize=7)
    ax.set_xlim(0.6, 5.0)
    ax.set_xlabel('Diesel-to-solar LCOE ratio (Southampton, 5%)')
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, 'fig06_tornado.png')


# ================================================================ Fig 7
def fig_monte_carlo(n=5000):
    mc5 = run_monte_carlo(n=n, discount_rate=0.05)
    mc8 = run_monte_carlo(n=n, discount_rate=0.08)
    fig, axs = plt.subplots(1, 2, figsize=(PAGE_W, 2.55), layout=None,
                            gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axs[0]
    for j, (site, colour) in enumerate(SITE_C.items()):
        ax.hist(mc5[f'solar_lcoe_{site}'], bins=np.linspace(4, 11, 50),
                density=True, histtype='step', lw=1.1, color=colour, label=site,
                ls=['-', '--', '-.', ':'][j])
    ax.hist(mc5['diesel_lcoe'], bins=np.linspace(14, 45, 50), density=True,
            histtype='stepfilled', alpha=0.4, color=DIESEL_A2, lw=0,
            label='Diesel A2')
    ax.set_xlim(4, 45); ax.set_ylim(0, 0.42)
    ax.set_xlabel('Levelised cost of electricity at 5% (£/kWh)')
    ax.set_ylabel('Probability density (kWh/£)')
    ax.legend(loc='upper right', ncol=1)
    ax.text(26, 0.16, 'no overlap in\n5,000 joint draws', fontsize=7,
            ha='center', style='italic', color=GREY)
    _panel_below(ax, '(a)', y=-0.30)

    ax = axs[1]
    sites = list(SITE_DESIGN); yb = np.arange(len(sites))
    for k, (mc, mk) in enumerate([(mc5, 'o'), (mc8, 's')]):
        off = -0.17 + 0.34 * k
        for i, s in enumerate(sites):
            r = mc['diesel_lcoe'] / mc[f'solar_lcoe_{s}']
            p10, p50, p90 = np.percentile(r, [10, 50, 90])
            ax.plot([p10, p90], [i + off] * 2, color=INK, lw=1.0)
            ax.plot(p50, i + off, mk, color=SITE_C[s], ms=5.5, mec=INK, mew=0.5)
    ax.axvline(1, color=GREY, lw=0.7, ls=':')
    ax.text(1.08, -0.45, 'parity', fontsize=7, color=GREY, ha='left')
    ax.set_yticks(yb); ax.set_yticklabels(sites); ax.invert_yaxis()
    ax.set_xlim(0.5, 5.0)
    ax.set_xlabel('Diesel-to-solar LCOE ratio (P10, P50, P90)')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#adb5bd',
                              mec=INK, label='5 %'),
                       Line2D([], [], marker='s', ls='', color='#adb5bd',
                              mec=INK, label='8 %')],
              loc='lower left')
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    _panel_below(ax, '(b)', y=-0.30)
    fig.subplots_adjust(wspace=0.45, bottom=0.30, left=0.08, right=0.99, top=0.97)
    return _save(fig, 'fig07_monte_carlo.png', tight=True)


# ================================================================ Fig 8
def fig_sweep2d():
    fig, axs = plt.subplots(1, 4, figsize=(PAGE_W, 2.3), sharey=True)
    batt = BATT_MULTS * CENTRAL['battery_capex_gbp_per_kwh']
    mesh = None
    for i, (ax, site) in enumerate(zip(axs, SITE_DESIGN)):
        m = ratio_matrix(site).values
        mesh = ax.pcolormesh(batt, RATES * 100, m, cmap='viridis', vmin=2.3,
                             vmax=4.3, shading='auto')
        cs = ax.contour(batt, RATES * 100, m, levels=[2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0],
                        colors='white', linewidths=0.6)
        ax.clabel(cs, fontsize=7, fmt='%.2f')
        ax.plot(700, 5, marker='*', ms=9, color='white', mec=INK, mew=0.6)
        ax.set_title(f'{site}', fontsize=8, pad=3)
        ax.set_xlabel('Battery capital cost (£/kWh)')
        ax.set_xticks([400, 800, 1200])
        _panel_below(ax, f'({chr(97 + i)})', y=-0.42)
    axs[0].set_ylabel('Discount rate (%)')
    cb = fig.colorbar(mesh, ax=axs, pad=0.015, fraction=0.03)
    cb.set_label('Diesel-to-solar LCOE ratio', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    return _save(fig, 'fig08_sweep2d.png')


if __name__ == '__main__':
    print('paper figures →', FIG)
    fig_ghi()
    fig_architecture()
    _, win = fig_winter_week()
    print('  winter window:', win[0].date(), '→', win[1].date())
    fig_sizing_landscape()
    fig_lcoe_and_breakdown()
    fig_tornado()
    fig_monte_carlo()
    fig_sweep2d()
