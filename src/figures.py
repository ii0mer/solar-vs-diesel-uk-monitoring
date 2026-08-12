"""Generate all dissertation figures.

Seven publication-quality figures, all regenerable on demand:

    Fig 1 — System architecture block diagram (PV-battery vs diesel)
    Fig 2 — Monthly GHI per site
    Fig 3 — Load profile breakdown
    Fig 4 — Edinburgh winter week stress case
    Fig 5 — Sizing landscape heatmap (PV × battery, LOLP)
    Fig 6 — LCOE comparison (linear and log scale, 3 systems)
    Fig 7 — Sensitivity tornado chart

When real PVGIS data is available, set USE_REAL_DATA = True (or pass
prefer='real' to weather functions); the script will swap synthetic for
real seamlessly.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

from .sites import SITES, Site
from .weather import get_or_create_tmy, annual_ghi_kwh_per_m2
from .load_profile import LoadProfile
from .pv_model import PVDesign, simulate_pv_dc
from .battery import BatteryDesign
from .simulation import run_simulation
from .economics import (
    build_pv_battery_cashflows, build_diesel_cashflows,
    DEFAULT_DISCOUNT_RATE, DEFAULT_PROJECT_YEARS,
)
from .diesel import DieselArchitecture1, DieselArchitecture2
from .sensitivity import tornado_data, CENTRAL


FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SITE_COLOURS = {
    'Southampton': '#d62728',
    'Birmingham':  '#2ca02c',
    'Liverpool':   '#9467bd',   # purple — distinct from green/red/blue
    'Edinburgh':   '#1f77b4',
}
SYSTEM_COLOURS = {
    'pv_battery': '#2a9d8f',
    'diesel_a2':  '#e76f51',
    'diesel_a1':  '#9b2226',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.titlesize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '-',
    'figure.dpi': 110,
})

WATERMARK_SYNTHETIC = "(SYNTHETIC calibrated weather — fallback only; not for dissertation figures)"


def fig01_architecture() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    def draw_box(ax, x, y, w, h, label, color='#e9ecef', edge='#212529'):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.04,rounding_size=0.08",
                             linewidth=1.2, edgecolor=edge, facecolor=color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=9, fontweight='medium')

    def draw_arrow(ax, x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#495057'))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.1, label,
                    ha='center', fontsize=8, color='#495057', style='italic')

    # Panel A: PV-Battery
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('(a) Solar PV–Battery System', fontsize=12, fontweight='bold')
    draw_box(ax, 0.5, 6.5, 2.5, 1.0, 'Solar Array\n(PV modules)', '#fff3cd')
    draw_box(ax, 4.0, 6.5, 2.5, 1.0, 'MPPT Charge\nController', '#d1ecf1')
    draw_box(ax, 4.0, 4.0, 2.5, 1.0, 'LFP Battery\n(1–5 kWh)', '#d4edda')
    draw_box(ax, 4.0, 1.5, 2.5, 1.0, 'DC Bus &\nDC-DC Converter', '#e2e3e5')
    draw_box(ax, 7.5, 1.5, 2.0, 1.0, 'Monitoring\nLoad (14.6 W)', '#f8d7da')
    draw_arrow(ax, 3.0, 7.0, 4.0, 7.0, 'DC')
    draw_arrow(ax, 5.25, 6.5, 5.25, 5.0, 'DC')
    draw_arrow(ax, 5.25, 4.0, 5.25, 2.5, 'DC')
    draw_arrow(ax, 6.5, 2.0, 7.5, 2.0, '')
    ax.text(5.0, 0.5, '~2 site visits/year (cleaning + inspection)',
            ha='center', fontsize=8, color='#198754', style='italic')

    # Panel B: Diesel
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('(b) Diesel Generator System (Architecture 2)',
                 fontsize=12, fontweight='bold')
    draw_box(ax, 0.5, 6.5, 2.5, 1.0, 'Fuel Tank\n(200 L)', '#f8d7da')
    draw_box(ax, 4.0, 6.5, 2.5, 1.0, 'Diesel Genset\n(1 kW)', '#fff3cd')
    draw_box(ax, 4.0, 4.0, 2.5, 1.0, 'AC-DC\nRectifier', '#d1ecf1')
    draw_box(ax, 4.0, 1.5, 2.5, 1.0, 'Lead-Acid\nBattery (1 kWh)', '#d4edda')
    draw_box(ax, 7.5, 1.5, 2.0, 1.0, 'Monitoring\nLoad (14.6 W)', '#f8d7da')
    draw_arrow(ax, 3.0, 7.0, 4.0, 7.0, 'fuel')
    draw_arrow(ax, 5.25, 6.5, 5.25, 5.0, 'AC')
    draw_arrow(ax, 5.25, 4.0, 5.25, 2.5, 'DC')
    draw_arrow(ax, 6.5, 2.0, 7.5, 2.0, '')
    ax.text(5.0, 0.5, '12 site visits/year (fuel delivery + inspection)',
            ha='center', fontsize=8, color='#dc3545', style='italic')

    fig.tight_layout()
    out = FIG_DIR / 'fig01_architecture.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def fig02_monthly_ghi(synthetic: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for key, site in SITES.items():
        df = get_or_create_tmy(site, prefer='synthetic' if synthetic else 'real')
        monthly = (df['ghi'].resample('ME').sum() / 1000.0)
        months = range(1, 13)
        ax.plot(months, monthly.values, '-o',
                color=SITE_COLOURS[site.name], linewidth=2, markersize=5,
                label=f'{site.name} ({annual_ghi_kwh_per_m2(df):.0f} kWh/m²/yr)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Global horizontal irradiation (kWh/m² per month)')
    ax.set_title('Monthly solar resource at four UK study sites')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.legend(loc='upper right', frameon=False)
    if synthetic:
        ax.text(0.5, -0.18, WATERMARK_SYNTHETIC, ha='center', va='top',
                transform=ax.transAxes, fontsize=8, style='italic', color='gray')
    fig.tight_layout()
    out = FIG_DIR / 'fig02_monthly_ghi.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def fig03_load_profile() -> Path:
    lp = LoadProfile()
    components = {
        'Data logger\n(Campbell CR1000X)': lp.datalogger_w,
        'Sensors ×4\n(env, T/RH, wind)': lp.sensors_w_each * lp.sensor_count,
        'Cellular modem\n(Digi IX20)': lp.modem_w,
        'Ancillary\n(LED, relay, heater)': lp.ancillary_w,
        '+20% margin\n(losses, derate)': lp.base_power_w() * (lp.margin_factor - 1),
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(components.keys(), components.values(),
                  color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f'],
                  edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, components.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.1f} W', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Continuous power draw (W)')
    ax.set_title(f'Load profile composition — {lp.design_power_w():.1f} W continuous, '
                 f'{lp.daily_energy_wh():.0f} Wh/day, {lp.annual_energy_kwh():.0f} kWh/yr')
    ax.set_ylim(0, max(components.values()) * 1.25)
    fig.tight_layout()
    out = FIG_DIR / 'fig03_load_profile.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def fig04_edinburgh_winter_week(synthetic: bool = True) -> Path:
    site = SITES['edinburgh']
    df = get_or_create_tmy(site, prefer='synthetic' if synthetic else 'real')
    # Strip timezone for uniform slicing
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    pv = PVDesign(nameplate_w=600, n_modules=1)   # updated to real-data optimal
    bat = BatteryDesign(capacity_kwh=1.0)
    load = LoadProfile()
    r1 = run_simulation(df, site, pv, bat, load)
    r = run_simulation(df, site, pv, bat, load,
                       initial_soc_frac=r1.final_soc_kwh / bat.capacity_kwh)
    # Slice by month/day position (works regardless of reference year)
    year = r.hourly.index[0].year
    week = r.hourly.loc[f'{year}-01-14':f'{year}-01-21']

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axs[0].fill_between(week.index, 0, week['pv_wh'], color='#f4a261',
                        alpha=0.7, label='PV generation')
    axs[0].plot(week.index, week['load_wh'], color='#264653', linewidth=1.5, label='Load')
    axs[0].set_ylabel('Power (Wh per hour)')
    axs[0].set_title(f'Edinburgh — winter week ({pv.total_nameplate_w:.0f} Wp PV, '
                     f'{bat.capacity_kwh:.1f} kWh battery)')
    axs[0].legend(loc='upper right', frameon=False)

    axs[1].fill_between(week.index, 0, week['soc_frac']*100,
                        color='#2a9d8f', alpha=0.6, label='Battery SoC')
    axs[1].axhline(20, color='#e76f51', linestyle='--', linewidth=1,
                   alpha=0.7, label='DoD floor (20%)')
    axs[1].set_ylabel('State of charge (%)')
    axs[1].set_ylim(0, 105)
    axs[1].legend(loc='lower left', frameon=False)

    axs[2].fill_between(week.index, 0, week['unmet_wh'],
                        color='#e63946', alpha=0.85, label='Unmet load')
    axs[2].set_ylabel('Unmet load\n(Wh per hour)')
    axs[2].set_xlabel('Date (UTC)')
    axs[2].legend(loc='upper right', frameon=False)
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

    if synthetic:
        fig.text(0.5, 0.005, WATERMARK_SYNTHETIC, ha='center', va='bottom',
                 fontsize=8, style='italic', color='gray')
    fig.tight_layout()
    out = FIG_DIR / 'fig04_edinburgh_winter_week.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def fig05_sizing_landscape(synthetic: bool = True) -> Path:
    from .sizing import size_system
    load = LoadProfile()
    pv_grid = np.array([100, 200, 300, 400, 500, 700, 1000])
    bat_grid = np.array([0.5, 1, 2, 3, 5, 10, 20])

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)
    for ax, (key, site) in zip(axs, SITES.items()):
        df = get_or_create_tmy(site, prefer='synthetic' if synthetic else 'real')
        if df.index.tz is not None:
            df = df.copy(); df.index = df.index.tz_localize(None)
        _best, sweep = size_system(df, site, load, target_lolp=0.01,
                                   pv_grid=pv_grid, battery_grid=bat_grid)
        pivot = sweep.pivot(index='battery_kwh', columns='pv_w', values='lolp') * 100
        im = ax.imshow(pivot.values, origin='lower', aspect='auto',
                       cmap='RdYlGn_r', vmin=0, vmax=15)
        ax.set_xticks(range(len(pv_grid)))
        ax.set_xticklabels([f'{int(p)}' for p in pv_grid], rotation=45)
        ax.set_yticks(range(len(bat_grid)))
        ax.set_yticklabels([f'{b}' for b in bat_grid])
        ax.set_title(f'{site.name}')
        ax.set_xlabel('PV (Wp)')
        for i, b in enumerate(bat_grid):
            for j, p in enumerate(pv_grid):
                v = pivot.values[i, j]
                col = 'white' if v > 5 else 'black'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=7, color=col)
    axs[0].set_ylabel('Battery (kWh)')
    cbar = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    cbar.set_label('Loss-of-load probability (%)')
    fig.suptitle('System sizing landscape — LOLP across PV × battery combinations')
    if synthetic:
        fig.text(0.5, 0.005, WATERMARK_SYNTHETIC, ha='center', va='bottom',
                 fontsize=8, style='italic', color='gray')
    out = FIG_DIR / 'fig05_sizing_landscape.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def fig06_lcoe_comparison() -> Path:
    site_results = {}
    # Real PVGIS validated sizing — updated after real TMY data retrieved
    # Southampton unchanged; Birmingham/Liverpool/Edinburgh each need +100Wp
    sz = {
        'southampton': (400, 1.0),
        'birmingham':  (500, 1.0),
        'liverpool':   (500, 1.0),
        'edinburgh':   (600, 1.0),
    }
    for key, site in SITES.items():
        pv_w, bat_kwh = sz[key]
        pv_cf = build_pv_battery_cashflows(
            pv_capex_gbp_per_wp=4.50, pv_size_wp=pv_w,
            battery_capex_gbp_per_kwh=700.0, battery_kwh=bat_kwh,
            annual_energy_delivered_kwh=128.2,
        )
        site_results[site.name] = pv_cf.lcoe_gbp_per_kwh(0.05)

    arch1 = DieselArchitecture1()
    arch2 = DieselArchitecture2()
    di1_lcoe = build_diesel_cashflows(arch1, fuel_price_ppl=76.02).lcoe_gbp_per_kwh(0.05)
    di2_lcoe = build_diesel_cashflows(arch2, fuel_price_ppl=76.02).lcoe_gbp_per_kwh(0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sites = list(site_results.keys())
    pv_lcoes = [site_results[s] for s in sites]
    x = np.arange(len(sites))
    width = 0.25

    for ax, scale_label in [(ax1, 'linear'), (ax2, 'log')]:
        n = len(sites)
        bars1 = ax.bar(x - width, pv_lcoes, width, label='Solar–Battery',
                       color=SYSTEM_COLOURS['pv_battery'], edgecolor='black')
        bars2 = ax.bar(x, [di2_lcoe]*n, width, label='Diesel — Architecture 2 (intermittent)',
                       color=SYSTEM_COLOURS['diesel_a2'], edgecolor='black')
        bars3 = ax.bar(x + width, [di1_lcoe]*n, width, label='Diesel — Architecture 1 (24/7)',
                       color=SYSTEM_COLOURS['diesel_a1'], edgecolor='black')
        for bars, vals in [(bars1, pv_lcoes), (bars2, [di2_lcoe]*n), (bars3, [di1_lcoe]*n)]:
            for bar, v in zip(bars, vals):
                offset = v * 1.06 if scale_label == 'log' else v + 1
                ax.text(bar.get_x() + bar.get_width()/2, offset,
                        f'£{v:.1f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sites)
        ax.set_ylabel('Levelised cost of electricity (£/kWh)')
        if scale_label == 'log':
            ax.set_yscale('log')
            ax.set_title('LCOE comparison (logarithmic scale)')
        else:
            ax.set_title('LCOE comparison (linear scale)')
        ax.legend(loc='upper left' if scale_label == 'linear' else 'upper right',
                  frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('25-year levelised cost of electricity, four UK sites '
                 '(IEA/NEA 2020 methodology, 5% real discount rate)', fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / 'fig06_lcoe_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame({
        'Site': sites,
        'Solar-Battery LCOE (£/kWh)': [round(v, 3) for v in pv_lcoes],
        'Diesel A2 LCOE (£/kWh)': round(di2_lcoe, 3),
        'Diesel A1 LCOE (£/kWh)': round(di1_lcoe, 3),
        'Ratio A2/PV': [round(di2_lcoe/v, 1) for v in pv_lcoes],
        'Ratio A1/PV': [round(di1_lcoe/v, 1) for v in pv_lcoes],
    }).to_csv(RESULTS_DIR / 'lcoe_comparison.csv', index=False)
    return out


def fig07_tornado_sensitivity() -> Path:
    td = tornado_data(architecture=2)
    td = td[td['range'] > 0.001]
    td = td.sort_values('range', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    central = td.iloc[0]['central_ratio']

    for i, (_, row) in enumerate(td.iterrows()):
        low = row['low_ratio']
        high = row['high_ratio']
        left_width = central - low if low < central else 0
        right_width = high - central if high > central else 0
        ax.barh(i, -left_width, left=central, height=0.6,
                color='#e76f51', edgecolor='black', linewidth=0.6)
        ax.barh(i, right_width, left=central, height=0.6,
                color='#2a9d8f', edgecolor='black', linewidth=0.6)
        ax.text(low - 0.04, i, f'{low:.2f}×', va='center', ha='right',
                fontsize=9, color='#9b2226')
        ax.text(high + 0.04, i, f'{high:.2f}×', va='center', ha='left',
                fontsize=9, color='#1d3557')

    ax.set_yticks(np.arange(len(td)))
    ax.set_yticklabels(td['parameter'].values)
    ax.axvline(central, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(central, len(td) - 0.3, f'Central: {central:.2f}×',
            ha='center', fontsize=9, fontweight='medium')
    ax.set_xlabel('Diesel-to-PV LCOE ratio')
    ax.set_title('Sensitivity tornado — robustness of the solar advantage\n'
                 '(parameters ranked by impact on diesel-vs-PV LCOE ratio)')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(left=ax.get_xlim()[0] - 0.3)

    fig.text(0.5, -0.01,
             'One-at-a-time minimum: 3.61× (site-visit cost halved). '
             'Combined worst-case stress test: 2.6× (see §3.3 of report).',
             ha='center', fontsize=9, style='italic', color='#212529')
    fig.tight_layout()
    out = FIG_DIR / 'fig07_tornado_sensitivity.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def main(synthetic: bool = False):
    """Generate all dissertation figures.

    Now defaults to real=True (synthetic=False) since PVGIS TMY data has been
    retrieved and cached in data/pvgis/. Pass synthetic=True only for offline
    reproducibility testing.
    """
    print("Generating dissertation figures...")
    print(f"  Using {'synthetic' if synthetic else 'real PVGIS'} weather data\n")
    figs = [
        ("Architecture diagram", fig01_architecture()),
        ("Monthly GHI", fig02_monthly_ghi(synthetic)),
        ("Load profile", fig03_load_profile()),
        ("Edinburgh winter week", fig04_edinburgh_winter_week(synthetic)),
        ("Sizing landscape", fig05_sizing_landscape(synthetic)),
        ("LCOE comparison", fig06_lcoe_comparison()),
        ("Tornado sensitivity", fig07_tornado_sensitivity()),
    ]
    for label, p in figs:
        print(f"  ✓ {label}: {p.name}")
    print(f"\nAll figures: {FIG_DIR}")


if __name__ == '__main__':
    main(synthetic=False)
