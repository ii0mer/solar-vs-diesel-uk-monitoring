"""Figures for the final-dissertation upgrades (Monte Carlo, 2-D sweeps).

    Fig 8 — Monte Carlo LCOE distributions + diesel/solar ratio percentiles
    Fig 9 — 2-D sensitivity: discount rate x battery cost, ratio contours

Run:  python -m src.figures_upgrades
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .monte_carlo import SITE_DESIGN, SITE_PV_WP, run_monte_carlo
from .sweeps_2d import RATES, BATT_MULTS, ratio_matrix
from .sensitivity import CENTRAL

FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'
FIG_DIR.mkdir(exist_ok=True)

SITE_COLOURS = {
    'Southampton': '#d62728',
    'Birmingham':  '#2ca02c',
    'Liverpool':   '#9467bd',
    'Edinburgh':   '#1f77b4',
}
DIESEL_COLOUR = '#9b2226'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'figure.dpi': 300,
})


def make_fig08_monte_carlo(n: int = 5000) -> Path:
    mc5 = run_monte_carlo(n=n, discount_rate=0.05)
    mc8 = run_monte_carlo(n=n, discount_rate=0.08)

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.6))

    # (a) LCOE distributions at 5% — single shared density axis
    ax = axs[0]
    bins_solar = np.linspace(4, 11, 55)
    for site, colour in SITE_COLOURS.items():
        ax.hist(mc5[f'solar_lcoe_{site}'], bins=bins_solar, density=True,
                histtype='step', linewidth=1.6, color=colour, label=site)
    ax.hist(mc5['diesel_lcoe'], bins=np.linspace(15, 45, 55), density=True,
            histtype='stepfilled', alpha=0.35, color=DIESEL_COLOUR,
            label='Diesel A2')
    ax.set_xlim(4, 45)
    ax.set_ylim(0, 0.40)
    ax.set_xlabel('LCOE (£/kWh), 5% discount rate')
    ax.set_ylabel('Probability density')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.set_title('(a) Monte Carlo LCOE distributions (5,000 draws)',
                 fontsize=10)
    ax.text(26, 0.14,
            'distributions never overlap:\nP(solar < diesel) = 100%',
            fontsize=8, style='italic', ha='center')

    # (b) ratio percentiles at 5% and 8%
    ax = axs[1]
    sites = list(SITE_PV_WP)
    y = np.arange(len(sites))
    for dy, (mc, marker) in enumerate([(mc5, 'o'), (mc8, 's')]):
        offs = -0.16 + 0.32 * dy
        for i, site in enumerate(sites):
            ratio = mc['diesel_lcoe'] / mc[f'solar_lcoe_{site}']
            p10, p50, p90 = np.percentile(ratio, [10, 50, 90])
            ax.plot([p10, p90], [i + offs] * 2, '-', color='#495057',
                    linewidth=1.4, zorder=1)
            ax.plot(p50, i + offs, marker, color=SITE_COLOURS[site],
                    markersize=7, markeredgecolor='black',
                    markeredgewidth=0.6, zorder=2)
    # rate legend with neutral markers (colour encodes site, marker = rate)
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([], [], marker='o', linestyle='', color='#adb5bd',
               markeredgecolor='black', markeredgewidth=0.6, markersize=7,
               label='5% (Green Book + premium)'),
        Line2D([], [], marker='s', linestyle='', color='#adb5bd',
               markeredgecolor='black', markeredgewidth=0.6, markersize=7,
               label='8% (commercial WACC)'),
    ]
    ax.legend(handles=proxies, fontsize=8, frameon=False, loc='lower left')
    ax.axvline(1.0, color=DIESEL_COLOUR, linestyle='--', linewidth=1.2)
    ax.text(1.10, 1.5, 'parity', fontsize=8, color=DIESEL_COLOUR,
            rotation=90, va='center')
    ax.set_yticks(y)
    ax.set_yticklabels(sites)
    ax.set_xlim(0.5, 5.2)
    ax.set_xlabel('Diesel-to-solar LCOE ratio (P10–P50–P90)')
    ax.set_title('(b) Ratio percentiles by site and discount framing',
                 fontsize=10)
    ax.invert_yaxis()

    fig.tight_layout()
    out = FIG_DIR / 'fig08_monte_carlo_lcoe.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def make_fig09_sweep2d() -> Path:
    fig, axs = plt.subplots(2, 2, figsize=(10.0, 6.8), sharex=True,
                            sharey=True)
    levels = [2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
    batt_gbp = BATT_MULTS * CENTRAL['battery_capex_gbp_per_kwh']
    mesh = None
    for ax, site in zip(axs.flat, SITE_DESIGN):
        mat = ratio_matrix(site).values
        mesh = ax.pcolormesh(batt_gbp, RATES * 100, mat, cmap='viridis',
                             vmin=2.4, vmax=4.4, shading='auto')
        cs = ax.contour(batt_gbp, RATES * 100, mat, levels=levels,
                        colors='white', linewidths=0.8)
        ax.clabel(cs, fontsize=7, fmt='%.2f×')
        ax.plot(CENTRAL['battery_capex_gbp_per_kwh'], 5.0, 'w*',
                markersize=11, markeredgecolor='black')
        wp, kwh = SITE_DESIGN[site]
        ax.set_title(f'{site} ({wp:.0f} Wp + {kwh:.1f} kWh)', fontsize=10)
    for ax in axs[1, :]:
        ax.set_xlabel('Battery capex (£/kWh)')
    for ax in axs[:, 0]:
        ax.set_ylabel('Discount rate (%)')
    cbar = fig.colorbar(mesh, ax=axs, shrink=0.85, pad=0.02)
    cbar.set_label('Diesel-to-solar LCOE ratio (×)')
    fig.suptitle('Diesel-to-solar LCOE ratio: discount rate × battery cost '
                 '(★ = central case; parity requires £9,600–19,000/kWh '
                 'battery capex, 14–27× the 2025 market price)',
                 fontsize=10)
    out = FIG_DIR / 'fig09_sweep2d_ratio.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == '__main__':
    make_fig08_monte_carlo()
    make_fig09_sweep2d()
