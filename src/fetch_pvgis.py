"""Standalone script to fetch real PVGIS-SARAH2 TMY data for all sites.

Run this on your local machine (which has internet to re.jrc.ec.europa.eu).
Once it completes, real data is cached in data/pvgis/<site>_tmy.csv and
all subsequent simulations use the real data automatically.

Usage:
    python -m src.fetch_pvgis
"""
from __future__ import annotations
import sys
from .sites import SITES
from .weather import fetch_pvgis_tmy, DATA_DIR


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching real PVGIS-SARAH2 TMY data → {DATA_DIR}\n")
    for key, site in SITES.items():
        print(f"  [{site.name}] lat={site.latitude} lon={site.longitude} ...",
              end=' ', flush=True)
        try:
            df = fetch_pvgis_tmy(site)
            out = DATA_DIR / f'{site.name.lower()}_tmy.csv'
            df.to_csv(out)
            ann_ghi = df['ghi'].sum() / 1000.0
            print(f"OK  ({len(df)} rows, GHI={ann_ghi:.1f} kWh/m²/yr → {out.name})")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            print("\n  Note: PVGIS API requires internet access to re.jrc.ec.europa.eu.")
            print("  If on a restricted network, run from a different connection.")
            sys.exit(1)
    print("\nAll sites fetched. You can now run:")
    print("  python -m src.simulation")
    print("  python -m src.figures")


if __name__ == '__main__':
    main()
