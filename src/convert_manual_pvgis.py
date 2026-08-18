"""Convert manually-downloaded PVGIS TMY CSVs into the format the simulation uses.

WHEN TO USE THIS:
  When `python -m src.fetch_pvgis` fails due to API issues or pvlib version
  mismatches, you can download the TMY CSV manually from the PVGIS website
  and run this script to convert the files into the standardised cache format.

HOW TO DOWNLOAD MANUALLY:
  1. Visit https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY
  2. For each site, enter coordinates (or click on the map):
       Southampton:  50.9097, -1.4044
       Birmingham:   52.4862, -1.8904
       Liverpool:    53.4084, -2.9916
       Edinburgh:    55.9533, -3.1883
  3. Click "Download CSV" to get the TMY data.
  4. Save each file into data/pvgis/ as:
       southampton_tmy_raw.csv
       birmingham_tmy_raw.csv
       liverpool_tmy_raw.csv
       edinburgh_tmy_raw.csv

THEN RUN:
  python -m src.convert_manual_pvgis

This will read each *_tmy_raw.csv, convert to the standardised format, and
save as *_tmy.csv in data/pvgis/. All downstream code uses *_tmy.csv.
"""
from __future__ import annotations
import sys
from .sites import SITES
from .weather import load_pvgis_csv_manual, DATA_DIR


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Converting manual PVGIS CSVs in {DATA_DIR}\n")
    converted = 0
    missing = []
    for key, site in SITES.items():
        raw = DATA_DIR / f'{site.name.lower()}_tmy_raw.csv'
        out = DATA_DIR / f'{site.name.lower()}_tmy.csv'
        if not raw.exists():
            missing.append(raw.name)
            print(f"  [{site.name}] SKIP — {raw.name} not found")
            continue
        try:
            df = load_pvgis_csv_manual(site, raw)
            df.to_csv(out)
            ann_ghi = df['ghi'].sum() / 1000.0
            print(f"  [{site.name}] OK  ({len(df)} rows, GHI={ann_ghi:.1f} kWh/m²/yr → {out.name})")
            converted += 1
        except Exception as e:
            print(f"  [{site.name}] FAILED: {type(e).__name__}: {e}")
    print()
    print(f"Converted {converted} of {len(SITES)} sites.")
    if missing:
        print(f"Missing raw files: {', '.join(missing)}")
        print("Download them from https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY")
        sys.exit(1)
    print("All sites converted. You can now run:")
    print("  python -m src.simulation")
    print("  python -m src.figures_paper")


if __name__ == '__main__':
    main()
