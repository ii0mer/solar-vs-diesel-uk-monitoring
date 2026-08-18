"""Fetch the PVGIS 5.2 hourly series (2005-2020) used by src/validation.py
and src/multiyear.py.

    python -m src.fetch_pvgis_series

Downloads five seriescalc CSV files from the European Commission JRC PVGIS
API into data/pvgis_series/ (they are already committed; re-run only to
refresh). Requests: PVGIS-SARAH2, 2005-2020, pvcalculation=1 at a
nameplate of 0.50/0.55/0.65/0.65 kWp (output is normalised per kWp, so
the value is immaterial), system loss 0, fixed mount at
latitude tilt facing south, crystalline silicon, free-standing, horizon
off (a second Edinburgh request with horizon on quantifies terrain
shading), irradiance components on. Requires internet access to
re.jrc.ec.europa.eu and the 'requests' package.
"""
import os
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("Please run:  pip install requests   then re-run this script.")

BASE = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
import pathlib
OUTDIR = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "pvgis_series")

# (filename, lat, lon, peak kWp, tilt, usehorizon)
JOBS = [
    ("southampton_seriescalc_2005_2020.csv", 50.9097, -1.4044, 0.50, 50.91, 0),
    ("birmingham_seriescalc_2005_2020.csv",  52.4862, -1.8904, 0.55, 52.49, 0),
    ("liverpool_seriescalc_2005_2020.csv",   53.4084, -2.9916, 0.65, 53.41, 0),
    ("edinburgh_seriescalc_2005_2020.csv",   55.9533, -3.1883, 0.65, 55.95, 0),
    ("edinburgh_seriescalc_2005_2020_horizon.csv", 55.9533, -3.1883, 0.65, 55.95, 1),
]


def fetch(fname, lat, lon, kwp, tilt, horizon):
    params = dict(
        lat=lat, lon=lon, raddatabase="PVGIS-SARAH2",
        startyear=2005, endyear=2020, pvcalculation=1, peakpower=kwp,
        loss=0, trackingtype=0, angle=tilt, aspect=0,
        pvtechchoice="crystSi", mountingplace="free",
        usehorizon=horizon, components=1, outputformat="csv",
    )
    path = os.path.join(OUTDIR, fname)
    for attempt in range(1, 6):
        try:
            print(f"  downloading {fname} (attempt {attempt}) ...", flush=True)
            r = requests.get(BASE, params=params, timeout=180)
            if r.status_code == 200 and len(r.content) > 100_000:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"    OK  {len(r.content)/1e6:.1f} MB")
                return True
            print(f"    server said {r.status_code}; retrying in 20 s")
        except Exception as e:  # noqa: BLE001
            print(f"    error: {e}; retrying in 20 s")
        time.sleep(20)
    return False


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Saving into ./{OUTDIR}/")
    ok = all(fetch(*job) for job in JOBS)
    if not ok:
        sys.exit("\nSome files failed. Re-run the script; it skips nothing but "
                 "PVGIS is usually fine on the second try.")
    print(f"\nDONE. Files in {OUTDIR}")


if __name__ == "__main__":
    main()
