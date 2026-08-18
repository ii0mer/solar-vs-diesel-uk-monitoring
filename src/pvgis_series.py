"""Loader for PVGIS 5.2 ``seriescalc`` hourly CSV files (2005-2020).

The files were requested from https://re.jrc.ec.europa.eu/api/v5_2/seriescalc
with the parameters recorded in ``src/fetch_pvgis_series.py``:

    raddatabase=PVGIS-SARAH2, startyear=2005, endyear=2020,
    pvcalculation=1, peakpower=<design kWp>, loss=0, trackingtype=0,
    angle=<site latitude>, aspect=0, pvtechchoice=crystSi,
    mountingplace=free, usehorizon=0 (1 for the Edinburgh horizon case),
    components=1, outputformat=csv

Columns in the file (PVGIS names):
    time      UTC time stamp, ``YYYYMMDD:HHMM`` (SARAH2 stamps at :10/:11)
    P         PV system power, W (PVGIS model, system losses = 0)
    Gb(i)     beam irradiance in the plane of the array, W/m2
    Gd(i)     sky-diffuse irradiance in the plane of the array, W/m2
    Gr(i)     ground-reflected irradiance in the plane of the array, W/m2
    H_sun     sun elevation, degrees
    T2m       2 m air temperature, degC
    WS10m     10 m wind speed, m/s
    Int       1 = value reconstructed (satellite gap)

Nothing here calls the network: the CSVs are read from ``data/pvgis_series``.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import io
import re

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'pvgis_series'

# kWp requested from PVGIS for each file. The CSV header prints the nominal
# power to one decimal place (0.55 shows as 0.6, 0.65 as 0.7); the value
# actually used by PVGIS is verified in tests/test_validation_multiyear.py by
# reproducing P from the irradiance columns with PVGIS's own power model.
REQUESTED_KWP = {
    'southampton': 0.50,
    'birmingham': 0.55,
    'liverpool': 0.65,
    'edinburgh': 0.65,
}

FILES = {
    'southampton': 'southampton_seriescalc_2005_2020.csv',
    'birmingham': 'birmingham_seriescalc_2005_2020.csv',
    'liverpool': 'liverpool_seriescalc_2005_2020.csv',
    'edinburgh': 'edinburgh_seriescalc_2005_2020.csv',
}
EDINBURGH_HORIZON_FILE = 'edinburgh_seriescalc_2005_2020_horizon.csv'


@dataclass
class PvgisSeries:
    site_key: str
    kwp: float                  # nameplate used for P (kW)
    slope_deg: float
    elevation_m: float
    latitude: float
    longitude: float
    horizon: bool
    data: pd.DataFrame          # hourly, UTC index

    @property
    def years(self):
        return sorted(set(self.data.index.year))


def _parse_header(text: str) -> dict:
    meta = {}
    m = re.search(r'Latitude \(decimal degrees\):\s*([-\d.]+)', text)
    meta['latitude'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'Longitude \(decimal degrees\):\s*([-\d.]+)', text)
    meta['longitude'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'Elevation \(m\):\s*([-\d.]+)', text)
    meta['elevation_m'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'Slope:\s*([-\d.]+)\s*deg', text)
    meta['slope_deg'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'Nominal power of the PV system \(c-Si\) \(kWp\):\s*([-\d.]+)', text)
    meta['kwp_printed'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'System losses \(%\):\s*([-\d.]+)', text)
    meta['loss_pct'] = float(m.group(1)) if m else float('nan')
    return meta


def load_seriescalc(path, site_key: str | None = None,
                    kwp: float | None = None,
                    horizon: bool = False) -> PvgisSeries:
    """Read one PVGIS seriescalc CSV into a :class:`PvgisSeries`.

    ``kwp`` overrides the nominal power used to normalise ``P``; when None
    the value in :data:`REQUESTED_KWP` for ``site_key`` is used, falling
    back to the printed header value.
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8', errors='ignore')
    meta = _parse_header(text)
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('time,'))
    end = start + 1
    while end < len(lines) and lines[end].strip() and lines[end][0].isdigit():
        end += 1
    df = pd.read_csv(io.StringIO('\n'.join(lines[start:end])))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', utc=True)
    df = df.set_index('time')
    df = df.rename(columns={
        'P': 'p_pvgis_w', 'Gb(i)': 'poa_direct', 'Gd(i)': 'poa_sky_diffuse',
        'Gr(i)': 'poa_ground_diffuse', 'H_sun': 'sun_elevation',
        'T2m': 'temp_air', 'WS10m': 'wind_speed', 'Int': 'reconstructed'})
    df['poa_global'] = (df['poa_direct'] + df['poa_sky_diffuse']
                        + df['poa_ground_diffuse'])
    if site_key is None:
        site_key = path.stem.split('_')[0]
    if kwp is None:
        kwp = REQUESTED_KWP.get(site_key, meta['kwp_printed'])
    return PvgisSeries(site_key=site_key, kwp=float(kwp),
                       slope_deg=meta['slope_deg'],
                       elevation_m=meta['elevation_m'],
                       latitude=meta['latitude'], longitude=meta['longitude'],
                       horizon=horizon, data=df)


def load_all(data_dir=DATA_DIR) -> dict:
    """Load the four no-horizon series keyed by site."""
    data_dir = Path(data_dir)
    return {k: load_seriescalc(data_dir / f, site_key=k)
            for k, f in FILES.items()}


def load_edinburgh_horizon(data_dir=DATA_DIR) -> PvgisSeries:
    return load_seriescalc(Path(data_dir) / EDINBURGH_HORIZON_FILE,
                           site_key='edinburgh', horizon=True)


# ---------------------------------------------------------------------------
# PVGIS's own PV power model (used only to verify the nameplate the files
# were computed with; not part of the study's PV chain)
# ---------------------------------------------------------------------------

def huld_relative_efficiency(g_wm2, t_mod_c):
    """Huld et al. (2011) relative efficiency for crystalline silicon as
    implemented in PVGIS 5 (coefficients from the PVGIS documentation)."""
    import numpy as np
    k1, k2, k3, k4, k5, k6 = (-0.017237, -0.040465, -0.004702,
                              0.000149, 0.000170, 0.000005)
    g = np.asarray(g_wm2, dtype=float)
    t = np.asarray(t_mod_c, dtype=float) - 25.0
    with np.errstate(divide='ignore', invalid='ignore'):
        gp = np.log(np.where(g > 0, g / 1000.0, np.nan))
    eta = (1 + k1 * gp + k2 * gp ** 2 + t * (k3 + k4 * gp + k5 * gp ** 2)
           + k6 * t ** 2)
    return np.where(g > 0, eta, 0.0)


def faiman_module_temperature(g_wm2, temp_air_c, wind_ms,
                              u0: float = 26.9, u1: float = 6.20):
    """Faiman (2008) module temperature with the free-standing c-Si
    coefficients used by PVGIS (Koehl et al., 2011)."""
    import numpy as np
    return (np.asarray(temp_air_c, dtype=float)
            + np.asarray(g_wm2, dtype=float)
            / (u0 + u1 * np.asarray(wind_ms, dtype=float)))


if __name__ == '__main__':
    for k, s in load_all().items():
        d = s.data
        print(f"{k:12s} rows={len(d)} years={s.years[0]}-{s.years[-1]} "
              f"kWp={s.kwp} slope={s.slope_deg} "
              f"mean annual yield={d['p_pvgis_w'].sum()/1000/s.kwp/len(s.years):.0f} kWh/kWp "
              f"reconstructed={int(d['reconstructed'].sum())} h")
