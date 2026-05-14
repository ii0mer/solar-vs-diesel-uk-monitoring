"""Weather data acquisition.

Two modes:
1. ``fetch_pvgis_tmy(site)`` — pulls real SARAH2 TMY data from the JRC API.
   Use this on your local machine. Requires internet access to
   re.jrc.ec.europa.eu (free, no API key).
2. ``synthesise_tmy(site)`` — generates a physically realistic TMY-like
   dataset using pvlib's clear-sky model + a deterministic seasonal cloud
   factor calibrated to the site's annual GHI. Used for offline development
   and CI smoke tests. Annual GHI matches the proposal table within ±2%.

The columns produced by both functions are identical:
    ghi, dni, dhi, temp_air, wind_speed
indexed by hourly UTC timestamps for one TMY year.

This contract means downstream code is agnostic to source.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib

from .sites import Site, SITES

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'pvgis'


# ---------------------------------------------------------------------------
# Real PVGIS fetch (works on any machine with internet)
# ---------------------------------------------------------------------------

def fetch_pvgis_tmy(site: Site) -> pd.DataFrame:
    """Pull TMY hourly data from PVGIS-SARAH2.

    Uses pvlib.iotools.get_pvgis_tmy. Returns a DataFrame with the standard
    pvlib variable names (ghi, dni, dhi, temp_air, wind_speed) indexed by
    UTC hour.

    Notes
    -----
    PVGIS-SARAH2 covers Europe and Africa. For UK sites this is the
    appropriate database (vs. ERA5, which is also available but coarser).
    """
    # Handle pvlib API changes across versions:
    #   - pvlib <= 0.10:  returns (data, months_dict, inputs_dict, meta_dict)  → 4-tuple
    #   - pvlib 0.11-0.13: returns (data, meta_dict)                            → 2-tuple
    #   - pvlib >= 0.15:   may return (data, meta_dict) or a dict
    result = pvlib.iotools.get_pvgis_tmy(
        latitude=site.latitude,
        longitude=site.longitude,
        outputformat='json',
        map_variables=True,  # rename PVGIS columns to pvlib conventions
    )
    # Unpack robustly
    if isinstance(result, tuple):
        if len(result) == 4:
            data, _months, _inputs, _meta = result
        elif len(result) == 2:
            data, _meta = result
        else:
            data = result[0]
    elif isinstance(result, dict):
        data = result.get('data', result.get('outputs', result))
    else:
        data = result
    # Standardise: keep just the columns we use
    cols = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
    return data[cols].copy()


# ---------------------------------------------------------------------------
# Synthetic fallback (offline development)
# ---------------------------------------------------------------------------

def synthesise_tmy(site: Site, seed: int = 1181101) -> pd.DataFrame:
    """Generate a physically plausible TMY-like dataset.

    Method:
      1. Build a TMY-style hourly index for a non-leap year.
      2. Use pvlib's Ineichen clear-sky model for theoretical max GHI/DNI/DHI.
      3. Apply a seasonal cloud-factor envelope: ~0.45 in winter, ~0.55 in
         summer (UK reality — winter is *both* dimmer and cloudier; summer
         has more diffuse fraction).
      4. Add deterministic stochastic noise (cloud passes) seeded by student ID.
      5. Scale the final annual GHI to match the proposal's site value within
         ±2% by adjusting the mean cloud factor.

    Output is signed off as "realistic test data" — NOT for dissertation
    results. Replace with fetch_pvgis_tmy on your machine.
    """
    rng = np.random.default_rng(seed)

    # 1. Hourly index, one TMY year (use 2019 — non-leap)
    idx = pd.date_range('2019-01-01 00:30', '2019-12-31 23:30',
                        freq='h', tz='UTC')

    # 2. Clear-sky model
    location = pvlib.location.Location(
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.altitude,
        tz='UTC',
    )
    cs = location.get_clearsky(idx, model='ineichen')

    # 3. Seasonal cloud factor (1.0 = perfect clear, 0.0 = total overcast)
    # UK winter sun is *very* attenuated, summer sun moderately so.
    day_of_year = idx.dayofyear.values
    seasonal = 0.50 + 0.10 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    # 4. Stochastic noise — synthetic cloud passes
    noise = rng.normal(0.0, 0.18, size=len(idx))
    cloud_factor = np.clip(seasonal + noise, 0.05, 1.0)

    ghi = cs['ghi'].values * cloud_factor
    # When cloudy, more of the irradiance is diffuse
    diffuse_frac = 0.30 + 0.55 * (1 - cloud_factor)
    diffuse_frac = np.clip(diffuse_frac, 0.20, 1.0)
    dhi = ghi * diffuse_frac
    # DNI from GHI = DHI + DNI*cos(zenith), so:
    sp = location.get_solarposition(idx)
    cos_zen = np.cos(np.radians(sp['zenith'].values))
    cos_zen = np.where(cos_zen > 0.01, cos_zen, np.nan)
    dni = (ghi - dhi) / cos_zen
    dni = np.nan_to_num(dni, nan=0.0)
    dni = np.clip(dni, 0, cs['dni'].values)

    # 5. Scale to match proposal annual GHI within ±2%
    sim_annual = ghi.sum() / 1000.0  # Wh/m²/h * h → Wh/m²/yr → kWh/m²/yr
    target = site.annual_ghi_proposal
    scale = target / sim_annual
    ghi = ghi * scale
    dhi = dhi * scale
    dni = dni * scale

    # Air temperature: simple sinusoidal annual + diurnal pattern
    # UK mean ~10°C, seasonal amplitude ~7°C, diurnal amplitude ~4°C
    annual_temp = 10.0 - 7.0 * np.cos(2 * np.pi * (day_of_year - 30) / 365)
    diurnal_temp = -4.0 * np.cos(2 * np.pi * idx.hour.values / 24)
    # Edinburgh ~2°C cooler than Southampton
    site_offset = {'southampton': 1.0, 'birmingham': 0.0, 'edinburgh': -1.5}
    offset = site_offset.get(site.name.lower(), 0.0)
    temp_air = annual_temp + diurnal_temp + offset + rng.normal(0, 1.2, len(idx))

    # Wind speed: lognormal, mean ~4 m/s
    wind_speed = rng.lognormal(mean=1.2, sigma=0.4, size=len(idx))
    wind_speed = np.clip(wind_speed, 0.5, 25.0)

    df = pd.DataFrame({
        'ghi': ghi,
        'dni': dni,
        'dhi': dhi,
        'temp_air': temp_air,
        'wind_speed': wind_speed,
    }, index=idx)
    df.index.name = 'time(UTC)'
    return df


# ---------------------------------------------------------------------------
# Manual download fallback — for when the API is misbehaving
# ---------------------------------------------------------------------------

def load_pvgis_csv_manual(site: Site, csv_path) -> pd.DataFrame:
    """Load a PVGIS TMY CSV downloaded manually from the PVGIS web interface.

    Use this when ``fetch_pvgis_tmy`` fails (network issue or API change).

    To download manually:
      1. Visit https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY
      2. Enter site coordinates (or click on map).
      3. Click "Download .csv" (the standard TMY CSV format).
      4. Save as data/pvgis/<sitename>_tmy_raw.csv (e.g. liverpool_tmy_raw.csv).
      5. Run:
         from src.weather import load_pvgis_csv_manual, DATA_DIR
         from src.sites import SITES
         df = load_pvgis_csv_manual(SITES['liverpool'],
                                    DATA_DIR / 'liverpool_tmy_raw.csv')
         df.to_csv(DATA_DIR / 'liverpool_tmy.csv')

    The PVGIS TMY CSV has 18 metadata rows at the top, then a header row,
    then 8,760 hourly data rows, then footer rows. This function strips all
    that and returns a clean DataFrame matching ``fetch_pvgis_tmy``'s output.
    """
    import io
    from pathlib import Path
    csv_path = Path(csv_path)
    text = csv_path.read_text(encoding='utf-8', errors='ignore')
    # PVGIS TMY CSV has header rows starting with "time(UTC)"
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('time(UTC)'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find 'time(UTC)' header in {csv_path}. "
                         f"Make sure you downloaded the TMY CSV (not hourly).")
    # Find footer — first blank line after header
    footer_idx = len(lines)
    for i in range(header_idx + 1, len(lines)):
        if not lines[i].strip() or lines[i].startswith('PVGIS'):
            footer_idx = i
            break
    data_text = '\n'.join(lines[header_idx:footer_idx])
    df = pd.read_csv(io.StringIO(data_text))
    # PVGIS CSV columns: time(UTC), T2m, RH, G(h), Gb(n), Gd(h), IR(h), WS10m, WD10m, SP
    # Map to pvlib convention
    df = df.rename(columns={
        'time(UTC)': 'time',
        'T2m': 'temp_air',
        'G(h)': 'ghi',
        'Gb(n)': 'dni',
        'Gd(h)': 'dhi',
        'WS10m': 'wind_speed',
    })
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
    df = df.dropna(subset=['time']).set_index('time')
    cols = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
    return df[cols].copy()


# ---------------------------------------------------------------------------
# Local cache layer
# ---------------------------------------------------------------------------

def get_or_create_tmy(site: Site, prefer: str = 'real') -> pd.DataFrame:
    """Return TMY data for a site, caching on disk.

    Parameters
    ----------
    prefer : 'real' or 'synthetic'
        'real' → try PVGIS fetch first, fall back to synthetic if it fails.
        'synthetic' → always use synthetic (for offline reproducibility).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = 'tmy' if prefer == 'real' else 'tmy_synthetic'
    cache = DATA_DIR / f'{site.name.lower()}_{suffix}.csv'

    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        # PVGIS real data has UTC-aware timestamps; strip tz for uniform downstream use
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    if prefer == 'real':
        try:
            df = fetch_pvgis_tmy(site)
            df.to_csv(cache)
            return df
        except Exception as e:
            print(f"[warn] PVGIS fetch failed for {site.name}: {e}")
            print(f"[warn] falling back to synthetic data — replace on local machine")

    df = synthesise_tmy(site)
    cache_syn = DATA_DIR / f'{site.name.lower()}_tmy_synthetic.csv'
    df.to_csv(cache_syn)
    return df


def annual_ghi_kwh_per_m2(df: pd.DataFrame) -> float:
    """Sum GHI over the year, returning kWh/m²/yr."""
    return float(df['ghi'].sum() / 1000.0)


if __name__ == '__main__':
    for key, site in SITES.items():
        df = get_or_create_tmy(site, prefer='synthetic')
        annual = annual_ghi_kwh_per_m2(df)
        print(f"{site.name:12s}  rows={len(df)}  annual GHI = {annual:.1f} kWh/m²/yr "
              f"(proposal: {site.annual_ghi_proposal:.0f})")
