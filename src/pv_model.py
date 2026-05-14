"""PV system model — thin wrapper around pvlib.ModelChain.

For an off-grid monitoring station the array is small (50–500 W) and DC-coupled
through a charge controller to the battery. We don't need a full inverter
model for the AC side here because the load is mostly DC-native (router,
logger, sensors). For methodological cleanliness we still simulate the AC
side and use a 92% DC-DC converter efficiency on the load.

Key choices and why:
- Tilt = latitude (rule of thumb for year-round off-grid; we may revisit).
- Azimuth = 180° (true south).
- Module: representative 400 W mono-Si from CEC database. Final dissertation
  will fix specific commercial module + cite datasheet.
- Temperature model: SAPM open-rack glass/glass.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pvlib
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.location import Location

from .sites import Site


@dataclass
class PVDesign:
    """Sizing decisions for a PV array."""
    nameplate_w: float = 400.0          # single-module nominal at STC
    n_modules: int = 1
    tilt_deg: float | None = None       # None → use site latitude
    azimuth_deg: float = 180.0          # south-facing
    module_efficiency: float = 0.20     # for area calc only

    @property
    def total_nameplate_w(self) -> float:
        return self.nameplate_w * self.n_modules

    @property
    def total_area_m2(self) -> float:
        # nameplate / (efficiency * 1000 W/m² STC irradiance)
        return self.total_nameplate_w / (self.module_efficiency * 1000.0)


def _generic_module_params(nameplate_w: float) -> dict:
    """Return CEC-style module params for a generic mono-Si module.

    Values below are representative of a modern ~400 W residential module.
    For the dissertation we should pick a real commercial module from the CEC
    database (`pvlib.pvsystem.retrieve_sam('CECMod')`) and cite it.
    """
    return {
        'pdc0': nameplate_w,
        'gamma_pdc': -0.0035,           # -0.35%/°C, typical mono-Si
    }


def _generic_inverter_params(ac_w: float) -> dict:
    return {
        'pdc0': ac_w / 0.96,            # DC capacity feeding inverter
        'eta_inv_nom': 0.96,
    }


def simulate_pv_dc(weather: pd.DataFrame, site: Site,
                   design: PVDesign) -> pd.Series:
    """Return hourly DC power output of the PV array, in watts.

    We use the simpler PVWatts-style model (pvlib.pvsystem.pvwatts_dc) because
    it's the right level of detail for a system-level techno-economic study
    and avoids over-fitting to a specific module's I-V curve. The dissertation
    methodology section should justify this choice explicitly.
    """
    tilt = design.tilt_deg if design.tilt_deg is not None else site.latitude
    location = Location(latitude=site.latitude, longitude=site.longitude,
                        altitude=site.altitude, tz='UTC')

    # Solar position
    times = weather.index
    sp = location.get_solarposition(times)

    # Plane-of-array irradiance via Hay-Davies transposition.
    # dni_extra (top-of-atmosphere) is required for anisotropic transposition.
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=design.azimuth_deg,
        solar_zenith=sp['apparent_zenith'],
        solar_azimuth=sp['azimuth'],
        dni=weather['dni'],
        ghi=weather['ghi'],
        dhi=weather['dhi'],
        dni_extra=dni_extra,
        model='haydavies',
    )

    # Cell temperature (SAPM open-rack glass-glass)
    cell_temp = pvlib.temperature.sapm_cell(
        poa_global=poa['poa_global'],
        temp_air=weather['temp_air'],
        wind_speed=weather['wind_speed'],
        a=-3.47, b=-0.0594, deltaT=3,   # open-rack glass-glass
    )

    # PVWatts DC
    p_dc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa['poa_global'],
        temp_cell=cell_temp,
        pdc0=design.total_nameplate_w,
        gamma_pdc=-0.0035,
    )
    p_dc = p_dc.fillna(0.0).clip(lower=0.0)
    p_dc.name = 'pv_dc_w'
    return p_dc


def annual_yield_kwh_per_kwp(p_dc_w: pd.Series, design: PVDesign) -> float:
    """Specific yield: kWh per installed kWp per year. UK typical: 850–1050."""
    annual_kwh = p_dc_w.sum() / 1000.0
    kwp = design.total_nameplate_w / 1000.0
    return annual_kwh / kwp


if __name__ == '__main__':
    from .weather import get_or_create_tmy
    from .sites import SITES

    design = PVDesign(nameplate_w=400, n_modules=1)
    print(f"Array: {design.total_nameplate_w:.0f} Wp, ~{design.total_area_m2:.1f} m²\n")

    for key, site in SITES.items():
        w = get_or_create_tmy(site, prefer='synthetic')
        p = simulate_pv_dc(w, site, design)
        print(f"{site.name:12s}  annual = {p.sum()/1000:.1f} kWh  "
              f"specific yield = {annual_yield_kwh_per_kwp(p, design):.1f} kWh/kWp")
