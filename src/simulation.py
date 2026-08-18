"""Hourly energy balance simulation for an off-grid PV-battery system.

For each of 8,760 hours we compute:
    PV generation → DC bus
    Load draw    ← DC bus
    Battery: charge if surplus, discharge if deficit
    Track unmet load (hours where battery couldn't cover deficit)
    Track curtailed energy (PV output that couldn't be stored)

Loss-of-Load Probability (LOLP) = (hours unmet) / 8760.
Capacity factor and self-consumption metrics also returned.

This is vectorised where possible, but the battery state is sequential
so the SoC loop is a Python for-loop. 8,760 iterations runs in <100ms.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from .pv_model import PVDesign, simulate_pv_dc
from .battery import BatteryDesign
from .load_profile import LoadProfile
from .sites import Site


@dataclass
class SimulationResult:
    """Container for one year's simulation outputs."""
    site_name: str
    pv_design: PVDesign
    battery_design: BatteryDesign
    load_profile: LoadProfile
    hourly: pd.DataFrame = field(repr=False)  # full timeseries

    # Aggregate metrics
    annual_pv_kwh: float = 0.0
    annual_load_kwh: float = 0.0
    annual_unmet_kwh: float = 0.0
    annual_curtailed_kwh: float = 0.0
    lolp: float = 0.0                          # fraction of hours
    self_consumption: float = 0.0              # frac of PV that served load
    self_sufficiency: float = 0.0              # frac of load served by PV
    final_soc_kwh: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.site_name}\n"
            f"  PV       : {self.pv_design.total_nameplate_w:.0f} Wp\n"
            f"  Battery  : {self.battery_design.capacity_kwh:.2f} kWh "
            f"(usable {self.battery_design.usable_kwh:.2f})\n"
            f"  PV gen   : {self.annual_pv_kwh:.1f} kWh/yr\n"
            f"  Load     : {self.annual_load_kwh:.1f} kWh/yr\n"
            f"  Unmet    : {self.annual_unmet_kwh:.2f} kWh/yr  "
            f"(LOLP = {self.lolp*100:.3f}%)\n"
            f"  Curtail  : {self.annual_curtailed_kwh:.1f} kWh/yr\n"
            f"  Self-cons: {self.self_consumption*100:.1f}%\n"
            f"  Self-suff: {self.self_sufficiency*100:.1f}%\n"
        )


def run_simulation(weather: pd.DataFrame,
                   site: Site,
                   pv: PVDesign,
                   battery: BatteryDesign,
                   load: LoadProfile,
                   load_multiplier: float = 1.0,
                   initial_soc_frac: float = 0.5,
                   pv_ageing_factor: float = 1.0,
                   charge_temp_limit_c: float | None = None,
                   pv_series_w: Optional[pd.Series] = None,
                   battery_soh: float = 1.0
                   ) -> SimulationResult:
    """Run an 8,760-hour energy balance.

    Parameters
    ----------
    initial_soc_frac : float
        Starting SoC as a fraction of NAMEPLATE capacity (capped at the
        SoH-reduced ceiling). Default 0.5 (mid-charge).
        For sizing studies we typically run two passes — the second uses the
        first pass's final SoC as initial — to remove start-condition bias.
    pv_ageing_factor : float
        Multiplies PV output to represent module degradation at a given
        service year, e.g. (1 - 0.005)**24 for year-25 output at 0.5%/yr.
        Sizing is performed at end-of-life so the LOLP criterion holds for
        the whole project life, not only year 1.
    charge_temp_limit_c : float | None
        If set (e.g. 0.0), battery CHARGING is disabled in hours where
        ambient temperature is at or below this limit — a conservative
        proxy for the LFP low-temperature charging restriction (ambient
        used in place of enclosure temperature; discharge unaffected).
        None (default) disables the constraint; the delta between runs
        bounds the effect for the Limitations section.
    pv_series_w : pd.Series, optional
        Precomputed hourly PV output in W (post-loss-chain). PVWatts DC is
        linear in nameplate, so sizing sweeps compute the chain once per
        site at a reference size and scale — this skips the pvlib chain.
    battery_soh : float
        Battery state of health as a fraction of nameplate capacity
        (1.0 = new; 0.8 = end of warranty life). The usable window scales
        with SoH; the DoD floor fraction is unchanged.
    """
    # Hourly PV in W → convert to Wh per hour (1h timestep)
    pv_w = pv_series_w if pv_series_w is not None \
        else simulate_pv_dc(weather, site, pv)
    pv_wh = pv_w.values * pv_ageing_factor  # 1-hour timestep: W·1h = Wh

    # Load in W → Wh
    load_wh = load.hourly_series(weather.index, multiplier=load_multiplier).values

    # Battery state in Wh
    cap_wh = battery.capacity_kwh * 1000.0 * battery_soh
    soc_min_wh = cap_wh * (1.0 - battery.max_dod)   # floor
    soc_max_wh = cap_wh                              # ceiling
    # initial_soc_frac is a fraction of NAMEPLATE capacity (callers pass
    # final_soc_kwh / capacity_kwh from a settling pass), capped at the
    # SoH-reduced ceiling so a second pass starts exactly where the first
    # pass ended.
    soc_wh = min(cap_wh, battery.capacity_kwh * 1000.0 * initial_soc_frac)
    eta_chg = np.sqrt(battery.round_trip_eff)        # split RT eff
    eta_dis = np.sqrt(battery.round_trip_eff)
    self_dis_h = battery.hourly_self_discharge_frac

    n = len(pv_wh)
    out_soc = np.empty(n)
    out_unmet = np.empty(n)
    out_curtail = np.empty(n)
    out_charge = np.empty(n)
    out_discharge = np.empty(n)

    if charge_temp_limit_c is not None:
        charge_blocked = (weather['temp_air'].values <= charge_temp_limit_c)
    else:
        charge_blocked = np.zeros(n, dtype=bool)

    for i in range(n):
        # 1. Self-discharge first (always)
        soc_wh *= (1.0 - self_dis_h)

        # 2. Net energy at the bus this hour
        net_wh = pv_wh[i] - load_wh[i]

        unmet = 0.0
        curtail = 0.0
        charged = 0.0
        discharged = 0.0

        if net_wh >= 0:
            # Surplus → charge battery (unless cold-charge-blocked)
            headroom = 0.0 if charge_blocked[i] else (soc_max_wh - soc_wh)
            energy_to_battery = min(net_wh, headroom / eta_chg)
            soc_wh += energy_to_battery * eta_chg
            charged = energy_to_battery * eta_chg
            curtail = net_wh - energy_to_battery
        else:
            # Deficit → discharge battery
            deficit = -net_wh
            available = max(0.0, soc_wh - soc_min_wh) * eta_dis
            energy_from_battery = min(deficit, available)
            soc_wh -= energy_from_battery / eta_dis
            discharged = energy_from_battery
            unmet = deficit - energy_from_battery

        out_soc[i] = soc_wh
        out_unmet[i] = unmet
        out_curtail[i] = curtail
        out_charge[i] = charged
        out_discharge[i] = discharged

    hourly = pd.DataFrame({
        'pv_wh': pv_wh,
        'load_wh': load_wh,
        'soc_wh': out_soc,
        'soc_frac': out_soc / cap_wh,
        'unmet_wh': out_unmet,
        'curtail_wh': out_curtail,
        'charge_wh': out_charge,
        'discharge_wh': out_discharge,
    }, index=weather.index)

    annual_pv = hourly['pv_wh'].sum() / 1000.0
    annual_load = hourly['load_wh'].sum() / 1000.0
    annual_unmet = hourly['unmet_wh'].sum() / 1000.0
    annual_curtail = hourly['curtail_wh'].sum() / 1000.0
    hours_unmet = (hourly['unmet_wh'] > 0).sum()

    pv_to_load = max(0.0, annual_pv - annual_curtail)
    self_cons = pv_to_load / annual_pv if annual_pv > 0 else 0.0
    served = annual_load - annual_unmet
    self_suff = served / annual_load if annual_load > 0 else 0.0

    return SimulationResult(
        site_name=site.name,
        pv_design=pv,
        battery_design=battery,
        load_profile=load,
        hourly=hourly,
        annual_pv_kwh=annual_pv,
        annual_load_kwh=annual_load,
        annual_unmet_kwh=annual_unmet,
        annual_curtailed_kwh=annual_curtail,
        lolp=hours_unmet / n,
        self_consumption=self_cons,
        self_sufficiency=self_suff,
        final_soc_kwh=soc_wh / 1000.0,
    )


if __name__ == '__main__':
    from .weather import get_or_create_tmy
    from .sites import SITES

    pv = PVDesign(nameplate_w=200, n_modules=1)   # 200 Wp starter
    bat = BatteryDesign(capacity_kwh=2.0)         # 2 kWh starter
    load = LoadProfile()

    print(f"Test config: {pv.total_nameplate_w:.0f} Wp PV + "
          f"{bat.capacity_kwh:.1f} kWh battery, load {load.daily_energy_wh():.0f} Wh/day\n")

    for key, site in SITES.items():
        w = get_or_create_tmy(site, prefer='synthetic')
        # Two-pass: settle the battery
        r1 = run_simulation(w, site, pv, bat, load)
        r2 = run_simulation(w, site, pv, bat, load,
                            initial_soc_frac=r1.final_soc_kwh / bat.capacity_kwh)
        print(r2.summary())
