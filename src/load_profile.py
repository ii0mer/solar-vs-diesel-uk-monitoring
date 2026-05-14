"""Load profile for a representative remote industrial monitoring station.

Components and powers from your proposal Table 2 — values are continuous
average draws, not nameplate. All components run 24/7 (monitoring is by
definition non-flexible).

Sources for the values:
- Campbell Scientific CR1000X: ~1.2 W typical, full datasheet to be added to data/load/
- Digi IX20 cellular router: ~6.0 W with periodic transmission peaks
- Environmental sensors (temp, humidity, wind): ~0.5 W each, x4
- Ancillary (status LED, relay, heater standby): ~3.0 W

Total nominal: 12 W continuous → 288 Wh/day raw → ~351 Wh/day with 20% margin.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class LoadProfile:
    """Constant 24/7 monitoring load.

    The proposal commits to a non-flexible profile; we keep that.
    Parameters are exposed so we can sensitivity-test ±20% on total demand later.
    """
    datalogger_w: float = 1.2
    sensors_w_each: float = 0.5
    sensor_count: int = 4
    modem_w: float = 6.0
    ancillary_w: float = 3.0
    margin_factor: float = 1.20   # +20% for losses, derate, future-proofing

    def base_power_w(self) -> float:
        """Average continuous power draw in watts, before margin."""
        return (self.datalogger_w
                + self.sensors_w_each * self.sensor_count
                + self.modem_w
                + self.ancillary_w)

    def design_power_w(self) -> float:
        """Power including margin — what we size against."""
        return self.base_power_w() * self.margin_factor

    def daily_energy_wh(self) -> float:
        return self.design_power_w() * 24.0

    def annual_energy_kwh(self) -> float:
        return self.daily_energy_wh() * 365 / 1000.0

    def hourly_series(self, index: pd.DatetimeIndex,
                      multiplier: float = 1.0) -> pd.Series:
        """Return hourly load (in W) as a pandas Series aligned to `index`.

        Parameters
        ----------
        multiplier : float
            Scale factor for sensitivity analysis (e.g. 0.8 / 1.0 / 1.2).
        """
        p = self.design_power_w() * multiplier
        return pd.Series(p, index=index, name='load_w')


def summary() -> str:
    lp = LoadProfile()
    return (
        f"Base draw         : {lp.base_power_w():.2f} W\n"
        f"With +20% margin  : {lp.design_power_w():.2f} W\n"
        f"Daily energy      : {lp.daily_energy_wh():.1f} Wh\n"
        f"Annual energy     : {lp.annual_energy_kwh():.1f} kWh\n"
    )


if __name__ == '__main__':
    print(summary())
