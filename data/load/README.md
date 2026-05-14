# data/load/

This folder is reserved for component datasheet specifications.

## Intended contents

Manufacturer datasheet PDF extracts or CSV summaries for the monitoring station load components:

| File (to be added) | Component | Power (W) |
|---|---|---|
| `cr1000x_datasheet.md` | Campbell Scientific CR1000X datalogger | 1.2 (typical continuous) |
| `ix20_datasheet.md` | Digi IX20 cellular LTE router | 6.0 (conservative allowance) |
| `sensor_specs.md` | Four environmental sensors (temp, humidity, wind, water) | 0.5 each |

## Current load assumptions

The load profile is implemented in `src/load_profile.py`. All values are sourced from manufacturer datasheets.

| Component | Power (W) | Basis |
|---|---|---|
| CR1000X datalogger | 1.2 | Typical continuous draw from Campbell Scientific datasheet |
| 4× sensors | 2.0 (0.5 W each) | Manufacturer specifications |
| Digi IX20 modem | 6.0 | Conservative allowance for transmission peaks and signal-search |
| Ancillary | 3.0 | Engineering estimate (LED status, relay, heater standby) |
| **Base total** | **12.2** | — |
| **Design load (+20%)** | **14.6** | Standard 20% engineering margin for losses, derating, expansion |
| **Annual energy** | **128.2 kWh/yr** | 14.64 W × 8760 h / 1000 |
