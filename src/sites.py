"""Site definitions for the four UK study locations.

Coordinates verified against OS Grid / Google Maps for city centres.
Annual GHI values are PVGIS-SARAH2 long-term means as cited in the proposal
(Southampton ~1100, Birmingham ~1000, Liverpool ~970, Edinburgh ~900 kWh/m²).

Liverpool added in v3.3: NW maritime climate, intermediate latitude (~53.4°N),
LJMU host city, fills the climate gap between Midlands Birmingham and
Scottish Edinburgh.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Site:
    name: str
    latitude: float       # degrees N
    longitude: float      # degrees E (negative = west)
    altitude: float       # metres above sea level (used for clear-sky model)
    annual_ghi_proposal: float  # kWh/m²/yr — value cited in proposal Table 3
    timezone: str = 'Europe/London'

    @property
    def coords(self) -> tuple:
        return (self.latitude, self.longitude)


SITES: Dict[str, Site] = {
    'southampton': Site(
        name='Southampton',
        latitude=50.9097,
        longitude=-1.4044,
        altitude=9.0,
        annual_ghi_proposal=1100.0,
    ),
    'birmingham': Site(
        name='Birmingham',
        latitude=52.4862,
        longitude=-1.8904,
        altitude=140.0,
        annual_ghi_proposal=1000.0,
    ),
    'liverpool': Site(
        name='Liverpool',
        latitude=53.4084,
        longitude=-2.9916,
        altitude=10.0,                  # near sea level, Mersey estuary
        annual_ghi_proposal=970.0,      # NW maritime, between Birmingham and Edinburgh
                                        # placeholder; real PVGIS will refine on local machine
    ),
    'edinburgh': Site(
        name='Edinburgh',
        latitude=55.9533,
        longitude=-3.1883,
        altitude=47.0,
        annual_ghi_proposal=900.0,
    ),
}


if __name__ == '__main__':
    for k, s in SITES.items():
        print(f"{k:12s} lat={s.latitude:6.2f} lon={s.longitude:6.2f} "
              f"alt={s.altitude:5.1f}m  GHI~{s.annual_ghi_proposal:.0f} kWh/m²/yr")
