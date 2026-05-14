"""Battery state-of-charge tracker.

Simple energy bucket model — appropriate for system sizing studies. Not
a cell-level electrochemical model. Captures: capacity, round-trip
efficiency, depth-of-discharge limit, self-discharge, calendar/cycle
ageing as linear fade.

Defaults are LFP (lithium iron phosphate) — currently the dominant
chemistry for stationary storage per BNEF 2025. NMC (nickel-manganese-cobalt)
parameters are also recorded so we can sensitivity-test chemistry choice.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BatteryDesign:
    """Battery sizing and performance parameters."""
    capacity_kwh: float = 1.0           # nameplate energy
    chemistry: str = 'LFP'              # 'LFP' or 'NMC'
    round_trip_eff: float = 0.92        # AC→AC, includes converter losses
    max_dod: float = 0.80               # max depth of discharge per cycle
    self_discharge_pct_month: float = 2.0
    end_of_life_capacity: float = 0.80  # cap fade after warranty period
    cycle_life: int = 5000              # to 80% capacity, LFP

    @property
    def usable_kwh(self) -> float:
        return self.capacity_kwh * self.max_dod

    @property
    def hourly_self_discharge_frac(self) -> float:
        # Convert % per month to fraction per hour
        # 30 days * 24 h = 720 h
        return (self.self_discharge_pct_month / 100.0) / 720.0


def chemistry_presets() -> dict:
    """Reference values for sensitivity analysis."""
    return {
        'LFP': dict(round_trip_eff=0.92, max_dod=0.90, cycle_life=5000,
                    cost_per_kwh_usd=70),  # BNEF 2025 stationary pack
        'NMC': dict(round_trip_eff=0.94, max_dod=0.80, cycle_life=3000,
                    cost_per_kwh_usd=108),  # BNEF 2025 average pack
    }
