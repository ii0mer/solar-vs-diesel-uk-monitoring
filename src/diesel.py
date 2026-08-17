"""Diesel generator model — small standby gensets at very low load.

Two architectures modelled:

  ARCHITECTURE 1: Continuous 24/7 operation
    Smallest practical genset (1 kW class) runs continuously, powering the
    14.6 W load directly through a small AC-DC converter. Genset operates
    at <2% of rated capacity — extreme low-load regime causing wet stacking,
    oil contamination, and accelerated wear.

  ARCHITECTURE 2: Diesel + small battery (intermittent)
    Genset runs ~4 hours/day at ~30% of rated capacity to recharge a small
    lead-acid battery (~1 kWh). Battery powers the load between starts.
    More realistic engineering deployment; better fuel efficiency per kWh
    delivered; less mechanical stress on the engine.

The dissertation reports BOTH and uses Architecture 2 as central case for
the headline LCOE figure, with Architecture 1 as a worst-case sensitivity
bound. This frames the comparison honestly: even on the realistic case,
diesel is uneconomic vs. solar-battery for this load class.

Sources:
- Solent Power (2026) "How much fuel does a generator use?" — industry rule of
  thumb: kVA × 0.25 = L/hr at full load. Available at:
  https://www.solentpower.co.uk/how-much-fuel-does-a-generator-use/
- AHDB (2026) Fuel prices, updated 6 May 2026. UK red diesel 2025 mean: 76.02 ppl
  excl. VAT.
- DESNZ (2025) GHG conversion factors 2025: gas oil = 2.75766 kgCO2e/litre Scope 1.
- Honda EU2200i specifications, manufacturer datasheet (gasoline reference).
- Generic 1 kW diesel manufacturer specs (Hyundai DHY1500SE class).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DieselArchitecture1:
    """Continuous 24/7 diesel — worst-case scenario."""

    # --- Hardware ---
    # No ~1 kW diesel genset is sold; the smallest common air-cooled diesel
    # sets are 2 kVA class (e.g. Yanmar YDG2700N, 2.0 kVA continuous;
    # Hatz 1B-engined sets), i.e. about 1.6 kW at 0.8 pf.
    genset_rated_kw: float = 1.6              # 2 kVA-class air-cooled diesel
    capex_genset_gbp: float = 2200.0          # 2 kVA-class set, ex VAT (assumption;
                                              # a 4.8 kVA Pramac P6000 lists at
                                              # GBP 3,453 ex VAT, 2026)
    capex_fuel_tank_gbp: float = 350.0        # 200 L weatherproof bunded tank
    capex_enclosure_gbp: float = 800.0        # housing, anti-theft, ventilation
    capex_install_gbp: float = 500.0          # site levelling, electrical, commissioning
    capex_acdc_converter_gbp: float = 80.0    # AC→DC converter for the load

    # --- Operating point ---
    # Load is 14.64 W on a 1.6 kW genset = 0.9% of rated load.
    # Fuel at part load from the Skarstein–Uhlen linear model
    # (Skarstein & Uhlen, Wind Engineering 13(2), 1989), the standard
    # part-load fuel curve in the hybrid-systems literature:
    #   F [L/h] = 0.08415 · P_rated [kW] + 0.246 · P_out [kW]
    # At P_rated = 1.6 kW, P_out = 0.01464 kW → 0.138 L/h. The dominant
    # term is the no-load (rated-capacity) intercept — exactly the
    # wet-stacking regime. Cross-check: an Onan QD3200 (3.2 kW) datasheet
    # gives 0.7 L/h at NO load vs 0.27 L/h from S–U, so S–U UNDERSTATES
    # small-set consumption by roughly 25–50%: conservative for the
    # solar-vs-diesel finding.
    su_intercept_l_per_h_per_kw: float = 0.08415
    su_slope_l_per_kwh: float = 0.246
    electrical_load_kw: float = 0.01464
    annual_runtime_hours: int = 8760

    @property
    def fuel_litres_per_hour_at_op_point(self) -> float:
        return (self.su_intercept_l_per_h_per_kw * self.genset_rated_kw
                + self.su_slope_l_per_kwh * self.electrical_load_kw)

    # --- Maintenance ---
    oil_change_interval_hours: int = 100      # frequent due to wet stacking
    cost_per_oil_change_gbp: float = 25.0
    # Industrial gensets achieve 10,000–30,000 h at proper load (interim
    # Table 1); 5,000 h is the derated life under sustained <25–30% load
    # operation (wet stacking, bore glazing — NFPA 110 / Hamilton et al.).
    genset_lifetime_hours: int = 5000         # severe wear at extreme low load

    # --- Logistics (per visit) ---
    fuel_delivery_cost_per_visit_gbp: float = 80.0
    inspection_cost_per_visit_gbp: float = 120.0
    annual_site_visits: int = 12              # monthly for fuel + check

    @property
    def total_capex_gbp(self) -> float:
        return (self.capex_genset_gbp + self.capex_fuel_tank_gbp
                + self.capex_enclosure_gbp + self.capex_install_gbp
                + self.capex_acdc_converter_gbp)

    @property
    def annual_fuel_litres(self) -> float:
        return self.fuel_litres_per_hour_at_op_point * self.annual_runtime_hours

    def annual_fuel_cost_gbp(self, fuel_price_ppl: float) -> float:
        return self.annual_fuel_litres * (fuel_price_ppl / 100.0)

    @property
    def annual_oil_changes(self) -> int:
        return self.annual_runtime_hours // self.oil_change_interval_hours

    @property
    def annual_oil_cost_gbp(self) -> float:
        return self.annual_oil_changes * self.cost_per_oil_change_gbp

    @property
    def annual_visit_cost_gbp(self) -> float:
        return self.annual_site_visits * (
            self.fuel_delivery_cost_per_visit_gbp
            + self.inspection_cost_per_visit_gbp)

    @property
    def genset_replacements_per_year(self) -> float:
        return self.annual_runtime_hours / self.genset_lifetime_hours

    @property
    def annualised_genset_replacement_gbp(self) -> float:
        return self.genset_replacements_per_year * self.capex_genset_gbp


@dataclass
class DieselArchitecture2:
    """Diesel + small battery, intermittent operation — realistic case."""

    # --- Hardware ---
    genset_rated_kw: float = 1.6              # 2 kVA-class air-cooled diesel
    capex_genset_gbp: float = 2200.0
    capex_fuel_tank_gbp: float = 350.0        # 200 L bunded tank (at the OSR 2001 threshold)
    capex_enclosure_gbp: float = 800.0
    capex_install_gbp: float = 500.0
    capex_battery_gbp: float = 400.0          # 200 Ah / 12 V lead-acid (~2.4 kWh nominal),
                                              # sized so the 480 W bulk charge is C/5 (40 A)
    capex_charge_controller_gbp: float = 100.0

    # --- Operating point ---
    # The genset recharges the buffer battery at 30% of rated load
    # (480 W on a 1.6 kW set), the manufacturer minimum-load regime
    # (Kubota/Caterpillar/Cummins 30%; Hatz 25%; Hatz-E/Onan 15%).
    # Runtime is DERIVED from the daily energy balance: the load draws
    # 351.4 Wh/day, delivered through the lead-acid buffer at ~80%
    # round-trip efficiency, so the genset must generate ≈439 Wh/day →
    # 439/480 ≈ 0.91 h/day. Note that at a fixed load fraction the
    # Skarstein–Uhlen annual fuel is independent of the rating (the
    # per-hour rate scales with P_rated, the hours inversely).
    charge_power_kw: float = 0.48             # 30% of 1.6 kW
    daily_load_wh: float = 351.36             # 14.64 W × 24 h
    battery_path_efficiency: float = 0.80     # PbA round-trip, PSoC duty
    # Skarstein–Uhlen fuel at 30% load, 1.6 kW machine:
    # 0.08415·1.6 + 0.246·0.48 = 0.253 L/h.
    su_intercept_l_per_h_per_kw: float = 0.08415
    su_slope_l_per_kwh: float = 0.246

    @property
    def daily_runtime_hours(self) -> float:
        gen_wh_needed = self.daily_load_wh / self.battery_path_efficiency
        return gen_wh_needed / (self.charge_power_kw * 1000.0)

    @property
    def fuel_litres_per_hour_at_op_point(self) -> float:
        return (self.su_intercept_l_per_h_per_kw * self.genset_rated_kw
                + self.su_slope_l_per_kwh * self.charge_power_kw)

    @property
    def annual_runtime_hours(self) -> int:
        return int(self.daily_runtime_hours * 365)

    # --- Maintenance ---
    oil_change_interval_hours: int = 250      # Hatz 1B / Onan QD: 250 h or 12 months
    cost_per_oil_change_gbp: float = 25.0
    genset_lifetime_hours: int = 8000         # assumption (no manufacturer hours-life found)

    # --- Battery replacement ---
    battery_lifetime_years: float = 4.0       # lead-acid daily cycling

    # --- Logistics ---
    fuel_delivery_cost_per_visit_gbp: float = 80.0
    inspection_cost_per_visit_gbp: float = 120.0
    annual_site_visits: int = 12

    @property
    def total_capex_gbp(self) -> float:
        return (self.capex_genset_gbp + self.capex_fuel_tank_gbp
                + self.capex_enclosure_gbp + self.capex_install_gbp
                + self.capex_battery_gbp + self.capex_charge_controller_gbp)

    @property
    def annual_fuel_litres(self) -> float:
        return self.fuel_litres_per_hour_at_op_point * self.annual_runtime_hours

    def annual_fuel_cost_gbp(self, fuel_price_ppl: float) -> float:
        return self.annual_fuel_litres * (fuel_price_ppl / 100.0)

    @property
    def annual_oil_changes(self) -> int:
        # manufacturer interval in hours, with the manuals' 12-month floor
        return max(1, self.annual_runtime_hours // self.oil_change_interval_hours)

    @property
    def annual_oil_cost_gbp(self) -> float:
        return self.annual_oil_changes * self.cost_per_oil_change_gbp

    @property
    def annual_visit_cost_gbp(self) -> float:
        return self.annual_site_visits * (
            self.fuel_delivery_cost_per_visit_gbp
            + self.inspection_cost_per_visit_gbp)

    @property
    def genset_replacements_per_year(self) -> float:
        return self.annual_runtime_hours / self.genset_lifetime_hours

    @property
    def annualised_genset_replacement_gbp(self) -> float:
        return self.genset_replacements_per_year * self.capex_genset_gbp

    @property
    def annualised_battery_replacement_gbp(self) -> float:
        return self.capex_battery_gbp / self.battery_lifetime_years


# --- DEFRA 2025 gas oil emission factor (Scope 1, kgCO2e per litre) ---
DEFRA_2025_GAS_OIL_KGCO2E_PER_LITRE = 2.75766


def annual_co2e_kg(arch, fuel_factor: float = DEFRA_2025_GAS_OIL_KGCO2E_PER_LITRE) -> float:
    """Annual CO2e from diesel combustion (Scope 1 only)."""
    return arch.annual_fuel_litres * fuel_factor


def lifetime_summary(arch, project_years: int = 25,
                     fuel_price_ppl: float = 76.02) -> dict:
    """Compute a lifetime-cost summary for a diesel architecture.

    Pure cash-flow, no discounting (that happens in economics.py).
    """
    annual_fuel_cost = arch.annual_fuel_cost_gbp(fuel_price_ppl)
    annual_oil = arch.annual_oil_cost_gbp
    annual_visits = arch.annual_visit_cost_gbp
    annual_genset_repl = arch.annualised_genset_replacement_gbp
    annual_battery_repl = getattr(arch, 'annualised_battery_replacement_gbp', 0.0)

    annual_opex = (annual_fuel_cost + annual_oil + annual_visits
                   + annual_genset_repl + annual_battery_repl)
    lifetime_opex = annual_opex * project_years
    lifetime_total_cost = arch.total_capex_gbp + lifetime_opex

    annual_co2 = annual_co2e_kg(arch)
    lifetime_co2 = annual_co2 * project_years

    return {
        'capex_gbp': arch.total_capex_gbp,
        'annual_fuel_litres': arch.annual_fuel_litres,
        'annual_fuel_cost_gbp': annual_fuel_cost,
        'annual_oil_cost_gbp': annual_oil,
        'annual_visit_cost_gbp': annual_visits,
        'annual_genset_replacement_gbp': annual_genset_repl,
        'annual_battery_replacement_gbp': annual_battery_repl,
        'annual_opex_gbp': annual_opex,
        'lifetime_opex_gbp': lifetime_opex,
        'lifetime_total_cost_gbp': lifetime_total_cost,
        'annual_co2e_kg': annual_co2,
        'lifetime_co2e_kg': lifetime_co2,
        'lifetime_co2e_tonnes': lifetime_co2 / 1000.0,
    }


if __name__ == '__main__':
    arch1 = DieselArchitecture1()
    arch2 = DieselArchitecture2()

    s1 = lifetime_summary(arch1, fuel_price_ppl=76.02)
    s2 = lifetime_summary(arch2, fuel_price_ppl=76.02)

    print("ARCHITECTURE 1: Continuous 24/7 (worst case)")
    print(f"  CapEx                  : £{s1['capex_gbp']:>8,.0f}")
    print(f"  Annual fuel            : {s1['annual_fuel_litres']:>8.0f} L")
    print(f"  Annual fuel cost       : £{s1['annual_fuel_cost_gbp']:>8,.0f}")
    print(f"  Annual oil             : £{s1['annual_oil_cost_gbp']:>8,.0f}")
    print(f"  Annual site visits     : £{s1['annual_visit_cost_gbp']:>8,.0f}")
    print(f"  Annual genset repl     : £{s1['annual_genset_replacement_gbp']:>8,.0f}")
    print(f"  Annual OpEx total      : £{s1['annual_opex_gbp']:>8,.0f}")
    print(f"  25-year total cost     : £{s1['lifetime_total_cost_gbp']:>8,.0f}")
    print(f"  25-year CO2e           : {s1['lifetime_co2e_tonnes']:>8.1f} t")

    print("\nARCHITECTURE 2: Diesel + battery (realistic)")
    print(f"  CapEx                  : £{s2['capex_gbp']:>8,.0f}")
    print(f"  Annual fuel            : {s2['annual_fuel_litres']:>8.0f} L")
    print(f"  Annual fuel cost       : £{s2['annual_fuel_cost_gbp']:>8,.0f}")
    print(f"  Annual oil             : £{s2['annual_oil_cost_gbp']:>8,.0f}")
    print(f"  Annual site visits     : £{s2['annual_visit_cost_gbp']:>8,.0f}")
    print(f"  Annual genset repl     : £{s2['annual_genset_replacement_gbp']:>8,.0f}")
    print(f"  Annual battery repl    : £{s2['annual_battery_replacement_gbp']:>8,.0f}")
    print(f"  Annual OpEx total      : £{s2['annual_opex_gbp']:>8,.0f}")
    print(f"  25-year total cost     : £{s2['lifetime_total_cost_gbp']:>8,.0f}")
    print(f"  25-year CO2e           : {s2['lifetime_co2e_tonnes']:>8.1f} t")
