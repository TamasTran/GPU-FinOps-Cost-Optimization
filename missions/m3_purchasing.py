"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []

    # Extension 5 tracking: interruptible jobs power & carbon
    interruptible_kwh_monthly = 0.0

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # Track energy for interruptible jobs
        if interruptible:
            watts = num(c.get("watts", 500))
            interruptible_kwh_monthly += (gpu_hours * watts) / 1000.0

        # Extension 1 specific interruption rates
        irate = 0.03 if gtype in ("H100", "H200", "B200") else 0.08
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od, interrupt_rate=irate)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: Multi-region carbon and electricity economics for flexible jobs
    carbon_by_region = {}
    for region, gco2_kwh in sustainability.REGION_CARBON.items():
        co2_kg = (interruptible_kwh_monthly * gco2_kwh) / 1000.0
        elec_cost = interruptible_kwh_monthly * sustainability.REGION_PRICE_KWH.get(region, 0.12)
        carbon_by_region[region] = {
            "gco2_kwh": gco2_kwh,
            "carbon_kg": round(co2_kg, 1),
            "electricity_cost_usd": round(elec_cost, 2),
            "price_kwh": sustainability.REGION_PRICE_KWH.get(region, 0.12),
        }

    base_co2_kg = carbon_by_region["us-east-1"]["carbon_kg"]
    green_co2_kg = carbon_by_region["europe-north1"]["carbon_kg"]
    carbon_saved_kg = base_co2_kg - green_co2_kg
    carbon_saved_pct = (carbon_saved_kg / base_co2_kg * 100) if base_co2_kg else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n[Extension 5] Carbon-Aware Scheduling for Interruptible Training Workloads:")
        print(f"  Total interruptible compute energy: {interruptible_kwh_monthly:,.1f} kWh/month")
        print(f"{'Region':18}{'gCO2/kWh':>10}{'Electricity ($/kWh)':>22}{'Monthly Carbon (kg)':>22}{'Elec Cost ($)':>15}")
        for reg, data in sorted(carbon_by_region.items(), key=lambda x: x[1]["gco2_kwh"]):
            print(f"  {reg:16}{data['gco2_kwh']:>10}  ${data['price_kwh']:>20.3f}{data['carbon_kg']:>20,.1f} kg  ${data['electricity_cost_usd']:>12,.2f}")
        print(f"\n  Impact of scheduling to europe-north1: Saves {carbon_saved_kg:,.1f} kg CO2e/month ({carbon_saved_pct:.1f}% carbon reduction)")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "carbon_scheduling": {
            "interruptible_kwh_monthly": round(interruptible_kwh_monthly, 1),
            "carbon_by_region": carbon_by_region,
            "carbon_saved_kg": round(carbon_saved_kg, 1),
            "carbon_saved_pct": round(carbon_saved_pct, 1),
        }
    }


if __name__ == "__main__":
    run()
