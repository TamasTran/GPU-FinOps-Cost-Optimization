"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # Extension 2: Right-sizing analysis for memory-bound GPUs
    mbu_rightsizing = []
    for lie in lies:
        cur_type = lie["gpu_type"]
        cur_info = cat[cur_type]
        cur_od = num(cur_info["on_demand_hr"])
        cur_bw = num(cur_info["peak_bw_tbs"])
        cur_vram = num(cur_info["hbm_gb"])
        achieved_bw = lie["mbu"] * cur_bw

        # Search catalog for cheaper GPU meeting achieved memory bandwidth & capacity
        candidates = []
        for gname, ginfo in cat.items():
            god = num(ginfo["on_demand_hr"])
            gbw = num(ginfo["peak_bw_tbs"])
            gvram = num(ginfo["hbm_gb"])
            if god < cur_od and gbw >= achieved_bw * 0.9 and gvram >= 24:
                candidates.append((god, gname, gbw, gvram))
        if candidates:
            best_od, best_name, best_bw, best_vram = min(candidates, key=lambda x: x[0])
            monthly_savings = (cur_od - best_od) * 24 * 30
            mbu_rightsizing.append({
                "gpu_id": lie["gpu_id"],
                "current_type": cur_type,
                "current_cost_hr": cur_od,
                "recommended_type": best_name,
                "recommended_cost_hr": best_od,
                "achieved_bw_tbs": round(achieved_bw, 2),
                "target_peak_bw_tbs": best_bw,
                "monthly_savings_usd": round(monthly_savings, 2),
            })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        if mbu_rightsizing:
            print("\n[Extension 2] MBU Right-Sizing Recommendations for Memory-Bound GPUs:")
            print(f"{'GPU':14}{'Current':9}{'Target':9}{'Achieved BW':>13}{'Monthly Savings':>18}")
            for r in mbu_rightsizing:
                print(f"{r['gpu_id']:14}{r['current_type']:9}{r['recommended_type']:9}{r['achieved_bw_tbs']:>10.2f} TB/s ${r['monthly_savings_usd']:>15,.2f}")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "mbu_rightsizing": mbu_rightsizing,
    }


if __name__ == "__main__":
    run()
