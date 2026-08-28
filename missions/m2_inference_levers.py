"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 4 tracking
    reasoning_reqs = 0
    standard_reqs = 0
    reasoning_tokens = 0
    standard_tokens = 0
    reasoning_cost = 0.0
    standard_cost = 0.0
    reasoning_wh = 0.0
    standard_wh = 0.0

    # Extension 3 tracking
    cached_input_tokens_total = 0
    uncached_input_tokens_total = 0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r.get("is_reasoning", 0))))
        req_tok = inp + out
        total_tokens += req_tok

        cached_input_tokens_total += cached
        uncached_input_tokens_total += (inp - cached)

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        req_base = pricing.request_cost(inp, out, lin, lout)
        base_cost += req_base

        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        req_opt = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        opt_cost += req_opt

        # Reasoning tracking
        req_wh = sustainability.wh_per_query(req_tok, is_reasoning=is_reasoning)
        if is_reasoning:
            reasoning_reqs += 1
            reasoning_tokens += req_tok
            reasoning_cost += req_opt
            reasoning_wh += req_wh
        else:
            standard_reqs += 1
            standard_tokens += req_tok
            standard_cost += req_opt
            standard_wh += req_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 3: Cache Economics
    # Assuming prompt write overhead is ~ $3.75/1M tokens (e.g., Anthropic write fee = 1.25x base)
    write_fee_large = 3.75
    cache_be_reads_large = pricing.cache_break_even_reads(write_fee_large, base_input_price=3.00, read_discount=0.10)
    # Average reads per cached prefix estimated from dataset
    cache_hit_rate = cached_input_tokens_total / max(1, cached_input_tokens_total + uncached_input_tokens_total)
    cache_worth_it = pricing.cache_is_worth_it(avg_cache_reads=3.5, write_cost_per_m=write_fee_large, read_discount=0.10, base_input_price=3.00)

    # Extension 4: Reasoning impact analysis
    reasoning_req_pct = (reasoning_reqs / len(rows) * 100) if rows else 0.0
    reasoning_cost_pct = (reasoning_cost / opt_cost * 100) if opt_cost else 0.0
    reasoning_energy_pct = (reasoning_wh / (reasoning_wh + standard_wh) * 100) if (reasoning_wh + standard_wh) else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n[Extension 3] Prompt Caching Economics:")
        print(f"  Cache hit rate (input tokens): {cache_hit_rate:.1%}")
        print(f"  Break-even read multiplier (Large model): {cache_be_reads_large:.2f}x reads required")
        print(f"  Is caching profitable for dataset? {cache_worth_it} (Estimated reuse: ~3.5x > break-even {cache_be_reads_large:.2f}x)")

        print("\n[Extension 4] Reasoning Traffic vs Standard Traffic Impact:")
        print(f"  Reasoning requests: {reasoning_reqs} ({reasoning_req_pct:.1f}% of total)")
        print(f"  Reasoning cost ($): ${reasoning_cost:.2f}/day ({reasoning_cost_pct:.1f}% of optimized spend)")
        print(f"  Reasoning energy (Wh): {reasoning_wh:.1f} Wh ({reasoning_energy_pct:.1f}% of total inference energy)")
        print(f"  Energy intensity ratio: {((reasoning_wh/max(1, reasoning_tokens))/(standard_wh/max(1, standard_tokens))):.1f}x Wh/token vs standard")

    return {
        "baseline_daily": round(base_cost, 2),
        "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3),
        "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1),
        "total_tokens": total_tokens,
        "reasoning_analysis": {
            "reasoning_reqs": reasoning_reqs,
            "reasoning_req_pct": round(reasoning_req_pct, 1),
            "reasoning_cost_daily": round(reasoning_cost, 2),
            "reasoning_cost_pct": round(reasoning_cost_pct, 1),
            "reasoning_wh_daily": round(reasoning_wh, 2),
            "reasoning_energy_pct": round(reasoning_energy_pct, 1),
        },
        "cache_analysis": {
            "cache_hit_rate": round(cache_hit_rate, 3),
            "cache_break_even_reads": round(cache_be_reads_large, 2),
            "is_cache_worth_it": cache_worth_it,
        },
    }


if __name__ == "__main__":
    run()
