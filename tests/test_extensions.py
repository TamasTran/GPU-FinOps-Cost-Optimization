import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, metrics, sustainability


def test_extension1_recommend_tier_advanced():
    assert pricing.recommend_tier(20, True, gpu_type="H100") == "spot"
    assert pricing.recommend_tier(24, False, gpu_type="A10G") == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_extension2_metrics_cost_efficiency():
    bw_eff = metrics.bandwidth_cost_efficiency(2.5, 3.35)
    assert 0.7 < bw_eff < 0.8
    vram_eff = metrics.vram_cost_efficiency(2.5, 80)
    assert 0.03 < vram_eff < 0.04


def test_extension3_cache_economics():
    be_reads = pricing.cache_break_even_reads(write_cost_per_m=3.75, base_input_price=3.0, read_discount=0.10)
    assert abs(be_reads - 1.388888) < 1e-3
    assert pricing.cache_is_worth_it(avg_cache_reads=3.5, write_cost_per_m=3.75, base_input_price=3.0, read_discount=0.10) is True
    assert pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost_per_m=3.75, base_input_price=3.0, read_discount=0.10) is False


def test_extension4_reasoning_energy_multiplier():
    std_wh = sustainability.wh_per_query(1000, is_reasoning=False)
    res_wh = sustainability.wh_per_query(1000, is_reasoning=True)
    assert abs(res_wh / std_wh - 80.0) < 1e-6


def test_extension5_regional_carbon():
    wh = 1000.0
    c_norway = sustainability.carbon_g(wh, "europe-north1")
    c_us_east = sustainability.carbon_g(wh, "us-east-1")
    assert c_norway < c_us_east
    assert abs(c_norway - 30.0) < 1e-6
