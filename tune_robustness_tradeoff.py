#!/usr/bin/env python3
"""
Tune JAMAdvanced for robustness-focused design.

Goal: Sacrifice ~10% performance (107.25 → ~96.5) to gain:
- Higher frequency capability
- Better robustness (higher stress tolerance)
- More conservative safety margins
"""

from compare_greedy_vs_intrinsic import run_experiments
from test_proper_robustness import test_stress_resilience, calculate_robustness_score
from test_softmin_jam import SoftminJAMAgent
import numpy as np

print("="*80)
print("TUNING JAMAdvanced: Performance vs Robustness Trade-off")
print("="*80)
print("Goal: ~10% performance reduction (107.25 → ~96.5)")
print("Target: Higher robustness + better frequency")
print("="*80)

# Test different λ values (higher λ = more conservative = more robust)
lambda_candidates = [0.5, 1.0, 2.0, 5.0, 10.0]

results = []

for lambda_val in lambda_candidates:
    print(f"\n{'='*80}")
    print(f"Testing λ={lambda_val}")
    print('='*80)

    # Test robustness and get design metrics from single run
    print(f"\nTesting robustness and design quality...")
    rob_results = test_stress_resilience(
        "JAMAdvanced",
        SoftminJAMAgent,
        {"lambda_weight": lambda_val, "beta": 5.0},
        design_steps=40,
        seed=42,
    )

    # Get design metrics from the test (uses seed 42)
    from advanced_chip_simulator import AdvancedDesignSpace, ProcessTechnology
    space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
    space.initialize_actions()
    agent = SoftminJAMAgent(lambda_weight=lambda_val, beta=5.0)
    agent.initialize(space)

    # Design phase
    for _ in range(40):
        agent.step()

    avg_perf = space.calculate_performance()
    constraints = space.calculate_constraints()
    avg_power = constraints['total_power_w']
    headrooms = space.get_headrooms(include_performance=False)
    avg_headroom = min(headrooms.values())

    rob_score = calculate_robustness_score(rob_results)

    # Extract specific tolerances
    power_tol = None
    perf_tol = None
    for stress_level, survived in rob_results['tighten_power']:
        if not survived:
            power_tol = stress_level
            break
    else:
        power_tol = 1.0  # Survived all

    for stress_level, survived in rob_results['increase_performance']:
        if not survived:
            perf_tol = stress_level
            break
    else:
        perf_tol = 1.0

    results.append({
        'lambda': lambda_val,
        'performance': avg_perf,
        'power': avg_power,
        'headroom': avg_headroom,
        'robustness': rob_score,
        'power_tolerance': power_tol,
        'perf_tolerance': perf_tol,
    })

    print(f"\n📊 Results for λ={lambda_val}:")
    print(f"  Performance:      {avg_perf:.2f}")
    print(f"  Power:           {avg_power:.2f}W")
    print(f"  Min Headroom:    {avg_headroom:.4f}")
    print(f"  Robustness:      {rob_score:.1%}")
    print(f"  Power Tolerance:  {power_tol:.0%}")
    print(f"  Perf Tolerance:   {perf_tol:.0%}")

    # Check if this meets the goal
    perf_reduction = ((107.25 - avg_perf) / 107.25) * 100
    print(f"  Performance reduction: {perf_reduction:.1f}% (target: ~10%)")

# Summary
print("\n" + "="*80)
print("SUMMARY - λ Parameter Sweep")
print("="*80)
print(f"{'λ':>6} | {'Perf':>6} | {'Power':>6} | {'Headroom':>9} | {'Rob%':>5} | {'PwrTol':>7} | {'PerfTol':>8}")
print("-" * 80)

for r in results:
    print(f"{r['lambda']:>6.1f} | {r['performance']:>6.2f} | {r['power']:>6.2f}W | {r['headroom']:>9.4f} | {r['robustness']:>5.1%} | {r['power_tolerance']:>7.0%} | {r['perf_tolerance']:>8.0%}")

# Find best candidate for ~10% performance reduction
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

target_perf = 107.25 * 0.9  # 10% reduction
best = min(results, key=lambda r: abs(r['performance'] - target_perf))

print(f"Target performance: ~{target_perf:.1f} (10% reduction)")
print(f"\nBest match: λ={best['lambda']}")
print(f"  Performance:      {best['performance']:.2f} ({((best['performance']-107.25)/107.25*100):+.1f}%)")
print(f"  Power:           {best['power']:.2f}W")
print(f"  Min Headroom:    {best['headroom']:.4f} (vs 0.748 before)")
print(f"  Robustness:      {best['robustness']:.1%} (vs 40.0% before)")
print(f"  Power Tolerance:  {best['power_tolerance']:.0%} (vs 10% before)")
print(f"  Perf Tolerance:   {best['perf_tolerance']:.0%} (vs 35% before)")

print(f"\n✓ Update compare_greedy_vs_intrinsic.py:")
print(f"  SoftminJAMAgent(lambda_weight={best['lambda']}, beta=5.0)")
print("="*80)
