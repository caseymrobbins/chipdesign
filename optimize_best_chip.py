#!/usr/bin/env python3
"""
Optimize for the BEST chip: maximum robustness + high frequency
Push λ higher to maximize safety margins and frequency headroom
"""

from advanced_chip_simulator import AdvancedDesignSpace, ProcessTechnology
from test_softmin_jam import SoftminJAMAgent
from test_proper_robustness import test_stress_resilience, calculate_robustness_score
import numpy as np

print("="*80)
print("OPTIMIZING FOR BEST CHIP POSSIBLE")
print("="*80)
print("Goals:")
print("  1. Maximum robustness (highest stress tolerance)")
print("  2. High frequency capability")
print("  3. Maintain competitive performance (>100)")
print("="*80)

# Test very high λ values for maximum robustness
lambda_candidates = [200, 500, 1000, 2000, 5000]

results = []

for lambda_val in lambda_candidates:
    print(f"\n{'='*80}")
    print(f"Testing λ={lambda_val}")
    print('='*80)

    # Design the chip
    space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
    space.initialize_actions()
    agent = SoftminJAMAgent(lambda_weight=lambda_val, beta=5.0)
    agent.initialize(space)

    # Design phase
    for _ in range(40):
        agent.step()

    # Get metrics
    perf = space.calculate_performance()
    constraints = space.calculate_constraints()
    power = constraints['total_power_w']
    headrooms = space.get_headrooms(include_performance=False)
    min_headroom = min(headrooms.values())

    # Calculate efficiency and margins
    efficiency = perf / power
    power_margin = (12.0 - power) / 12.0  # % unused power budget

    print(f"\n📊 Design Metrics:")
    print(f"  Performance:      {perf:.2f}")
    print(f"  Power:           {power:.2f}W (margin: {power_margin*100:.1f}%)")
    print(f"  Efficiency:      {efficiency:.2f} perf/W")
    print(f"  Min Headroom:    {min_headroom:.4f}")

    # Quick robustness test (power and perf only for speed)
    print(f"\n🔬 Testing Robustness...")

    # Test power cuts
    power_tol = 0
    for stress in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        test_space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
        test_space.initialize_actions()
        test_agent = SoftminJAMAgent(lambda_weight=lambda_val, beta=5.0)
        test_agent.initialize(test_space)
        for _ in range(40):
            test_agent.step()
        test_space.limits.max_power_watts *= (1.0 - stress)
        if test_space.is_feasible():
            power_tol = stress
        else:
            break

    # Test performance increases
    perf_tol = 0
    for stress in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        test_space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
        test_space.initialize_actions()
        test_agent = SoftminJAMAgent(lambda_weight=lambda_val, beta=5.0)
        test_agent.initialize(test_space)
        for _ in range(40):
            test_agent.step()
        test_space.limits.min_frequency_ghz *= (1.0 + stress)
        if test_space.is_feasible():
            perf_tol = stress
        else:
            break

    print(f"  Power Tolerance:  {power_tol:.0%} (can handle {power_tol:.0%} power cuts)")
    print(f"  Perf Tolerance:   {perf_tol:.0%} (can handle {perf_tol:.0%} perf increases)")
    print(f"  Frequency Margin: {power_margin*100:.1f}% (unused power = freq headroom)")

    results.append({
        'lambda': lambda_val,
        'performance': perf,
        'power': power,
        'power_margin': power_margin,
        'efficiency': efficiency,
        'min_headroom': min_headroom,
        'power_tol': power_tol,
        'perf_tol': perf_tol,
    })

    # Check if this meets "best chip" criteria
    if perf > 100 and power_tol >= 0.20 and min_headroom > 0.8:
        print(f"\n  ✓ EXCELLENT: High perf ({perf:.1f}), 20%+ power tol, 0.8+ headroom")

# Summary
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)
print(f"{'λ':>6} | {'Perf':>6} | {'Power':>7} | {'PwrMgn':>7} | {'Eff':>6} | {'Headrm':>7} | {'PwrTol':>7} | {'PerfTol':>8}")
print("-" * 80)

for r in results:
    print(f"{r['lambda']:>6.0f} | {r['performance']:>6.2f} | {r['power']:>6.2f}W | {r['power_margin']*100:>6.1f}% | {r['efficiency']:>6.2f} | {r['min_headroom']:>7.4f} | {r['power_tol']:>7.0%} | {r['perf_tol']:>8.0%}")

# Find best overall (balance of perf, robustness, margins)
print("\n" + "="*80)
print("RECOMMENDATION: BEST CHIP DESIGN")
print("="*80)

# Score: weighted combination of performance and robustness
best = max(results, key=lambda r: (
    r['performance'] * 0.3 +           # 30% weight on performance
    r['power_tol'] * 100 * 0.3 +       # 30% weight on power tolerance
    r['perf_tol'] * 100 * 0.2 +        # 20% weight on perf tolerance
    r['min_headroom'] * 100 * 0.2      # 20% weight on safety margin
))

print(f"\nBest Configuration: λ={best['lambda']}")
print(f"\n📈 Performance:")
print(f"  Score:           {best['performance']:.2f}")
print(f"  Efficiency:      {best['efficiency']:.2f} perf/W")

print(f"\n⚡ Power & Frequency:")
print(f"  Power:           {best['power']:.2f}W / 12.0W")
print(f"  Power Margin:    {best['power_margin']*100:.1f}% (headroom for frequency boost)")
print(f"  Power Tolerance: {best['power_tol']:.0%}")

print(f"\n🛡️ Robustness:")
print(f"  Min Headroom:    {best['min_headroom']:.4f}")
print(f"  Power Tolerance:  {best['power_tol']:.0%} (can cut power by {best['power_tol']:.0%})")
print(f"  Perf Tolerance:   {best['perf_tol']:.0%} (can boost perf reqs by {best['perf_tol']:.0%})")

print(f"\n✓ Configuration: SoftminJAMAgent(lambda_weight={best['lambda']}, beta=5.0)")
print("="*80)
