#!/usr/bin/env python3
"""
Aggressive tuning: Force 10% performance reduction for robustness
Try much higher λ values to create conservative designs
"""

from test_proper_robustness import test_stress_resilience, calculate_robustness_score
from test_softmin_jam import SoftminJAMAgent
from advanced_chip_simulator import AdvancedDesignSpace, ProcessTechnology

print("="*80)
print("AGGRESSIVE ROBUSTNESS TUNING")
print("="*80)
print("Target: 10% performance reduction (107.25 → ~96.5)")
print("Method: High λ values to force conservative designs")
print("="*80)

# Try much higher λ values
lambda_candidates = [20.0, 50.0, 100.0, 200.0, 500.0]

results = []

for lambda_val in lambda_candidates:
    print(f"\n{'='*80}")
    print(f"Testing λ={lambda_val}")
    print('='*80)

    # Get design metrics
    space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
    space.initialize_actions()
    agent = SoftminJAMAgent(lambda_weight=lambda_val, beta=5.0)
    agent.initialize(space)

    # Design phase
    for _ in range(40):
        agent.step()

    perf = space.calculate_performance()
    constraints = space.calculate_constraints()
    power = constraints['total_power_w']
    headrooms = space.get_headrooms(include_performance=False)
    min_headroom = min(headrooms.values())
    freq = constraints.get('frequency_ghz', 0)

    print(f"\n Design Metrics:")
    print(f"  Performance:  {perf:.2f}")
    print(f"  Power:       {power:.2f}W")
    print(f"  Frequency:   {freq:.2f}GHz")
    print(f"  Min Headroom: {min_headroom:.4f}")

    # Test robustness (just power and perf, skip area/thermal for speed)
    print(f"\n Testing power tolerance...")
    from test_proper_robustness import ShiftType
    import numpy as np

    # Test power cuts
    power_tol = 0.05
    for stress_level in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        test_space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
        test_space.initialize_actions()
        test_agent = SoftminJAMAgent(lambda_weight=lambda_val, beta=5.0)
        test_agent.initialize(test_space)

        for _ in range(40):
            test_agent.step()

        # Apply stress
        test_space.limits.max_power_watts *= (1.0 - stress_level)
        survived = test_space.is_feasible()

        if not survived:
            power_tol = stress_level
            break
        else:
            power_tol = stress_level + 0.05

    # Test performance increases
    perf_tol = 0.05
    for stress_level in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        test_space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
        test_space.initialize_actions()
        test_agent = SoftminJAMAgent(lambda_weight=lambda_val, beta=5.0)
        test_agent.initialize(test_space)

        for _ in range(40):
            test_agent.step()

        # Apply stress
        test_space.limits.min_frequency_ghz *= (1.0 + stress_level)
        survived = test_space.is_feasible()

        if not survived:
            perf_tol = stress_level
            break
        else:
            perf_tol = stress_level + 0.05

    results.append({
        'lambda': lambda_val,
        'performance': perf,
        'power': power,
        'frequency': freq,
        'headroom': min_headroom,
        'power_tolerance': power_tol,
        'perf_tolerance': perf_tol,
    })

    print(f"\n Robustness:")
    print(f"  Power Tolerance:  {power_tol:.0%}")
    print(f"  Perf Tolerance:   {perf_tol:.0%}")

    perf_reduction = ((107.25 - perf) / 107.25) * 100
    print(f"\n Performance reduction: {perf_reduction:.1f}% (target: ~10%)")

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"{'λ':>6} | {'Perf':>6} | {'Δ%':>6} | {'Power':>6} | {'Freq':>6} | {'Headroom':>9} | {'PwrTol':>7} | {'PerfTol':>8}")
print("-" * 80)

for r in results:
    perf_change = ((r['performance'] - 107.25) / 107.25) * 100
    print(f"{r['lambda']:>6.0f} | {r['performance']:>6.2f} | {perf_change:>+5.1f}% | {r['power']:>6.2f}W | {r['frequency']:>6.2f} | {r['headroom']:>9.4f} | {r['power_tolerance']:>7.0%} | {r['perf_tolerance']:>8.0%}")

# Find best for target
target_perf = 107.25 * 0.9
best = min(results, key=lambda r: abs(r['performance'] - target_perf))

print("\n" + "="*80)
print("RECOMMENDATION FOR 10% REDUCTION")
print("="*80)
print(f"Best match: λ={best['lambda']}")
print(f"  Performance:      {best['performance']:.2f} ({((best['performance']-107.25)/107.25*100):+.1f}%)")
print(f"  Power:           {best['power']:.2f}W")
print(f"  Frequency:       {best['frequency']:.2f}GHz")
print(f"  Min Headroom:    {best['headroom']:.4f}")
print(f"  Power Tolerance:  {best['power_tolerance']:.0%}")
print(f"  Perf Tolerance:   {best['perf_tolerance']:.0%}")
print("="*80)
