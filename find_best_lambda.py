#!/usr/bin/env python3
"""
Find lambda that beats IndustryBest at EVERYTHING
"""

from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedGreedyPerformanceAgent,
    ProcessTechnology,
    ShiftType,
)
from test_softmin_jam import SoftminJAMAgent

def test_single_agent(agent_class, agent_kwargs, design_steps=50):
    """Test agent and return robustness breakdown"""
    results = {}

    # Test each stress type
    for stress_type, stress_level in [
        (ShiftType.TIGHTEN_POWER, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        (ShiftType.INCREASE_PERFORMANCE, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]),
        (ShiftType.TIGHTEN_THERMAL, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]),
    ]:
        max_survived = 0
        for level in stress_level:
            space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
            space.initialize_actions()
            agent = agent_class(**agent_kwargs)
            agent.initialize(space)

            for _ in range(design_steps):
                agent.step()

            if stress_type == ShiftType.TIGHTEN_POWER:
                space.limits.max_power_watts *= (1.0 - level)
            elif stress_type == ShiftType.INCREASE_PERFORMANCE:
                space.limits.min_frequency_ghz *= (1.0 + level)
            elif stress_type == ShiftType.TIGHTEN_THERMAL:
                space.limits.max_temperature_c -= (level * 50)

            if space.is_feasible():
                max_survived = level
            else:
                break

        results[stress_type.value] = max_survived

    # Get final metrics
    space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
    space.initialize_actions()
    agent = agent_class(**agent_kwargs)
    agent.initialize(space)
    for _ in range(design_steps):
        agent.step()

    perf = space.calculate_performance()
    constraints = space.calculate_constraints()
    power = constraints['total_power_w']

    return {
        'performance': perf,
        'power': power,
        'power_tolerance': results[ShiftType.TIGHTEN_POWER.value] * 100,
        'perf_headroom': results[ShiftType.INCREASE_PERFORMANCE.value] * 100,
        'thermal_tolerance': results[ShiftType.TIGHTEN_THERMAL.value] * 100,
    }

# Get IndustryBest baseline
print("Testing IndustryBest baseline...")
baseline = test_single_agent(AdvancedGreedyPerformanceAgent, {})
print(f"IndustryBest: perf={baseline['performance']:.1f}, power={baseline['power']:.2f}W")
print(f"  Power tolerance: {baseline['power_tolerance']:.0f}%")
print(f"  Perf headroom:   {baseline['perf_headroom']:.0f}%")
print(f"  Thermal:         {baseline['thermal_tolerance']:.0f}%")

print("\n" + "="*80)
print("SEARCHING FOR LAMBDA THAT BEATS INDUSTRYBEST AT EVERYTHING...")
print("="*80)

# Target: beat IndustryBest at ALL metrics
target = {
    'performance': baseline['performance'],
    'power': baseline['power'],
    'power_tolerance': baseline['power_tolerance'],
    'perf_headroom': baseline['perf_headroom'],
    'thermal_tolerance': baseline['thermal_tolerance'],
}

# Test lambda values from 0.01 to 500
lambda_values = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]

best_lambda = None
best_score = -1

for lam in lambda_values:
    result = test_single_agent(SoftminJAMAgent, {'lambda_weight': lam, 'beta': 5.0})

    # Check if beats IndustryBest at EVERYTHING
    beats_all = (
        result['performance'] > target['performance'] and
        result['power'] < target['power'] and
        result['power_tolerance'] > target['power_tolerance'] and
        result['perf_headroom'] > target['perf_headroom'] and
        result['thermal_tolerance'] >= target['thermal_tolerance']
    )

    # Count how many metrics beat IndustryBest
    wins = sum([
        result['performance'] > target['performance'],
        result['power'] < target['power'],
        result['power_tolerance'] > target['power_tolerance'],
        result['perf_headroom'] > target['perf_headroom'],
        result['thermal_tolerance'] >= target['thermal_tolerance'],
    ])

    status = "✓✓✓ BEATS ALL ✓✓✓" if beats_all else f"({wins}/5 wins)"

    print(f"\nλ={lam:6.2f}: {status}")
    print(f"  Perf: {result['performance']:6.1f} vs {target['performance']:6.1f} {'✓' if result['performance'] > target['performance'] else '✗'}")
    print(f"  Power: {result['power']:5.2f}W vs {target['power']:5.2f}W {'✓' if result['power'] < target['power'] else '✗'}")
    print(f"  Power tol: {result['power_tolerance']:3.0f}% vs {target['power_tolerance']:3.0f}% {'✓' if result['power_tolerance'] > target['power_tolerance'] else '✗'}")
    print(f"  Perf head: {result['perf_headroom']:3.0f}% vs {target['perf_headroom']:3.0f}% {'✓' if result['perf_headroom'] > target['perf_headroom'] else '✗'}")
    print(f"  Thermal:   {result['thermal_tolerance']:3.0f}% vs {target['thermal_tolerance']:3.0f}% {'✓' if result['thermal_tolerance'] >= target['thermal_tolerance'] else '✗'}")

    if beats_all:
        print(f"\n🎯 FOUND IT! λ={lam} beats IndustryBest at EVERYTHING!")
        best_lambda = lam
        break

    if wins > best_score:
        best_score = wins
        best_lambda = lam

if best_score < 5:
    print("\n" + "="*80)
    print(f"⚠️  NO LAMBDA BEATS INDUSTRYBEST AT EVERYTHING")
    print(f"⚠️  Best found: λ={best_lambda} with {best_score}/5 wins")
    print("="*80)
