#!/usr/bin/env python3
"""
Test forcing constraints: Does JAM achieve high performance AND efficiency when REQUIRED?

This test compares all four agents with minimum resource constraints to see if
forcing JAM to use power/area helps it beat GreedyPerf on BOTH metrics.
"""

from advanced_chip_simulator import (
    AdvancedDesignSpace,
    ConstraintLimits,
    ProcessTechnology,
    AdvancedGreedyPerformanceAgent,
    JAMAgent,
    AdaptiveJAM,
    HybridJAM,
)


def run_comparison(steps: int = 50):
    """Run four-way comparison with forcing constraints."""

    process = ProcessTechnology.create_7nm()

    # Create four design spaces (one per agent)
    # Limits are created internally with forcing constraints
    spaces = {
        'GreedyPerf': AdvancedDesignSpace(process=process),
        'JAM': AdvancedDesignSpace(process=process),
        'AdaptiveJAM': AdvancedDesignSpace(process=process),
        'HybridJAM': AdvancedDesignSpace(process=process),
    }

    # Create agents
    agents = {
        'GreedyPerf': AdvancedGreedyPerformanceAgent(),
        'JAM': JAMAgent(),
        'AdaptiveJAM': AdaptiveJAM(margin_target=10.0),
        'HybridJAM': HybridJAM(performance_weight=0.05),
    }

    # Assign agents to spaces
    for name in spaces:
        agents[name].design_space = spaces[name]

    print("=" * 80)
    print("FOUR-WAY COMPARISON WITH FORCING CONSTRAINTS")
    print("=" * 80)
    print(f"\nConstraints:")
    print(f"  Power: {spaces['JAM'].limits.min_power_watts:.1f}W - {spaces['JAM'].limits.max_power_watts:.1f}W")
    print(f"  Area: {spaces['JAM'].limits.min_area_mm2:.1f}mm² - {spaces['JAM'].limits.max_area_mm2:.1f}mm²")
    print(f"  Performance: ≥{spaces['JAM'].limits.min_performance_score:.1f}")
    print(f"  Frequency: ≥{spaces['JAM'].limits.min_frequency_ghz:.1f}GHz")
    print()

    # Check starting feasibility
    print(f"Starting configuration:")
    for name, space in spaces.items():
        perf = space.calculate_performance()
        constraints = space.calculate_constraints()
        headroom = space.get_min_headroom()
        feasible = space.is_feasible()
        status = "✓" if feasible else "✗"
        print(f"  {name:12s} {status} Perf={perf:5.1f} | {constraints['total_power_w']:4.1f}W | "
              f"{constraints['area_mm2']:5.1f}mm² | head={headroom:6.2f}")
    print()

    # Run optimization
    print(f"Running {steps} optimization steps...")
    print(f"{'Step':>5s}  {'Greedy':>6s}  {'JAM':>6s}  {'Adaptive':>6s}  {'Hybrid':>6s}")

    for step in range(steps):
        # Each agent takes a step
        for name, agent in agents.items():
            action = agent.select_action()
            if action:
                spaces[name].apply_action(action)

        # Print progress
        if step % 10 == 0 or step == steps - 1:
            perfs = [spaces[name].calculate_performance() for name in ['GreedyPerf', 'JAM', 'AdaptiveJAM', 'HybridJAM']]
            print(f"{step:5d}:  {perfs[0]:6.1f}  {perfs[1]:6.1f}  {perfs[2]:6.1f}  {perfs[3]:6.1f}")

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()

    # Print detailed results
    results = []
    for name in ['GreedyPerf', 'JAM', 'AdaptiveJAM', 'HybridJAM']:
        space = spaces[name]
        perf = space.calculate_performance()
        constraints = space.calculate_constraints()
        headroom = space.get_min_headroom()
        feasible = space.is_feasible()

        power = constraints['total_power_w']
        area = constraints['area_mm2']
        efficiency = perf / power if power > 0 else 0

        results.append({
            'name': name,
            'perf': perf,
            'power': power,
            'area': area,
            'headroom': headroom,
            'efficiency': efficiency,
            'feasible': feasible,
        })

        status = "✓" if feasible else "✗"
        print(f"{name:12s} {status} Perf={perf:6.2f} | {power:4.1f}W | {area:5.1f}mm² | "
              f"head={headroom:6.2f} | {efficiency:4.1f} perf/W")

    print()

    # Compare metrics
    print("COMPARISONS:")
    print()

    greedy = next(r for r in results if r['name'] == 'GreedyPerf')
    jam = next(r for r in results if r['name'] == 'JAM')

    print(f"JAM vs GreedyPerf:")
    perf_ratio = (jam['perf'] / greedy['perf']) * 100 if greedy['perf'] > 0 else 0
    eff_ratio = (jam['efficiency'] / greedy['efficiency']) * 100 if greedy['efficiency'] > 0 else 0
    print(f"  Performance: JAM = {perf_ratio:.1f}% of GreedyPerf")
    print(f"  Efficiency:  JAM = {eff_ratio:.1f}% of GreedyPerf")
    print()

    if perf_ratio > 90 and eff_ratio > 110:
        print("🎯 SUCCESS! JAM achieves both high performance (>90%) AND better efficiency (>110%)!")
    elif perf_ratio > 90:
        print("✓ JAM matches performance but not efficiency advantage")
    elif eff_ratio > 110:
        print("✓ JAM has efficiency advantage but not performance")
    else:
        print("⚠ JAM doesn't beat GreedyPerf on both metrics yet")

    print()

    # Best of each metric
    best_perf = max(results, key=lambda r: r['perf'])
    best_eff = max(results, key=lambda r: r['efficiency'])
    best_balanced = max(results, key=lambda r: r['perf'] * r['efficiency'])

    print(f"Best Performance: {best_perf['name']:12s} ({best_perf['perf']:.2f})")
    print(f"Best Efficiency:  {best_eff['name']:12s} ({best_eff['efficiency']:.1f} perf/W)")
    print(f"Best Balanced:    {best_balanced['name']:12s} (perf×eff = {best_balanced['perf']*best_balanced['efficiency']:.1f})")
    print()


if __name__ == "__main__":
    run_comparison(steps=50)
