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


def run_comparison(steps: int = 200):
    """Run comprehensive comparison testing different optimization strategies."""

    process = ProcessTechnology.create_7nm()

    # Create design spaces for all agent variants
    spaces = {
        'GreedyPerf': AdvancedDesignSpace(process=process),
        'JAM': AdvancedDesignSpace(process=process),
        'Hybrid_0.05': AdvancedDesignSpace(process=process),
        'Hybrid_0.10': AdvancedDesignSpace(process=process),
        'Hybrid_0.15': AdvancedDesignSpace(process=process),
        'Hybrid_0.20': AdvancedDesignSpace(process=process),
    }

    # Create agents with different configurations
    agents = {
        'GreedyPerf': AdvancedGreedyPerformanceAgent(),
        'JAM': JAMAgent(),
        'Hybrid_0.05': HybridJAM(performance_weight=0.05),  # Baseline
        'Hybrid_0.10': HybridJAM(performance_weight=0.10),  # Option 1a
        'Hybrid_0.15': HybridJAM(performance_weight=0.15),  # Option 1b
        'Hybrid_0.20': HybridJAM(performance_weight=0.20),  # Option 1c
    }

    # Assign agents to spaces
    for name in spaces:
        agents[name].design_space = spaces[name]

    print("=" * 80)
    print("COMPREHENSIVE COMPARISON: Testing Different HybridJAM Weights")
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
    agent_names = list(spaces.keys())
    header = f"{'Step':>5s}  " + "  ".join([f"{name:>8s}" for name in agent_names])
    print(header)

    for step in range(steps):
        # Each agent takes a step
        for name, agent in agents.items():
            action = agent.select_action()
            if action:
                spaces[name].apply_action(action)

        # Print progress every 20 steps for 200-step run
        if step % 20 == 0 or step == steps - 1:
            perfs = [spaces[name].calculate_performance() for name in agent_names]
            perf_str = "  ".join([f"{p:8.1f}" for p in perfs])
            print(f"{step:5d}:  {perf_str}")

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()

    # Print detailed results
    results = []
    for name in agent_names:
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

    # Compare all agents vs GreedyPerf
    print("COMPARISONS VS GREEDYPERF:")
    print()

    greedy = next(r for r in results if r['name'] == 'GreedyPerf')

    for result in results:
        if result['name'] == 'GreedyPerf':
            continue

        name = result['name']
        perf_ratio = (result['perf'] / greedy['perf']) * 100 if greedy['perf'] > 0 else 0
        eff_ratio = (result['efficiency'] / greedy['efficiency']) * 100 if greedy['efficiency'] > 0 else 0
        perf_diff = result['perf'] - greedy['perf']
        eff_diff = result['efficiency'] - greedy['efficiency']

        print(f"{name:12s}:")
        print(f"  Performance: {result['perf']:6.2f} vs {greedy['perf']:6.2f} = {perf_ratio:5.1f}% ({perf_diff:+5.2f})")
        print(f"  Efficiency:  {result['efficiency']:6.2f} vs {greedy['efficiency']:6.2f} = {eff_ratio:5.1f}% ({eff_diff:+4.2f})")

        if perf_diff > 0 and eff_diff > 0:
            print(f"  ✓✓ WINS on BOTH metrics!")
        elif perf_ratio > 95 and eff_diff > 0:
            print(f"  ✓ Very close perf + better efficiency")
        print()

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
    run_comparison(steps=200)
