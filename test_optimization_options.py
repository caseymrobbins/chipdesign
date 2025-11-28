#!/usr/bin/env python3
"""
Test all optimization options to push HybridJAM performance higher.

This test compares different architectural choices:
- Option 1: Different performance weights (DONE - all converge to same solution)
- Option 2: Tighter bounds (min_power=8W, min_area=30mm²)
- Option 3: Higher floor threshold (6.0, 8.0)
- Option 4: Adjusted target configuration (power=11.5W, area=48mm²)
"""

from dataclasses import replace
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    ConstraintLimits,
    ProcessTechnology,
    AdvancedGreedyPerformanceAgent,
    JAMAgent,
    HybridJAM,
)


def create_option2_limits() -> ConstraintLimits:
    """Option 2: Tighter bounds to push toward higher performance region."""
    limits = ConstraintLimits()
    # Tighter bounds: min_power 6→8W, min_area 25→30mm²
    limits.min_power_watts = 8.0  # Force higher power usage
    limits.min_area_mm2 = 30.0     # Force larger area usage

    # Recalculate weights for tighter bounds at target (11W, 46mm²)
    # Floor = 4.0 (same as baseline)
    limits.constraint_weights = {
        # Power bounds: At 11W target between [8W, 12W]
        'power_max': 4.0 / 1.0,      # 12 - 11 = 1W headroom
        'power_min': 4.0 / 3.0,      # 11 - 8 = 3W headroom (tighter!)

        # Area bounds: At 46mm² target between [30mm², 50mm²]
        'area_max': 4.0 / 4.0,       # 50 - 46 = 4mm² headroom
        'area_min': 4.0 / 16.0,      # 46 - 30 = 16mm² headroom (tighter!)

        # Physics constraints (same as baseline)
        'temperature': 4.0 / 10.0,
        'frequency': 4.0 / 0.2,
        'timing_slack': 4.0 / 20.0,
        'ir_drop': 4.0 / 10.0,
        'power_density': 4.0 / 0.1,
        'wire_delay': 4.0 / 20.0,
        'yield': 1.0,
        'signal_integrity': 1.0,
    }
    return limits


def create_option3_limits(floor: float = 6.0) -> ConstraintLimits:
    """Option 3: Higher floor threshold to push performance."""
    limits = ConstraintLimits()
    # Same bounds as baseline (min_power=6W, min_area=25mm²)
    # But higher floor threshold: 4.0 → 6.0 or 8.0

    limits.constraint_weights = {
        # Power bounds: At 11W target between [6W, 12W]
        'power_max': floor / 1.0,    # Higher floor!
        'power_min': floor / 5.0,

        # Area bounds: At 46mm² target between [25mm², 50mm²]
        'area_max': floor / 4.0,     # Higher floor!
        'area_min': floor / 21.0,

        # Physics constraints: higher floor pushes all constraints
        'temperature': floor / 10.0,
        'frequency': floor / 0.2,
        'timing_slack': floor / 20.0,
        'ir_drop': floor / 10.0,
        'power_density': floor / 0.1,
        'wire_delay': floor / 20.0,
        'yield': 1.0,
        'signal_integrity': 1.0,
    }
    return limits


def create_option4_limits() -> ConstraintLimits:
    """Option 4: Adjusted target configuration (11.5W, 48mm²) - more aggressive."""
    limits = ConstraintLimits()
    # Same bounds as baseline (min_power=6W, min_area=25mm²)
    # But target higher: (11W, 46mm²) → (11.5W, 48mm²)

    # Floor = 4.0 (same as baseline)
    limits.constraint_weights = {
        # Power bounds: At 11.5W target between [6W, 12W]
        'power_max': 4.0 / 0.5,      # 12 - 11.5 = 0.5W headroom (very tight!)
        'power_min': 4.0 / 5.5,      # 11.5 - 6 = 5.5W headroom

        # Area bounds: At 48mm² target between [25mm², 50mm²]
        'area_max': 4.0 / 2.0,       # 50 - 48 = 2mm² headroom (tighter!)
        'area_min': 4.0 / 23.0,      # 48 - 25 = 23mm² headroom

        # Physics constraints: expect tighter margins at higher performance
        'temperature': 4.0 / 8.0,    # 8°C headroom (tighter thermal)
        'frequency': 4.0 / 0.15,     # 0.15GHz above min (tighter)
        'timing_slack': 4.0 / 15.0,  # 15ps slack (tighter)
        'ir_drop': 4.0 / 8.0,        # 8mV headroom (tighter)
        'power_density': 4.0 / 0.08, # 0.08 W/mm² margin (tighter)
        'wire_delay': 4.0 / 15.0,    # 15ps headroom (tighter)
        'yield': 1.0,
        'signal_integrity': 1.0,
    }
    return limits


def run_comparison(steps: int = 200):
    """Run comprehensive comparison testing all optimization options."""

    process = ProcessTechnology.create_7nm()

    # Create design spaces with different constraint configurations
    spaces = {}

    # GreedyPerf and Baseline use default limits
    spaces['GreedyPerf'] = AdvancedDesignSpace(process=process)
    spaces['Baseline'] = AdvancedDesignSpace(process=process)

    # Option 2: Tighter bounds
    spaces['Option2'] = AdvancedDesignSpace(process=process)
    spaces['Option2'].limits = create_option2_limits()
    spaces['Option2'].initial_limits = spaces['Option2'].limits.clone()

    # Option 3.6: Higher floor (6.0)
    spaces['Option3_6.0'] = AdvancedDesignSpace(process=process)
    spaces['Option3_6.0'].limits = create_option3_limits(floor=6.0)
    spaces['Option3_6.0'].initial_limits = spaces['Option3_6.0'].limits.clone()

    # Option 3.8: Highest floor (8.0)
    spaces['Option3_8.0'] = AdvancedDesignSpace(process=process)
    spaces['Option3_8.0'].limits = create_option3_limits(floor=8.0)
    spaces['Option3_8.0'].initial_limits = spaces['Option3_8.0'].limits.clone()

    # Option 4: Adjusted target
    spaces['Option4'] = AdvancedDesignSpace(process=process)
    spaces['Option4'].limits = create_option4_limits()
    spaces['Option4'].initial_limits = spaces['Option4'].limits.clone()

    # Create agents
    agents = {
        'GreedyPerf': AdvancedGreedyPerformanceAgent(),
        'Baseline': HybridJAM(performance_weight=0.05),
        'Option2': HybridJAM(performance_weight=0.05),
        'Option3_6.0': HybridJAM(performance_weight=0.05),
        'Option3_8.0': HybridJAM(performance_weight=0.05),
        'Option4': HybridJAM(performance_weight=0.05),
    }

    # Assign agents to spaces
    for name in spaces:
        agents[name].design_space = spaces[name]

    print("=" * 90)
    print("COMPREHENSIVE OPTIMIZATION OPTIONS TEST")
    print("=" * 90)
    print()
    print("Testing:")
    print("  Baseline:    min_power=6W, min_area=25mm², floor=4.0, target=(11W, 46mm²)")
    print("  Option 2:    min_power=8W, min_area=30mm², floor=4.0, target=(11W, 46mm²)  [TIGHTER BOUNDS]")
    print("  Option 3.6:  min_power=6W, min_area=25mm², floor=6.0, target=(11W, 46mm²)  [HIGHER FLOOR]")
    print("  Option 3.8:  min_power=6W, min_area=25mm², floor=8.0, target=(11W, 46mm²)  [HIGHEST FLOOR]")
    print("  Option 4:    min_power=6W, min_area=25mm², floor=4.0, target=(11.5W, 48mm²) [AGGRESSIVE TARGET]")
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
    header = f"{'Step':>5s}  " + "  ".join([f"{name:>11s}" for name in agent_names])
    print(header)

    for step in range(steps):
        # Each agent takes a step
        for name, agent in agents.items():
            action = agent.select_action()
            if action:
                spaces[name].apply_action(action)

        # Print progress every 20 steps
        if step % 20 == 0 or step == steps - 1:
            perfs = [spaces[name].calculate_performance() for name in agent_names]
            perf_str = "  ".join([f"{p:11.1f}" for p in perfs])
            print(f"{step:5d}:  {perf_str}")

    print()
    print("=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)
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

    # Compare all options vs GreedyPerf AND Baseline
    print("COMPARISONS VS GREEDYPERF:")
    print()

    greedy = next(r for r in results if r['name'] == 'GreedyPerf')
    baseline = next(r for r in results if r['name'] == 'Baseline')

    for result in results:
        if result['name'] in ['GreedyPerf', 'Baseline']:
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
    print("COMPARISONS VS BASELINE (HybridJAM 0.05):")
    print()

    for result in results:
        if result['name'] in ['GreedyPerf', 'Baseline']:
            continue

        name = result['name']
        perf_diff = result['perf'] - baseline['perf']
        eff_diff = result['efficiency'] - baseline['efficiency']
        perf_pct = (perf_diff / baseline['perf']) * 100 if baseline['perf'] > 0 else 0
        eff_pct = (eff_diff / baseline['efficiency']) * 100 if baseline['efficiency'] > 0 else 0

        print(f"{name:12s}:")
        print(f"  Performance: {result['perf']:6.2f} vs {baseline['perf']:6.2f} = {perf_diff:+5.2f} ({perf_pct:+5.1f}%)")
        print(f"  Efficiency:  {result['efficiency']:6.2f} vs {baseline['efficiency']:6.2f} = {eff_diff:+4.2f} ({eff_pct:+5.1f}%)")

        if perf_diff > 0:
            print(f"  ✓ IMPROVEMENT over baseline!")
        elif abs(perf_diff) < 0.5:
            print(f"  ≈ Similar to baseline")
        else:
            print(f"  ✗ Worse than baseline")
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
