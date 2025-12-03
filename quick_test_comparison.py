#!/usr/bin/env python3
"""
Quick test to verify all agents work with pure intrinsic optimization
"""

from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedGreedyPerformanceAgent,
    JAMAgent,
    AdaptiveJAM,
    HybridJAM,
    ProcessTechnology,
)
from test_softmin_jam import SoftminJAMAgent

def test_all_agents():
    """Test that all agents can run without errors"""

    print("\n" + "="*80)
    print("QUICK TEST: All Agents with Pure Intrinsic Optimization")
    print("="*80)

    # Create design space
    process = ProcessTechnology.create_7nm()
    base_space = AdvancedDesignSpace(process=process, seed=42)
    base_space.initialize_actions()

    # Create all agents
    agents = [
        ("Greedy", AdvancedGreedyPerformanceAgent()),
        ("JAM (hard min)", JAMAgent()),
        ("AdaptiveJAM", AdaptiveJAM(margin_target=10.0)),
        ("HybridJAM (λ=1000)", HybridJAM(lambda_reg=1000.0)),
        ("SoftminJAM (λ=200,β=2.5)", SoftminJAMAgent(lambda_weight=200.0, beta=2.5)),
        ("SoftminJAM (λ=1000,β=5.0)", SoftminJAMAgent(lambda_weight=1000.0, beta=5.0)),
        ("SoftminJAM (λ=5000,β=10.0)", SoftminJAMAgent(lambda_weight=5000.0, beta=10.0)),
    ]

    print("\nTesting each agent for 10 optimization steps...\n")

    results = []
    for name, agent in agents:
        space = base_space.clone()
        agent.initialize(space)

        # Get initial state
        initial_perf = space.calculate_performance()
        initial_headroom = space.get_min_headroom()
        initial_constraints = space.calculate_constraints()
        initial_power = initial_constraints['total_power_w']
        initial_eff = initial_perf / initial_power

        # Run 10 steps
        for step in range(10):
            action = agent.select_action()
            if action:
                space.apply_action(action)

        # Get final state
        final_perf = space.calculate_performance()
        final_headroom = space.get_min_headroom()
        final_constraints = space.calculate_constraints()
        final_power = final_constraints['total_power_w']
        final_eff = final_perf / final_power

        # Calculate improvements
        perf_improvement = ((final_perf - initial_perf) / initial_perf) * 100
        eff_improvement = ((final_eff - initial_eff) / initial_eff) * 100

        results.append({
            'name': name,
            'initial_perf': initial_perf,
            'final_perf': final_perf,
            'perf_improvement': perf_improvement,
            'initial_eff': initial_eff,
            'final_eff': final_eff,
            'eff_improvement': eff_improvement,
            'initial_headroom': initial_headroom,
            'final_headroom': final_headroom,
            'final_power': final_power,
        })

        print(f"✓ {name:30s}: Perf={final_perf:6.1f} (+{perf_improvement:+5.1f}%), "
              f"Eff={final_eff:5.2f} (+{eff_improvement:+5.1f}%), "
              f"Power={final_power:5.1f}W, Headroom={final_headroom:5.1f}")

    print("\n" + "="*80)
    print("INITIAL COMPARISON (10 steps)")
    print("="*80)

    # Sort by final performance
    results_sorted = sorted(results, key=lambda x: x['final_perf'], reverse=True)

    print(f"\n{'Rank':<5} {'Agent':<30} {'Performance':<12} {'Efficiency':<12} {'Power':<10}")
    print("-" * 80)

    for rank, res in enumerate(results_sorted, 1):
        print(f"{rank:<5} {res['name']:<30} "
              f"{res['final_perf']:6.1f} (+{res['perf_improvement']:+5.1f}%)  "
              f"{res['final_eff']:5.2f} (+{res['eff_improvement']:+5.1f}%)  "
              f"{res['final_power']:5.1f}W")

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)

    # Find best performers
    best_perf = max(results_sorted, key=lambda x: x['final_perf'])
    best_eff = max(results_sorted, key=lambda x: x['final_eff'])

    print(f"\n🏆 Best Performance: {best_perf['name']}")
    print(f"   Score: {best_perf['final_perf']:.1f} (+{best_perf['perf_improvement']:+.1f}% improvement)")

    print(f"\n⚡ Best Efficiency: {best_eff['name']}")
    print(f"   Score: {best_eff['final_eff']:.2f} perf/W (+{best_eff['eff_improvement']:+.1f}% improvement)")

    print(f"\n✓ All {len(agents)} agents running successfully with pure intrinsic optimization!")
    print("✓ NO external constraints, NO threshold checks")
    print("✓ Trusting log(min(v)) and log(softmin(v)) to prevent catastrophic failures")

    print("\n" + "="*80)
    print("To generate full comparison with visualization:")
    print("  python compare_greedy_vs_intrinsic.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_all_agents()
