#!/usr/bin/env python3
"""
Diagnostic script to debug why agents are performing poorly
"""

import numpy as np
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedGreedyPerformanceAgent,
    JAMAgent,
    AdaptiveJAM,
    HybridJAM,
    ProcessTechnology,
)
from test_softmin_jam import SoftminJAMAgent

def diagnose_agents():
    """Run detailed diagnostics on each agent"""

    print("\n" + "="*80)
    print("AGENT DIAGNOSTIC ANALYSIS")
    print("="*80)

    # Create design space
    process = ProcessTechnology.create_7nm()
    base_space = AdvancedDesignSpace(process=process, seed=42)
    base_space.initialize_actions()

    print(f"\nInitial Design Space State:")
    print(f"  Performance: {base_space.calculate_performance():.2f}")
    print(f"  Min Headroom: {base_space.get_min_headroom():.2f}")
    print(f"  Is Feasible: {base_space.is_feasible()}")
    print(f"  Available Actions: {len(base_space.actions)}")

    constraints = base_space.calculate_constraints()
    print(f"  Power: {constraints['total_power_w']:.2f}W")
    print(f"  Temperature: {constraints['temperature_c']:.2f}°C")
    print(f"  Area: {constraints['area_mm2']:.2f}mm²")

    headrooms = base_space.get_headrooms()
    print(f"\n  Headrooms:")
    for name, value in headrooms.items():
        print(f"    {name}: {value:.2f}")

    # Test each agent
    agents = [
        ("Greedy", AdvancedGreedyPerformanceAgent()),
        ("JAM", JAMAgent()),
        ("AdaptiveJAM", AdaptiveJAM(margin_target=10.0)),
        ("HybridJAM", HybridJAM(lambda_reg=1000.0)),
        ("SoftminJAM (λ=200)", SoftminJAMAgent(lambda_weight=200.0, beta=2.5)),
        ("SoftminJAM (λ=1000)", SoftminJAMAgent(lambda_weight=1000.0, beta=5.0)),
        ("SoftminJAM (λ=5000)", SoftminJAMAgent(lambda_weight=5000.0, beta=10.0)),
    ]

    print("\n" + "="*80)
    print("TESTING EACH AGENT")
    print("="*80)

    for name, agent in agents:
        print(f"\n{'─'*80}")
        print(f"Testing: {name}")
        print(f"{'─'*80}")

        # Clone space for this agent
        space = base_space.clone()
        agent.initialize(space)

        # Try to select first action
        try:
            action = agent.select_action()
            if action is None:
                print(f"  ❌ ERROR: Agent returned None for first action!")
                print(f"     This means it couldn't find any feasible action")
                continue
            else:
                print(f"  ✓ First action selected: {action.name}")
        except Exception as e:
            print(f"  ❌ EXCEPTION selecting action: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Try to apply action
        try:
            feasible = space.apply_action(action)
            print(f"  ✓ Action applied, feasible={feasible}")
        except Exception as e:
            print(f"  ❌ EXCEPTION applying action: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Check state after action
        perf = space.calculate_performance()
        min_h = space.get_min_headroom()

        print(f"  Performance: {perf:.2f}")
        print(f"  Min Headroom: {min_h:.2f}")
        print(f"  Is Feasible: {space.is_feasible()}")

        # Try 10 more steps
        print(f"\n  Running 10 optimization steps...")
        successes = 0
        failures = 0

        for step in range(10):
            try:
                action = agent.select_action()
                if action is None:
                    failures += 1
                    print(f"    Step {step+1}: No action available")
                    break

                feasible = space.apply_action(action)
                if not feasible:
                    failures += 1
                    print(f"    Step {step+1}: Action made design infeasible!")
                    break

                successes += 1

            except Exception as e:
                failures += 1
                print(f"    Step {step+1}: Exception - {e}")
                break

        # Final state
        final_perf = space.calculate_performance()
        final_headroom = space.get_min_headroom()
        improvement = ((final_perf - base_space.calculate_performance()) /
                      base_space.calculate_performance()) * 100

        print(f"\n  Results after {successes} successful steps:")
        print(f"    Final Performance: {final_perf:.2f} ({improvement:+.1f}%)")
        print(f"    Final Min Headroom: {final_headroom:.2f}")
        print(f"    Successes: {successes}, Failures: {failures}")

        if successes == 10:
            print(f"    ✓ Agent completed all 10 steps successfully!")
        elif successes > 0:
            print(f"    ⚠ Agent stopped after {successes} steps")
        else:
            print(f"    ❌ Agent failed immediately!")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    diagnose_agents()
