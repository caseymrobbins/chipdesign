#!/usr/bin/env python3
"""
Debug script to understand why JAMAdvanced performance plateaus at 36.62
while JAM reaches 110.12.

This script will:
1. Run both agents side-by-side with the same initial state
2. Print detailed action selection information at each step
3. Compare objective values for each potential action
4. Identify where the agents diverge in their decisions
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedAgent,
    JAMAgent,
    DesignAction,
    ProcessTechnology,
)
from test_softmin_jam import SoftminJAMAgent


def debug_action_selection(agent: AdvancedAgent, step: int):
    """Print detailed information about action selection"""
    if not agent.design_space:
        return

    print(f"\n{'='*80}")
    print(f"Step {step}: {agent.name}")
    print(f"{'='*80}")

    # Current state
    perf = agent.design_space.calculate_performance()
    constraints = agent.design_space.calculate_constraints()
    headrooms = agent.design_space.get_headrooms(include_performance=False)
    min_headroom = min(headrooms.values())

    print(f"Current State:")
    print(f"  Performance: {perf:.2f}")
    print(f"  Power: {constraints['total_power_w']:.2f}W / {agent.design_space.limits.max_power_watts:.2f}W ({constraints['total_power_w']/agent.design_space.limits.max_power_watts*100:.1f}%)")
    print(f"  Area: {constraints['area_mm2']:.2f}mm² / {agent.design_space.limits.max_area_mm2:.2f}mm²")
    print(f"  Min Headroom: {min_headroom:.4f}")

    # Evaluate all actions
    action_scores = []
    for action in agent.design_space.actions:
        test_space = agent.design_space.clone()
        test_space.apply_action(action)

        test_headrooms = test_space.get_headrooms(include_performance=False)

        # Calculate objective based on agent type
        # Save original design_space
        original_space = agent.design_space
        agent.design_space = test_space

        if isinstance(agent, SoftminJAMAgent):
            objective = agent.calculate_objective(test_headrooms)
        elif isinstance(agent, JAMAgent):
            # JAM uses a different objective
            test_perf = test_space.calculate_performance()
            test_min_headroom = min(test_headrooms.values())
            # JAM formula: perf * 0.8 + log(min_headroom) * 0.2
            if test_min_headroom <= 0:
                objective = -float('inf')
            else:
                objective = test_perf * 0.8 + np.log(test_min_headroom) * 0.2
        else:
            objective = 0.0

        # Restore original design_space
        agent.design_space = original_space

        test_perf = test_space.calculate_performance()
        test_constraints = test_space.calculate_constraints()
        test_min_headroom = min(test_headrooms.values())

        action_scores.append({
            'action': action,
            'objective': objective,
            'perf': test_perf,
            'power': test_constraints['total_power_w'],
            'min_headroom': test_min_headroom,
        })

    # Sort by objective (best first)
    action_scores.sort(key=lambda x: x['objective'], reverse=True)

    # Print top 5 actions
    print(f"\nTop 5 Actions (by objective):")
    for i, score in enumerate(action_scores[:5]):
        action = score['action']
        print(f"  {i+1}. {action.name}")
        print(f"     Category: {action.category}")
        print(f"     Objective: {score['objective']:.4f}")
        print(f"     → Perf: {score['perf']:.2f} (Δ={score['perf']-perf:+.2f})")
        print(f"     → Power: {score['power']:.2f}W (Δ={score['power']-constraints['total_power_w']:+.2f}W)")
        print(f"     → Min Headroom: {score['min_headroom']:.4f} (Δ={score['min_headroom']-min_headroom:+.4f})")

    # Print worst action for comparison
    print(f"\nWorst Action (for comparison):")
    worst = action_scores[-1]
    action = worst['action']
    print(f"  {action.name}")
    print(f"  Category: {action.category}")
    print(f"  Objective: {worst['objective']:.4f}")
    print(f"  → Perf: {worst['perf']:.2f} (Δ={worst['perf']-perf:+.2f})")

    return action_scores[0]['action']  # Return best action


def run_debug_comparison(design_steps: int = 20, seed: int = 42):
    """Run side-by-side comparison with detailed debugging"""

    print(f"\n{'='*80}")
    print(f"DEBUG: JAM vs JAMAdvanced Performance Plateau")
    print(f"{'='*80}")
    print(f"Design steps: {design_steps}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Create identical initial states
    rng = np.random.RandomState(seed)
    space_seed = rng.randint(0, 1000000)

    # Create design spaces
    space_jam = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=space_seed)
    space_jam.initialize_actions()

    space_jamadv = space_jam.clone()

    # Create agents
    agent_jam = JAMAgent()
    agent_jam.initialize(space_jam)

    agent_jamadv = SoftminJAMAgent(lambda_weight=0.1, beta=5.0)
    agent_jamadv.initialize(space_jamadv)

    print(f"Testing Parameters:")
    print(f"  JAM: Standard formula (perf * 0.8 + log(min_headroom) * 0.2)")
    print(f"  JAMAdvanced: R = perf + λ·log(min_headroom), λ=0.1, β=5.0")

    # Run design phase with detailed logging
    divergence_step = None

    for step in range(design_steps):
        print(f"\n\n{'#'*80}")
        print(f"DESIGN STEP {step}")
        print(f"{'#'*80}")

        # Debug JAM action selection
        jam_action = debug_action_selection(agent_jam, step)
        agent_jam.step()

        # Debug JAMAdvanced action selection
        jamadv_action = debug_action_selection(agent_jamadv, step)
        agent_jamadv.step()

        # Check if agents diverged
        if divergence_step is None:
            jam_perf = agent_jam.design_space.calculate_performance()
            jamadv_perf = agent_jamadv.design_space.calculate_performance()
            if abs(jam_perf - jamadv_perf) > 1.0:
                divergence_step = step
                print(f"\n{'!'*80}")
                print(f"DIVERGENCE DETECTED at step {step}")
                print(f"  JAM Performance: {jam_perf:.2f}")
                print(f"  JAMAdvanced Performance: {jamadv_perf:.2f}")
                print(f"  Difference: {abs(jam_perf - jamadv_perf):.2f}")
                print(f"{'!'*80}")

    # Final results
    print(f"\n\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")

    jam_perf = agent_jam.design_space.calculate_performance()
    jam_constraints = agent_jam.design_space.calculate_constraints()
    jam_headrooms = agent_jam.design_space.get_headrooms(include_performance=False)

    jamadv_perf = agent_jamadv.design_space.calculate_performance()
    jamadv_constraints = agent_jamadv.design_space.calculate_constraints()
    jamadv_headrooms = agent_jamadv.design_space.get_headrooms(include_performance=False)

    print(f"\nJAM:")
    print(f"  Performance: {jam_perf:.2f}")
    print(f"  Power: {jam_constraints['total_power_w']:.2f}W ({jam_constraints['total_power_w']/agent_jam.design_space.limits.max_power_watts*100:.1f}%)")
    print(f"  Min Headroom: {min(jam_headrooms.values()):.4f}")

    print(f"\nJAMAdvanced:")
    print(f"  Performance: {jamadv_perf:.2f}")
    print(f"  Power: {jamadv_constraints['total_power_w']:.2f}W ({jamadv_constraints['total_power_w']/agent_jamadv.design_space.limits.max_power_watts*100:.1f}%)")
    print(f"  Min Headroom: {min(jamadv_headrooms.values()):.4f}")

    print(f"\nPerformance Gap: {jam_perf - jamadv_perf:.2f} ({(jam_perf - jamadv_perf) / jam_perf * 100:.1f}%)")

    if divergence_step is not None:
        print(f"\nDivergence first detected at step: {divergence_step}")
    else:
        print(f"\nNo significant divergence detected (agents made similar choices)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Run debug comparison with limited steps for detailed analysis
    run_debug_comparison(design_steps=10, seed=42)
