#!/usr/bin/env python3
"""
Performance-First Softmin JAM: Beat Greedy in performance AND efficiency

Key insight: Current Softmin JAM is TOO conservative (47 perf vs Greedy's 94).
We need to AGGRESSIVELY pursue performance while maintaining survival.

New formula: R = α·perf + sum(headrooms) + λ·log(softmin(headrooms; β) + ε)

Where:
- α (alpha): Performance weight - NEW! Makes us competitive with Greedy
- sum term: Encourages improving ALL headrooms
- λ·softmin term: Smooth bottleneck focus (reduced weight vs original)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedAgent,
    AdvancedGreedyPerformanceAgent,
    DesignAction,
    ProcessTechnology,
    AdvancedSimulation,
    ShiftType,
)
from test_softmin_jam import SoftminJAMAgent, softmin
import json


class PerformanceFirstJAMAgent(AdvancedAgent):
    """
    Performance-First JAM: Aggressively pursue performance while maintaining survival

    Objective: R = α·perf + sum(headrooms) + λ·log(softmin(headrooms; β) + ε)

    The key difference from conservative Softmin JAM:
    1. α term: Explicitly rewards raw performance (like Greedy does)
    2. Lower λ: Reduces over-emphasis on margins
    3. Lower min_threshold: Allows tighter operation for higher performance
    """

    def __init__(
        self,
        alpha: float = 2.0,           # Performance weight (NEW!)
        lambda_weight: float = 0.1,    # Softmin weight (REDUCED from 0.2)
        beta: float = 1.5,             # Softmin temperature
        min_margin_threshold: float = 0.5,  # LOWER threshold for aggression
        epsilon: float = 0.01,
    ):
        super().__init__(f"PerfFirstJAM(α={alpha},λ={lambda_weight},β={beta})")
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        self.beta = beta
        self.min_margin_threshold = min_margin_threshold
        self.epsilon = epsilon

    def calculate_objective(
        self,
        headrooms_dict: Dict[str, float],
        performance: float,
    ) -> float:
        """
        Calculate the performance-first objective function.

        R = α·perf + sum(headrooms) + λ·log(softmin(headrooms; β) + ε)
        """
        # Get weighted headrooms
        weights = self.design_space.limits.constraint_weights
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms_dict.items()
        }

        headroom_values = np.array(list(weighted_headrooms.values()))

        # Ensure all headrooms are positive for log
        if np.any(headroom_values <= 0):
            return -np.inf

        # Compute components
        performance_term = self.alpha * performance  # NEW: Explicit performance reward
        sum_term = np.sum(headroom_values)
        softmin_val = softmin(headroom_values, beta=self.beta)
        softmin_term = self.lambda_weight * np.log(softmin_val + self.epsilon)

        return performance_term + sum_term + softmin_term

    def select_action(self) -> Optional[DesignAction]:
        """Select action that maximizes the performance-first objective"""
        if not self.design_space:
            return None

        current_headrooms = self.design_space.get_headrooms()
        current_perf = self.design_space.calculate_performance()
        current_objective = self.calculate_objective(current_headrooms, current_perf)

        best_action = None
        best_objective = current_objective

        for action in self.design_space.actions:
            # Simulate applying the action
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            headrooms = test_space.get_headrooms()
            min_headroom = min(headrooms.values())

            # Skip if below minimum threshold (but threshold is low!)
            if min_headroom < self.min_margin_threshold:
                continue

            perf = test_space.calculate_performance()
            objective_score = self.calculate_objective(headrooms, perf)

            if objective_score > best_objective:
                best_objective = objective_score
                best_action = action

        return best_action


def run_performance_comparison(
    num_runs: int = 50,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    seed: Optional[int] = None,
    verbose: bool = False,
):
    """
    Compare performance-focused agents:
    1. Greedy (baseline - high perf, low survival)
    2. Conservative Softmin JAM (low perf, high survival)
    3. Performance-First JAM variants (trying to beat both!)
    """

    print(f"\n{'='*80}")
    print(f"PERFORMANCE-FIRST JAM EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Goal: Beat Greedy in BOTH performance AND efficiency")
    print(f"Greedy baseline: ~94 perf, ~8.55 perf/W, 28% survival")
    print(f"Current Softmin: ~47 perf, ~7.20 perf/W, 84% survival")
    print(f"{'='*80}\n")

    rng = np.random.RandomState(seed)
    all_results = []

    for run_id in range(num_runs):
        run_seed = rng.randint(0, 1000000)

        # Create base space
        base_space = AdvancedDesignSpace(
            process=ProcessTechnology.create_7nm(),
            seed=run_seed
        )
        base_space.initialize_actions()

        # Create agents
        agents = [
            ("Greedy", AdvancedGreedyPerformanceAgent()),
            ("Softmin JAM (conservative)", SoftminJAMAgent(
                lambda_weight=0.2,
                beta=1.5,
                min_margin_threshold=1.0
            )),
            ("PerfFirst JAM (α=2.0)", PerformanceFirstJAMAgent(
                alpha=2.0,
                lambda_weight=0.1,
                beta=1.5,
                min_margin_threshold=0.5
            )),
            ("PerfFirst JAM (α=3.0)", PerformanceFirstJAMAgent(
                alpha=3.0,
                lambda_weight=0.05,
                beta=1.5,
                min_margin_threshold=0.3
            )),
            ("PerfFirst JAM (α=5.0,aggressive)", PerformanceFirstJAMAgent(
                alpha=5.0,
                lambda_weight=0.02,
                beta=1.0,
                min_margin_threshold=0.2
            )),
        ]

        spaces = []
        for name, agent in agents:
            space = base_space.clone()
            agent.initialize(space)
            spaces.append((name, agent, space))

        # DESIGN PHASE
        for step in range(design_steps):
            for name, agent, space in spaces:
                agent.step()

        # Collect design results
        design_results = {}
        for name, agent, space in spaces:
            constraints = space.calculate_constraints()
            design_results[name] = {
                'performance': space.calculate_performance(),
                'power': constraints['total_power_w'],
                'min_headroom': space.get_min_headroom(),
                'efficiency': space.calculate_performance() / constraints['total_power_w'],
            }

        # REQUIREMENT SHIFT
        shift_type = rng.choice(list(ShiftType))
        survival = {}

        for name, agent, space in spaces:
            space.apply_requirement_shift(shift_type, rng)
            survival[name] = space.is_feasible()

        # ADAPTATION PHASE
        for step in range(adaptation_steps):
            for name, agent, space in spaces:
                if survival[name]:
                    action, feasible = agent.step()
                    if not feasible:
                        survival[name] = False

        # Collect final results
        run_result = {
            'run_id': run_id,
            'shift_type': shift_type.value,
            'agents': {}
        }

        for name, agent, space in spaces:
            constraints = space.calculate_constraints()
            run_result['agents'][name] = {
                'design_performance': design_results[name]['performance'],
                'design_power': design_results[name]['power'],
                'design_efficiency': design_results[name]['efficiency'],
                'design_min_headroom': design_results[name]['min_headroom'],
                'survived': survival[name],
                'final_performance': space.calculate_performance() if survival[name] else 0.0,
                'final_power': constraints['total_power_w'] if survival[name] else 0.0,
            }

        all_results.append(run_result)

        if not verbose and (run_id + 1) % 10 == 0:
            print(f"Completed {run_id + 1}/{num_runs} runs...")

    # AGGREGATE STATISTICS
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}\n")

    agent_names = [name for name, _, _ in spaces]

    stats = {}
    for agent_name in agent_names:
        design_perf = []
        design_power = []
        design_eff = []
        design_headroom = []
        survived_count = 0
        final_perf = []
        final_eff = []

        for result in all_results:
            agent_data = result['agents'][agent_name]
            design_perf.append(agent_data['design_performance'])
            design_power.append(agent_data['design_power'])
            design_eff.append(agent_data['design_efficiency'])
            design_headroom.append(agent_data['design_min_headroom'])

            if agent_data['survived']:
                survived_count += 1
                final_perf.append(agent_data['final_performance'])
                if agent_data['final_power'] > 0:
                    final_eff.append(agent_data['final_performance'] / agent_data['final_power'])

        stats[agent_name] = {
            'design_perf': {'mean': np.mean(design_perf), 'std': np.std(design_perf)},
            'design_power': {'mean': np.mean(design_power), 'std': np.std(design_power)},
            'design_efficiency': {'mean': np.mean(design_eff), 'std': np.std(design_eff)},
            'design_headroom': {'mean': np.mean(design_headroom), 'std': np.std(design_headroom)},
            'survival_rate': survived_count / num_runs,
            'survival_count': survived_count,
            'final_perf': {'mean': np.mean(final_perf) if final_perf else 0.0,
                           'std': np.std(final_perf) if final_perf else 0.0},
            'final_efficiency': {'mean': np.mean(final_eff) if final_eff else 0.0,
                                'std': np.std(final_eff) if final_eff else 0.0},
        }

    # Print comparison table
    print(f"{'Agent':<35s} | {'Perf':>7s} | {'Eff':>7s} | {'Survival':>8s} | {'MinHead':>7s}")
    print("-" * 80)

    for agent_name in agent_names:
        s = stats[agent_name]
        print(f"{agent_name:<35s} | "
              f"{s['design_perf']['mean']:7.1f} | "
              f"{s['design_efficiency']['mean']:7.2f} | "
              f"{s['survival_rate']*100:6.0f}% | "
              f"{s['design_headroom']['mean']:7.2f}")

    # Highlight the best
    print(f"\n{'='*80}")
    print("PERFORMANCE RANKINGS")
    print(f"{'='*80}")

    # Rank by design performance
    by_perf = sorted(agent_names,
                     key=lambda n: stats[n]['design_perf']['mean'],
                     reverse=True)
    print("\nBy Design Performance:")
    for i, name in enumerate(by_perf, 1):
        perf = stats[name]['design_perf']['mean']
        print(f"  {i}. {name:35s}: {perf:7.1f}")

    # Rank by efficiency
    by_eff = sorted(agent_names,
                    key=lambda n: stats[n]['design_efficiency']['mean'],
                    reverse=True)
    print("\nBy Efficiency (perf/W):")
    for i, name in enumerate(by_eff, 1):
        eff = stats[name]['design_efficiency']['mean']
        print(f"  {i}. {name:35s}: {eff:7.2f}")

    # Rank by survival
    by_surv = sorted(agent_names,
                     key=lambda n: stats[n]['survival_rate'],
                     reverse=True)
    print("\nBy Survival Rate:")
    for i, name in enumerate(by_surv, 1):
        rate = stats[name]['survival_rate']
        count = stats[name]['survival_count']
        print(f"  {i}. {name:35s}: {rate*100:5.0f}% ({count}/{num_runs})")

    # Find the WINNER: best combo of performance + efficiency + survival
    print(f"\n{'='*80}")
    print("WINNER ANALYSIS")
    print(f"{'='*80}\n")

    greedy_perf = stats['Greedy']['design_perf']['mean']
    greedy_eff = stats['Greedy']['design_efficiency']['mean']

    print(f"Greedy baseline: {greedy_perf:.1f} perf, {greedy_eff:.2f} eff")
    print(f"\nAgents that BEAT Greedy in BOTH perf AND eff:")

    winners = []
    for name in agent_names:
        if name == "Greedy":
            continue
        s = stats[name]
        perf = s['design_perf']['mean']
        eff = s['design_efficiency']['mean']
        survival = s['survival_rate']

        if perf >= greedy_perf and eff >= greedy_eff:
            winners.append(name)
            print(f"  ✓ {name:35s}: {perf:7.1f} perf (+{perf-greedy_perf:4.1f}), "
                  f"{eff:6.2f} eff (+{eff-greedy_eff:4.2f}), "
                  f"{survival*100:4.0f}% survival")

    if not winners:
        print("  (No agent beat Greedy in BOTH metrics)")
        print(f"\nClosest performers:")
        for name in agent_names:
            if name == "Greedy":
                continue
            s = stats[name]
            perf = s['design_perf']['mean']
            eff = s['design_efficiency']['mean']
            survival = s['survival_rate']
            print(f"  - {name:35s}: {perf:7.1f} perf ({perf/greedy_perf*100:4.0f}%), "
                  f"{eff:6.2f} eff ({eff/greedy_eff*100:4.0f}%), "
                  f"{survival*100:4.0f}% survival")

    # Save results
    output = {
        'parameters': {
            'num_runs': num_runs,
            'design_steps': design_steps,
            'adaptation_steps': adaptation_steps,
        },
        'stats': {k: {
            'design_performance': v['design_perf'],
            'design_efficiency': v['design_efficiency'],
            'survival_rate': v['survival_rate'],
            'survival_count': v['survival_count'],
        } for k, v in stats.items()},
        'results': all_results,
    }

    with open('performance_jam_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print("Results saved to: performance_jam_results.json")
    print(f"{'='*80}\n")

    return output


if __name__ == "__main__":
    run_performance_comparison(
        num_runs=50,
        design_steps=75,
        adaptation_steps=25,
        seed=42,
        verbose=False,
    )
