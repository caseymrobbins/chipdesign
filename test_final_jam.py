#!/usr/bin/env python3
"""
Final Optimized JAM: Add performance term to Conservative Softmin JAM

INSIGHT: Conservative Softmin works (82% survival) but gets only 47 perf because
it doesn't optimize for performance. Let's fix that!

Objective: R = α·perf + sum(headrooms) + λ·log(softmin(headrooms; β) + ε)

Where α controls how much we value raw performance.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedAgent,
    AdvancedGreedyPerformanceAgent,
    DesignAction,
    ProcessTechnology,
    ShiftType,
)
from test_softmin_jam import SoftminJAMAgent, softmin
import json


class OptimizedJAMAgent(AdvancedAgent):
    """
    Optimized JAM: Conservative Softmin + Performance term

    R = α·perf + sum(headrooms) + λ·log(softmin(headrooms; β) + ε)
    """

    def __init__(
        self,
        alpha: float = 0.5,               # Performance weight
        lambda_weight: float = 0.2,
        beta: float = 1.5,
        min_margin_threshold: float = 0.8,  # Slightly lower than conservative (1.0)
        epsilon: float = 0.01,
    ):
        super().__init__(f"OptJAM(α={alpha},λ={lambda_weight},β={beta})")
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        self.beta = beta
        self.min_margin_threshold = min_margin_threshold
        self.epsilon = epsilon

    def calculate_objective(self, headrooms_dict: Dict[str, float], performance: float) -> float:
        """
        R = α·perf + sum(headrooms) + λ·log(softmin(headrooms; β) + ε)
        """
        weights = self.design_space.limits.constraint_weights
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms_dict.items()
        }

        headroom_values = np.array(list(weighted_headrooms.values()))

        if np.any(headroom_values <= 0):
            return -np.inf

        # Components
        perf_term = self.alpha * performance
        sum_term = np.sum(headroom_values)
        softmin_val = softmin(headroom_values, beta=self.beta)
        softmin_term = self.lambda_weight * np.log(softmin_val + self.epsilon)

        return perf_term + sum_term + softmin_term

    def select_action(self) -> Optional[DesignAction]:
        """Select action maximizing the objective"""
        if not self.design_space:
            return None

        current_headrooms = self.design_space.get_headrooms()
        current_perf = self.design_space.calculate_performance()

        safe_actions = []
        risky_actions = []

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            headrooms = test_space.get_headrooms()
            min_headroom = min(headrooms.values())
            perf = test_space.calculate_performance()
            objective_score = self.calculate_objective(headrooms, perf)

            action_data = (action, objective_score, perf, min_headroom)

            if min_headroom >= self.min_margin_threshold:
                safe_actions.append(action_data)
            else:
                risky_actions.append(action_data)

        # Prefer safe actions
        if safe_actions:
            best = max(safe_actions, key=lambda x: x[1])  # Just use objective score
            return best[0]

        # No safe actions - try risky ones
        elif risky_actions:
            improving = [a for a in risky_actions if a[3] >= min(current_headrooms.values())]
            if improving:
                best = max(improving, key=lambda x: x[1])
                return best[0]

        return None


def run_final_comparison(
    num_runs: int = 50,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    seed: Optional[int] = None,
):
    """Final showdown: Can we beat Greedy?"""

    print(f"\n{'='*80}")
    print(f"FINAL OPTIMIZED JAM COMPARISON")
    print(f"{'='*80}")
    print(f"Strategy: Conservative Softmin + Performance awareness")
    print(f"Goal: Beat Greedy (93.9 perf, 8.54 eff, 64% survival)")
    print(f"{'='*80}\n")

    rng = np.random.RandomState(seed)
    all_results = []

    for run_id in range(num_runs):
        run_seed = rng.randint(0, 1000000)

        base_space = AdvancedDesignSpace(
            process=ProcessTechnology.create_7nm(),
            seed=run_seed
        )
        base_space.initialize_actions()

        # Test various alpha values (performance weight)
        agents = [
            ("Greedy", AdvancedGreedyPerformanceAgent()),
            ("Conservative Softmin", SoftminJAMAgent(
                lambda_weight=0.2,
                beta=1.5,
                min_margin_threshold=1.0
            )),
            ("OptJAM (α=0.1)", OptimizedJAMAgent(
                alpha=0.1,
                lambda_weight=0.2,
                beta=1.5,
                min_margin_threshold=0.8
            )),
            ("OptJAM (α=0.3)", OptimizedJAMAgent(
                alpha=0.3,
                lambda_weight=0.2,
                beta=1.5,
                min_margin_threshold=0.6
            )),
            ("OptJAM (α=0.5)", OptimizedJAMAgent(
                alpha=0.5,
                lambda_weight=0.15,
                beta=1.5,
                min_margin_threshold=0.5
            )),
            ("OptJAM (α=1.0,aggressive)", OptimizedJAMAgent(
                alpha=1.0,
                lambda_weight=0.1,
                beta=1.5,
                min_margin_threshold=0.4
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
            'agents': {}
        }

        for name, agent, space in spaces:
            run_result['agents'][name] = {
                'design_performance': design_results[name]['performance'],
                'design_efficiency': design_results[name]['efficiency'],
                'design_min_headroom': design_results[name]['min_headroom'],
                'survived': survival[name],
            }

        all_results.append(run_result)

        if (run_id + 1) % 10 == 0:
            print(f"Completed {run_id + 1}/{num_runs} runs...")

    # AGGREGATE
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")

    agent_names = [name for name, _, _ in spaces]
    stats = {}

    for agent_name in agent_names:
        perf = [r['agents'][agent_name]['design_performance'] for r in all_results]
        eff = [r['agents'][agent_name]['design_efficiency'] for r in all_results]
        headroom = [r['agents'][agent_name]['design_min_headroom'] for r in all_results]
        survived = sum(1 for r in all_results if r['agents'][agent_name]['survived'])

        stats[agent_name] = {
            'perf': np.mean(perf),
            'eff': np.mean(eff),
            'headroom': np.mean(headroom),
            'survival': survived / num_runs,
        }

    # Print table
    print(f"{'Agent':<30s} | {'Perf':>7s} | {'Eff':>7s} | {'Surv':>6s} | {'MinHead':>7s}")
    print("-" * 75)

    for name in agent_names:
        s = stats[name]
        print(f"{name:<30s} | "
              f"{s['perf']:7.1f} | "
              f"{s['eff']:7.2f} | "
              f"{s['survival']*100:5.0f}% | "
              f"{s['headroom']:7.2f}")

    # Find winners
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA")
    print(f"{'='*80}\n")

    greedy = stats['Greedy']
    print(f"Greedy baseline: {greedy['perf']:.1f} perf, {greedy['eff']:.2f} eff, {greedy['survival']*100:.0f}% survival\n")

    winners = []
    for name in agent_names:
        if name == "Greedy":
            continue
        s = stats[name]

        beats_perf = s['perf'] >= greedy['perf']
        beats_eff = s['eff'] >= greedy['eff']
        beats_surv = s['survival'] >= greedy['survival']

        wins = sum([beats_perf, beats_eff, beats_surv])
        if wins >= 2:
            winners.append(name)
            print(f"✓ {name:<30s} BEATS GREEDY IN {wins}/3 METRICS!")
            if beats_perf:
                print(f"  ✓ Performance: {s['perf']:.1f} vs {greedy['perf']:.1f} (+{s['perf']-greedy['perf']:.1f})")
            if beats_eff:
                print(f"  ✓ Efficiency: {s['eff']:.2f} vs {greedy['eff']:.2f} (+{s['eff']-greedy['eff']:.2f})")
            if beats_surv:
                print(f"  ✓ Survival: {s['survival']*100:.0f}% vs {greedy['survival']*100:.0f}% (+{(s['survival']-greedy['survival'])*100:.0f}%)")
            print()

    if not winners:
        print("No agent beats Greedy in 2+ metrics yet. Best performers:\n")
        for name in sorted(agent_names, key=lambda n: stats[n]['perf'], reverse=True):
            if name == "Greedy":
                continue
            s = stats[name]
            print(f"  {name:<30s}: {s['perf']:5.1f} perf ({s['perf']/greedy['perf']*100:3.0f}%), "
                  f"{s['eff']:5.2f} eff ({s['eff']/greedy['eff']*100:3.0f}%), "
                  f"{s['survival']*100:3.0f}% surv")

    # Save
    with open('final_jam_results.json', 'w') as f:
        json.dump({'stats': stats, 'results': all_results}, f, indent=2)

    print(f"\n{'='*80}")
    print("Results saved to: final_jam_results.json")
    print(f"{'='*80}\n")

    return stats


if __name__ == "__main__":
    run_final_comparison(
        num_runs=50,
        design_steps=75,
        adaptation_steps=25,
        seed=42,
    )
