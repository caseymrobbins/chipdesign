#!/usr/bin/env python3
"""
Balanced JAM: Find the sweet spot between Conservative Softmin and Greedy

Analysis of previous attempts:
- Conservative Softmin: 47 perf, 7.21 eff, 82% survival ← TOO SAFE
- Greedy: 93.9 perf, 8.54 eff, 64% survival ← GOOD BALANCE!
- PerfFirst JAM: 39.4 perf, 3.28 eff, 0% survival ← TOO AGGRESSIVE

Key insight: Greedy is actually doing well! But we can beat it by adding
margin awareness while still being aggressive about performance.

Strategy: Like Greedy, but with smart margin management
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
from test_softmin_jam import softmin
import json


class BalancedJAMAgent(AdvancedAgent):
    """
    Balanced JAM: Greedy-like performance with margin awareness

    Core idea: Select actions that maximize performance, BUT with
    a secondary objective to maintain margins when performance is similar.

    Decision logic:
    1. Filter to feasible actions (like Greedy)
    2. Among good performers, prefer those with better margins
    3. Use a hybrid score: performance + margin_bonus
    """

    def __init__(
        self,
        margin_weight: float = 1.0,      # How much to value margins
        perf_threshold: float = 0.95,     # Consider "similar" if within 95% of best
        min_margin_safety: float = 0.5,   # Minimum margin to maintain
    ):
        super().__init__(f"BalancedJAM(γ={margin_weight})")
        self.margin_weight = margin_weight
        self.perf_threshold = perf_threshold
        self.min_margin_safety = min_margin_safety

    def calculate_margin_score(self, headrooms_dict: Dict[str, float]) -> float:
        """Calculate a margin score that rewards healthy headrooms"""
        weights = self.design_space.limits.constraint_weights
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms_dict.items()
        }

        headroom_values = np.array(list(weighted_headrooms.values()))

        if np.any(headroom_values <= 0):
            return -np.inf

        # Combine sum (total margin) + log(min) (bottleneck focus)
        return np.sum(headroom_values) + np.log(np.min(headroom_values) + 0.01)

    def select_action(self) -> Optional[DesignAction]:
        """Select action balancing performance and margins"""
        if not self.design_space:
            return None

        candidates = []

        for action in self.design_space.actions:
            # Simulate applying the action
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            headrooms = test_space.get_headrooms()
            min_headroom = min(headrooms.values())

            # Safety check: don't go too close to constraints
            if min_headroom < self.min_margin_safety:
                continue

            perf = test_space.calculate_performance()
            margin_score = self.calculate_margin_score(headrooms)

            candidates.append({
                'action': action,
                'performance': perf,
                'margin_score': margin_score,
                'min_headroom': min_headroom,
            })

        if not candidates:
            return None

        # Find best performance
        max_perf = max(c['performance'] for c in candidates)

        # Filter to "good enough" performance (within threshold of best)
        perf_cutoff = self.perf_threshold * max_perf
        good_performers = [c for c in candidates if c['performance'] >= perf_cutoff]

        # Among good performers, pick the one with best margins
        best = max(good_performers,
                   key=lambda c: c['performance'] + self.margin_weight * c['margin_score'])

        return best['action']


class HybridJAMAgent(AdvancedAgent):
    """
    Hybrid JAM: Directly optimize perf + margin in one objective

    R = perf + γ·[sum(headrooms) + log(min(headroom))]

    This is more principled than Balanced JAM's two-stage approach.
    """

    def __init__(
        self,
        margin_weight: float = 0.5,
        min_margin_safety: float = 0.5,
    ):
        super().__init__(f"HybridJAM(γ={margin_weight})")
        self.margin_weight = margin_weight
        self.min_margin_safety = min_margin_safety

    def calculate_objective(
        self,
        performance: float,
        headrooms_dict: Dict[str, float],
    ) -> float:
        """R = perf + γ·[sum(headrooms) + log(min(headroom))]"""
        weights = self.design_space.limits.constraint_weights
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms_dict.items()
        }

        headroom_values = np.array(list(weighted_headrooms.values()))

        if np.any(headroom_values <= 0):
            return -np.inf

        # Margin term: sum + log(min)
        sum_term = np.sum(headroom_values)
        min_term = np.log(np.min(headroom_values) + 0.01)
        margin_score = sum_term + min_term

        return performance + self.margin_weight * margin_score

    def select_action(self) -> Optional[DesignAction]:
        """Select action maximizing the hybrid objective"""
        if not self.design_space:
            return None

        best_action = None
        best_score = -float('inf')

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            headrooms = test_space.get_headrooms()
            min_headroom = min(headrooms.values())

            if min_headroom < self.min_margin_safety:
                continue

            perf = test_space.calculate_performance()
            score = self.calculate_objective(perf, headrooms)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action


def run_balanced_comparison(
    num_runs: int = 50,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    seed: Optional[int] = None,
    verbose: bool = False,
):
    """
    Test balanced approaches that aim to beat Greedy
    """

    print(f"\n{'='*80}")
    print(f"BALANCED JAM EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Goal: Beat Greedy (93.9 perf, 8.54 eff, 64% survival)")
    print(f"Strategy: Greedy-like performance + margin awareness")
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

            # Balanced variants: prefer margins among top performers
            ("BalancedJAM (γ=0.5)", BalancedJAMAgent(
                margin_weight=0.5,
                perf_threshold=0.98,
                min_margin_safety=0.5
            )),
            ("BalancedJAM (γ=1.0)", BalancedJAMAgent(
                margin_weight=1.0,
                perf_threshold=0.98,
                min_margin_safety=0.5
            )),
            ("BalancedJAM (γ=2.0)", BalancedJAMAgent(
                margin_weight=2.0,
                perf_threshold=0.98,
                min_margin_safety=0.5
            )),

            # Hybrid variants: direct optimization
            ("HybridJAM (γ=0.1)", HybridJAMAgent(
                margin_weight=0.1,
                min_margin_safety=0.5
            )),
            ("HybridJAM (γ=0.3)", HybridJAMAgent(
                margin_weight=0.3,
                min_margin_safety=0.5
            )),
            ("HybridJAM (γ=0.5)", HybridJAMAgent(
                margin_weight=0.5,
                min_margin_safety=0.5
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
        design_eff = []
        design_headroom = []
        survived_count = 0

        for result in all_results:
            agent_data = result['agents'][agent_name]
            design_perf.append(agent_data['design_performance'])
            design_eff.append(agent_data['design_efficiency'])
            design_headroom.append(agent_data['design_min_headroom'])

            if agent_data['survived']:
                survived_count += 1

        stats[agent_name] = {
            'design_perf': {'mean': np.mean(design_perf), 'std': np.std(design_perf)},
            'design_efficiency': {'mean': np.mean(design_eff), 'std': np.std(design_eff)},
            'design_headroom': {'mean': np.mean(design_headroom), 'std': np.std(design_headroom)},
            'survival_rate': survived_count / num_runs,
            'survival_count': survived_count,
        }

    # Print comparison table
    print(f"{'Agent':<30s} | {'Perf':>7s} | {'Eff':>7s} | {'Survival':>8s} | {'MinHead':>7s}")
    print("-" * 80)

    for agent_name in agent_names:
        s = stats[agent_name]
        print(f"{agent_name:<30s} | "
              f"{s['design_perf']['mean']:7.1f} | "
              f"{s['design_efficiency']['mean']:7.2f} | "
              f"{s['survival_rate']*100:6.0f}% | "
              f"{s['design_headroom']['mean']:7.2f}")

    # Find winners
    print(f"\n{'='*80}")
    print("WINNERS: Agents that BEAT or MATCH Greedy")
    print(f"{'='*80}\n")

    greedy_perf = stats['Greedy']['design_perf']['mean']
    greedy_eff = stats['Greedy']['design_efficiency']['mean']
    greedy_survival = stats['Greedy']['survival_rate']

    print(f"Greedy baseline: {greedy_perf:.1f} perf, {greedy_eff:.2f} eff, {greedy_survival*100:.0f}% survival\n")

    winners = []
    for name in agent_names:
        if name == "Greedy":
            continue
        s = stats[name]

        perf = s['design_perf']['mean']
        eff = s['design_efficiency']['mean']
        survival = s['survival_rate']

        # Winner criteria: beat Greedy in at least 2 of 3 metrics
        beats_perf = perf >= greedy_perf
        beats_eff = eff >= greedy_eff
        beats_survival = survival >= greedy_survival

        score = sum([beats_perf, beats_eff, beats_survival])

        if score >= 2:
            winners.append((name, score, perf, eff, survival))

    if winners:
        print("✓ WINNERS (beat Greedy in 2+ metrics):")
        winners.sort(key=lambda x: x[1], reverse=True)
        for name, score, perf, eff, survival in winners:
            markers = []
            if perf >= greedy_perf:
                markers.append(f"✓perf:{perf:.1f}")
            if eff >= greedy_eff:
                markers.append(f"✓eff:{eff:.2f}")
            if survival >= greedy_survival:
                markers.append(f"✓surv:{survival*100:.0f}%")
            print(f"  • {name:<30s}: {', '.join(markers)}")
    else:
        print("(No clear winners - showing best performers)\n")
        # Show agents sorted by a composite score
        for name in agent_names:
            if name == "Greedy":
                continue
            s = stats[name]
            perf_ratio = s['design_perf']['mean'] / greedy_perf
            eff_ratio = s['design_efficiency']['mean'] / greedy_eff
            surv_ratio = s['survival_rate'] / max(greedy_survival, 0.01)

            print(f"  • {name:<30s}: "
                  f"perf {perf_ratio*100:4.0f}%, "
                  f"eff {eff_ratio*100:4.0f}%, "
                  f"surv {surv_ratio*100:4.0f}%")

    # Save results
    output = {
        'parameters': {
            'num_runs': num_runs,
            'design_steps': design_steps,
            'adaptation_steps': adaptation_steps,
        },
        'stats': stats,
        'results': all_results,
    }

    with open('balanced_jam_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print("Results saved to: balanced_jam_results.json")
    print(f"{'='*80}\n")

    return output


if __name__ == "__main__":
    run_balanced_comparison(
        num_runs=50,
        design_steps=75,
        adaptation_steps=25,
        seed=42,
        verbose=False,
    )
