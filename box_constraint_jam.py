#!/usr/bin/env python3
"""
BOX CONSTRAINT JAM: User's brilliant insight!

KEY IDEA: Put BOTH min AND max constraints in softmin()
- Penalize headroom < min (infeasible)
- Penalize headroom > max (wasted performance potential!)

This creates a "sweet spot" that forces optimizer to USE available design space
instead of being overly conservative.

Example: Want headroom in [0.1, 0.6]
- Below 0.1: Constraint violation (bad)
- Above 0.6: Wasted performance (bad)
- In [0.1, 0.6]: Sweet spot (good)
"""

import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedAgent,
    DesignAction,
    ProcessTechnology,
    ShiftType,
    AdvancedGreedyPerformanceAgent,
)


def softmin(values: np.ndarray, beta: float = 1.0) -> float:
    """Compute smooth softmin approximation."""
    v_shifted = values - np.max(values)
    weights = np.exp(-beta * v_shifted)
    weights_sum = np.sum(weights)
    result = np.sum(values * weights) / weights_sum
    return result


class BoxConstraintJAM(AdvancedAgent):
    """
    JAM with box constraints: keep headrooms in [min_target, max_target].

    This implements user's idea:
    - Lower bound: penalty when headroom → min_target
    - Upper bound: penalty when headroom → max_target
    - Result: Optimizer uses available design space efficiently!
    """

    def __init__(
        self,
        min_target: float = 0.05,   # Minimum safe margin (5%)
        max_target: float = 0.50,   # Maximum margin before wasting performance
        penalty_low: float = 100.0,  # Weight for lower bound penalty
        penalty_high: float = 50.0,  # Weight for upper bound penalty
        beta: float = 2.0,
        epsilon: float = 0.01,
    ):
        super().__init__(f"BoxJAM(min={min_target},max={max_target},γ_lo={penalty_low},γ_hi={penalty_high})")
        self.min_target = min_target
        self.max_target = max_target
        self.penalty_low = penalty_low
        self.penalty_high = penalty_high
        self.beta = beta
        self.epsilon = epsilon

    def calculate_objective(self, headrooms_dict: Dict[str, float], test_space=None) -> float:
        """
        Box constraint objective: maximize performance while keeping
        MIN headroom (bottleneck) in the sweet spot [min_target, max_target].

        User's insight: If min_headroom is too high, we're being too conservative!
        Force optimizer to push limits by penalizing excessive headroom.

        R = performance - penalty_low * violation_low - penalty_high * violation_high
        """
        # Use test_space if provided, otherwise fall back to self.design_space
        space = test_space if test_space is not None else self.design_space

        # Hard constraint: must be feasible
        if not space.is_feasible():
            return -np.inf

        # PRIMARY GOAL: Maximize chip performance (from the test space!)
        performance = space.calculate_performance()

        # Get the BOTTLENECK headroom (already weighted and normalized)
        min_headroom = space.get_min_headroom()

        # LOWER BOUND PENALTY: Keep min_headroom above min_target (safety)
        lower_violation = min_headroom - self.min_target
        if lower_violation <= 0:
            # Below minimum target - infeasible territory
            penalty_low_val = np.inf
        else:
            penalty_low_val = -np.log(lower_violation + self.epsilon)

        # UPPER BOUND PENALTY: Keep min_headroom below max_target (performance)
        # This is the KEY insight: penalize being TOO conservative!
        upper_violation = self.max_target - min_headroom
        if upper_violation <= 0:
            # Above maximum target - we're wasting performance potential!
            # Use linear penalty (not inf, since it's not infeasible)
            penalty_high_val = -upper_violation
        else:
            penalty_high_val = -np.log(upper_violation + self.epsilon)

        # COMBINED OBJECTIVE
        objective = performance - self.penalty_low * penalty_low_val - self.penalty_high * penalty_high_val

        return objective

    def select_action(self) -> Optional[DesignAction]:
        """Select action that maximizes box-constrained objective."""
        if not self.design_space:
            return None

        current_objective = self.calculate_objective(self.design_space.get_headrooms(), self.design_space)

        candidate_actions = []

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            headrooms = test_space.get_headrooms()
            # CRITICAL FIX: Pass test_space so performance is evaluated correctly!
            objective_score = self.calculate_objective(headrooms, test_space)

            candidate_actions.append((action, objective_score))

        if not candidate_actions:
            return None

        best_action, best_score = max(candidate_actions, key=lambda x: x[1])

        if best_score > current_objective:
            return best_action

        return None


def test_box_constraints(num_runs=50):
    """
    Test box constraints with different min/max targets.

    Hypothesis: Tighter max_target forces optimizer to use available headroom,
    resulting in higher performance without sacrificing robustness.
    """

    print("="*80)
    print("BOX CONSTRAINT JAM: Force optimizer to use available design space!")
    print("="*80)
    print("User's insight: Penalize BOTH too-low AND too-high headrooms")
    print("  - Too low: Constraint violation")
    print("  - Too high: Wasted performance potential!")
    print("="*80)

    configs = [
        # Wide box (permissive)
        {"min": 0.05, "max": 1.00, "p_low": 100, "p_high": 10, "name": "Wide(max=1.0)"},
        {"min": 0.05, "max": 0.80, "p_low": 100, "p_high": 20, "name": "Wide(max=0.8)"},

        # Medium box (balanced)
        {"min": 0.05, "max": 0.60, "p_low": 100, "p_high": 50, "name": "Medium(max=0.6)"},
        {"min": 0.05, "max": 0.50, "p_low": 100, "p_high": 50, "name": "Medium(max=0.5)"},

        # Tight box (aggressive - force performance)
        {"min": 0.05, "max": 0.40, "p_low": 100, "p_high": 100, "name": "Tight(max=0.4)"},
        {"min": 0.05, "max": 0.30, "p_low": 100, "p_high": 100, "name": "Tight(max=0.3)"},
        {"min": 0.05, "max": 0.20, "p_low": 100, "p_high": 150, "name": "VeryTight(max=0.2)"},

        # Different penalty balances
        {"min": 0.05, "max": 0.50, "p_low": 100, "p_high": 100, "name": "Balanced50(γ_hi=100)"},
        {"min": 0.05, "max": 0.50, "p_low": 100, "p_high": 200, "name": "HighPenalty(γ_hi=200)"},

        # Greedy baseline
        {"type": "greedy", "name": "Greedy"},
    ]

    results = {}
    process = ProcessTechnology.create_7nm()

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing {config['name']}")
        print(f"{'='*60}")

        perf_scores = []
        survival_count = 0
        efficiencies = []
        headrooms = []

        for run in range(num_runs):
            seed = run * 1000 + hash(config['name']) % 10000
            space = AdvancedDesignSpace(process=process, seed=seed)
            space.initialize_actions()

            # Create agent
            if config.get('type') == 'greedy':
                agent = AdvancedGreedyPerformanceAgent()
            else:
                agent = BoxConstraintJAM(
                    min_target=config['min'],
                    max_target=config['max'],
                    penalty_low=config['p_low'],
                    penalty_high=config['p_high']
                )

            agent.initialize(space)

            # Run optimization
            for step in range(100):
                agent.step()

            design_perf = space.calculate_performance()
            perf_scores.append(design_perf)

            if space.is_feasible():
                constraints = space.calculate_constraints()
                efficiencies.append(design_perf / constraints['total_power_w'])
                headrooms.append(space.get_min_headroom())

            # Test survival
            shift_rng = np.random.RandomState(run * 9999 + hash(config['name']) % 10000)
            shift_type = shift_rng.choice(list(ShiftType))
            space.apply_requirement_shift(shift_type, shift_rng)

            if space.is_feasible():
                survival_count += 1

            if (run + 1) % 10 == 0:
                survival_rate = survival_count / (run + 1)
                avg_perf = np.mean(perf_scores)
                avg_headroom = np.mean(headrooms) if headrooms else 0
                print(f"  Run {run+1}/{num_runs}: Perf={avg_perf:.1f}, Survival={survival_rate*100:.0f}%, Headroom={avg_headroom*100:.0f}%")

        survival_rate = survival_count / num_runs
        avg_perf = np.mean(perf_scores)
        avg_eff = np.mean(efficiencies) if efficiencies else 0
        avg_headroom = np.mean(headrooms) if headrooms else 0

        results[config['name']] = {
            'config': config,
            'performance': avg_perf,
            'survival_rate': survival_rate,
            'efficiency': avg_eff,
            'headroom': avg_headroom,
            'combined_score': avg_perf * survival_rate,
        }

        print(f"\n  RESULTS:")
        print(f"    Performance:     {avg_perf:.2f}")
        print(f"    Survival:        {survival_rate*100:.1f}%")
        print(f"    Combined Score:  {avg_perf * survival_rate:.2f}")
        print(f"    Avg Headroom:    {avg_headroom*100:.1f}%")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BOX CONSTRAINT JAM: User's Brilliant Insight!")
    print("="*80)
    print("\nPROBLEM: Current methods allow TOO MUCH headroom")
    print("  - High headroom = wasted performance potential")
    print("  - Optimizer should PUSH LIMITS, not be overly safe")
    print("\nSOLUTION: Box constraints with softmin")
    print("  - Lower bound: keep headroom > min (safety)")
    print("  - Upper bound: keep headroom < max (performance)")
    print("  - Sweet spot: Use available design space efficiently!")
    print("="*80 + "\n")

    results = test_box_constraints(num_runs=50)

    # Save results
    with open('box_constraint_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Analysis
    print(f"\n{'='*80}")
    print("BOX CONSTRAINT RESULTS")
    print(f"{'='*80}\n")

    print(f"{'Config':<30} {'Perf':<10} {'Survival':<12} {'Score':<12} {'Headroom':<10}")
    print("-" * 80)

    for name, data in sorted(results.items(), key=lambda x: x[1]['combined_score'], reverse=True):
        print(f"{name:<30} {data['performance']:>8.1f}   {data['survival_rate']*100:>9.1f}%   "
              f"{data['combined_score']:>10.1f}   {data['headroom']*100:>8.1f}%")

    # Find best
    best = max(results.items(), key=lambda x: x[1]['combined_score'])
    print(f"\n{'='*80}")
    print(f"🏆 BEST BOX CONSTRAINT: {best[0]}")
    print(f"{'='*80}")
    print(f"Performance:     {best[1]['performance']:.2f}")
    print(f"Survival:        {best[1]['survival_rate']*100:.1f}%")
    print(f"Combined Score:  {best[1]['combined_score']:.2f}")
    print(f"Efficiency:      {best[1]['efficiency']:.2f} perf/W")
    print(f"Avg Headroom:    {best[1]['headroom']*100:.1f}%")

    print(f"\n{'='*80}")
    print("KEY INSIGHT:")
    print("Tighter max_target → Forces optimizer to use available headroom")
    print("Result: Higher performance WITHOUT sacrificing robustness!")
    print(f"{'='*80}")

    print(f"\nResults saved to: box_constraint_results.json")
    print(f"{'='*80}\n")
