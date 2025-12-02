#!/usr/bin/env python3
"""
PERFORMANCE-FIRST JAM: Fix the objective function!

KEY INSIGHTS (from user analysis):
1. Hard margin constraint creates cliff - ALL constraints must go in softmin()
2. sum(headrooms) optimizes for CONSERVATIVE designs, not chip performance
3. We should optimize for PERFORMANCE with soft penalties for constraint violations

NEW OBJECTIVE:
R = performance - penalty(low_headrooms)

Where penalty uses log-barrier or softmin to keep headrooms positive
WITHOUT rewarding excessively high headrooms (which waste performance potential)
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


class PerformanceFirstJAM(AdvancedAgent):
    """
    JAM agent that optimizes for PERFORMANCE with soft margin constraints.

    Key differences from original:
    1. NO hard margin threshold (no cliff!)
    2. Objective = performance - penalty(headrooms)
    3. Penalty only kicks in when headrooms get LOW, doesn't reward HIGH headrooms
    """

    def __init__(
        self,
        penalty_weight: float = 10.0,
        penalty_beta: float = 2.0,
        target_margin: float = 0.1,  # Soft target, not hard threshold
        epsilon: float = 0.01,
    ):
        super().__init__(f"PerfFirstJAM(penalty={penalty_weight},β={penalty_beta},target={target_margin})")
        self.penalty_weight = penalty_weight
        self.penalty_beta = penalty_beta
        self.target_margin = target_margin
        self.epsilon = epsilon

    def calculate_objective(self, headrooms_dict: Dict[str, float]) -> float:
        """
        NEW OBJECTIVE: Maximize performance with soft margin penalties

        R = performance - penalty_weight * violation_penalty

        Where violation_penalty uses log-barrier to softly penalize low headrooms
        """
        # Get weighted headrooms
        weights = self.design_space.limits.constraint_weights
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms_dict.items()
        }

        headroom_values = np.array(list(weighted_headrooms.values()))

        # If any headroom is negative, this is infeasible
        if np.any(headroom_values <= 0):
            return -np.inf

        # PRIMARY GOAL: Maximize chip performance!
        performance = self.design_space.calculate_performance()

        # SOFT PENALTY: Log-barrier to keep headrooms positive
        # penalty = -log(softmin(headrooms))
        # This goes to +inf as any headroom → 0, but doesn't care if headroom is HIGH
        softmin_val = softmin(headroom_values, beta=self.penalty_beta)
        barrier_penalty = -np.log(softmin_val + self.epsilon)

        # Combine: maximize performance, minimize barrier penalty
        objective = performance - self.penalty_weight * barrier_penalty

        return objective

    def select_action(self) -> Optional[DesignAction]:
        """
        Select action that maximizes objective.

        KEY DIFFERENCE: NO hard margin threshold!
        All actions considered based purely on objective value.
        """
        if not self.design_space:
            return None

        current_objective = self.calculate_objective(self.design_space.get_headrooms())

        # Evaluate ALL feasible actions (no safe/risky split!)
        candidate_actions = []

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            # Only check basic feasibility (constraints met)
            if not test_space.is_feasible():
                continue

            headrooms = test_space.get_headrooms()
            objective_score = self.calculate_objective(headrooms)

            candidate_actions.append((action, objective_score))

        if not candidate_actions:
            return None

        # Pick action with BEST objective (performance - penalty)
        best_action, best_score = max(candidate_actions, key=lambda x: x[1])

        # Only take action if it improves objective
        if best_score > current_objective:
            return best_action

        return None


class HybridObjectiveJAM(AdvancedAgent):
    """
    Test alternative objective formulations to find what works best.
    """

    def __init__(
        self,
        objective_type: str = "performance_first",
        alpha: float = 1.0,  # Weight for performance
        beta: float = 1.0,   # Softmin temperature
        gamma: float = 10.0, # Penalty weight
    ):
        super().__init__(f"HybridJAM({objective_type},α={alpha},β={beta},γ={gamma})")
        self.objective_type = objective_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = 0.01

    def calculate_objective(self, headrooms_dict: Dict[str, float]) -> float:
        """Test different objective formulations"""
        weights = self.design_space.limits.constraint_weights
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms_dict.items()
        }

        headroom_values = np.array(list(weighted_headrooms.values()))

        if np.any(headroom_values <= 0):
            return -np.inf

        performance = self.design_space.calculate_performance()
        softmin_val = softmin(headroom_values, beta=self.beta)
        sum_headrooms = np.sum(headroom_values)

        if self.objective_type == "performance_first":
            # Maximize performance with log-barrier penalty
            return self.alpha * performance - self.gamma * (-np.log(softmin_val + self.epsilon))

        elif self.objective_type == "performance_ratio":
            # Maximize performance / headroom_penalty ratio
            # This naturally trades off performance vs margins
            return self.alpha * performance / (1.0 + self.gamma * (-np.log(softmin_val + self.epsilon)))

        elif self.objective_type == "original":
            # Original formulation for comparison
            return sum_headrooms + self.gamma * np.log(softmin_val + self.epsilon)

        elif self.objective_type == "performance_only":
            # Pure performance (greedy-like) with minimal penalty
            return performance - self.gamma * (-np.log(softmin_val + self.epsilon))

        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")

    def select_action(self) -> Optional[DesignAction]:
        """Select action - no hard constraints!"""
        if not self.design_space:
            return None

        current_objective = self.calculate_objective(self.design_space.get_headrooms())

        candidate_actions = []

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            headrooms = test_space.get_headrooms()
            objective_score = self.calculate_objective(headrooms)

            candidate_actions.append((action, objective_score))

        if not candidate_actions:
            return None

        best_action, best_score = max(candidate_actions, key=lambda x: x[1])

        if best_score > current_objective:
            return best_action

        return None


def test_objective_formulations(num_runs=50):
    """
    Test different objective formulations to see which maximizes performance.

    Hypotheses:
    1. Performance-first should beat original (sum-based) formulation
    2. Removing hard margin threshold should unlock higher performance
    3. Optimal penalty weight balances performance vs robustness
    """

    print("="*80)
    print("OBJECTIVE FUNCTION AUDIT")
    print("="*80)
    print("Testing if sum(headrooms) is holding us back!")
    print("Goal: Optimize for PERFORMANCE, not conservative margins")
    print("="*80)

    configs = [
        # Original baseline (for comparison)
        {"type": "original", "alpha": 1.0, "beta": 2.0, "gamma": 1.0, "name": "Original(sum+softmin)"},

        # Performance-first with varying penalty weights
        {"type": "performance_first", "alpha": 1.0, "beta": 2.0, "gamma": 5.0, "name": "PerfFirst(γ=5)"},
        {"type": "performance_first", "alpha": 1.0, "beta": 2.0, "gamma": 10.0, "name": "PerfFirst(γ=10)"},
        {"type": "performance_first", "alpha": 1.0, "beta": 2.0, "gamma": 20.0, "name": "PerfFirst(γ=20)"},
        {"type": "performance_first", "alpha": 1.0, "beta": 2.0, "gamma": 50.0, "name": "PerfFirst(γ=50)"},

        # Performance ratio (natural trade-off)
        {"type": "performance_ratio", "alpha": 1.0, "beta": 2.0, "gamma": 1.0, "name": "PerfRatio(γ=1)"},
        {"type": "performance_ratio", "alpha": 1.0, "beta": 2.0, "gamma": 5.0, "name": "PerfRatio(γ=5)"},

        # Performance-only (minimal penalty)
        {"type": "performance_only", "alpha": 1.0, "beta": 2.0, "gamma": 0.1, "name": "PerfOnly(γ=0.1)"},
        {"type": "performance_only", "alpha": 1.0, "beta": 2.0, "gamma": 1.0, "name": "PerfOnly(γ=1)"},

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

            # Create agent based on config
            if config['type'] == 'greedy':
                agent = AdvancedGreedyPerformanceAgent()
            else:
                agent = HybridObjectiveJAM(
                    objective_type=config['type'],
                    alpha=config['alpha'],
                    beta=config['beta'],
                    gamma=config['gamma']
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
                print(f"  Run {run+1}/{num_runs}: Perf={avg_perf:.1f}, Survival={survival_rate*100:.0f}%")

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
        print(f"    Headroom:        {avg_headroom*100:.1f}%")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PERFORMANCE-FIRST JAM: Fix the objective function!")
    print("="*80)
    print("\nHYPOTHESES:")
    print("1. sum(headrooms) rewards CONSERVATIVE designs (problem!)")
    print("2. Hard margin threshold creates CLIFF (problem!)")
    print("3. Optimizing for PERFORMANCE should beat optimizing for HEADROOM")
    print("="*80 + "\n")

    results = test_objective_formulations(num_runs=50)

    # Save results
    with open('objective_audit_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Analysis
    print(f"\n{'='*80}")
    print("OBJECTIVE FUNCTION COMPARISON")
    print(f"{'='*80}\n")

    print(f"{'Objective':<30} {'Perf':<10} {'Survival':<12} {'Score':<12} {'Headroom':<10}")
    print("-" * 80)

    for name, data in sorted(results.items(), key=lambda x: x[1]['combined_score'], reverse=True):
        print(f"{name:<30} {data['performance']:>8.1f}   {data['survival_rate']*100:>9.1f}%   "
              f"{data['combined_score']:>10.1f}   {data['headroom']*100:>8.1f}%")

    # Find best
    best = max(results.items(), key=lambda x: x[1]['combined_score'])
    print(f"\n{'='*80}")
    print(f"🏆 BEST OBJECTIVE: {best[0]}")
    print(f"{'='*80}")
    print(f"Performance:     {best[1]['performance']:.2f}")
    print(f"Survival:        {best[1]['survival_rate']*100:.1f}%")
    print(f"Combined Score:  {best[1]['combined_score']:.2f}")
    print(f"Efficiency:      {best[1]['efficiency']:.2f} perf/W")
    print(f"Headroom:        {best[1]['headroom']*100:.1f}%")

    print(f"\n{'='*80}")
    print("Results saved to: objective_audit_results.json")
    print(f"{'='*80}\n")
