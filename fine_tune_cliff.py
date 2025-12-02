#!/usr/bin/env python3
"""
FINE-TUNE THE CLIFF: Find exact performance boundary

Question: Is margin=0.5 the cliff, or do we have room between 0.5-0.75?
Goal: Test finer granularity to find the EXACT cliff location
Then: Squeeze out every last % of performance!

Current knowledge:
- margin=0.50: 70.02 perf, 100% survival ✅
- margin=0.75: 47.04 perf, 82% survival ❌ (-33% drop!)
- GAP: We never tested 0.55, 0.60, 0.65, 0.70!
"""

import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedAgent,
    DesignAction,
    ProcessTechnology,
    AdvancedSimulation,
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


class SoftminJAMAgent(AdvancedAgent):
    """JAM agent using softmin instead of hard min."""

    def __init__(
        self,
        lambda_weight: float = 1.0,
        beta: float = 2.0,
        min_margin_threshold: float = 2.0,
        epsilon: float = 0.01,
    ):
        super().__init__(f"SoftminJAM(λ={lambda_weight},β={beta},m={min_margin_threshold})")
        self.lambda_weight = lambda_weight
        self.beta = beta
        self.min_margin_threshold = min_margin_threshold
        self.epsilon = epsilon

    def calculate_objective(self, headrooms_dict: Dict[str, float]) -> float:
        """Calculate R = sum(headrooms) + λ * log(softmin(headrooms; β) + ε)"""
        weights = self.design_space.limits.constraint_weights
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms_dict.items()
        }

        headroom_values = np.array(list(weighted_headrooms.values()))

        if np.any(headroom_values <= 0):
            return -np.inf

        sum_term = np.sum(headroom_values)
        softmin_val = softmin(headroom_values, beta=self.beta)
        softmin_term = self.lambda_weight * np.log(softmin_val + self.epsilon)

        return sum_term + softmin_term

    def select_action(self) -> Optional[DesignAction]:
        """Select action that maximizes the new objective"""
        if not self.design_space:
            return None

        current_headrooms = self.design_space.get_headrooms()
        current_objective = self.calculate_objective(current_headrooms)

        safe_actions = []
        risky_actions = []

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            headrooms = test_space.get_headrooms()
            min_headroom = min(headrooms.values())
            objective_score = self.calculate_objective(headrooms)
            perf = test_space.calculate_performance()

            action_data = (action, objective_score, perf, min_headroom)

            if min_headroom >= self.min_margin_threshold:
                safe_actions.append(action_data)
            else:
                risky_actions.append(action_data)

        if safe_actions:
            best = max(safe_actions, key=lambda x: (x[1], x[2]))
            return best[0]

        elif risky_actions:
            improving = [a for a in risky_actions if a[3] >= min(current_headrooms.values())]
            if improving:
                best = max(improving, key=lambda x: (x[1], x[2]))
                return best[0]
            else:
                best = max(risky_actions, key=lambda x: x[3])
                return best[0]

        return None


def test_cliff_location(num_runs=50):
    """Fine-grained sweep to find exact cliff location"""

    print("="*80)
    print("PHASE 1: FIND THE CLIFF")
    print("="*80)
    print("Testing margins: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75")
    print("Goal: Find exact boundary between high-perf and low-perf regions")
    print("="*80)

    margins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    LAMBDA = 0.05
    BETA = 1.0
    STEPS = 100

    results = {}
    process = ProcessTechnology.create_7nm()

    for margin in margins:
        print(f"\n{'='*60}")
        print(f"Testing margin={margin:.2f}")
        print(f"{'='*60}")

        perf_scores = []
        survival_count = 0
        efficiencies = []
        headrooms = []

        for run in range(num_runs):
            seed = run * 1000 + int(margin * 10000)
            space = AdvancedDesignSpace(process=process, seed=seed)
            space.initialize_actions()

            agent = SoftminJAMAgent(
                lambda_weight=LAMBDA,
                beta=BETA,
                min_margin_threshold=margin
            )
            agent.initialize(space)

            for step in range(STEPS):
                agent.step()

            design_perf = space.calculate_performance()
            perf_scores.append(design_perf)

            if space.is_feasible():
                constraints = space.calculate_constraints()
                efficiencies.append(design_perf / constraints['total_power_w'])
                headrooms.append(space.get_min_headroom())

            shift_rng = np.random.RandomState(run * 9999 + int(margin * 10000))
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

        results[f"margin={margin:.2f}"] = {
            'margin': margin,
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


def test_other_knobs(optimal_margin, num_runs=50):
    """Test other parameters at the optimal margin to squeeze out more performance"""

    print(f"\n{'='*80}")
    print("PHASE 2: SQUEEZE OUT MORE PERFORMANCE")
    print(f"{'='*80}")
    print(f"Using optimal margin={optimal_margin:.2f}")
    print("Testing other knobs:")
    print("  1. More design steps (100 → 150)")
    print("  2. Different λ values (0.05 → 0.10)")
    print("  3. Different β values (1.0 → 2.0)")
    print("  4. Lower epsilon (0.01 → 0.001)")
    print("="*80)

    configs = [
        # Baseline (current best)
        {"name": "Baseline", "lambda": 0.05, "beta": 1.0, "steps": 100, "epsilon": 0.01},

        # More optimization steps
        {"name": "MoreSteps(150)", "lambda": 0.05, "beta": 1.0, "steps": 150, "epsilon": 0.01},
        {"name": "MoreSteps(200)", "lambda": 0.05, "beta": 1.0, "steps": 200, "epsilon": 0.01},

        # Different λ
        {"name": "Lambda=0.10", "lambda": 0.10, "beta": 1.0, "steps": 100, "epsilon": 0.01},
        {"name": "Lambda=0.15", "lambda": 0.15, "beta": 1.0, "steps": 100, "epsilon": 0.01},

        # Different β
        {"name": "Beta=2.0", "lambda": 0.05, "beta": 2.0, "steps": 100, "epsilon": 0.01},
        {"name": "Beta=1.5", "lambda": 0.05, "beta": 1.5, "steps": 100, "epsilon": 0.01},

        # Lower epsilon
        {"name": "Epsilon=0.001", "lambda": 0.05, "beta": 1.0, "steps": 100, "epsilon": 0.001},

        # Combo: more steps + different params
        {"name": "Combo(steps=150,λ=0.10)", "lambda": 0.10, "beta": 1.0, "steps": 150, "epsilon": 0.01},
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

        for run in range(num_runs):
            seed = run * 1000 + hash(config['name']) % 10000
            space = AdvancedDesignSpace(process=process, seed=seed)
            space.initialize_actions()

            agent = SoftminJAMAgent(
                lambda_weight=config['lambda'],
                beta=config['beta'],
                min_margin_threshold=optimal_margin,
                epsilon=config['epsilon']
            )
            agent.initialize(space)

            for step in range(config['steps']):
                agent.step()

            design_perf = space.calculate_performance()
            perf_scores.append(design_perf)

            if space.is_feasible():
                constraints = space.calculate_constraints()
                efficiencies.append(design_perf / constraints['total_power_w'])

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

        results[config['name']] = {
            'config': config,
            'performance': avg_perf,
            'survival_rate': survival_rate,
            'efficiency': avg_eff,
            'combined_score': avg_perf * survival_rate,
        }

        print(f"\n  RESULTS:")
        print(f"    Performance:     {avg_perf:.2f}")
        print(f"    Survival:        {survival_rate*100:.1f}%")
        print(f"    Combined Score:  {avg_perf * survival_rate:.2f}")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FINE-TUNE OPTIMIZATION: Find the cliff + squeeze out every last %")
    print("="*80)

    # Phase 1: Find exact cliff location
    cliff_results = test_cliff_location(num_runs=50)

    # Find the best margin
    best_margin = max(cliff_results.items(), key=lambda x: x[1]['combined_score'])
    print(f"\n{'='*80}")
    print(f"CLIFF ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Best margin: {best_margin[1]['margin']:.2f}")
    print(f"  Performance: {best_margin[1]['performance']:.2f}")
    print(f"  Survival: {best_margin[1]['survival_rate']*100:.1f}%")
    print(f"  Combined Score: {best_margin[1]['combined_score']:.2f}")

    # Show the performance curve
    print(f"\nPerformance curve across margins:")
    for name, data in sorted(cliff_results.items(), key=lambda x: x[1]['margin']):
        print(f"  {name}: {data['performance']:.2f} perf, {data['survival_rate']*100:.0f}% survival")

    # Save cliff results
    with open('cliff_location_results.json', 'w') as f:
        json.dump(cliff_results, f, indent=2)

    # Phase 2: Optimize other parameters
    print(f"\n{'='*80}")
    print("STARTING PHASE 2: PARAMETER OPTIMIZATION")
    print(f"{'='*80}")

    optimal_margin = best_margin[1]['margin']
    param_results = test_other_knobs(optimal_margin, num_runs=50)

    # Show improvements
    baseline_score = param_results['Baseline']['combined_score']
    print(f"\n{'='*80}")
    print("FINAL OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Baseline (margin={optimal_margin:.2f}, λ=0.05, β=1.0, steps=100):")
    print(f"  Combined Score: {baseline_score:.2f}")

    print(f"\nImprovements over baseline:")
    improvements = []
    for name, data in param_results.items():
        if name != 'Baseline':
            delta = data['combined_score'] - baseline_score
            pct = (delta / baseline_score) * 100
            improvements.append((name, delta, pct, data))
            print(f"  {name:30s}: {delta:+.2f} ({pct:+.1f}%) → {data['combined_score']:.2f}")

    # Find best config
    best_config = max(param_results.items(), key=lambda x: x[1]['combined_score'])
    print(f"\n🏆 BEST CONFIGURATION: {best_config[0]}")
    print(f"  Performance:     {best_config[1]['performance']:.2f}")
    print(f"  Survival:        {best_config[1]['survival_rate']*100:.1f}%")
    print(f"  Combined Score:  {best_config[1]['combined_score']:.2f}")
    print(f"  Improvement:     {(best_config[1]['combined_score'] - baseline_score):.2f} (+{((best_config[1]['combined_score'] - baseline_score) / baseline_score * 100):.1f}%)")

    # Save parameter results
    with open('parameter_optimization_results.json', 'w') as f:
        json.dump(param_results, f, indent=2)

    print(f"\n{'='*80}")
    print("Results saved to:")
    print("  - cliff_location_results.json")
    print("  - parameter_optimization_results.json")
    print(f"{'='*80}")
