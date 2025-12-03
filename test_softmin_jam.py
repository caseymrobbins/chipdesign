#!/usr/bin/env python3
"""
Test new JAM formula with softmin instead of hard min.

New formula: R = sum(headrooms) + λ * log(softmin(headrooms; β) + ε)

This combines:
1. Sum term: Encourages improving ALL headrooms
2. Softmin term: Smooth approximation to min, focuses on bottleneck

Parameters to tune:
- λ (lambda): Weight of softmin term vs sum term
- β (beta): Temperature parameter (higher β → closer to hard min)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedAgent,
    DesignAction,
    ProcessTechnology,
    AdvancedSimulation,
    ShiftType,
)
from dataclasses import asdict
import json


def softmin(values: np.ndarray, beta: float = 1.0) -> float:
    """
    Compute smooth softmin approximation with numerical stability.

    softmin(v; β) = Σ_i v_i * exp(-β * v_i) / Σ_i exp(-β * v_i)

    As β → ∞, this approaches min(values)
    As β → 0, this approaches mean(values)

    Args:
        values: Array of values
        beta: Temperature parameter (higher = closer to hard min)

    Returns:
        Smooth minimum approximation
    """
    # Normalize to avoid numerical overflow by shifting to make min = 0
    # This way exp(-beta * 0) = 1 for min, and exp(-beta * positive) < 1 for others
    v_shifted = values - np.min(values)

    # Clip to prevent overflow even with high beta
    exponents = -beta * v_shifted
    exponents = np.clip(exponents, -700, 700)  # exp(700) is near float64 max

    weights = np.exp(exponents)
    weights_sum = np.sum(weights)

    # Weighted average
    result = np.sum(values * weights) / weights_sum

    return result


class SoftminJAMAgent(AdvancedAgent):
    """
    JAM agent using softmin instead of hard min - PURE intrinsic optimization.

    Objective: R = Σv + λ·log(softmin(v; β) + ε)

    Benefits of this formulation:
    1. Sum term: Encourages improving ALL headrooms, not just the minimum
    2. Softmin term: Smooth focus on bottleneck, differentiable
    3. λ controls balance between global improvement (sum) and bottleneck focus (softmin)
    4. β controls how "hard" the minimum is (β→∞ recovers original JAM)

    CRITICAL: NO external constraints or threshold checks!
    Trust that log(softmin(v)) → -∞ as any value → 0 prevents catastrophic failures.

    Optimal parameters (per guide):
    - For neural network/gradient-based: λ=200, β=0.05
    - For tabular/discrete: λ=1000-5000, β=2.0-5.0
    """

    def __init__(
        self,
        lambda_weight: float = 200.0,
        beta: float = 2.5,
        epsilon: float = 1e-10,
    ):
        super().__init__(f"SoftminJAM(λ={lambda_weight},β={beta})")
        self.lambda_weight = lambda_weight
        self.beta = beta
        self.epsilon = epsilon

    def calculate_objective(self, headrooms_dict: Dict[str, float]) -> float:
        """
        Calculate the intrinsic objective function.

        R = Σv + λ·log(softmin(v; β) + ε)

        CRITICAL: v includes EVERYTHING (all agency domains):
        - Performance: must do the job (prevents paralyzed agent)
        - Efficiency: must be efficient
        - Constraints: must satisfy all constraints

        All are treated as agency domains in softmin - doing nothing → perf=0 → log(0) → -∞
        """
        # Get ALL agency domains
        perf = self.design_space.calculate_performance()
        constraints = self.design_space.calculate_constraints()
        efficiency = perf / constraints['total_power_w']

        # Get weighted constraint headrooms
        weights = self.design_space.limits.constraint_weights
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms_dict.items()
        }

        # Build complete value vector: v = [performance, efficiency, ...all headrooms]
        # KEY INSIGHT: Headrooms (~0.4-1.0) dominate softmin compared to performance (~100)
        # Solution: Scale headrooms DOWN to same magnitude as performance/efficiency
        # This allows performance to compete fairly in the softmin
        HEADROOM_SCALE = 0.01  # Scale headrooms from ~1.0 to ~0.01 (same magnitude as performance/100)

        all_values = np.array(
            [perf, efficiency] +
            [h * HEADROOM_SCALE for h in weighted_headrooms.values()]
        )

        # Ensure all values are positive for log
        if np.any(all_values <= 0):
            return -np.inf

        # INTRINSIC MULTI-OBJECTIVE REWARD: R = Σv + λ·log(softmin(v; β))
        # Performance in softmin: agent must do job AND satisfy constraints
        sum_term = np.sum(all_values)
        softmin_val = softmin(all_values, beta=self.beta)
        log_softmin_term = self.lambda_weight * np.log(softmin_val + self.epsilon)

        return sum_term + log_softmin_term

    def select_action(self) -> Optional[DesignAction]:
        """Select action that maximizes intrinsic objective - pure optimization, no thresholds"""
        if not self.design_space:
            return None

        best_action = None
        best_objective = -float('inf')

        for action in self.design_space.actions:
            # Simulate applying the action
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            # NO feasibility check - pure intrinsic optimization!
            # log(softmin(v)) → -∞ naturally handles infeasible states
            headrooms = test_space.get_headrooms(include_performance=False)
            objective_score = self.calculate_objective(headrooms)

            # Select action with highest intrinsic objective
            if objective_score > best_objective:
                best_objective = objective_score
                best_action = action

        return best_action


class SoftminSimulation(AdvancedSimulation):
    """Extended simulation to test different JAM variants"""

    def run_comparison(self, run_id: int) -> Dict:
        """
        Run comparison between:
        1. Original JAM (hard min)
        2. Softmin JAM with various parameters
        """

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SOFTMIN JAM COMPARISON RUN {run_id}")
            print(f"{'='*70}")

        # Create design spaces
        seed1 = self.rng.randint(0, 1000000)

        # Space 1: Original hard-min JAM
        from advanced_chip_simulator import JAMAgent
        space1 = AdvancedDesignSpace(process=self.process, seed=seed1)
        space1.initialize_actions()
        agent1 = JAMAgent(min_margin_threshold=2.0)
        agent1.initialize(space1)

        # Space 2: Softmin JAM (β=2.0, λ=1.0) - moderate smoothing
        space2 = space1.clone()
        agent2 = SoftminJAMAgent(lambda_weight=1.0, beta=2.0, min_margin_threshold=2.0)
        agent2.initialize(space2)

        # Space 3: Softmin JAM (β=5.0, λ=1.0) - closer to hard min
        space3 = space1.clone()
        agent3 = SoftminJAMAgent(lambda_weight=1.0, beta=5.0, min_margin_threshold=2.0)
        agent3.initialize(space3)

        # Space 4: Softmin JAM (β=2.0, λ=0.5) - more emphasis on sum
        space4 = space1.clone()
        agent4 = SoftminJAMAgent(lambda_weight=0.5, beta=2.0, min_margin_threshold=2.0)
        agent4.initialize(space4)

        agents = [agent1, agent2, agent3, agent4]

        if self.verbose:
            print(f"\nTesting {len(agents)} JAM variants:")
            for i, agent in enumerate(agents, 1):
                print(f"  {i}. {agent.name}")

        # Run design phase
        if self.verbose:
            print(f"\nDESIGN PHASE ({self.design_steps} steps)")
            print("-" * 70)

        for step in range(self.design_steps):
            for agent in agents:
                agent.step()

            if self.verbose and step % 10 == 0:
                print(f"\nStep {step}:")
                for agent in agents:
                    space = agent.design_space
                    print(f"  {agent.name:30s}: Perf={space.calculate_performance():7.2f}, "
                          f"MinHead={space.get_min_headroom():7.2f}")

        # Collect design phase results
        design_results = []
        for agent in agents:
            space = agent.design_space
            design_results.append({
                'agent': agent.name,
                'performance': space.calculate_performance(),
                'min_headroom': space.get_min_headroom(),
                'power': space.calculate_constraints()['total_power_w'],
                'area': space.calculate_constraints()['area_mm2'],
                'is_feasible': space.is_feasible(),
            })

        if self.verbose:
            print(f"\n{'='*70}")
            print("DESIGN PHASE RESULTS")
            print(f"{'='*70}")
            for res in design_results:
                print(f"{res['agent']:30s}: Perf={res['performance']:7.2f}, "
                      f"MinHead={res['min_headroom']:6.2f}, "
                      f"Power={res['power']:5.1f}W, "
                      f"Area={res['area']:5.1f}mm²")

        # Apply requirement shift
        shift_type = self.shift_type or self.rng.choice(list(ShiftType))

        if self.verbose:
            print(f"\nREQUIREMENT SHIFT: {shift_type.value}")
            print("-" * 70)

        survival = []
        for agent in agents:
            shift_info = agent.design_space.apply_requirement_shift(shift_type, self.rng)
            survived = agent.design_space.is_feasible()
            survival.append(survived)

            if self.verbose:
                status = "✓ SURVIVED" if survived else "✗ FAILED"
                print(f"  {agent.name:30s}: {status}")

        # Adaptation phase
        if self.verbose:
            print(f"\nADAPTATION PHASE ({self.adaptation_steps} steps)")
            print("-" * 70)

        for step in range(self.adaptation_steps):
            for i, agent in enumerate(agents):
                if survival[i]:
                    action, feasible = agent.step()
                    if not feasible:
                        survival[i] = False

        # Collect final results
        final_results = []
        for i, agent in enumerate(agents):
            space = agent.design_space
            final_results.append({
                'agent': agent.name,
                'survived': survival[i],
                'performance': space.calculate_performance() if survival[i] else 0.0,
                'min_headroom': space.get_min_headroom() if survival[i] else -999.0,
                'power': space.calculate_constraints()['total_power_w'] if survival[i] else 0.0,
                'area': space.calculate_constraints()['area_mm2'] if survival[i] else 0.0,
            })

        if self.verbose:
            print(f"\n{'='*70}")
            print("FINAL RESULTS")
            print(f"{'='*70}")
            for res in final_results:
                status = "✓" if res['survived'] else "✗"
                print(f"{status} {res['agent']:30s}: Perf={res['performance']:7.2f}, "
                      f"MinHead={res['min_headroom']:6.2f}, "
                      f"Power={res['power']:5.1f}W")

        return {
            'run_id': run_id,
            'shift_type': shift_type.value,
            'design_results': design_results,
            'final_results': final_results,
        }


def run_softmin_experiments(
    num_runs: int = 20,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    shift_type: Optional[ShiftType] = None,
    seed: Optional[int] = None,
    output_file: str = "softmin_jam_results.json",
    verbose: bool = False,
):
    """Run experiments comparing softmin JAM variants"""

    print(f"\n{'='*80}")
    print(f"SOFTMIN JAM EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Runs: {num_runs}")
    print(f"Design steps: {design_steps}")
    print(f"Adaptation steps: {adaptation_steps}")
    print(f"Testing formula: R = sum(headrooms) + λ·log(softmin(headrooms; β) + ε)")
    print(f"{'='*80}\n")

    master_rng = np.random.RandomState(seed)
    all_results = []

    for run_id in range(num_runs):
        run_seed = master_rng.randint(0, 1000000)
        sim = SoftminSimulation(
            design_steps=design_steps,
            adaptation_steps=adaptation_steps,
            checkpoint_frequency=10,
            shift_type=shift_type,
            process=ProcessTechnology.create_7nm(),
            seed=run_seed,
            verbose=verbose,
        )

        result = sim.run_comparison(run_id)
        all_results.append(result)

        if not verbose and (run_id + 1) % 5 == 0:
            print(f"Completed {run_id + 1}/{num_runs} runs...")

    # Aggregate statistics
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}\n")

    # Count survival by agent type
    survival_counts = {}
    perf_design = {}
    perf_final = {}

    for result in all_results:
        for design_res in result['design_results']:
            agent = design_res['agent']
            if agent not in perf_design:
                perf_design[agent] = []
            perf_design[agent].append(design_res['performance'])

        for final_res in result['final_results']:
            agent = final_res['agent']
            if agent not in survival_counts:
                survival_counts[agent] = 0
                perf_final[agent] = []

            if final_res['survived']:
                survival_counts[agent] += 1
                perf_final[agent].append(final_res['performance'])

    # Print survival rates
    print("SURVIVAL RATES:")
    for agent in sorted(survival_counts.keys()):
        rate = survival_counts[agent] / num_runs
        print(f"  {agent:35s}: {survival_counts[agent]:3d}/{num_runs} ({rate:6.1%})")

    # Print design phase performance
    print("\nDESIGN PHASE PERFORMANCE:")
    for agent in sorted(perf_design.keys()):
        values = perf_design[agent]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {agent:35s}: {mean:7.2f} ± {std:5.2f}")

    # Print final performance (survivors only)
    print("\nFINAL PERFORMANCE (survivors):")
    for agent in sorted(perf_final.keys()):
        if perf_final[agent]:
            values = perf_final[agent]
            mean = np.mean(values)
            std = np.std(values)
            count = len(values)
            print(f"  {agent:35s}: {mean:7.2f} ± {std:5.2f} (n={count})")
        else:
            print(f"  {agent:35s}: No survivors")

    # Save results
    output_data = {
        'parameters': {
            'num_runs': num_runs,
            'design_steps': design_steps,
            'adaptation_steps': adaptation_steps,
            'shift_type': shift_type.value if shift_type else 'random',
        },
        'summary': {
            'survival_counts': survival_counts,
            'design_performance': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                                  for k, v in perf_design.items()},
            'final_performance': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                                 for k, v in perf_final.items() if v},
        },
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")

    return output_data


if __name__ == "__main__":
    # Run comparison
    run_softmin_experiments(
        num_runs=20,
        design_steps=75,
        adaptation_steps=25,
        seed=42,
        output_file="softmin_jam_results.json",
        verbose=False,
    )
