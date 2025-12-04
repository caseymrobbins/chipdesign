#!/usr/bin/env python3
"""
Proper robustness test with graduated stress levels.

Instead of random binary (all survive or all die), test agents across
a spectrum of stress levels to see which handles adversity better.
"""

import numpy as np
from typing import List, Dict, Tuple
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedGreedyPerformanceAgent,
    JAMAgent,
    ProcessTechnology,
    ShiftType,
)
from test_softmin_jam import SoftminJAMAgent
import matplotlib.pyplot as plt


def test_stress_resilience(
    agent_name: str,
    agent_class,
    agent_kwargs: dict,
    design_steps: int = 40,
    seed: int = 42,
) -> Dict[str, List[Tuple[float, bool]]]:
    """
    Test an agent's resilience across graduated stress levels.

    Returns dict mapping shift_type -> list of (stress_level, survived) tuples
    """
    results = {}

    # Test each shift type
    shift_types = [
        ShiftType.TIGHTEN_POWER,
        ShiftType.INCREASE_PERFORMANCE,
        ShiftType.REDUCE_AREA,
        ShiftType.TIGHTEN_THERMAL,
    ]

    for shift_type in shift_types:
        stress_results = []

        # Test graduated stress levels from 5% to 50%
        stress_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

        for stress_level in stress_levels:
            # Create fresh agent and design
            space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=seed)
            space.initialize_actions()

            agent = agent_class(**agent_kwargs)
            agent.initialize(space)

            # Design phase
            for _ in range(design_steps):
                agent.step()

            # Apply graduated stress
            if shift_type == ShiftType.TIGHTEN_POWER:
                space.limits.max_power_watts *= (1.0 - stress_level)
            elif shift_type == ShiftType.INCREASE_PERFORMANCE:
                space.limits.min_frequency_ghz *= (1.0 + stress_level)
            elif shift_type == ShiftType.REDUCE_AREA:
                space.limits.max_area_mm2 *= (1.0 - stress_level)
            elif shift_type == ShiftType.TIGHTEN_THERMAL:
                space.limits.max_temperature_c -= (stress_level * 50)  # Up to -25°C

            # Check if design survives
            survived = space.is_feasible()
            stress_results.append((stress_level, survived))

            print(f"  {agent_name} | {shift_type.value:20s} | stress={stress_level:.0%} | {'✓ SURVIVED' if survived else '✗ FAILED'}")

        results[shift_type.value] = stress_results

    return results


def calculate_robustness_score(results: Dict[str, List[Tuple[float, bool]]]) -> float:
    """
    Calculate overall robustness score.

    Score = average stress level where agent fails (higher is better)
    """
    failure_points = []

    for shift_type, stress_results in results.items():
        # Find the first stress level where agent fails
        for stress_level, survived in stress_results:
            if not survived:
                failure_points.append(stress_level)
                break
        else:
            # Survived all stress levels
            failure_points.append(1.0)

    return np.mean(failure_points)


if __name__ == "__main__":
    print("="*80)
    print("PROPER ROBUSTNESS TEST - Graduated Stress Levels")
    print("="*80)
    print("\nTesting each agent across 5% to 50% stress levels")
    print("This reveals true robustness differences between agents")
    print("="*80)

    # Test each agent
    agents_config = [
        ("IndustryBest", AdvancedGreedyPerformanceAgent, {}),
        ("JAM", JAMAgent, {}),
        ("JAMAdvanced", SoftminJAMAgent, {"lambda_weight": 0.1, "beta": 5.0}),
    ]

    all_results = {}
    scores = {}

    for agent_name, agent_class, kwargs in agents_config:
        print(f"\n{'='*80}")
        print(f"Testing: {agent_name}")
        print('='*80)
        results = test_stress_resilience(agent_name, agent_class, kwargs)
        all_results[agent_name] = results
        scores[agent_name] = calculate_robustness_score(results)

    # Print summary
    print("\n" + "="*80)
    print("ROBUSTNESS SCORES (Average stress level at failure)")
    print("="*80)
    for agent_name in ["IndustryBest", "JAM", "JAMAdvanced"]:
        score = scores[agent_name]
        print(f"{agent_name:20s}: {score:.1%} stress tolerance")

    # Determine winner
    winner = max(scores.items(), key=lambda x: x[1])
    print(f"\n🏆 Most Robust: {winner[0]} ({winner[1]:.1%} stress tolerance)")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Agent Robustness Under Graduated Stress', fontsize=14, fontweight='bold')

    shift_types = [
        ShiftType.TIGHTEN_POWER,
        ShiftType.INCREASE_PERFORMANCE,
        ShiftType.REDUCE_AREA,
        ShiftType.TIGHTEN_THERMAL,
    ]

    colors = {
        'IndustryBest': '#ff7f0e',
        'JAM': '#2ca02c',
        'JAMAdvanced': '#1f77b4'
    }

    for idx, shift_type in enumerate(shift_types):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(shift_type.value.replace('_', ' ').title())
        ax.set_xlabel('Stress Level')
        ax.set_ylabel('Design Survives')
        ax.set_ylim([-0.1, 1.1])
        ax.grid(alpha=0.3)

        for agent_name in ["IndustryBest", "JAM", "JAMAdvanced"]:
            stress_data = all_results[agent_name][shift_type.value]
            stress_levels = [s[0] for s in stress_data]
            survival = [1 if s[1] else 0 for s in stress_data]

            ax.plot(stress_levels, survival, 'o-', label=agent_name,
                   color=colors[agent_name], linewidth=2, markersize=6)

        ax.legend()
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Failed', 'Survived'])

    plt.tight_layout()
    plt.savefig('robustness_graduated_stress.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: robustness_graduated_stress.png")
    print("="*80)
