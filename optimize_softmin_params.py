#!/usr/bin/env python3
"""
Fine-tune Softmin parameters around the winning configuration.

Baseline (Winner): Softmin(λ=0.05, β=1.0, threshold=0.5)

Testing strategy:
1. Keep baseline as reference
2. Tweak λ (lambda): controls balance between sum and softmin
3. Tweak β (beta): controls sharpness of bottleneck focus
4. Tweak threshold: controls safety margin

Goal: Find the highest performance configuration through systematic exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from dataclasses import dataclass
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    AdvancedGreedyPerformanceAgent,
    JAMAgent,
    ProcessTechnology,
    ShiftType,
)
from test_softmin_jam import SoftminJAMAgent
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (24, 14)
plt.rcParams['font.size'] = 10


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    agent_type: str  # 'greedy', 'hardmin', 'softmin'
    params: Dict


@dataclass
class AgentResult:
    """Results for a single agent in one simulation"""
    name: str
    agent_type: str
    design_performance: float
    design_min_headroom: float
    design_power: float
    design_area: float
    design_frequency: float
    design_steps_taken: int
    survived_shift: bool
    final_performance: float
    final_min_headroom: float
    final_power: float


def create_parameter_sweep_configs() -> List[AgentConfig]:
    """
    Create 4 Softmin configurations: baseline winner + 3 tweaked variants.

    Baseline: λ=0.05, β=1.0, threshold=0.5

    Tweaking strategy:
    - Variant 1: Slightly more aggressive (lower λ, lower β)
    - Variant 2: Slightly more conservative (higher λ, higher β)
    - Variant 3: Sharper bottleneck focus (keep λ, increase β)
    """

    configs = []

    # Add Greedy for baseline comparison
    configs.append(AgentConfig(
        name="Greedy",
        agent_type="greedy",
        params={}
    ))

    # 🏆 BASELINE WINNER
    configs.append(AgentConfig(
        name="Softmin(λ=0.05,β=1.0)",
        agent_type="softmin",
        params={'lambda_weight': 0.05, 'beta': 1.0, 'min_margin_threshold': 0.5}
    ))

    # VARIANT 1: Slightly more aggressive (lower λ and β = more sum focus, softer min)
    configs.append(AgentConfig(
        name="Softmin(λ=0.04,β=0.9)",
        agent_type="softmin",
        params={'lambda_weight': 0.04, 'beta': 0.9, 'min_margin_threshold': 0.5}
    ))

    # VARIANT 2: Slightly more conservative (higher λ and β = more min focus, sharper)
    configs.append(AgentConfig(
        name="Softmin(λ=0.06,β=1.1)",
        agent_type="softmin",
        params={'lambda_weight': 0.06, 'beta': 1.1, 'min_margin_threshold': 0.5}
    ))

    # VARIANT 3: Sharper bottleneck focus (same λ, much higher β)
    configs.append(AgentConfig(
        name="Softmin(λ=0.05,β=1.3)",
        agent_type="softmin",
        params={'lambda_weight': 0.05, 'beta': 1.3, 'min_margin_threshold': 0.5}
    ))

    return configs


def create_agent_from_config(config: AgentConfig):
    """Create an agent instance from configuration"""

    if config.agent_type == "greedy":
        return AdvancedGreedyPerformanceAgent()
    elif config.agent_type == "hardmin":
        return JAMAgent(**config.params)
    elif config.agent_type == "softmin":
        return SoftminJAMAgent(**config.params)
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")


def run_single_comparison(
    run_id: int,
    agent_configs: List[AgentConfig],
    design_steps: int = 75,
    adaptation_steps: int = 25,
    shift_type: Optional[ShiftType] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[AgentResult]:
    """Run one comparison across all agents"""

    rng = np.random.RandomState(seed)

    # Create initial design space
    base_space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=seed)
    base_space.initialize_actions()

    # Create agents with identical starting conditions
    spaces = []
    for config in agent_configs:
        space = base_space.clone()
        agent = create_agent_from_config(config)
        agent.initialize(space)
        spaces.append((config, agent, space))

    if verbose:
        print(f"\n{'='*70}")
        print(f"RUN {run_id}")
        print(f"{'='*70}")
        print(f"Testing {len(agent_configs)} agents")

    # DESIGN PHASE
    steps_taken = {config.name: 0 for config, _, _ in spaces}

    for step in range(design_steps):
        for config, agent, space in spaces:
            action, feasible = agent.step()
            if action is not None:
                steps_taken[config.name] += 1

    # Collect design phase results
    design_results = []
    for config, agent, space in spaces:
        constraints = space.calculate_constraints()
        design_results.append({
            'config': config,
            'performance': space.calculate_performance(),
            'min_headroom': space.get_min_headroom(),
            'power': constraints['total_power_w'],
            'area': constraints['area_mm2'],
            'frequency': space.params.clock_freq_ghz,
            'steps_taken': steps_taken[config.name],
        })

    if verbose:
        print(f"\nDESIGN PHASE COMPLETE - Top 5 by performance:")
        sorted_results = sorted(design_results, key=lambda x: x['performance'], reverse=True)
        for i, res in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {res['config'].name:35s}: Perf={res['performance']:7.2f}, "
                  f"MinH={res['min_headroom']:6.2f}")

    # REQUIREMENT SHIFT
    if shift_type is None:
        shift_type = rng.choice(list(ShiftType))

    if verbose:
        print(f"\nREQUIREMENT SHIFT: {shift_type.value}")

    survival = []
    for config, agent, space in spaces:
        space.apply_requirement_shift(shift_type, rng)
        survived = space.is_feasible()
        survival.append(survived)

    survival_count = sum(survival)
    if verbose:
        print(f"  Survived: {survival_count}/{len(spaces)}")

    # ADAPTATION PHASE
    for step in range(adaptation_steps):
        for i, (config, agent, space) in enumerate(spaces):
            if survival[i]:
                action, feasible = agent.step()
                if not feasible:
                    survival[i] = False

    # Collect final results
    results = []
    for i, (config, agent, space) in enumerate(spaces):
        design_res = design_results[i]

        if survival[i]:
            final_perf = space.calculate_performance()
            final_headroom = space.get_min_headroom()
            final_power = space.calculate_constraints()['total_power_w']
        else:
            final_perf = 0.0
            final_headroom = -999.0
            final_power = 0.0

        results.append(AgentResult(
            name=config.name,
            agent_type=config.agent_type,
            design_performance=design_res['performance'],
            design_min_headroom=design_res['min_headroom'],
            design_power=design_res['power'],
            design_area=design_res['area'],
            design_frequency=design_res['frequency'],
            design_steps_taken=design_res['steps_taken'],
            survived_shift=survival[i],
            final_performance=final_perf,
            final_min_headroom=final_headroom,
            final_power=final_power,
        ))

    if verbose:
        print(f"\nFINAL RESULTS - Top 5 survivors:")
        survivors = [r for r in results if r.survived_shift]
        sorted_survivors = sorted(survivors, key=lambda x: x.final_performance, reverse=True)
        for i, res in enumerate(sorted_survivors[:5], 1):
            print(f"  {i}. {res.name:35s}: Perf={res.final_performance:7.2f}")

    return results


def run_experiments(
    num_runs: int = 100,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    seed: Optional[int] = None,
    test_all_shifts: bool = True,
    verbose: bool = False,
) -> List[List[AgentResult]]:
    """
    Run multiple comparison experiments with realistic scenarios.

    Args:
        num_runs: Total number of simulation runs
        design_steps: Steps in design phase
        adaptation_steps: Steps in adaptation phase after shift
        seed: Random seed for reproducibility
        test_all_shifts: If True, evenly distribute runs across all shift types
        verbose: Print detailed progress
    """

    agent_configs = create_parameter_sweep_configs()

    print(f"\n{'='*80}")
    print(f"SOFTMIN PARAMETER OPTIMIZATION - REALISTIC SIMULATION")
    print(f"{'='*80}")
    print(f"🏆 Baseline: Softmin(λ=0.05, β=1.0, threshold=0.5)")
    print(f"\nTesting {len(agent_configs)} configurations:")
    for i, config in enumerate(agent_configs, 1):
        marker = "🏆" if "λ=0.05,β=1.0)" in config.name and "Greedy" not in config.name else "  "
        print(f"  {marker} {i}. {config.name}")

    print(f"\nSimulation parameters:")
    print(f"  Total runs:        {num_runs}")
    print(f"  Design steps:      {design_steps}")
    print(f"  Adaptation steps:  {adaptation_steps}")

    if test_all_shifts:
        shift_types = list(ShiftType)
        runs_per_shift = num_runs // len(shift_types)
        remainder = num_runs % len(shift_types)
        print(f"  Shift distribution: {runs_per_shift} runs/shift type + {remainder} random")
        print(f"  Shift types tested: {', '.join([s.value for s in shift_types])}")
    else:
        print(f"  Shift distribution: Random")

    print(f"{'='*80}\n")

    rng = np.random.RandomState(seed)
    all_results = []

    # Create balanced shift type distribution for realism
    if test_all_shifts:
        shift_schedule = []
        shift_types = list(ShiftType)
        runs_per_shift = num_runs // len(shift_types)

        for shift in shift_types:
            shift_schedule.extend([shift] * runs_per_shift)

        # Add remaining runs with random shifts
        for _ in range(num_runs - len(shift_schedule)):
            shift_schedule.append(rng.choice(shift_types))

        # Shuffle the schedule
        rng.shuffle(shift_schedule)
    else:
        shift_schedule = [None] * num_runs

    for run_id in range(num_runs):
        run_seed = rng.randint(0, 1000000)
        shift_type = shift_schedule[run_id]

        results = run_single_comparison(
            run_id=run_id,
            agent_configs=agent_configs,
            design_steps=design_steps,
            adaptation_steps=adaptation_steps,
            shift_type=shift_type,
            seed=run_seed,
            verbose=verbose,
        )
        all_results.append(results)

        if not verbose and (run_id + 1) % 20 == 0:
            print(f"Progress: {run_id + 1}/{num_runs} runs completed ({(run_id+1)/num_runs*100:.0f}%)")

    return all_results


def analyze_results(all_results: List[List[AgentResult]]) -> Dict:
    """Analyze and aggregate results"""

    # Aggregate by agent name
    agent_data = {}

    for run_results in all_results:
        for agent_result in run_results:
            name = agent_result.name

            if name not in agent_data:
                agent_data[name] = {
                    'agent_type': agent_result.agent_type,
                    'design_perf': [],
                    'design_power': [],
                    'design_headroom': [],
                    'survival': [],
                    'final_perf': [],
                    'efficiency': [],
                }

            agent_data[name]['design_perf'].append(agent_result.design_performance)
            agent_data[name]['design_power'].append(agent_result.design_power)
            agent_data[name]['design_headroom'].append(agent_result.design_min_headroom)
            agent_data[name]['survival'].append(1 if agent_result.survived_shift else 0)

            if agent_result.survived_shift:
                agent_data[name]['final_perf'].append(agent_result.final_performance)
                agent_data[name]['efficiency'].append(
                    agent_result.design_performance / agent_result.design_power
                )

    # Calculate statistics for each agent
    agent_stats = {}
    for name, data in agent_data.items():
        agent_stats[name] = {
            'agent_type': data['agent_type'],
            'design_perf_mean': np.mean(data['design_perf']),
            'design_perf_std': np.std(data['design_perf']),
            'survival_rate': np.mean(data['survival']),
            'survival_count': sum(data['survival']),
            'final_perf_mean': np.mean(data['final_perf']) if data['final_perf'] else 0.0,
            'final_perf_std': np.std(data['final_perf']) if data['final_perf'] else 0.0,
            'efficiency_mean': np.mean(data['efficiency']) if data['efficiency'] else 0.0,
            'design_headroom_mean': np.mean(data['design_headroom']),
        }

    return agent_stats, agent_data


def create_visualization(agent_stats: Dict, agent_data: Dict, num_runs: int,
                        output_file: str = "optimized_softmin_comparison.png"):
    """Create comprehensive visualization"""

    fig = plt.figure(figsize=(24, 14))

    # Sort agents by survival rate, then final performance
    sorted_names = sorted(agent_stats.keys(),
                         key=lambda k: (agent_stats[k]['survival_rate'],
                                       agent_stats[k]['final_perf_mean']),
                         reverse=True)

    # Identify baseline and winner
    baseline_name = "Softmin(λ=0.05,β=1.0)"
    winner_name = sorted_names[0] if sorted_names[0] != "Greedy" else sorted_names[1]

    # Colors: baseline=blue, winner=gold, greedy=red, others=green
    colors = []
    for name in sorted_names:
        if name == "Greedy":
            colors.append('#e74c3c')  # Red
        elif name == baseline_name:
            colors.append('#3498db')  # Blue
        elif name == winner_name and name != baseline_name:
            colors.append('#f39c12')  # Gold
        else:
            colors.append('#95a5a6')  # Gray

    # 1. SURVIVAL RATE
    ax1 = plt.subplot(2, 4, 1)
    survival_rates = [agent_stats[name]['survival_rate'] * 100 for name in sorted_names]
    positions = range(len(sorted_names))

    bars = ax1.barh(positions, survival_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_yticks(positions)
    ax1.set_yticklabels([name for name in sorted_names], fontsize=9)
    ax1.set_xlabel('Survival Rate (%)', fontweight='bold')
    ax1.set_title('Survival After Requirement Shift', fontweight='bold', fontsize=12)
    ax1.set_xlim([0, 100])
    ax1.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for bar, rate in zip(bars, survival_rates):
        width = bar.get_width()
        ax1.text(width + 2, bar.get_y() + bar.get_height()/2.,
                f'{rate:.0f}%', ha='left', va='center', fontweight='bold', fontsize=9)

    # 2. DESIGN PHASE PERFORMANCE
    ax2 = plt.subplot(2, 4, 2)
    design_perf_data = [agent_data[name]['design_perf'] for name in sorted_names]

    bp2 = ax2.boxplot(design_perf_data, positions=positions, widths=0.6,
                      patch_artist=True, vert=False)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_yticks(positions)
    ax2.set_yticklabels([name for name in sorted_names], fontsize=9)
    ax2.set_xlabel('Performance', fontweight='bold')
    ax2.set_title('Design Phase Performance', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)

    # 3. FINAL PERFORMANCE (SURVIVORS)
    ax3 = plt.subplot(2, 4, 3)
    final_perf_means = [agent_stats[name]['final_perf_mean'] for name in sorted_names]

    bars3 = ax3.barh(positions, final_perf_means, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)

    ax3.set_yticks(positions)
    ax3.set_yticklabels([name for name in sorted_names], fontsize=9)
    ax3.set_xlabel('Performance', fontweight='bold')
    ax3.set_title('Post-Adaptation Performance (Survivors)', fontweight='bold', fontsize=12)
    ax3.grid(axis='x', alpha=0.3)

    # 4. EFFICIENCY
    ax4 = plt.subplot(2, 4, 4)
    efficiency_means = [agent_stats[name]['efficiency_mean'] for name in sorted_names]

    bars4 = ax4.barh(positions, efficiency_means, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)

    ax4.set_yticks(positions)
    ax4.set_yticklabels([name for name in sorted_names], fontsize=9)
    ax4.set_xlabel('Perf / Watt', fontweight='bold')
    ax4.set_title('Power Efficiency', fontweight='bold', fontsize=12)
    ax4.grid(axis='x', alpha=0.3)

    # 5. PARAMETER COMPARISON: Survival vs Performance Scatter
    ax5 = plt.subplot(2, 4, 5)

    for name in sorted_names:
        if name == "Greedy":
            continue
        stats = agent_stats[name]
        color = colors[sorted_names.index(name)]

        ax5.scatter(stats['survival_rate'] * 100, stats['final_perf_mean'],
                   c=color, s=200, alpha=0.8, edgecolors='black', linewidth=2)

        # Add labels
        label = name.replace("Softmin(", "").replace(")", "")
        ax5.annotate(label, (stats['survival_rate'] * 100, stats['final_perf_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax5.set_xlabel('Survival Rate (%)', fontweight='bold')
    ax5.set_ylabel('Final Performance', fontweight='bold')
    ax5.set_title('Pareto Front: Survival vs Performance', fontweight='bold', fontsize=12)
    ax5.grid(alpha=0.3)
    ax5.set_xlim([0, 100])

    # 6. DESIGN HEADROOM DISTRIBUTION
    ax6 = plt.subplot(2, 4, 6)
    headroom_data = [agent_data[name]['design_headroom'] for name in sorted_names if name != "Greedy"]
    headroom_labels = [name.replace("Softmin(", "").replace(")", "") for name in sorted_names if name != "Greedy"]

    bp6 = ax6.boxplot(headroom_data, positions=range(len(headroom_data)),
                     widths=0.6, patch_artist=True, vert=False)

    softmin_colors = [c for n, c in zip(sorted_names, colors) if n != "Greedy"]
    for patch, color in zip(bp6['boxes'], softmin_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax6.set_yticks(range(len(headroom_labels)))
    ax6.set_yticklabels(headroom_labels, fontsize=9)
    ax6.set_xlabel('Design Phase Min Headroom', fontweight='bold')
    ax6.set_title('Headroom Distribution (Softmin Only)', fontweight='bold', fontsize=12)
    ax6.grid(axis='x', alpha=0.3)

    # 7. EFFICIENCY COMPARISON
    ax7 = plt.subplot(2, 4, 7)

    # Group by configuration
    config_names = [name for name in sorted_names if name != "Greedy"]
    config_efficiency = [agent_stats[name]['efficiency_mean'] for name in config_names]
    config_colors = [c for n, c in zip(sorted_names, colors) if n != "Greedy"]

    bars7 = ax7.barh(range(len(config_names)), config_efficiency,
                     color=config_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax7.set_yticks(range(len(config_names)))
    ax7.set_yticklabels([n.replace("Softmin(", "").replace(")", "") for n in config_names], fontsize=9)
    ax7.set_xlabel('Power Efficiency (Perf/W)', fontweight='bold')
    ax7.set_title('Power Efficiency Comparison', fontweight='bold', fontsize=12)
    ax7.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, eff in zip(bars7, config_efficiency):
        width = bar.get_width()
        ax7.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                f'{eff:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)

    # 8. SUMMARY TABLE
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')

    # Top 5 configurations
    top5_names = sorted_names[:5]
    summary_data = []

    for name in top5_names:
        stats = agent_stats[name]
        marker = ""
        if name == baseline_name:
            marker = "📘 "
        elif name == winner_name:
            marker = "🏆 "

        summary_data.append([
            marker + name[:25],
            f"{stats['design_perf_mean']:.1f}",
            f"{stats['survival_rate']*100:.0f}%",
            f"{stats['final_perf_mean']:.1f}",
            f"{stats['efficiency_mean']:.2f}",
        ])

    table = ax8.table(cellText=summary_data,
                     colLabels=['Configuration', 'Design\nPerf', 'Survival', 'Final\nPerf', 'Eff'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(1, len(top5_names) + 1):
        name = top5_names[i-1]
        if name == baseline_name:
            color = '#3498db'
        elif name == winner_name:
            color = '#f39c12'
        elif name == "Greedy":
            color = '#e74c3c'
        else:
            color = '#95a5a6'

        for j in range(5):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)

    ax8.set_title('Top 5 Configurations\n📘=Baseline 🏆=Winner', fontweight='bold', fontsize=12, pad=20)

    # Overall title
    fig.suptitle(f'Softmin Parameter Optimization: {len(sorted_names)} Configurations × {num_runs} Simulations\n' +
                f'Baseline: {baseline_name} | Best: {winner_name}',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    return fig


def print_detailed_summary(agent_stats: Dict, num_runs: int):
    """Print detailed summary"""

    print(f"\n{'='*80}")
    print("PARAMETER OPTIMIZATION RESULTS")
    print(f"{'='*80}\n")

    # Sort by survival rate, then final performance
    sorted_names = sorted(agent_stats.keys(),
                         key=lambda k: (agent_stats[k]['survival_rate'],
                                       agent_stats[k]['final_perf_mean']),
                         reverse=True)

    baseline_name = "Softmin(λ=0.05,β=1.0)"
    baseline_stats = agent_stats[baseline_name]

    print(f"BASELINE (Original Winner): {baseline_name}")
    print(f"  Design Performance: {baseline_stats['design_perf_mean']:.2f} ± {baseline_stats['design_perf_std']:.2f}")
    print(f"  Survival Rate:      {baseline_stats['survival_rate']*100:.1f}% ({baseline_stats['survival_count']}/{num_runs})")
    print(f"  Final Performance:  {baseline_stats['final_perf_mean']:.2f} ± {baseline_stats['final_perf_std']:.2f}")
    print(f"  Efficiency:         {baseline_stats['efficiency_mean']:.3f} perf/W")
    print()

    print(f"{'='*80}")
    print("ALL CONFIGURATIONS (sorted by survival rate, then final performance)")
    print(f"{'='*80}\n")

    for rank, name in enumerate(sorted_names, 1):
        stats = agent_stats[name]
        marker = "🏆" if rank == 1 and name != "Greedy" else f"{rank:2d}."

        # Calculate improvement vs baseline
        if name != baseline_name and name != "Greedy":
            surv_diff = (stats['survival_rate'] - baseline_stats['survival_rate']) * 100
            perf_diff = stats['final_perf_mean'] - baseline_stats['final_perf_mean']
            surv_arrow = "↑" if surv_diff > 0 else "↓" if surv_diff < 0 else "="
            perf_arrow = "↑" if perf_diff > 0 else "↓" if perf_diff < 0 else "="
            improvement = f" ({surv_arrow}{abs(surv_diff):.1f}%, {perf_arrow}{abs(perf_diff):.1f})"
        else:
            improvement = ""

        print(f"{marker} {name:35s}")
        print(f"   Survival: {stats['survival_rate']*100:5.1f}% ({stats['survival_count']:2d}/{num_runs}) | "
              f"Final Perf: {stats['final_perf_mean']:6.2f} | "
              f"Efficiency: {stats['efficiency_mean']:.3f}{improvement}")

    print(f"\n{'='*80}")

    # Find best configuration (exclude Greedy)
    softmin_names = [n for n in sorted_names if n != "Greedy"]
    winner_name = softmin_names[0]
    winner_stats = agent_stats[winner_name]

    print("🏆 OPTIMAL CONFIGURATION")
    print(f"{'='*80}\n")
    print(f"{winner_name}")
    print(f"  Design Performance: {winner_stats['design_perf_mean']:.2f} ± {winner_stats['design_perf_std']:.2f}")
    print(f"  Survival Rate:      {winner_stats['survival_rate']*100:.1f}% ({winner_stats['survival_count']}/{num_runs})")
    print(f"  Final Performance:  {winner_stats['final_perf_mean']:.2f} ± {winner_stats['final_perf_std']:.2f}")
    print(f"  Efficiency:         {winner_stats['efficiency_mean']:.3f} perf/W")

    if winner_name != baseline_name:
        surv_improvement = (winner_stats['survival_rate'] - baseline_stats['survival_rate']) * 100
        perf_improvement = winner_stats['final_perf_mean'] - baseline_stats['final_perf_mean']
        print(f"\n  Improvement vs Baseline:")
        print(f"    Survival Rate:     {surv_improvement:+.1f}%")
        print(f"    Final Performance: {perf_improvement:+.2f}")
    else:
        print(f"\n  ✓ Baseline configuration is optimal!")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Run parameter optimization with realistic simulation
    print("Starting Softmin parameter optimization...")
    print("This will take several minutes...")

    NUM_RUNS = 100  # More runs for statistical confidence

    all_results = run_experiments(
        num_runs=NUM_RUNS,
        design_steps=75,
        adaptation_steps=25,
        seed=42,
        test_all_shifts=True,  # Test against all shift types evenly
        verbose=False,
    )

    # Analyze results
    print("\nAnalyzing results...")
    agent_stats, agent_data = analyze_results(all_results)

    # Print detailed summary
    print_detailed_summary(agent_stats, num_runs=NUM_RUNS)

    # Create visualization
    print("\nCreating visualization...")
    create_visualization(agent_stats, agent_data, num_runs=NUM_RUNS,
                        output_file="optimized_softmin_comparison.png")

    # Save data
    output_data = {
        'agent_stats': {k: {**v} for k, v in agent_stats.items()},
        'num_runs': NUM_RUNS,
        'test_all_shifts': True,
    }

    with open('optimized_softmin_data.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("PARAMETER OPTIMIZATION COMPLETE!")
    print(f"{'='*80}")
    print("Files created:")
    print("  - optimized_softmin_comparison.png (visualization)")
    print("  - optimized_softmin_data.json (statistics)")
    print(f"{'='*80}\n")
