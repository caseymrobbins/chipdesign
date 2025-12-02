#!/usr/bin/env python3
"""
Comprehensive comparison of multiple JAM agent configurations:
- 8 HardMin JAM agents with different thresholds
- 8 Softmin JAM agents with different λ and β parameters
- 1 Greedy agent

Run 100 realistic simulations and visualize the top performers.
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

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
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


def create_agent_configs() -> List[AgentConfig]:
    """Create all agent configurations to test"""

    configs = []

    # 1 Greedy agent
    configs.append(AgentConfig(
        name="Greedy",
        agent_type="greedy",
        params={}
    ))

    # NO HardMin JAM agents - REMOVED

    # 4 ULTRA-AGGRESSIVE Softmin JAM agents
    # Push the limits: lower λ, lower β, lower threshold
    softmin_configs = [
        {'lambda_weight': 0.01, 'beta': 0.5, 'min_margin_threshold': 0.2},   # ULTRA aggressive
        {'lambda_weight': 0.03, 'beta': 0.8, 'min_margin_threshold': 0.3},   # HYPER aggressive
        {'lambda_weight': 0.05, 'beta': 1.0, 'min_margin_threshold': 0.5},   # Very aggressive
        {'lambda_weight': 0.08, 'beta': 1.2, 'min_margin_threshold': 0.7},   # Aggressive
    ]

    for i, params in enumerate(softmin_configs, 1):
        configs.append(AgentConfig(
            name=f"Softmin(λ={params['lambda_weight']:.2f},β={params['beta']:.1f})",
            agent_type="softmin",
            params=params
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
            print(f"  {i}. {res['config'].name:30s}: Perf={res['performance']:7.2f}, "
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
            print(f"  {i}. {res.name:30s}: Perf={res.final_performance:7.2f}")

    return results


def run_experiments(
    num_runs: int = 100,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[List[AgentResult]]:
    """Run multiple comparison experiments"""

    agent_configs = create_agent_configs()

    print(f"\n{'='*80}")
    print(f"ULTRA-AGGRESSIVE SOFTMIN COMPARISON")
    print(f"{'='*80}")
    print(f"Runs: {num_runs}")
    print(f"Agents: {len(agent_configs)}")
    print(f"  - 1 Greedy agent")
    print(f"  - 0 HardMin JAM agents (REMOVED)")
    print(f"  - 4 ULTRA-AGGRESSIVE Softmin JAM agents")
    print(f"{'='*80}\n")

    rng = np.random.RandomState(seed)
    all_results = []

    for run_id in range(num_runs):
        run_seed = rng.randint(0, 1000000)
        results = run_single_comparison(
            run_id=run_id,
            agent_configs=agent_configs,
            design_steps=design_steps,
            adaptation_steps=adaptation_steps,
            seed=run_seed,
            verbose=verbose,
        )
        all_results.append(results)

        if not verbose and (run_id + 1) % 10 == 0:
            print(f"Completed {run_id + 1}/{num_runs} runs...")

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


def find_top_agents(agent_stats: Dict) -> Dict[str, str]:
    """Find the top agent for each type"""

    top_agents = {}

    # Find top greedy
    greedy_agents = {k: v for k, v in agent_stats.items() if v['agent_type'] == 'greedy'}
    if greedy_agents:
        top_agents['greedy'] = max(greedy_agents.keys(),
                                   key=lambda k: agent_stats[k]['final_perf_mean'])

    # Find top hardmin (by survival rate, then performance)
    hardmin_agents = {k: v for k, v in agent_stats.items() if v['agent_type'] == 'hardmin'}
    if hardmin_agents:
        top_agents['hardmin'] = max(hardmin_agents.keys(),
                                    key=lambda k: (agent_stats[k]['survival_rate'],
                                                  agent_stats[k]['final_perf_mean']))

    # Find top softmin (by survival rate, then performance)
    softmin_agents = {k: v for k, v in agent_stats.items() if v['agent_type'] == 'softmin'}
    if softmin_agents:
        top_agents['softmin'] = max(softmin_agents.keys(),
                                    key=lambda k: (agent_stats[k]['survival_rate'],
                                                  agent_stats[k]['final_perf_mean']))

    return top_agents


def create_visualization(agent_stats: Dict, agent_data: Dict, top_agents: Dict,
                        output_file: str = "comprehensive_comparison.png"):
    """Create comprehensive visualization comparing top agents"""

    fig = plt.figure(figsize=(20, 12))

    # Colors for agent types
    type_colors = {
        'greedy': '#e74c3c',    # Red
        'hardmin': '#3498db',   # Blue
        'softmin': '#2ecc71',   # Green
    }

    # Extract top agent data
    top_names = list(top_agents.values())
    top_stats = {name: agent_stats[name] for name in top_names}
    top_data = {name: agent_data[name] for name in top_names}

    # 1. DESIGN PHASE PERFORMANCE
    ax1 = plt.subplot(2, 4, 1)
    positions = range(len(top_names))
    perf_data = [top_data[name]['design_perf'] for name in top_names]
    colors = [type_colors[top_stats[name]['agent_type']] for name in top_names]

    bp1 = ax1.boxplot(perf_data, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([name.replace('(', '\n(') for name in top_names], fontsize=9)
    ax1.set_ylabel('Performance', fontweight='bold')
    ax1.set_title('Design Phase Performance', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # 2. SURVIVAL RATE
    ax2 = plt.subplot(2, 4, 2)
    survival_rates = [top_stats[name]['survival_rate'] * 100 for name in top_names]
    bars = ax2.bar(positions, survival_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax2.set_xticks(positions)
    ax2.set_xticklabels([name.replace('(', '\n(') for name in top_names], fontsize=9)
    ax2.set_ylabel('Survival Rate (%)', fontweight='bold')
    ax2.set_title('Adaptability: Survival After Shift', fontweight='bold', fontsize=12)
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bar, rate in zip(bars, survival_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 3. FINAL PERFORMANCE (SURVIVORS)
    ax3 = plt.subplot(2, 4, 3)
    final_perf_data = [top_data[name]['final_perf'] for name in top_names]

    bp3 = ax3.boxplot(final_perf_data, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_xticks(positions)
    ax3.set_xticklabels([name.replace('(', '\n(') for name in top_names], fontsize=9)
    ax3.set_ylabel('Final Performance', fontweight='bold')
    ax3.set_title('Post-Adaptation Performance (Survivors)', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    # 4. EFFICIENCY
    ax4 = plt.subplot(2, 4, 4)
    eff_data = [top_data[name]['efficiency'] for name in top_names]

    bp4 = ax4.boxplot(eff_data, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_xticks(positions)
    ax4.set_xticklabels([name.replace('(', '\n(') for name in top_names], fontsize=9)
    ax4.set_ylabel('Perf / Watt', fontweight='bold')
    ax4.set_title('Power Efficiency', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)

    # 5. ALL AGENTS SURVIVAL RATE COMPARISON
    ax5 = plt.subplot(2, 4, 5)

    # Group by type
    greedy_names = [n for n, s in agent_stats.items() if s['agent_type'] == 'greedy']
    hardmin_names = sorted([n for n, s in agent_stats.items() if s['agent_type'] == 'hardmin'])
    softmin_names = sorted([n for n, s in agent_stats.items() if s['agent_type'] == 'softmin'])

    greedy_survival = [agent_stats[n]['survival_rate'] * 100 for n in greedy_names]
    hardmin_survival = [agent_stats[n]['survival_rate'] * 100 for n in hardmin_names]
    softmin_survival = [agent_stats[n]['survival_rate'] * 100 for n in softmin_names]

    x_pos = 0
    if greedy_survival:
        ax5.bar([x_pos], greedy_survival, color=type_colors['greedy'],
               alpha=0.7, label='Greedy', width=0.8)
        x_pos += 1

    if hardmin_survival:
        ax5.bar(range(x_pos, x_pos + len(hardmin_survival)), hardmin_survival,
               color=type_colors['hardmin'], alpha=0.7, label='HardMin', width=0.8)
        x_pos += len(hardmin_survival)

    if softmin_survival:
        ax5.bar(range(x_pos, x_pos + len(softmin_survival)), softmin_survival,
               color=type_colors['softmin'], alpha=0.7, label='Softmin', width=0.8)

    ax5.set_ylabel('Survival Rate (%)', fontweight='bold')
    ax5.set_title('All Agents Survival Comparison', fontweight='bold', fontsize=12)
    ax5.set_ylim([0, 100])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # 6. ALL AGENTS PERFORMANCE COMPARISON
    ax6 = plt.subplot(2, 4, 6)

    greedy_perf = [agent_stats[n]['design_perf_mean'] for n in greedy_names]
    hardmin_perf = [agent_stats[n]['design_perf_mean'] for n in hardmin_names]
    softmin_perf = [agent_stats[n]['design_perf_mean'] for n in softmin_names]

    x_pos = 0
    if greedy_perf:
        ax6.bar([x_pos], greedy_perf, color=type_colors['greedy'],
               alpha=0.7, label='Greedy', width=0.8)
        x_pos += 1

    if hardmin_perf:
        ax6.bar(range(x_pos, x_pos + len(hardmin_perf)), hardmin_perf,
               color=type_colors['hardmin'], alpha=0.7, label='HardMin', width=0.8)
        x_pos += len(hardmin_perf)

    if softmin_perf:
        ax6.bar(range(x_pos, x_pos + len(softmin_perf)), softmin_perf,
               color=type_colors['softmin'], alpha=0.7, label='Softmin', width=0.8)

    ax6.set_ylabel('Design Performance', fontweight='bold')
    ax6.set_title('All Agents Design Performance', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    # 7. SUMMARY TABLE
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')

    summary_data = []
    for name in top_names:
        stats = top_stats[name]
        summary_data.append([
            name.split('(')[0],
            f"{stats['design_perf_mean']:.1f}",
            f"{stats['survival_rate']*100:.0f}%",
            f"{stats['final_perf_mean']:.1f}",
            f"{stats['efficiency_mean']:.2f}",
        ])

    table = ax7.table(cellText=summary_data,
                     colLabels=['Agent', 'Design\nPerf', 'Survival', 'Final\nPerf', 'Eff'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.2, 0.15, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows by agent type
    for i in range(1, len(top_names) + 1):
        agent_type = top_stats[top_names[i-1]]['agent_type']
        color = type_colors[agent_type]
        for j in range(5):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)

    ax7.set_title('Top Agents Summary', fontweight='bold', fontsize=12, pad=20)

    # 8. PERFORMANCE VS HEADROOM SCATTER (TOP AGENTS)
    ax8 = plt.subplot(2, 4, 8)

    for i, name in enumerate(top_names):
        agent_type = top_stats[name]['agent_type']
        color = type_colors[agent_type]
        perf = top_data[name]['design_perf']
        headroom = top_data[name]['design_headroom']
        ax8.scatter(headroom, perf, c=color, alpha=0.4, s=50,
                   edgecolors='black', linewidth=0.5, label=name.split('(')[0])

    ax8.set_xlabel('Design Headroom', fontweight='bold')
    ax8.set_ylabel('Design Performance', fontweight='bold')
    ax8.set_title('Performance vs Headroom Trade-off', fontweight='bold', fontsize=12)
    ax8.legend(loc='best', fontsize=9)
    ax8.grid(alpha=0.3)

    # Overall title
    fig.suptitle(f'ULTRA-AGGRESSIVE Softmin Comparison: 5 Agents Tested Across {len(agent_data[list(agent_data.keys())[0]]["survival"])} Simulations\n' +
                f'Top Performers: {", ".join([n.split("(")[0] for n in top_names])}',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    return fig


def print_summary(agent_stats: Dict, top_agents: Dict, num_runs: int):
    """Print detailed summary"""

    print(f"\n{'='*80}")
    print("TOP AGENTS BY CATEGORY")
    print(f"{'='*80}\n")

    for agent_type, name in top_agents.items():
        stats = agent_stats[name]
        print(f"{agent_type.upper()}: {name}")
        print(f"  Design Performance: {stats['design_perf_mean']:.2f} ± {stats['design_perf_std']:.2f}")
        print(f"  Survival Rate:      {stats['survival_rate']*100:.1f}% ({stats['survival_count']}/{num_runs})")
        print(f"  Final Performance:  {stats['final_perf_mean']:.2f} ± {stats['final_perf_std']:.2f}")
        print(f"  Efficiency:         {stats['efficiency_mean']:.2f} perf/W")
        print(f"  Design Headroom:    {stats['design_headroom_mean']:.2f}")
        print()

    # Identify overall winner
    print(f"{'='*80}")
    print("OVERALL WINNER")
    print(f"{'='*80}\n")

    # Winner = highest survival rate, then highest final performance
    winner_name = max(agent_stats.keys(),
                     key=lambda k: (agent_stats[k]['survival_rate'],
                                   agent_stats[k]['final_perf_mean']))
    winner_stats = agent_stats[winner_name]

    print(f"🏆 {winner_name}")
    print(f"  Survival Rate:      {winner_stats['survival_rate']*100:.1f}%")
    print(f"  Final Performance:  {winner_stats['final_perf_mean']:.2f}")
    print(f"  Efficiency:         {winner_stats['efficiency_mean']:.2f} perf/W")
    print()


if __name__ == "__main__":
    # Run comprehensive experiments
    print("Starting comprehensive agent comparison...")
    print("This will take several minutes...")

    all_results = run_experiments(
        num_runs=25,
        design_steps=75,
        adaptation_steps=25,
        seed=42,
        verbose=False,
    )

    # Analyze results
    print("\nAnalyzing results...")
    agent_stats, agent_data = analyze_results(all_results)

    # Find top agents
    top_agents = find_top_agents(agent_stats)

    # Print summary
    print_summary(agent_stats, top_agents, num_runs=25)

    # Create visualization
    print("\nCreating visualization...")
    create_visualization(agent_stats, agent_data, top_agents,
                        "comprehensive_comparison.png")

    # Save data
    output_data = {
        'top_agents': top_agents,
        'agent_stats': agent_stats,
    }

    with open('comprehensive_comparison_data.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print("Files created:")
    print("  - comprehensive_comparison.png (visualization)")
    print("  - comprehensive_comparison_data.json (statistics)")
    print(f"{'='*80}\n")
