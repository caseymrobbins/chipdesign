#!/usr/bin/env python3
"""
Comprehensive comparison: Greedy vs JAM vs Softmin JAM

Focus: Maximum performance and speed while maintaining robustness
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
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11


@dataclass
class AgentResult:
    """Results for a single agent"""
    name: str
    design_performance: float
    design_min_headroom: float
    design_power: float
    design_area: float
    design_frequency: float
    design_voltage: float
    design_cores: int
    design_steps_taken: int
    survived_shift: bool
    final_performance: float
    final_min_headroom: float
    final_power: float
    adaptation_steps: int


def run_single_comparison(
    run_id: int,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    shift_type: Optional[ShiftType] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[AgentResult]:
    """Run one comparison across all three agents"""

    rng = np.random.RandomState(seed)

    # Create initial design space
    base_space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=seed)
    base_space.initialize_actions()

    # Create agents with identical starting conditions
    agents = [
        ("Greedy", AdvancedGreedyPerformanceAgent()),
        ("JAM (hard-min)", JAMAgent(min_margin_threshold=2.0)),
        ("Softmin JAM (λ=0.2,β=1.5)", SoftminJAMAgent(lambda_weight=0.2, beta=1.5, min_margin_threshold=1.0)),
    ]

    spaces = []
    for name, agent in agents:
        space = base_space.clone()
        agent.initialize(space)
        spaces.append((name, agent, space))

    if verbose:
        print(f"\n{'='*70}")
        print(f"RUN {run_id}")
        print(f"{'='*70}")

    # DESIGN PHASE
    steps_taken = {name: 0 for name, _, _ in spaces}

    for step in range(design_steps):
        for name, agent, space in spaces:
            action, feasible = agent.step()
            if action is not None:
                steps_taken[name] += 1

            if verbose and step % 25 == 0:
                perf = space.calculate_performance()
                min_h = space.get_min_headroom()
                power = space.calculate_constraints()['total_power_w']
                print(f"Step {step:3d} | {name:25s}: Perf={perf:7.2f}, MinH={min_h:6.2f}, Power={power:5.1f}W")

    # Collect design phase results
    design_results = []
    for name, agent, space in spaces:
        constraints = space.calculate_constraints()
        design_results.append({
            'name': name,
            'performance': space.calculate_performance(),
            'min_headroom': space.get_min_headroom(),
            'power': constraints['total_power_w'],
            'area': constraints['area_mm2'],
            'frequency': space.params.clock_freq_ghz,
            'voltage': space.params.supply_voltage,
            'cores': space.params.num_cores,
            'steps_taken': steps_taken[name],
        })

    if verbose:
        print(f"\n{'='*70}")
        print("DESIGN PHASE COMPLETE")
        print(f"{'='*70}")
        for res in design_results:
            print(f"{res['name']:25s}: Perf={res['performance']:7.2f}, Power={res['power']:5.1f}W, "
                  f"Freq={res['frequency']:.2f}GHz, Steps={res['steps_taken']}")

    # REQUIREMENT SHIFT
    if shift_type is None:
        shift_type = rng.choice(list(ShiftType))

    if verbose:
        print(f"\nREQUIREMENT SHIFT: {shift_type.value}")

    survival = []
    for name, agent, space in spaces:
        space.apply_requirement_shift(shift_type, rng)
        survived = space.is_feasible()
        survival.append(survived)
        if verbose:
            status = "✓ SURVIVED" if survived else "✗ FAILED"
            print(f"  {name:25s}: {status}")

    # ADAPTATION PHASE
    if verbose:
        print(f"\nADAPTATION PHASE ({adaptation_steps} steps)")

    for step in range(adaptation_steps):
        for i, (name, agent, space) in enumerate(spaces):
            if survival[i]:
                action, feasible = agent.step()
                if not feasible:
                    survival[i] = False

    # Collect final results
    results = []
    for i, (name, agent, space) in enumerate(spaces):
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
            name=name,
            design_performance=design_res['performance'],
            design_min_headroom=design_res['min_headroom'],
            design_power=design_res['power'],
            design_area=design_res['area'],
            design_frequency=design_res['frequency'],
            design_voltage=design_res['voltage'],
            design_cores=design_res['cores'],
            design_steps_taken=design_res['steps_taken'],
            survived_shift=survival[i],
            final_performance=final_perf,
            final_min_headroom=final_headroom,
            final_power=final_power,
            adaptation_steps=adaptation_steps,
        ))

    if verbose:
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        for res in results:
            status = "✓" if res.survived_shift else "✗"
            print(f"{status} {res.name:25s}: Final Perf={res.final_performance:7.2f}")

    return results


def run_experiments(
    num_runs: int = 50,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[List[AgentResult]]:
    """Run multiple comparison experiments"""

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE AGENT COMPARISON")
    print(f"{'='*80}")
    print(f"Runs: {num_runs}")
    print(f"Agents: Greedy, JAM (hard-min), Softmin JAM")
    print(f"Focus: Maximum performance + speed + robustness")
    print(f"{'='*80}\n")

    rng = np.random.RandomState(seed)
    all_results = []

    for run_id in range(num_runs):
        run_seed = rng.randint(0, 1000000)
        results = run_single_comparison(
            run_id=run_id,
            design_steps=design_steps,
            adaptation_steps=adaptation_steps,
            seed=run_seed,
            verbose=verbose,
        )
        all_results.append(results)

        if not verbose and (run_id + 1) % 10 == 0:
            print(f"Completed {run_id + 1}/{num_runs} runs...")

    return all_results


def create_comprehensive_visualization(all_results: List[List[AgentResult]], output_file: str = "agent_comparison.png"):
    """Create comprehensive visualization comparing all agents"""

    # Extract data by agent
    agent_names = ["Greedy", "JAM (hard-min)", "Softmin JAM (λ=0.2,β=1.5)"]
    data = {name: {
        'design_perf': [],
        'design_power': [],
        'design_freq': [],
        'design_headroom': [],
        'survival': [],
        'final_perf': [],
        'efficiency': [],  # perf/watt
    } for name in agent_names}

    for run_results in all_results:
        for agent_result in run_results:
            name = agent_result.name
            data[name]['design_perf'].append(agent_result.design_performance)
            data[name]['design_power'].append(agent_result.design_power)
            data[name]['design_freq'].append(agent_result.design_frequency)
            data[name]['design_headroom'].append(agent_result.design_min_headroom)
            data[name]['survival'].append(1 if agent_result.survived_shift else 0)

            if agent_result.survived_shift:
                data[name]['final_perf'].append(agent_result.final_performance)
                data[name]['efficiency'].append(agent_result.design_performance / agent_result.design_power)

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red (Greedy), Blue (JAM), Green (Softmin)

    # 1. PERFORMANCE COMPARISON (Design Phase)
    ax1 = plt.subplot(3, 3, 1)
    positions = [0, 1, 2]
    perf_data = [data[name]['design_perf'] for name in agent_names]
    bp1 = ax1.boxplot(perf_data, positions=positions, widths=0.6, patch_artist=True,
                      boxprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='darkred'),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Greedy', 'JAM\n(hard-min)', 'Softmin\nJAM'], fontsize=10)
    ax1.set_ylabel('Performance', fontweight='bold', fontsize=11)
    ax1.set_title('Design Phase Performance', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Add mean values as text
    for i, (pos, perf) in enumerate(zip(positions, perf_data)):
        mean_val = np.mean(perf)
        ax1.text(pos, mean_val, f'{mean_val:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9, color=colors[i])

    # 2. SURVIVAL RATE
    ax2 = plt.subplot(3, 3, 2)
    survival_rates = [np.mean(data[name]['survival']) * 100 for name in agent_names]
    bars = ax2.bar(positions, survival_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Greedy', 'JAM\n(hard-min)', 'Softmin\nJAM'], fontsize=10)
    ax2.set_ylabel('Survival Rate (%)', fontweight='bold', fontsize=11)
    ax2.set_title('Adaptability: Survival After Requirement Shift', fontweight='bold', fontsize=12)
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, (bar, rate) in enumerate(zip(bars, survival_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=11, color=colors[i])

    # 3. EFFICIENCY (Performance / Watt)
    ax3 = plt.subplot(3, 3, 3)
    eff_data = [data[name]['efficiency'] for name in agent_names]
    bp3 = ax3.boxplot(eff_data, positions=positions, widths=0.6, patch_artist=True,
                      boxprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='darkred'),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(['Greedy', 'JAM\n(hard-min)', 'Softmin\nJAM'], fontsize=10)
    ax3.set_ylabel('Perf / Watt', fontweight='bold', fontsize=11)
    ax3.set_title('Power Efficiency', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    # Add mean values
    for i, (pos, eff) in enumerate(zip(positions, eff_data)):
        mean_val = np.mean(eff)
        ax3.text(pos, mean_val, f'{mean_val:.2f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9, color=colors[i])

    # 4. POWER CONSUMPTION
    ax4 = plt.subplot(3, 3, 4)
    power_data = [data[name]['design_power'] for name in agent_names]
    bp4 = ax4.boxplot(power_data, positions=positions, widths=0.6, patch_artist=True,
                      boxprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='darkred'),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xticks(positions)
    ax4.set_xticklabels(['Greedy', 'JAM\n(hard-min)', 'Softmin\nJAM'], fontsize=10)
    ax4.set_ylabel('Power (W)', fontweight='bold', fontsize=11)
    ax4.set_title('Power Consumption (Design Phase)', fontweight='bold', fontsize=12)
    ax4.axhline(y=12.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='12W Max')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # Add mean values
    for i, (pos, pwr) in enumerate(zip(positions, power_data)):
        mean_val = np.mean(pwr)
        ax4.text(pos, mean_val, f'{mean_val:.1f}W', ha='center', va='bottom',
                fontweight='bold', fontsize=9, color=colors[i])

    # 5. CLOCK FREQUENCY
    ax5 = plt.subplot(3, 3, 5)
    freq_data = [data[name]['design_freq'] for name in agent_names]
    bp5 = ax5.boxplot(freq_data, positions=positions, widths=0.6, patch_artist=True,
                      boxprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='darkred'),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    for patch, color in zip(bp5['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_xticks(positions)
    ax5.set_xticklabels(['Greedy', 'JAM\n(hard-min)', 'Softmin\nJAM'], fontsize=10)
    ax5.set_ylabel('Clock Frequency (GHz)', fontweight='bold', fontsize=11)
    ax5.set_title('Clock Speed', fontweight='bold', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)

    # Add mean values
    for i, (pos, freq) in enumerate(zip(positions, freq_data)):
        mean_val = np.mean(freq)
        ax5.text(pos, mean_val, f'{mean_val:.2f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9, color=colors[i])

    # 6. MIN HEADROOM (Robustness Indicator)
    ax6 = plt.subplot(3, 3, 6)
    headroom_data = [data[name]['design_headroom'] for name in agent_names]
    bp6 = ax6.boxplot(headroom_data, positions=positions, widths=0.6, patch_artist=True,
                      boxprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='darkred'),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    for patch, color in zip(bp6['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_xticks(positions)
    ax6.set_xticklabels(['Greedy', 'JAM\n(hard-min)', 'Softmin\nJAM'], fontsize=10)
    ax6.set_ylabel('Min Headroom', fontweight='bold', fontsize=11)
    ax6.set_title('Design Margin (Higher = More Robust)', fontweight='bold', fontsize=12)
    ax6.grid(axis='y', alpha=0.3)

    # Add mean values
    for i, (pos, hr) in enumerate(zip(positions, headroom_data)):
        mean_val = np.mean(hr)
        ax6.text(pos, mean_val, f'{mean_val:.2f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9, color=colors[i])

    # 7. PERFORMANCE vs POWER SCATTER
    ax7 = plt.subplot(3, 3, 7)
    for i, name in enumerate(agent_names):
        perf = data[name]['design_perf']
        power = data[name]['design_power']
        ax7.scatter(power, perf, c=colors[i], alpha=0.6, s=100, edgecolors='black',
                   linewidth=1, label=name.replace(' (λ=0.2,β=1.5)', ''))
    ax7.set_xlabel('Power Consumption (W)', fontweight='bold', fontsize=11)
    ax7.set_ylabel('Performance', fontweight='bold', fontsize=11)
    ax7.set_title('Performance vs Power Trade-off', fontweight='bold', fontsize=12)
    ax7.axvline(x=12.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax7.text(12.1, ax7.get_ylim()[1]*0.95, '12W limit', fontsize=9, color='red')
    ax7.legend(loc='lower right', fontsize=9)
    ax7.grid(alpha=0.3)

    # 8. WINNER DISTRIBUTION
    ax8 = plt.subplot(3, 3, 8)

    # Determine winner for each run (by final performance if survived, else DNF)
    winners = {'Greedy': 0, 'JAM (hard-min)': 0, 'Softmin JAM (λ=0.2,β=1.5)': 0, 'Tie': 0}

    for run_results in all_results:
        # Get survivors
        survivors = [r for r in run_results if r.survived_shift]

        if len(survivors) == 0:
            winners['Tie'] += 1
        elif len(survivors) == 1:
            winners[survivors[0].name] += 1
        else:
            # Multiple survivors - pick highest final performance
            best = max(survivors, key=lambda r: r.final_performance)
            # Check if there's a clear winner (>1% difference)
            second_best = sorted(survivors, key=lambda r: r.final_performance, reverse=True)[1]
            if (best.final_performance - second_best.final_performance) / best.final_performance > 0.01:
                winners[best.name] += 1
            else:
                winners['Tie'] += 1

    winner_names = ['Greedy', 'JAM\n(hard-min)', 'Softmin\nJAM', 'Tie']
    winner_counts = [winners['Greedy'], winners['JAM (hard-min)'], winners['Softmin JAM (λ=0.2,β=1.5)'], winners['Tie']]
    winner_colors = colors + ['#95a5a6']

    bars = ax8.bar(range(4), winner_counts, color=winner_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax8.set_xticks(range(4))
    ax8.set_xticklabels(winner_names, fontsize=10)
    ax8.set_ylabel('Number of Wins', fontweight='bold', fontsize=11)
    ax8.set_title('Overall Winners (Survived + Best Performance)', fontweight='bold', fontsize=12)
    ax8.grid(axis='y', alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, winner_counts):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 9. SUMMARY STATISTICS TABLE
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Create summary table
    summary_data = []
    for i, name in enumerate(agent_names):
        short_name = name.replace(' (λ=0.2,β=1.5)', '').replace(' (hard-min)', '')
        summary_data.append([
            short_name,
            f"{np.mean(data[name]['design_perf']):.1f}",
            f"{survival_rates[i]:.0f}%",
            f"{np.mean(data[name]['efficiency']):.2f}",
            f"{np.mean(data[name]['design_power']):.1f}W",
        ])

    table = ax9.table(cellText=summary_data,
                     colLabels=['Agent', 'Avg Perf', 'Survival', 'Eff', 'Power'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows by agent
    for i in range(1, 4):
        for j in range(5):
            table[(i, j)].set_facecolor(colors[i-1])
            table[(i, j)].set_alpha(0.3)

    ax9.set_title('Summary Statistics', fontweight='bold', fontsize=12, pad=20)

    # Overall title
    fig.suptitle('Agent Comparison: Greedy vs JAM vs Softmin JAM\nFocus: Performance, Speed, and Robustness',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    return fig


def print_summary_stats(all_results: List[List[AgentResult]]):
    """Print detailed summary statistics"""

    agent_names = ["Greedy", "JAM (hard-min)", "Softmin JAM (λ=0.2,β=1.5)"]

    print(f"\n{'='*80}")
    print("DETAILED SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    for name in agent_names:
        results = [r for run in all_results for r in run if r.name == name]

        design_perf = [r.design_performance for r in results]
        design_power = [r.design_power for r in results]
        design_freq = [r.design_frequency for r in results]
        survival = [r.survived_shift for r in results]
        efficiency = [r.design_performance / r.design_power for r in results]
        final_perf = [r.final_performance for r in results if r.survived_shift]

        print(f"{name}")
        print(f"{'-'*len(name)}")
        print(f"  Design Phase:")
        print(f"    Performance:     {np.mean(design_perf):7.2f} ± {np.std(design_perf):5.2f}")
        print(f"    Power:           {np.mean(design_power):7.2f} ± {np.std(design_power):5.2f} W")
        print(f"    Clock Freq:      {np.mean(design_freq):7.2f} ± {np.std(design_freq):5.2f} GHz")
        print(f"    Efficiency:      {np.mean(efficiency):7.2f} ± {np.std(efficiency):5.2f} perf/W")
        print(f"  Adaptation:")
        print(f"    Survival Rate:   {np.mean(survival)*100:6.1f}% ({sum(survival)}/{len(survival)})")
        if final_perf:
            print(f"    Final Perf:      {np.mean(final_perf):7.2f} ± {np.std(final_perf):5.2f} (survivors only)")
        else:
            print(f"    Final Perf:      N/A (no survivors)")
        print()


if __name__ == "__main__":
    # Run experiments
    all_results = run_experiments(
        num_runs=50,
        design_steps=75,
        adaptation_steps=25,
        seed=42,
        verbose=False,
    )

    # Print statistics
    print_summary_stats(all_results)

    # Create visualization
    create_comprehensive_visualization(all_results, "agent_comparison.png")

    # Save raw data
    data_to_save = []
    for run_results in all_results:
        run_data = []
        for result in run_results:
            run_data.append({
                'name': result.name,
                'design_performance': result.design_performance,
                'design_power': result.design_power,
                'design_frequency': result.design_frequency,
                'survived': result.survived_shift,
                'final_performance': result.final_performance,
            })
        data_to_save.append(run_data)

    with open('agent_comparison_data.json', 'w') as f:
        json.dump(data_to_save, f, indent=2)

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print("Files created:")
    print("  - agent_comparison.png (visualization)")
    print("  - agent_comparison_data.json (raw data)")
    print(f"{'='*80}\n")
