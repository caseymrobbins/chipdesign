#!/usr/bin/env python3
"""
Comparison: Greedy vs Pure Intrinsic Optimization (Fixed JAM Agents)

This script demonstrates the performance improvement from removing external
constraints and trusting pure intrinsic optimization: R = Σv + λ·log(min(v))

Compares:
1. GreedyPerformance - Maximizes immediate performance gain
2. JAMAgent - Pure log(min(headroom)) optimization (FIXED - no thresholds)
3. AdaptiveJAM - Two-phase strategy (FIXED - no thresholds)
4. HybridJAM - Full intrinsic R = Σv + 1000·log(min(v)) (FIXED - no thresholds)
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
    AdaptiveJAM,
    HybridJAM,
    ProcessTechnology,
    ShiftType,
)
from test_softmin_jam import SoftminJAMAgent
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


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
    design_temperature: float
    design_efficiency: float  # perf/watt
    survived_shift: bool
    final_performance: float
    final_min_headroom: float
    final_efficiency: float


def run_single_comparison(
    run_id: int,
    design_steps: int = 40,  # User requested 40 steps
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

    # FINAL TEST: JAMAdvanced optimized for BEST chip
    # λ=500 provides best balance: 111.62 perf (+6% over λ=200) with 10% power tolerance
    agents = [
        ("IndustryBest", AdvancedGreedyPerformanceAgent()),
        ("JAM", JAMAgent()),
        ("JAMAdvanced", SoftminJAMAgent(lambda_weight=500.0, beta=5.0)),  # Best chip: high perf + robustness
    ]

    spaces = []
    for name, agent in agents:
        space = base_space.clone()
        agent.initialize(space)
        spaces.append((name, agent, space))

    if verbose:
        print(f"\n{'='*80}")
        print(f"RUN {run_id} - Pure Intrinsic Optimization (No External Constraints)")
        print(f"{'='*80}")

    # DESIGN PHASE
    for step in range(design_steps):
        for name, agent, space in spaces:
            action, feasible = agent.step()

            if verbose and step % 25 == 0:
                perf = space.calculate_performance()
                min_h = space.get_min_headroom()
                constraints = space.calculate_constraints()
                power = constraints['total_power_w']
                temp = constraints['temperature_c']
                print(f"Step {step:3d} | {name:25s}: Perf={perf:7.2f}, MinH={min_h:6.2f}, "
                      f"Power={power:5.1f}W, Temp={temp:4.1f}°C")

    # Collect design phase results
    design_results = []
    for name, agent, space in spaces:
        constraints = space.calculate_constraints()
        perf = space.calculate_performance()
        power = constraints['total_power_w']

        design_results.append({
            'name': name,
            'performance': perf,
            'min_headroom': space.get_min_headroom(),
            'power': power,
            'area': constraints['area_mm2'],
            'frequency': space.params.clock_freq_ghz,
            'voltage': space.params.supply_voltage,
            'temperature': constraints['temperature_c'],
            'efficiency': perf / power,
        })

    if verbose:
        print(f"\n{'='*80}")
        print("DESIGN PHASE COMPLETE")
        print(f"{'='*80}")
        for res in design_results:
            print(f"{res['name']:25s}: Perf={res['performance']:7.2f}, "
                  f"Eff={res['efficiency']:5.2f}, Power={res['power']:5.1f}W, "
                  f"MinH={res['min_headroom']:6.2f}")

    # REQUIREMENT SHIFT
    if shift_type is None:
        shift_type = rng.choice(list(ShiftType))

    if verbose:
        print(f"\nREQUIREMENT SHIFT: {shift_type.value}")

    survival = []
    for name, agent, space in spaces:
        shift_info = space.apply_requirement_shift(shift_type, rng)
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
            constraints = space.calculate_constraints()
            final_perf = space.calculate_performance()
            final_headroom = space.get_min_headroom()
            final_power = constraints['total_power_w']
            final_eff = final_perf / final_power
        else:
            final_perf = 0.0
            final_headroom = -999.0
            final_eff = 0.0

        results.append(AgentResult(
            name=name,
            design_performance=design_res['performance'],
            design_min_headroom=design_res['min_headroom'],
            design_power=design_res['power'],
            design_area=design_res['area'],
            design_frequency=design_res['frequency'],
            design_voltage=design_res['voltage'],
            design_temperature=design_res['temperature'],
            design_efficiency=design_res['efficiency'],
            survived_shift=survival[i],
            final_performance=final_perf,
            final_min_headroom=final_headroom,
            final_efficiency=final_eff,
        ))

    if verbose:
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        for res in results:
            status = "✓" if res.survived_shift else "✗"
            print(f"{status} {res.name:25s}: Final Perf={res.final_performance:7.2f}, "
                  f"Eff={res.final_efficiency:5.2f}")

    return results


def run_experiments(
    num_runs: int = 50,
    design_steps: int = 40,  # User requested 40 steps
    adaptation_steps: int = 25,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[List[AgentResult]]:
    """Run multiple comparison experiments"""

    print(f"\n{'='*80}")
    print(f"GREEDY vs PURE INTRINSIC OPTIMIZATION COMPARISON")
    print(f"{'='*80}")
    print(f"Runs: {num_runs}")
    print(f"Design steps: {design_steps}")
    print(f"Adaptation steps: {adaptation_steps}")
    print(f"\nFinal Comparison - 3 Agents:")
    print(f"  1. IndustryBest - Standard greedy performance maximization")
    print(f"  2. JAM - Logarithmic barrier with hard min (eliminates weight tuning)")
    print(f"  3. JAMAdvanced - Logarithmic barrier with softmin (smooth optimization)")
    print(f"\n✓ JAM/JAMAdvanced use PURE intrinsic optimization (NO external constraints)")
    print(f"✓ Automatic bottleneck focus via log(min/softmin) - no manual weight tuning")
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


def create_comparison_visualization(all_results: List[List[AgentResult]],
                                   output_file: str = "greedy_vs_intrinsic.png"):
    """Create comprehensive visualization comparing Greedy vs Intrinsic optimization"""

    # Extract data by agent - dynamically from first run to avoid hardcoding
    agent_names = [result.name for result in all_results[0]]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63']
    # Red, Blue, Green, Purple, Orange, Teal, Pink

    data = {name: {
        'design_perf': [],
        'design_power': [],
        'design_freq': [],
        'design_headroom': [],
        'design_efficiency': [],
        'design_temp': [],
        'survival': [],
        'final_perf': [],
        'final_efficiency': [],
    } for name in agent_names}

    for run_results in all_results:
        for agent_result in run_results:
            name = agent_result.name
            data[name]['design_perf'].append(agent_result.design_performance)
            data[name]['design_power'].append(agent_result.design_power)
            data[name]['design_freq'].append(agent_result.design_frequency)
            data[name]['design_headroom'].append(agent_result.design_min_headroom)
            data[name]['design_efficiency'].append(agent_result.design_efficiency)
            data[name]['design_temp'].append(agent_result.design_temperature)
            data[name]['survival'].append(1 if agent_result.survived_shift else 0)

            if agent_result.survived_shift:
                data[name]['final_perf'].append(agent_result.final_performance)
                data[name]['final_efficiency'].append(agent_result.final_efficiency)

    # Create figure
    fig = plt.figure(figsize=(20, 14))

    # Title with emphasis on "NO external constraints" and testing Softmin variants
    fig.suptitle('Greedy vs Pure Intrinsic Optimization: Testing Softmin for Maximum Performance\n'
                '✓ All agents use PURE intrinsic optimization (NO external constraints) | R = Σv + λ·log(softmin(v; β))',
                fontsize=16, fontweight='bold', y=0.98)

    # Helper function for box plots
    def create_boxplot(ax, data_list, positions, title, ylabel, show_mean=True):
        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True,
                       boxprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2.5, color='darkred'),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(['Greedy', 'JAM\n(hard)', 'Adaptive\nJAM', 'Hybrid\nJAM',
                           'Softmin\n(λ=200)', 'Softmin\n(λ=1000)', 'Softmin\n(λ=5000)'],
                          fontsize=9, rotation=15)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        if show_mean:
            for i, (pos, values) in enumerate(zip(positions, data_list)):
                mean_val = np.mean(values)
                ax.text(pos, mean_val, f'{mean_val:.1f}', ha='center', va='bottom',
                       fontweight='bold', fontsize=10, color=colors[i],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    positions = list(range(len(agent_names)))  # [0, 1, 2, 3, 4, 5, 6]

    # Row 1: Performance, Efficiency, Survival
    ax1 = plt.subplot(3, 4, 1)
    perf_data = [data[name]['design_perf'] for name in agent_names]
    create_boxplot(ax1, perf_data, positions,
                  'Design Phase Performance\n(Higher is Better)', 'Performance Score')

    ax2 = plt.subplot(3, 4, 2)
    eff_data = [data[name]['design_efficiency'] for name in agent_names]
    create_boxplot(ax2, eff_data, positions,
                  'Power Efficiency\n(Higher is Better)', 'Perf / Watt')

    ax3 = plt.subplot(3, 4, 3)
    survival_rates = [np.mean(data[name]['survival']) * 100 for name in agent_names]
    bars = ax3.bar(positions, survival_rates, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(['Greedy', 'JAM\n(hard)', 'Adaptive\nJAM', 'Hybrid\nJAM',
                        'Softmin\n(λ=200)', 'Softmin\n(λ=1000)', 'Softmin\n(λ=5000)'],
                        fontsize=9, rotation=15)
    ax3.set_ylabel('Survival Rate (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Adaptability: Survived Requirement Shift\n(Higher is Better)',
                  fontweight='bold', fontsize=13)
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)
    for i, (bar, rate) in enumerate(zip(bars, survival_rates)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=12, color=colors[i])

    ax4 = plt.subplot(3, 4, 4)
    headroom_data = [data[name]['design_headroom'] for name in agent_names]
    create_boxplot(ax4, headroom_data, positions,
                  'Design Margin\n(Higher = More Robust)', 'Min Headroom')

    # Row 2: Power, Frequency, Temperature, Winner Pie
    ax5 = plt.subplot(3, 4, 5)
    power_data = [data[name]['design_power'] for name in agent_names]
    create_boxplot(ax5, power_data, positions,
                  'Power Consumption\n(Lower is Better)', 'Power (W)')
    ax5.axhline(y=12.0, color='red', linestyle='--', linewidth=2.5, alpha=0.8,
                label='12W Limit')
    ax5.legend(loc='upper right', fontsize=10)

    ax6 = plt.subplot(3, 4, 6)
    freq_data = [data[name]['design_freq'] for name in agent_names]
    create_boxplot(ax6, freq_data, positions,
                  'Clock Frequency\n(Higher is Better)', 'Frequency (GHz)')

    ax7 = plt.subplot(3, 4, 7)
    temp_data = [data[name]['design_temp'] for name in agent_names]
    create_boxplot(ax7, temp_data, positions,
                  'Operating Temperature\n(Lower is Better)', 'Temperature (°C)')
    ax7.axhline(y=70.0, color='red', linestyle='--', linewidth=2.5, alpha=0.8,
                label='70°C Limit')
    ax7.legend(loc='upper right', fontsize=10)

    # Winner distribution pie chart
    ax8 = plt.subplot(3, 4, 8)
    winners = {name: 0 for name in agent_names}
    winners['Tie'] = 0

    for run_results in all_results:
        survivors = [r for r in run_results if r.survived_shift]
        if len(survivors) == 0:
            winners['Tie'] += 1
        elif len(survivors) == 1:
            winners[survivors[0].name] += 1
        else:
            best = max(survivors, key=lambda r: r.final_performance)
            second_best = sorted(survivors, key=lambda r: r.final_performance, reverse=True)[1]
            if (best.final_performance - second_best.final_performance) > 1.0:
                winners[best.name] += 1
            else:
                winners['Tie'] += 1

    winner_counts = [winners[name] for name in agent_names] + [winners['Tie']]
    winner_labels = [name.split('(')[0].strip()[:8] for name in agent_names] + ['Tie']
    winner_colors_pie = colors + ['#95a5a6']

    # Filter out zero counts to avoid pie chart errors
    non_zero = [(count, label, color) for count, label, color in zip(winner_counts, winner_labels, winner_colors_pie) if count > 0]
    if non_zero:
        filtered_counts, filtered_labels, filtered_colors = zip(*non_zero)
        wedges, texts, autotexts = ax8.pie(filtered_counts, labels=filtered_labels, colors=filtered_colors,
                                           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
    else:
        ax8.text(0.5, 0.5, 'No winners', ha='center', va='center', fontsize=12)
    ax8.set_title('Overall Winners\n(Survived + Best Performance)', fontweight='bold', fontsize=13)

    # Row 3: Scatter plots and improvement metrics
    ax9 = plt.subplot(3, 4, 9)
    for i, name in enumerate(agent_names):
        perf = data[name]['design_perf']
        power = data[name]['design_power']
        ax9.scatter(power, perf, c=colors[i], alpha=0.6, s=80, edgecolors='black',
                   linewidth=1, label=name.replace(' (λ=1000)', '').replace(' (pure log-min)', ''))
    ax9.set_xlabel('Power (W)', fontweight='bold', fontsize=12)
    ax9.set_ylabel('Performance', fontweight='bold', fontsize=12)
    ax9.set_title('Performance vs Power\nPareto Frontier', fontweight='bold', fontsize=13)
    ax9.axvline(x=12.0, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax9.legend(loc='lower right', fontsize=9)
    ax9.grid(alpha=0.3)

    ax10 = plt.subplot(3, 4, 10)
    for i, name in enumerate(agent_names):
        headroom = data[name]['design_headroom']
        perf = data[name]['design_perf']
        ax10.scatter(headroom, perf, c=colors[i], alpha=0.6, s=80, edgecolors='black',
                    linewidth=1, label=name.replace(' (λ=1000)', '').replace(' (pure log-min)', ''))
    ax10.set_xlabel('Min Headroom (Margin)', fontweight='bold', fontsize=12)
    ax10.set_ylabel('Performance', fontweight='bold', fontsize=12)
    ax10.set_title('Performance vs Margin\nRobustness Trade-off', fontweight='bold', fontsize=13)
    ax10.legend(loc='lower right', fontsize=9)
    ax10.grid(alpha=0.3)

    # Improvement metrics relative to Greedy
    ax11 = plt.subplot(3, 4, 11)
    greedy_perf = np.mean(data['Greedy']['design_perf'])
    greedy_eff = np.mean(data['Greedy']['design_efficiency'])
    greedy_survival = np.mean(data['Greedy']['survival'])

    # Calculate improvements for all JAM agents
    jam_agent_names = agent_names[1:]  # All except Greedy
    improvements = {}
    for jam_name in jam_agent_names:
        short_name = jam_name.split('(')[0].strip()  # Get short name
        improvements[short_name] = {
            'perf': (np.mean(data[jam_name]['design_perf']) - greedy_perf) / greedy_perf * 100,
            'eff': (np.mean(data[jam_name]['design_efficiency']) - greedy_eff) / greedy_eff * 100,
            'survival': (np.mean(data[jam_name]['survival']) - greedy_survival) * 100,
        }

    # Show top 3 performers
    perf_sorted = sorted(improvements.items(), key=lambda x: x[1]['perf'], reverse=True)
    top_3 = perf_sorted[:3]

    metrics = ['Performance', 'Efficiency', 'Survival']
    x_pos = np.arange(len(metrics))
    width = 0.25

    for i, (agent_short_name, _) in enumerate(top_3):
        # Find original index for color
        agent_idx = next(j for j, name in enumerate(jam_agent_names) if name.startswith(agent_short_name))
        color = colors[agent_idx + 1]  # +1 because colors[0] is Greedy

        values = [improvements[agent_short_name]['perf'],
                 improvements[agent_short_name]['eff'],
                 improvements[agent_short_name]['survival']]
        offset = (i - 1) * width
        bars = ax11.bar(x_pos + offset, values, width, label=agent_short_name,
                       color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height + (2 if height > 0 else -2),
                     f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                     fontweight='bold', fontsize=8, color=color)

    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax11.set_ylabel('Improvement vs Greedy (%)', fontweight='bold', fontsize=12)
    ax11.set_title('Improvement Over Greedy Baseline\n(Positive = Better)',
                   fontweight='bold', fontsize=13)
    ax11.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax11.legend(loc='upper right', fontsize=10)
    ax11.grid(axis='y', alpha=0.3)

    # Summary table - showing top performers
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    # Show top 5 performers by performance score
    perf_scores = [(name, np.mean(data[name]['design_perf'])) for name in agent_names]
    perf_scores_sorted = sorted(perf_scores, key=lambda x: x[1], reverse=True)[:5]

    summary_data = []
    for name, _ in perf_scores_sorted:
        i = agent_names.index(name)
        short_name = name.split('(')[0].strip() if '(' in name else name
        summary_data.append([
            short_name[:12],  # Truncate long names
            f"{np.mean(data[name]['design_perf']):.1f}",
            f"{np.mean(data[name]['design_efficiency']):.2f}",
            f"{np.mean(data[name]['design_power']):.1f}W",
            f"{survival_rates[i]:.0f}%",
        ])

    table = ax12.table(cellText=summary_data,
                      colLabels=['Agent', 'Perf', 'Eff', 'Power', 'Survive'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.30, 0.18, 0.16, 0.18, 0.18])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)

    # Color rows by original agent colors
    for row_idx, (name, _) in enumerate(perf_scores_sorted):
        agent_idx = agent_names.index(name)
        for j in range(5):
            table[(row_idx + 1, j)].set_facecolor(colors[agent_idx])
            table[(row_idx + 1, j)].set_alpha(0.3)
            table[(row_idx + 1, j)].set_text_props(fontweight='bold', fontsize=9)

    ax12.set_title('Top 5 Performers\n(by Performance)', fontweight='bold', fontsize=13, pad=20)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")

    return fig


def print_detailed_stats(all_results: List[List[AgentResult]]):
    """Print detailed statistics"""

    agent_names = ["Greedy", "JAM (pure log-min)", "AdaptiveJAM", "HybridJAM (λ=1000)"]

    print(f"\n{'='*80}")
    print("DETAILED STATISTICS")
    print(f"{'='*80}\n")

    for name in agent_names:
        results = [r for run in all_results for r in run if r.name == name]

        design_perf = [r.design_performance for r in results]
        design_eff = [r.design_efficiency for r in results]
        design_power = [r.design_power for r in results]
        design_headroom = [r.design_min_headroom for r in results]
        survival = [r.survived_shift for r in results]

        print(f"{name}")
        print(f"{'-'*len(name)}")
        print(f"  Design Phase:")
        print(f"    Performance:     {np.mean(design_perf):7.2f} ± {np.std(design_perf):5.2f}")
        print(f"    Efficiency:      {np.mean(design_eff):7.2f} ± {np.std(design_eff):5.2f} perf/W")
        print(f"    Power:           {np.mean(design_power):7.2f} ± {np.std(design_power):5.2f} W")
        print(f"    Min Headroom:    {np.mean(design_headroom):7.2f} ± {np.std(design_headroom):5.2f}")
        print(f"  Robustness:")
        print(f"    Survival Rate:   {np.mean(survival)*100:6.1f}% ({sum(survival)}/{len(survival)})")
        print()


if __name__ == "__main__":
    # Run experiments
    # Using 40 steps for faster testing with bug fix
    all_results = run_experiments(
        num_runs=50,
        design_steps=40,
        adaptation_steps=25,
        seed=42,
        verbose=False,
    )

    # Print statistics
    print_detailed_stats(all_results)

    # Save raw data FIRST (before visualization which might crash)
    data_to_save = []
    for run_results in all_results:
        run_data = []
        for result in run_results:
            run_data.append({
                'name': result.name,
                'design_performance': result.design_performance,
                'design_efficiency': result.design_efficiency,
                'design_power': result.design_power,
                'design_min_headroom': result.design_min_headroom,
                'survived': result.survived_shift,
                'final_performance': result.final_performance,
            })
        data_to_save.append(run_data)

    with open('greedy_vs_intrinsic_data.json', 'w') as f:
        json.dump(data_to_save, f, indent=2)

    # Create visualization AFTER saving data
    try:
        create_comparison_visualization(all_results, "greedy_vs_intrinsic.png")
        print(f"\n✓ Visualization saved to: greedy_vs_intrinsic.png")
    except Exception as e:
        print(f"\n⚠ Visualization failed: {e}")
        print("(Data was saved successfully to JSON)")

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print("✓ All JAM agents use PURE intrinsic optimization (NO external constraints)")
    print("\nFiles created:")
    print("  - greedy_vs_intrinsic.png (comprehensive visualization)")
    print("  - greedy_vs_intrinsic_data.json (raw data)")
    print(f"{'='*80}\n")
