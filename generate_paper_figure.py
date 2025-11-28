#!/usr/bin/env python3
"""
Generate publication-quality figure for JAM optimization paper.

Shows:
1. Performance vs Efficiency (Pareto frontier)
2. Optimization trajectory over time
3. Final results comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    ConstraintLimits,
    ProcessTechnology,
    AdvancedGreedyPerformanceAgent,
    JAMAgent,
    HybridJAM,
)


def create_option3_limits(floor: float = 6.0) -> ConstraintLimits:
    """Option 3: Higher floor threshold to push performance."""
    limits = ConstraintLimits()
    limits.constraint_weights = {
        'power_max': floor / 1.0,
        'power_min': floor / 5.0,
        'area_max': floor / 4.0,
        'area_min': floor / 21.0,
        'temperature': floor / 10.0,
        'frequency': floor / 0.2,
        'timing_slack': floor / 20.0,
        'ir_drop': floor / 10.0,
        'power_density': floor / 0.1,
        'wire_delay': floor / 20.0,
        'yield': 1.0,
        'signal_integrity': 1.0,
    }
    return limits


def run_experiment(steps: int = 200):
    """Run experiment and collect data for plotting."""
    process = ProcessTechnology.create_7nm()

    # Create design spaces with different configurations
    configs = {
        'GreedyPerf': ('default', None),
        'Floor_4.0': ('default', None),  # Current default (floor=8.0, but we'll override)
        'Floor_6.0': ('option3', 6.0),
        'Floor_8.0': ('option3', 8.0),
    }

    spaces = {}
    agents = {}
    trajectories = {}

    for name, (config_type, floor) in configs.items():
        spaces[name] = AdvancedDesignSpace(process=process)

        if name == 'GreedyPerf':
            agents[name] = AdvancedGreedyPerformanceAgent()
        elif name == 'Floor_4.0':
            # Override to use floor=4.0
            limits = ConstraintLimits()
            limits.constraint_weights = {
                'power_max': 4.0 / 1.0,
                'power_min': 4.0 / 5.0,
                'area_max': 4.0 / 4.0,
                'area_min': 4.0 / 21.0,
                'temperature': 4.0 / 10.0,
                'frequency': 4.0 / 0.2,
                'timing_slack': 4.0 / 20.0,
                'ir_drop': 4.0 / 10.0,
                'power_density': 4.0 / 0.1,
                'wire_delay': 4.0 / 20.0,
                'yield': 1.0,
                'signal_integrity': 1.0,
            }
            spaces[name].limits = limits
            spaces[name].initial_limits = limits.clone()
            agents[name] = HybridJAM(performance_weight=0.05)
        else:
            spaces[name].limits = create_option3_limits(floor=floor)
            spaces[name].initial_limits = spaces[name].limits.clone()
            agents[name] = HybridJAM(performance_weight=0.05)

        agents[name].design_space = spaces[name]
        trajectories[name] = {'step': [], 'perf': [], 'power': [], 'efficiency': []}

    # Run optimization and collect trajectory data
    for step in range(steps):
        for name, agent in agents.items():
            action = agent.select_action()
            if action:
                spaces[name].apply_action(action)

            # Record trajectory every 5 steps
            if step % 5 == 0 or step == steps - 1:
                perf = spaces[name].calculate_performance()
                constraints = spaces[name].calculate_constraints()
                power = constraints['total_power_w']
                efficiency = perf / power if power > 0 else 0

                trajectories[name]['step'].append(step)
                trajectories[name]['perf'].append(perf)
                trajectories[name]['power'].append(power)
                trajectories[name]['efficiency'].append(efficiency)

    # Collect final results
    results = {}
    for name in configs.keys():
        space = spaces[name]
        perf = space.calculate_performance()
        constraints = space.calculate_constraints()
        power = constraints['total_power_w']
        area = constraints['area_mm2']
        efficiency = perf / power if power > 0 else 0

        results[name] = {
            'perf': perf,
            'power': power,
            'area': area,
            'efficiency': efficiency,
        }

    return results, trajectories


def create_figure(results, trajectories):
    """Create publication-quality figure."""

    # Set publication style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (12, 4),
        'figure.dpi': 300,
    })

    fig = plt.figure(figsize=(14, 4.5))

    # Colors for different methods
    colors = {
        'GreedyPerf': '#1f77b4',      # Blue
        'Floor_4.0': '#ff7f0e',       # Orange
        'Floor_6.0': '#2ca02c',       # Green
        'Floor_8.0': '#d62728',       # Red
    }

    markers = {
        'GreedyPerf': 'o',
        'Floor_4.0': 's',
        'Floor_6.0': '^',
        'Floor_8.0': '*',
    }

    labels = {
        'GreedyPerf': 'GreedyPerf',
        'Floor_4.0': 'HybridJAM (floor=4.0)',
        'Floor_6.0': 'HybridJAM (floor=6.0)',
        'Floor_8.0': 'HybridJAM (floor=8.0)',
    }

    # Panel A: Performance vs Efficiency (Pareto frontier)
    ax1 = plt.subplot(1, 3, 1)
    for name in ['GreedyPerf', 'Floor_4.0', 'Floor_6.0', 'Floor_8.0']:
        r = results[name]
        ax1.scatter(r['efficiency'], r['perf'],
                   c=colors[name], marker=markers[name], s=200,
                   label=labels[name], zorder=3, edgecolors='black', linewidth=1)

    ax1.set_xlabel('Efficiency (perf/W)', fontweight='bold')
    ax1.set_ylabel('Performance', fontweight='bold')
    ax1.set_title('(a) Performance-Efficiency Trade-off', fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', framealpha=0.95)

    # Add arrow showing improvement direction
    ax1.annotate('', xy=(9.2, 102), xytext=(8.2, 92),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
    ax1.text(8.5, 96, 'Better', fontsize=9, color='gray', style='italic')

    # Panel B: Optimization trajectory (Performance over time)
    ax2 = plt.subplot(1, 3, 2)
    for name in ['GreedyPerf', 'Floor_4.0', 'Floor_6.0', 'Floor_8.0']:
        traj = trajectories[name]
        ax2.plot(traj['step'], traj['perf'],
                color=colors[name], marker=markers[name],
                markevery=4, markersize=6, linewidth=2,
                label=labels[name], alpha=0.8)

    ax2.set_xlabel('Optimization Step', fontweight='bold')
    ax2.set_ylabel('Performance', fontweight='bold')
    ax2.set_title('(b) Optimization Trajectory', fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', framealpha=0.95)
    ax2.set_xlim(-5, 205)

    # Panel C: Final results comparison (bar chart)
    ax3 = plt.subplot(1, 3, 3)

    methods = ['GreedyPerf', 'Floor_4.0', 'Floor_6.0', 'Floor_8.0']
    x = np.arange(len(methods))
    width = 0.35

    perfs = [results[m]['perf'] for m in methods]
    effs = [results[m]['efficiency'] for m in methods]

    # Normalize to GreedyPerf
    perf_norm = [p / results['GreedyPerf']['perf'] * 100 for p in perfs]
    eff_norm = [e / results['GreedyPerf']['efficiency'] * 100 for e in effs]

    bars1 = ax3.bar(x - width/2, perf_norm, width, label='Performance',
                    color='#5DA5DA', edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x + width/2, eff_norm, width, label='Efficiency',
                    color='#FAA43A', edgecolor='black', linewidth=1)

    # Add reference line at 100%
    ax3.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax3.set_ylabel('Relative to GreedyPerf (%)', fontweight='bold')
    ax3.set_title('(c) Final Results (Normalized)', fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Greedy', 'floor=4.0', 'floor=6.0', 'floor=8.0'], rotation=15, ha='right')
    ax3.legend(loc='upper left', framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax3.set_ylim(85, 110)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    return fig


def main():
    print("Running optimization experiment...")
    results, trajectories = run_experiment(steps=200)

    print("\nGenerating figure...")
    fig = create_figure(results, trajectories)

    # Save figure
    output_file = 'jam_optimization_results.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")

    # Also save as PDF for publication
    output_pdf = 'jam_optimization_results.pdf'
    fig.savefig(output_pdf, bbox_inches='tight')
    print(f"Figure saved to: {output_pdf}")

    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"{'Method':<25s} {'Perf':>8s} {'Power':>8s} {'Efficiency':>10s}")
    print("-"*80)

    for name in ['GreedyPerf', 'Floor_4.0', 'Floor_6.0', 'Floor_8.0']:
        r = results[name]
        label = {
            'GreedyPerf': 'GreedyPerf',
            'Floor_4.0': 'HybridJAM (floor=4.0)',
            'Floor_6.0': 'HybridJAM (floor=6.0)',
            'Floor_8.0': 'HybridJAM (floor=8.0)',
        }[name]
        print(f"{label:<25s} {r['perf']:8.2f} {r['power']:7.1f}W {r['efficiency']:9.2f}")

    print("\n" + "="*80)
    print("IMPROVEMENTS OVER GREEDYPERF")
    print("="*80)

    greedy_perf = results['GreedyPerf']['perf']
    greedy_eff = results['GreedyPerf']['efficiency']

    for name in ['Floor_4.0', 'Floor_6.0', 'Floor_8.0']:
        r = results[name]
        perf_improv = ((r['perf'] / greedy_perf) - 1) * 100
        eff_improv = ((r['efficiency'] / greedy_eff) - 1) * 100

        label = {
            'Floor_4.0': 'HybridJAM (floor=4.0)',
            'Floor_6.0': 'HybridJAM (floor=6.0)',
            'Floor_8.0': 'HybridJAM (floor=8.0)',
        }[name]

        print(f"{label:<25s} Perf: {perf_improv:+6.2f}%  Eff: {eff_improv:+6.2f}%")


if __name__ == "__main__":
    main()
