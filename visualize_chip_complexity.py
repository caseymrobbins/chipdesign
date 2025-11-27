#!/usr/bin/env python3
"""
Visualize chip complexity and demonstrate that JAM is a GLASS BOX strategy.

This script:
1. Runs a single simulation with both agents
2. Extracts design parameters and constraints over time
3. Creates visualizations showing:
   - What each agent is optimizing (their objective functions)
   - How chip parameters evolve
   - The decision-making transparency of JAM vs GreedyPerformance
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from advanced_chip_simulator import (
    AdvancedSimulation,
    ProcessTechnology,
    ShiftType,
)


def extract_timeseries(checkpoints, agent_name):
    """Extract time series data for an agent"""
    agent_data = [cp for cp in checkpoints if cp['agent_name'] == agent_name]

    data = {
        'steps': [],
        'phase': [],
        'performance': [],
        'min_headroom': [],
        'frequency': [],
        'voltage': [],
        'cores': [],
        'power': [],
        'temperature': [],
        'timing_slack': [],
        'yield': [],
        'area': [],
        'headrooms': {},
        'action': [],
    }

    for cp in agent_data:
        data['steps'].append(cp['step'])
        data['phase'].append(cp['phase'])
        data['performance'].append(cp['performance'])
        data['min_headroom'].append(cp['min_headroom'])
        data['frequency'].append(cp['parameters']['clock_freq_ghz'])
        data['voltage'].append(cp['parameters']['supply_voltage'])
        data['cores'].append(cp['parameters']['num_cores'])
        data['power'].append(cp['constraints']['total_power_w'])
        data['temperature'].append(cp['constraints']['temperature_c'])
        data['timing_slack'].append(cp['constraints']['timing_slack_ps'])
        data['yield'].append(cp['constraints']['yield'])
        data['area'].append(cp['constraints']['area_mm2'])
        data['action'].append(cp.get('action_taken', ''))

        # Collect all headrooms
        for hr_name, hr_val in cp['headrooms'].items():
            if hr_name not in data['headrooms']:
                data['headrooms'][hr_name] = []
            data['headrooms'][hr_name].append(hr_val)

    return data


def create_glass_box_visualization(greedy_data, jam_data, result, output_file='chip_complexity.png'):
    """
    Create comprehensive visualization showing JAM as a glass box.

    This visualization demonstrates:
    1. What each agent optimizes (objective function)
    2. How chip parameters evolve
    3. The interpretability of JAM's decisions
    """

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Color scheme
    greedy_color = '#E74C3C'  # Red
    jam_color = '#3498DB'      # Blue
    shift_color = '#95A5A6'    # Gray

    design_steps = result['design_steps']
    adaptation_steps = result['adaptation_steps']
    shift_step = design_steps

    # ========================================================================
    # ROW 1: OBJECTIVE FUNCTIONS (Glass Box Demonstration)
    # ========================================================================

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('GREEDY: Maximizes Performance\n(Simple objective)', fontsize=12, fontweight='bold')
    ax1.plot(greedy_data['steps'], greedy_data['performance'], color=greedy_color, linewidth=2, label='Performance')
    ax1.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7, label='Requirement Shift')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Performance Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.text(0.5, 0.95, 'Black Box: Just chase performance',
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('JAM: Maximizes log(min(headroom))\n(Glass Box: Interpretable)', fontsize=12, fontweight='bold')
    ax2.plot(jam_data['steps'], jam_data['min_headroom'], color=jam_color, linewidth=2, label='Min Headroom (margin)')
    ax2.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7, label='Requirement Shift')
    ax2.axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Minimum Headroom')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.text(0.5, 0.95, 'Glass Box: Maintain safety margins',
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax3.plot(greedy_data['steps'], greedy_data['performance'], color=greedy_color, linewidth=3,
             label='Greedy (red)', linestyle='-', marker='o', markevery=max(1, len(greedy_data['steps'])//10), markersize=5)
    ax3.plot(jam_data['steps'], jam_data['performance'], color=jam_color, linewidth=3,
             label='JAM (blue)', linestyle='--', marker='s', markevery=max(1, len(jam_data['steps'])//10), markersize=5)
    ax3.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7, label='Shift')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Performance', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # ========================================================================
    # ROW 2: CHIP PARAMETERS (Design Knobs)
    # ========================================================================

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('Clock Frequency Evolution', fontsize=11, fontweight='bold')
    ax4.plot(greedy_data['steps'], greedy_data['frequency'], color=greedy_color, linewidth=3, label='Greedy (red)', linestyle='-')
    ax4.plot(jam_data['steps'], jam_data['frequency'], color=jam_color, linewidth=3, label='JAM (blue)', linestyle='--')
    ax4.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Step', fontsize=10)
    ax4.set_ylabel('Frequency (GHz)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('Supply Voltage Evolution', fontsize=11, fontweight='bold')
    ax5.plot(greedy_data['steps'], greedy_data['voltage'], color=greedy_color, linewidth=3, label='Greedy (red)', linestyle='-')
    ax5.plot(jam_data['steps'], jam_data['voltage'], color=jam_color, linewidth=3, label='JAM (blue)', linestyle='--')
    ax5.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Step', fontsize=10)
    ax5.set_ylabel('Voltage (V)', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('Number of Cores', fontsize=11, fontweight='bold')
    ax6.plot(greedy_data['steps'], greedy_data['cores'], color=greedy_color, linewidth=3, label='Greedy (red)',
             linestyle='-', marker='o', markevery=max(1, len(greedy_data['steps'])//15), markersize=4)
    ax6.plot(jam_data['steps'], jam_data['cores'], color=jam_color, linewidth=3, label='JAM (blue)',
             linestyle='--', marker='s', markevery=max(1, len(jam_data['steps'])//15), markersize=4)
    ax6.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7)
    ax6.set_xlabel('Step', fontsize=10)
    ax6.set_ylabel('Core Count', fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)

    # ========================================================================
    # ROW 3: DERIVED CONSTRAINTS (Physics)
    # ========================================================================

    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_title('Power Consumption (Physics)', fontsize=11, fontweight='bold')
    ax7.plot(greedy_data['steps'], greedy_data['power'], color=greedy_color, linewidth=3, label='Greedy (red)', linestyle='-')
    ax7.plot(jam_data['steps'], jam_data['power'], color=jam_color, linewidth=3, label='JAM (blue)', linestyle='--')
    ax7.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7)
    ax7.axhline(150, color='darkred', linestyle=':', linewidth=2, alpha=0.7, label='TDP Limit')
    ax7.set_xlabel('Step', fontsize=10)
    ax7.set_ylabel('Power (W)', fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=9)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_title('Temperature (Thermal Physics)', fontsize=11, fontweight='bold')
    ax8.plot(greedy_data['steps'], greedy_data['temperature'], color=greedy_color, linewidth=3, label='Greedy (red)', linestyle='-')
    ax8.plot(jam_data['steps'], jam_data['temperature'], color=jam_color, linewidth=3, label='JAM (blue)', linestyle='--')
    ax8.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7)
    ax8.axhline(95, color='darkred', linestyle=':', linewidth=2, alpha=0.7, label='Thermal Limit')
    ax8.set_xlabel('Step', fontsize=10)
    ax8.set_ylabel('Temperature (°C)', fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=9)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_title('Timing Slack (Can we close timing?)', fontsize=11, fontweight='bold')
    ax9.plot(greedy_data['steps'], greedy_data['timing_slack'], color=greedy_color, linewidth=3, label='Greedy (red)', linestyle='-')
    ax9.plot(jam_data['steps'], jam_data['timing_slack'], color=jam_color, linewidth=3, label='JAM (blue)', linestyle='--')
    ax9.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7)
    ax9.axhline(50, color='darkred', linestyle=':', linewidth=2, alpha=0.7, label='Min Slack')
    ax9.axhline(0, color='darkred', linestyle='-', linewidth=2, alpha=0.9, label='Violation')
    ax9.set_xlabel('Step', fontsize=10)
    ax9.set_ylabel('Timing Slack (ps)', fontsize=10)
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=9)

    # ========================================================================
    # ROW 4: HEADROOM ANALYSIS (JAM's Glass Box Property)
    # ========================================================================

    ax10 = fig.add_subplot(gs[3, :2])
    ax10.set_title('JAM GLASS BOX: All Constraint Headrooms (Interpretable!)', fontsize=12, fontweight='bold')

    # Define unique colors for each constraint
    constraint_colors = {
        'power': '#E74C3C',           # Red
        'area': '#3498DB',            # Blue
        'temperature': '#F39C12',     # Orange
        'frequency': '#2ECC71',       # Green
        'timing_slack': '#9B59B6',    # Purple
        'ir_drop': '#1ABC9C',         # Turquoise
        'yield': '#E67E22',           # Dark Orange
        'signal_integrity': '#34495E', # Dark Gray
        'power_density': '#E91E63',   # Pink
        'wire_delay': '#00BCD4',      # Cyan
    }

    # Plot all headrooms for JAM with distinct colors
    for hr_name, hr_values in jam_data['headrooms'].items():
        color = constraint_colors.get(hr_name, None)
        ax10.plot(jam_data['steps'], hr_values, linewidth=2, label=hr_name,
                 color=color, alpha=0.8, linestyle='-')

    # Highlight the minimum with thick black dashed line
    ax10.plot(jam_data['steps'], jam_data['min_headroom'],
              color='black', linewidth=4, label='MIN (JAM optimizes this)', linestyle='--', zorder=10)
    ax10.axvline(shift_step, color=shift_color, linestyle='--', linewidth=2, alpha=0.7, label='Requirement Shift')
    ax10.axhline(0, color='darkred', linestyle='-', linewidth=2, alpha=0.9, label='Violation Threshold', zorder=5)
    ax10.set_xlabel('Step', fontsize=11)
    ax10.set_ylabel('Headroom (positive = safe)', fontsize=11)
    ax10.grid(True, alpha=0.3)
    ax10.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.9)
    ax10.text(0.02, 0.98, 'You can SEE what JAM is doing:\nIt keeps all margins positive\nand maximizes the weakest one',
              transform=ax10.transAxes, ha='left', va='top', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax11 = fig.add_subplot(gs[3, 2])
    ax11.set_title('Final Outcome Comparison', fontsize=11, fontweight='bold')

    outcomes = {
        'Greedy\nSurvived': result['agent1_survived_shift'],
        'JAM\nSurvived': result['agent2_survived_shift'],
    }

    colors_outcome = [greedy_color if not result['agent1_survived_shift'] else 'green',
                     jam_color if not result['agent2_survived_shift'] else 'green']

    bars = ax11.bar(outcomes.keys(), [int(v) for v in outcomes.values()], color=colors_outcome, alpha=0.7)
    ax11.set_ylabel('Survived Shift? (1=Yes, 0=No)')
    ax11.set_ylim([0, 1.2])
    ax11.grid(True, alpha=0.3, axis='y')

    for i, (bar, survived) in enumerate(zip(bars, outcomes.values())):
        height = bar.get_height()
        label = 'SURVIVED ✓' if survived else 'FAILED ✗'
        ax11.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Overall title
    fig.suptitle(f'Chip Design Complexity: Greedy vs JAM (Glass Box)\n' +
                 f'Shift: {result["shift_info"]["description"]}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    print(f"\nKEY INSIGHT: JAM is a GLASS BOX strategy")
    print(f"  - You can see what it optimizes: log(min(headroom))")
    print(f"  - You can see why: maintaining safety margins")
    print(f"  - You can interpret every decision: which margin is limiting?")
    print(f"  - Contrast with Greedy: just chase performance (less interpretable)")

    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize chip complexity for Greedy vs JAM')
    parser.add_argument('--design-steps', type=int, default=50, help='Design phase steps (default: 50, realistic for reaching optimum)')
    parser.add_argument('--adapt-steps', type=int, default=30, help='Adaptation phase steps (default: 30 for more detail)')
    parser.add_argument('--output', type=str, default='chip_complexity.png', help='Output image file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shift-type', type=str, default='tighten_power',
                       choices=['tighten_power', 'increase_performance', 'reduce_area',
                               'tighten_thermal', 'yield_requirement', 'add_feature'])

    args = parser.parse_args()

    print("="*80)
    print("GENERATING CHIP COMPLEXITY VISUALIZATION")
    print("="*80)
    print(f"Design steps: {args.design_steps}")
    print(f"Adaptation steps: {args.adapt_steps}")
    print(f"Shift type: {args.shift_type}")
    print(f"Random seed: {args.seed}")
    print("="*80)

    # Run simulation
    # Use more frequent checkpoints for smoother graphs (every 2 steps)
    checkpoint_freq = 2
    sim = AdvancedSimulation(
        design_steps=args.design_steps,
        adaptation_steps=args.adapt_steps,
        checkpoint_frequency=checkpoint_freq,
        shift_type=ShiftType(args.shift_type),
        process=ProcessTechnology.create_7nm(),
        seed=args.seed,
        verbose=True,
    )

    print("\nRunning simulation...")
    result = sim.run_single_simulation(run_id=0)

    # Convert to dict for easier access
    result_dict = result.to_dict()

    # Extract time series
    print("\nExtracting time series data...")
    greedy_data = extract_timeseries(result_dict['checkpoints'], 'GreedyPerformance')
    jam_data = extract_timeseries(result_dict['checkpoints'], 'JAM')

    # Create visualization
    print("\nCreating visualization...")
    fig = create_glass_box_visualization(greedy_data, jam_data, result_dict, args.output)

    print("\n" + "="*80)
    print("JAM INTERPRETABILITY ANALYSIS")
    print("="*80)
    print("\nGLASS BOX PROPERTIES OF JAM:")
    print("1. ✓ Clear objective: maximize log(min(headroom))")
    print("2. ✓ Interpretable decisions: always choose action that improves worst margin")
    print("3. ✓ Transparent reasoning: can trace why each action was selected")
    print("4. ✓ Predictable behavior: will avoid actions that crush any single margin")
    print("5. ✓ Explainable outcomes: designs have balanced margins by construction")
    print("\nVS. GREEDY (MORE BLACK BOX):")
    print("1. ✗ Simpler objective but less interpretable outcomes")
    print("2. ✗ Can't predict which margins will be sacrificed")
    print("3. ✗ Less insight into why designs fail after requirement shifts")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
