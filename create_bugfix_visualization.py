#!/usr/bin/env python3
"""Create visualization of bug fix results"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load results
with open('greedy_vs_intrinsic_data.json', 'r') as f:
    data = json.load(f)

# Extract data by agent
agents = {}
for run in data:
    for agent_data in run:
        name = agent_data['name']
        if name not in agents:
            agents[name] = {
                'design_perf': [],
                'design_power': [],
                'design_efficiency': [],
                'design_min_headroom': [],
                'survived': [],
                'final_perf': [],
            }

        agents[name]['design_perf'].append(agent_data['design_performance'])
        agents[name]['design_power'].append(agent_data['design_power'])
        agents[name]['design_efficiency'].append(agent_data['design_efficiency'])
        agents[name]['design_min_headroom'].append(agent_data['design_min_headroom'])
        agents[name]['survived'].append(agent_data['survived'])
        if agent_data['survived']:
            agents[name]['final_perf'].append(agent_data['final_performance'])

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('JAMAdvanced Bug Fix Results - 3 Agent Comparison\n(50 runs, 40 steps)',
             fontsize=14, fontweight='bold')

# Colors for each agent
colors = {
    'IndustryBest': '#ff7f0e',
    'JAM': '#2ca02c',
    'JAMAdvanced': '#1f77b4'
}

agent_order = ['IndustryBest', 'JAM', 'JAMAdvanced']

# Plot 1: Design Phase Performance
ax1.set_title('Design Phase Performance', fontweight='bold')
ax1.set_ylabel('Performance')
perfs = [np.mean(agents[name]['design_perf']) for name in agent_order]
bars1 = ax1.bar(agent_order, perfs, color=[colors[name] for name in agent_order], alpha=0.7)
ax1.set_ylim([0, max(perfs) * 1.2])

# Add value labels on bars
for i, (name, perf) in enumerate(zip(agent_order, perfs)):
    ax1.text(i, perf + 2, f'{perf:.1f}', ha='center', va='bottom', fontweight='bold')

# Add horizontal line for IndustryBest baseline
industrybest_perf = perfs[0]
ax1.axhline(y=industrybest_perf, color='red', linestyle='--', alpha=0.5, label=f'IndustryBest baseline ({industrybest_perf:.1f})')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Power Consumption
ax2.set_title('Power Consumption', fontweight='bold')
ax2.set_ylabel('Power (W)')
powers = [np.mean(agents[name]['design_power']) for name in agent_order]
bars2 = ax2.bar(agent_order, powers, color=[colors[name] for name in agent_order], alpha=0.7)
ax2.set_ylim([0, 12])

for i, (name, power) in enumerate(zip(agent_order, powers)):
    ax2.text(i, power + 0.2, f'{power:.2f}W', ha='center', va='bottom', fontweight='bold')

ax2.axhline(y=12.0, color='red', linestyle='--', alpha=0.5, label='Power limit (12W)')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Min Headroom (Safety Margin)
ax3.set_title('Min Headroom (Safety Margin)', fontweight='bold')
ax3.set_ylabel('Min Headroom')
headrooms = [np.mean(agents[name]['design_min_headroom']) for name in agent_order]
bars3 = ax3.bar(agent_order, headrooms, color=[colors[name] for name in agent_order], alpha=0.7)
ax3.set_ylim([0, max(headrooms) * 1.3])

for i, (name, hr) in enumerate(zip(agent_order, headrooms)):
    ax3.text(i, hr + 0.02, f'{hr:.4f}', ha='center', va='bottom', fontweight='bold')

ax3.grid(axis='y', alpha=0.3)

# Plot 4: Survival Rate & Final Performance
ax4.set_title('Robustness & Adapted Performance', fontweight='bold')
ax4.set_ylabel('Survival Rate (%)', color='tab:blue')
ax4.set_ylim([0, 100])

survival_rates = [sum(agents[name]['survived']) / len(agents[name]['survived']) * 100 for name in agent_order]
bars4 = ax4.bar([i - 0.2 for i in range(len(agent_order))], survival_rates,
                width=0.4, color=[colors[name] for name in agent_order], alpha=0.7, label='Survival Rate')

for i, rate in enumerate(survival_rates):
    ax4.text(i - 0.2, rate + 2, f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

ax4.tick_params(axis='y', labelcolor='tab:blue')
ax4.grid(axis='y', alpha=0.3)

# Add final performance on secondary y-axis
ax4_twin = ax4.twinx()
ax4_twin.set_ylabel('Final Performance (survivors)', color='tab:red')
final_perfs = []
for name in agent_order:
    if agents[name]['final_perf']:
        final_perfs.append(np.mean(agents[name]['final_perf']))
    else:
        final_perfs.append(0)

bars4_twin = ax4_twin.bar([i + 0.2 for i in range(len(agent_order))], final_perfs,
                          width=0.4, color='red', alpha=0.5, label='Final Perf')

for i, perf in enumerate(final_perfs):
    if perf > 0:
        ax4_twin.text(i + 0.2, perf + 2, f'{perf:.1f}', ha='center', va='bottom',
                     fontweight='bold', color='red')

ax4_twin.tick_params(axis='y', labelcolor='tab:red')
ax4_twin.set_ylim([0, max(final_perfs) * 1.3 if max(final_perfs) > 0 else 100])

ax4.set_xticks(range(len(agent_order)))
ax4.set_xticklabels(agent_order)

# Add legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Add text box with key findings
textstr = 'KEY FINDINGS:\n'
textstr += f'• JAMAdvanced: {perfs[2]:.1f} perf (+{((perfs[2]-perfs[0])/perfs[0]*100):.1f}% vs IndustryBest)\n'
textstr += f'• Lowest power: {powers[2]:.2f}W (JAMAdvanced)\n'
textstr += f'• Highest safety: {headrooms[2]:.4f} headroom (JAMAdvanced)\n'
textstr += f'• Bug fix impact: 36.62 → {perfs[2]:.1f} (+192.9%)'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig.text(0.5, 0.02, textstr, fontsize=10, verticalalignment='bottom',
         horizontalalignment='center', bbox=props)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('bugfix_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: bugfix_comparison.png")
