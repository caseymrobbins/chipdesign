#!/usr/bin/env python3
"""Create mobile-friendly comprehensive results visualization"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Load performance data
with open('greedy_vs_intrinsic_data.json', 'r') as f:
    data = json.load(f)

# Extract agent data
agents = {}
for run in data:
    for agent_data in run:
        name = agent_data['name']
        if name not in agents:
            agents[name] = {
                'design_perf': [],
                'design_power': [],
                'design_min_headroom': [],
            }
        agents[name]['design_perf'].append(agent_data['design_performance'])
        agents[name]['design_power'].append(agent_data['design_power'])
        agents[name]['design_min_headroom'].append(agent_data['design_min_headroom'])

# Robustness data (from graduated stress test)
robustness = {
    'IndustryBest': {'power': 5, 'performance': 45, 'area': 0, 'thermal': 50},
    'JAM': {'power': 5, 'performance': 40, 'area': 0, 'thermal': 50},
    'JAMAdvanced': {'power': 10, 'performance': 35, 'area': 0, 'thermal': 50},
}

# Create mobile-friendly figure (portrait orientation)
fig = plt.figure(figsize=(8, 14))
gs = GridSpec(6, 1, height_ratios=[1.2, 1, 1, 1, 1, 2.5], hspace=0.4)

# Colors
colors = {
    'IndustryBest': '#ff7f0e',
    'JAM': '#2ca02c',
    'JAMAdvanced': '#1f77b4'
}
agent_order = ['IndustryBest', 'JAM', 'JAMAdvanced']

# Title
fig.suptitle('JAMAdvanced Bug Fix - Complete Results\n(50 runs, 40 design steps)',
             fontsize=16, fontweight='bold', y=0.98)

# ============================================================================
# PLOT 1: Performance Comparison
# ============================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_title('Design Performance', fontweight='bold', fontsize=12)
ax1.set_ylabel('Performance', fontsize=10)

perfs = [np.mean(agents[name]['design_perf']) for name in agent_order]
bars = ax1.bar(agent_order, perfs, color=[colors[name] for name in agent_order], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (name, perf) in enumerate(zip(agent_order, perfs)):
    ax1.text(i, perf + 2, f'{perf:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Baseline and improvement markers
baseline = perfs[0]
ax1.axhline(y=baseline, color='red', linestyle='--', alpha=0.4, linewidth=1)
ax1.text(2.5, baseline, f'Baseline\n{baseline:.1f}', ha='right', va='center', fontsize=8, color='red')

# Improvement annotation
improvement = ((perfs[2] - baseline) / baseline) * 100
ax1.annotate(f'+{improvement:.1f}%', xy=(2, perfs[2]), xytext=(2, perfs[2] + 8),
            ha='center', fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

ax1.set_ylim([0, max(perfs) * 1.15])
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticklabels(agent_order, fontsize=9)

# ============================================================================
# PLOT 2: Power Consumption
# ============================================================================
ax2 = fig.add_subplot(gs[1])
ax2.set_title('Power Consumption', fontweight='bold', fontsize=12)
ax2.set_ylabel('Power (W)', fontsize=10)

powers = [np.mean(agents[name]['design_power']) for name in agent_order]
bars = ax2.bar(agent_order, powers, color=[colors[name] for name in agent_order], alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (name, power) in enumerate(zip(agent_order, powers)):
    ax2.text(i, power + 0.2, f'{power:.2f}W', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax2.axhline(y=12.0, color='red', linestyle='--', alpha=0.4, label='Limit: 12W')
ax2.set_ylim([0, 12.5])
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticklabels(agent_order, fontsize=9)

# ============================================================================
# PLOT 3: Min Headroom (Safety Margin)
# ============================================================================
ax3 = fig.add_subplot(gs[2])
ax3.set_title('Safety Margin (Min Headroom)', fontweight='bold', fontsize=12)
ax3.set_ylabel('Headroom', fontsize=10)

headrooms = [np.mean(agents[name]['design_min_headroom']) for name in agent_order]
bars = ax3.bar(agent_order, headrooms, color=[colors[name] for name in agent_order], alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (name, hr) in enumerate(zip(agent_order, headrooms)):
    ax3.text(i, hr + 0.03, f'{hr:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax3.set_ylim([0, max(headrooms) * 1.2])
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticklabels(agent_order, fontsize=9)

# ============================================================================
# PLOT 4: Power Tolerance
# ============================================================================
ax4 = fig.add_subplot(gs[3])
ax4.set_title('Power Budget Tolerance', fontweight='bold', fontsize=12)
ax4.set_ylabel('Max Stress Survived (%)', fontsize=10)

power_tol = [robustness[name]['power'] for name in agent_order]
bars = ax4.bar(agent_order, power_tol, color=[colors[name] for name in agent_order], alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (name, tol) in enumerate(zip(agent_order, power_tol)):
    ax4.text(i, tol + 0.5, f'{tol}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax4.set_ylim([0, 15])
ax4.grid(axis='y', alpha=0.3)
ax4.set_xticklabels(agent_order, fontsize=9)

# ============================================================================
# PLOT 5: Performance Tolerance
# ============================================================================
ax5 = fig.add_subplot(gs[4])
ax5.set_title('Performance Requirement Tolerance', fontweight='bold', fontsize=12)
ax5.set_ylabel('Max Stress Survived (%)', fontsize=10)

perf_tol = [robustness[name]['performance'] for name in agent_order]
bars = ax5.bar(agent_order, perf_tol, color=[colors[name] for name in agent_order], alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (name, tol) in enumerate(zip(agent_order, perf_tol)):
    ax5.text(i, tol + 1, f'{tol}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax5.set_ylim([0, 55])
ax5.grid(axis='y', alpha=0.3)
ax5.set_xticklabels(agent_order, fontsize=9)

# ============================================================================
# ANALYSIS TEXT BOX
# ============================================================================
ax6 = fig.add_subplot(gs[5])
ax6.axis('off')

analysis_text = f"""ANALYSIS & KEY FINDINGS

BUG FIX IMPACT:
• Critical bug: select_action() using wrong design_space
• Before: 36.62 performance (stuck/paralyzed)
• After: 107.25 performance (+192.9% improvement!)

PERFORMANCE COMPARISON:
✓ JAMAdvanced beats IndustryBest by +14.2%
  (107.25 vs 93.90 performance)
• JAM slightly ahead at 109.06 (+1.7% over JAMAdvanced)

POWER EFFICIENCY:
✓ JAMAdvanced uses LOWEST power: 10.49W
• JAM uses most power: 11.37W (+8.4% vs JAMAdvanced)
• Lower power = better for mobile/battery applications

SAFETY MARGINS:
✓ JAMAdvanced has HIGHEST headroom: 0.748
• 77% more headroom than IndustryBest (0.422)
• More conservative = safer under uncertainty

ROBUSTNESS TRADE-OFFS:
✓ Power cuts: JAMAdvanced survives 10% (2× better)
  IndustryBest & JAM fail at 10% cuts
✗ Perf increases: IndustryBest survives 45% (best)
  JAMAdvanced only handles 35%

VERDICT:
JAMAdvanced achieves the goal: higher performance than
IndustryBest while maintaining better power efficiency and
safety margins. The Boltzmann softmin provides smooth
optimization with natural constraint handling.

Trade-off: Power-efficient conservative design vs
performance-maximizing aggressive design."""

ax6.text(0.05, 0.95, analysis_text, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))

plt.savefig('mobile_complete_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Mobile-friendly visualization saved to: mobile_complete_results.png")
