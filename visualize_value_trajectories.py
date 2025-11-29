#!/usr/bin/env python3
"""
Visualize value trajectories to show over-optimization problem
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Import from the main experiment
sys.path.insert(0, '/home/user/chipdesign')
from jam_lookahead_experiment import *

def plot_value_trajectories():
    """Plot value trajectories for each strategy on adversarial topology"""

    # Create adversarial topology
    env = AdversarialTopology(size=12)

    # Run each strategy
    strategies = [
        JAMGreedy(),
        JAMLookahead(2),
        JAMLookahead(5),
        GoalSeeker()
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        result = strategy.run_episode(env, max_steps=200)

        ax = axes[idx]

        # Plot min value over time
        steps = list(range(len(result.min_value_history)))
        ax.plot(steps, result.min_value_history, 'b-', linewidth=2, label='min(values)')

        # Mark goal if reached
        if result.reached_goal:
            ax.axvline(result.steps, color='green', linestyle='--', linewidth=2,
                      label=f'Goal reached (step {result.steps})')
            ax.set_facecolor('#f0fff0')  # Light green background
        else:
            ax.axvline(result.steps, color='red', linestyle='--', linewidth=2,
                      label=f'Stopped (step {result.steps})')
            ax.set_facecolor('#fff5f5')  # Light red background

        # Styling
        title_prefix = "✓" if result.reached_goal else "✗"
        ax.set_title(f'{title_prefix} {strategy.name}\n'
                    f'Nodes: {result.nodes_evaluated:,} | Final min: {result.min_value_history[-1]:.2f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('min(values) = log reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add annotation about behavior
        if not result.reached_goal and result.survived:
            # Over-optimization case
            ax.text(0.5, 0.95, 'OVER-OPTIMIZING VALUES\n(trapped in local maximum)',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   fontsize=10, fontweight='bold')

    plt.suptitle('Value Trajectories on Adversarial Topology:\nWhy Lookahead Fails',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/jam_value_trajectories.png', dpi=150, bbox_inches='tight')
    print("✓ Value trajectory plot saved to /mnt/user-data/outputs/jam_value_trajectories.png")

    # Create comparison of efficiency vs success
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Data
    strategy_names = ['JAM-Greedy', 'JAM-2step', 'JAM-5step', 'Goal-Seeker']
    nodes_evaluated = [705, 5169, 182255, 1750]
    success_rates = [100, 25, 0, 100]
    colors = ['green', 'orange', 'red', 'blue']

    # Scatter plot
    for i, (name, nodes, success, color) in enumerate(zip(strategy_names, nodes_evaluated, success_rates, colors)):
        marker = 'o' if success == 100 else 'x'
        size = 300 if success == 100 else 200
        ax.scatter(nodes, success, s=size, c=color, marker=marker, alpha=0.7,
                  edgecolors='black', linewidth=2, label=name)

        # Annotate
        offset_x = 1.15 if name == 'JAM-Greedy' else 1.05
        ax.text(nodes * offset_x, success + 3, name, fontsize=12, fontweight='bold')

    # Highlight JAM-Greedy
    ax.scatter([705], [100], s=500, facecolors='none', edgecolors='gold',
              linewidth=4, linestyle='--', label='OPTIMAL')

    ax.set_xscale('log')
    ax.set_xlabel('Computational Cost (Nodes Evaluated, log scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Efficiency vs Success: JAM-Greedy Dominates\n'
                'Lower-left = Good | Upper-right = Bad',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 110)

    # Add quadrant labels
    ax.text(10000, 85, 'EXPENSIVE\n& FAILS', ha='center', va='center',
           fontsize=14, alpha=0.3, fontweight='bold', color='red')
    ax.text(1000, 85, 'CHEAP\n& WORKS', ha='center', va='center',
           fontsize=14, alpha=0.3, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/jam_efficiency_vs_success.png', dpi=150, bbox_inches='tight')
    print("✓ Efficiency comparison saved to /mnt/user-data/outputs/jam_efficiency_vs_success.png")

if __name__ == "__main__":
    print("Creating value trajectory visualizations...")
    plot_value_trajectories()
    print("\n✓ All visualizations complete!")
