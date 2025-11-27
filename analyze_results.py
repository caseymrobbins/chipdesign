#!/usr/bin/env python3
"""
Analysis script for chip design optimization simulation results.

This script loads results from a JSON file and provides detailed analysis
and visualization of the simulation outcomes.
"""

import json
import sys
from typing import Dict, List
import argparse


def load_results(filename: str) -> Dict:
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' is not valid JSON", file=sys.stderr)
        sys.exit(1)


def print_summary(data: Dict):
    """Print a summary of the results"""
    params = data['parameters']
    stats = data['statistics']

    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)

    print("\nPARAMETERS:")
    print(f"  Total Runs:           {params['num_runs']}")
    print(f"  Design Steps:         {params['design_steps']}")
    print(f"  Adaptation Steps:     {params['adaptation_steps']}")
    print(f"  Checkpoint Frequency: {params['checkpoint_frequency']}")
    print(f"  Shift Type:           {params['shift_type']}")
    print(f"  Seed:                 {params['seed']}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Survival rates
    agent1_survival = stats['survival']['agent1_survival_rate']
    agent2_survival = stats['survival']['agent2_survival_rate']
    survival_advantage = agent2_survival - agent1_survival

    print(f"\nSURVIVAL RATES:")
    print(f"  GreedyPerformance:  {agent1_survival:.1%}")
    print(f"  LogMinHeadroom:     {agent2_survival:.1%}")
    print(f"  Advantage:          {survival_advantage:+.1%} (LogMinHeadroom)")

    if survival_advantage > 0.2:
        print(f"  ✓ STRONG evidence for adaptability hypothesis!")
    elif survival_advantage > 0.1:
        print(f"  ✓ MODERATE evidence for adaptability hypothesis")
    elif survival_advantage > 0:
        print(f"  ~ WEAK evidence for adaptability hypothesis")
    else:
        print(f"  ✗ NO evidence for adaptability hypothesis")

    # Win rates
    agent1_wins = stats['wins']['agent1_win_rate']
    agent2_wins = stats['wins']['agent2_win_rate']

    print(f"\nOVERALL WIN RATES:")
    print(f"  GreedyPerformance:  {agent1_wins:.1%}")
    print(f"  LogMinHeadroom:     {agent2_wins:.1%}")
    print(f"  Ties:               {stats['wins']['ties']}/{stats['total_runs']}")

    # Performance tradeoff
    agent1_perf = stats['performance_design']['agent1_mean']
    agent2_perf = stats['performance_design']['agent2_mean']
    perf_diff = (agent1_perf - agent2_perf) / agent2_perf * 100

    print(f"\nPERFORMANCE TRADEOFF (Design Phase):")
    print(f"  GreedyPerformance:  {agent1_perf:.1f}")
    print(f"  LogMinHeadroom:     {agent2_perf:.1f}")
    print(f"  Difference:         {perf_diff:+.1f}% (GreedyPerformance advantage)")

    # Headroom advantage
    agent1_headroom = stats['headroom_design']['agent1_mean']
    agent2_headroom = stats['headroom_design']['agent2_mean']

    print(f"\nHEADROOM PRESERVATION (Design Phase):")
    print(f"  GreedyPerformance:  {agent1_headroom:.2f}")
    print(f"  LogMinHeadroom:     {agent2_headroom:.2f}")

    if agent2_headroom > 0 and agent1_headroom > 0:
        headroom_ratio = agent2_headroom / agent1_headroom
        print(f"  Ratio:              {headroom_ratio:.2f}x (LogMinHeadroom advantage)")
    elif agent2_headroom > 0:
        print(f"  LogMinHeadroom maintains positive headroom!")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    # Overall interpretation
    if survival_advantage > 0.1 and agent2_wins > agent1_wins:
        print("\n✓ Hypothesis SUPPORTED: Log-Min-Headroom optimization produces")
        print("  more adaptable designs that survive requirement shifts better.")
        print(f"  Trade-off: {perf_diff:.0f}% lower initial performance for")
        print(f"  {survival_advantage:.0%} better survival rate.")

    elif survival_advantage > 0:
        print("\n~ Hypothesis PARTIALLY SUPPORTED: Log-Min-Headroom shows some")
        print("  adaptability advantages but the effect is modest.")

    else:
        print("\n✗ Hypothesis NOT SUPPORTED in this experiment.")
        print("  Consider: More runs, longer design phases, or harsher shifts.")

    print("\n" + "="*80)


def analyze_by_shift_type(data: Dict):
    """Analyze results broken down by shift type"""
    stats = data['statistics']
    shift_breakdown = stats['shift_type_breakdown']

    print("\n" + "="*80)
    print("ANALYSIS BY SHIFT TYPE")
    print("="*80)

    for shift_type, shift_data in shift_breakdown.items():
        print(f"\n{shift_type.upper().replace('_', ' ')}:")
        print(f"  Occurrences: {shift_data['count']}")

        agent1_survived = shift_data['agent1_survived']
        agent2_survived = shift_data['agent2_survived']
        count = shift_data['count']

        agent1_rate = agent1_survived / count if count > 0 else 0
        agent2_rate = agent2_survived / count if count > 0 else 0

        print(f"  GreedyPerformance survival: {agent1_survived}/{count} ({agent1_rate:.1%})")
        print(f"  LogMinHeadroom survival:    {agent2_survived}/{count} ({agent2_rate:.1%})")

        if agent2_rate > agent1_rate:
            advantage = agent2_rate - agent1_rate
            print(f"  → LogMinHeadroom advantage: {advantage:+.1%}")
        elif agent1_rate > agent2_rate:
            advantage = agent1_rate - agent2_rate
            print(f"  → GreedyPerformance advantage: {advantage:+.1%}")
        else:
            print(f"  → No difference")


def analyze_individual_runs(data: Dict, show_failures_only: bool = False):
    """Analyze individual runs"""
    results = data['results']

    print("\n" + "="*80)
    if show_failures_only:
        print("FAILURE CASES")
    else:
        print("INDIVIDUAL RUN DETAILS")
    print("="*80)

    failure_count = 0

    for result in results:
        agent1_survived = result['agent1_survived_shift']
        agent2_survived = result['agent2_survived_shift']

        # Skip if showing failures only and both survived
        if show_failures_only and agent1_survived and agent2_survived:
            continue

        failure_count += 1

        print(f"\nRun {result['run_id']} (seed={result['seed']}):")
        print(f"  Shift: {result['shift_info']['description']}")
        print(f"  GreedyPerformance: {'SURVIVED' if agent1_survived else 'FAILED'}")
        print(f"    Design: perf={result['agent1_final_performance_design']:.1f}, headroom={result['agent1_min_headroom_design']:.2f}")
        print(f"    Adapt:  perf={result['agent1_final_performance_adapt']:.1f}, headroom={result['agent1_min_headroom_adapt']:.2f}")
        print(f"  LogMinHeadroom: {'SURVIVED' if agent2_survived else 'FAILED'}")
        print(f"    Design: perf={result['agent2_final_performance_design']:.1f}, headroom={result['agent2_min_headroom_design']:.2f}")
        print(f"    Adapt:  perf={result['agent2_final_performance_adapt']:.1f}, headroom={result['agent2_min_headroom_adapt']:.2f}")
        print(f"  Winner: {result['winner']}")

    if show_failures_only and failure_count == 0:
        print("\nNo failures detected (both agents survived all shifts)")


def export_csv(data: Dict, output_file: str):
    """Export results to CSV format"""
    results = data['results']

    try:
        with open(output_file, 'w') as f:
            # Header
            f.write("run_id,seed,shift_type,shift_description,")
            f.write("agent1_survived,agent1_perf_design,agent1_perf_adapt,agent1_headroom_design,agent1_headroom_adapt,")
            f.write("agent2_survived,agent2_perf_design,agent2_perf_adapt,agent2_headroom_design,agent2_headroom_adapt,")
            f.write("winner\n")

            # Data rows
            for r in results:
                f.write(f"{r['run_id']},{r['seed']},{r['shift_info']['type']},\"{r['shift_info']['description']}\",")
                f.write(f"{r['agent1_survived_shift']},{r['agent1_final_performance_design']:.2f},{r['agent1_final_performance_adapt']:.2f},")
                f.write(f"{r['agent1_min_headroom_design']:.2f},{r['agent1_min_headroom_adapt']:.2f},")
                f.write(f"{r['agent2_survived_shift']},{r['agent2_final_performance_design']:.2f},{r['agent2_final_performance_adapt']:.2f},")
                f.write(f"{r['agent2_min_headroom_design']:.2f},{r['agent2_min_headroom_adapt']:.2f},")
                f.write(f"\"{r['winner']}\"\n")

        print(f"\nResults exported to: {output_file}")

    except IOError as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze chip design optimization simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Input JSON file with simulation results',
    )

    parser.add_argument(
        '--failures',
        action='store_true',
        help='Show only failure cases in individual run details',
    )

    parser.add_argument(
        '--csv',
        type=str,
        metavar='OUTPUT',
        help='Export results to CSV file',
    )

    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary output',
    )

    parser.add_argument(
        '--no-shift-analysis',
        action='store_true',
        help='Skip shift type analysis',
    )

    parser.add_argument(
        '--show-runs',
        action='store_true',
        help='Show individual run details',
    )

    args = parser.parse_args()

    # Load results
    data = load_results(args.input_file)

    # Generate outputs
    if not args.no_summary:
        print_summary(data)

    if not args.no_shift_analysis:
        analyze_by_shift_type(data)

    if args.show_runs or args.failures:
        analyze_individual_runs(data, show_failures_only=args.failures)

    if args.csv:
        export_csv(data, args.csv)


if __name__ == "__main__":
    main()
