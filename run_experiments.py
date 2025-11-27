#!/usr/bin/env python3
"""
Command-line interface for running chip design optimization experiments.

This script provides a convenient interface for running simulations comparing
two optimization strategies: Greedy Performance Maximization vs Log-Min-Headroom Optimization.
"""

import argparse
import sys
from chip_design_simulator import (
    run_multiple_simulations,
    ShiftType,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run chip design optimization simulations comparing two strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Quick test run (5 simulations, verbose output)
  python run_experiments.py --runs 5 --design-steps 20 --adapt-steps 10 --verbose

  # Standard experiment (100 runs, default parameters)
  python run_experiments.py --runs 100 --output results.json

  # Test specific shift type
  python run_experiments.py --runs 50 --shift-type tighten_constraint

  # Long design phase with frequent checkpoints
  python run_experiments.py --runs 100 --design-steps 150 --checkpoint-freq 5

  # Reproducible run with fixed seed
  python run_experiments.py --runs 100 --seed 42

METRICS EXPLAINED:
  Performance: Throughput/clock speed metric (higher is better)
  Headroom: Distance from constraint floor (higher = more margin)
  Min Headroom: Minimum headroom across all constraints (bottleneck metric)
  Survival Rate: Percentage of designs that remain feasible after requirement shift
  Win Rate: Percentage of runs where agent achieves better overall outcome

SHIFT TYPES:
  tighten_constraint: Tightens one random constraint floor by 20%
  performance_increase: Increases performance requirement by 15%
  add_constraint: Adds a new constraint dimension
  (omit --shift-type to randomly select shift type each run)
        """
    )

    # Simulation parameters
    parser.add_argument(
        '--runs',
        type=int,
        default=100,
        help='Number of simulation runs to perform (default: 100)',
    )

    parser.add_argument(
        '--design-steps',
        type=int,
        default=75,
        help='Number of steps in the design phase (default: 75)',
    )

    parser.add_argument(
        '--adapt-steps',
        type=int,
        default=25,
        help='Number of steps in the adaptation phase (default: 25)',
    )

    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='Checkpoint frequency in steps (default: 10)',
    )

    parser.add_argument(
        '--shift-type',
        type=str,
        choices=['tighten_constraint', 'performance_increase', 'add_constraint'],
        default=None,
        help='Type of requirement shift (default: random)',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: random)',
    )

    parser.add_argument(
        '--output',
        type=str,
        default='simulation_results.json',
        help='Output JSON file for results (default: simulation_results.json)',
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output for each simulation run',
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except final statistics',
    )

    args = parser.parse_args()

    # Validate arguments
    if args.runs < 1:
        print("Error: --runs must be at least 1", file=sys.stderr)
        return 1

    if args.design_steps < 1:
        print("Error: --design-steps must be at least 1", file=sys.stderr)
        return 1

    if args.adapt_steps < 1:
        print("Error: --adapt-steps must be at least 1", file=sys.stderr)
        return 1

    if args.checkpoint_freq < 1:
        print("Error: --checkpoint-freq must be at least 1", file=sys.stderr)
        return 1

    if args.verbose and args.quiet:
        print("Error: Cannot use both --verbose and --quiet", file=sys.stderr)
        return 1

    # Convert shift type string to enum
    shift_type = None
    if args.shift_type:
        shift_type = ShiftType(args.shift_type)

    # Run simulations
    try:
        results = run_multiple_simulations(
            num_runs=args.runs,
            design_steps=args.design_steps,
            adaptation_steps=args.adapt_steps,
            checkpoint_frequency=args.checkpoint_freq,
            shift_type=shift_type,
            seed=args.seed,
            output_file=args.output,
            verbose=args.verbose,
        )

        return 0

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"\nError running simulation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
