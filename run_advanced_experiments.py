#!/usr/bin/env python3
"""
Command-line interface for running ADVANCED chip design optimization experiments.

This enhanced simulator includes:
- Physics-based power modeling (P = CV²f + leakage)
- Thermal modeling with temperature feedback
- Realistic timing models
- Manufacturing yield calculations
- IR drop and signal integrity
- Process technology support (7nm, 5nm)
"""

import argparse
import sys
from advanced_chip_simulator import (
    run_advanced_simulations,
    ShiftType,
    ProcessTechnology,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run ADVANCED chip design optimization simulations with realistic physics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Quick test run (5 simulations, verbose output)
  python run_advanced_experiments.py --runs 5 --design-steps 20 --adapt-steps 10 --verbose

  # Standard experiment (100 runs, 7nm process)
  python run_advanced_experiments.py --runs 100 --output advanced_results.json

  # Test with 5nm process technology
  python run_advanced_experiments.py --runs 50 --process 5nm

  # Test specific shift type
  python run_advanced_experiments.py --runs 50 --shift-type tighten_power

PHYSICS MODELS:
  Power: P_dynamic = C·V²·f·α + P_leakage
  Thermal: T = T_ambient + P·R_thermal (with feedback loop)
  Timing: delay ∝ 1/(V-Vth) × temperature_factor
  Yield: Poisson model based on die area and defect density

DESIGN PARAMETERS (actual knobs):
  - Clock frequency (GHz)
  - Supply voltage (V)
  - Pipeline stages
  - Issue width
  - Cache sizes (L1, L2, L3)
  - Number of cores
  - Transistor sizing
  - Metal layers
  - Power gating coverage

DERIVED CONSTRAINTS (calculated from physics):
  - Total power (dynamic + static)
  - Junction temperature
  - Timing slack
  - IR drop
  - Manufacturing yield
  - Signal integrity

SHIFT TYPES:
  tighten_power: Reduce power budget by 20%
  increase_performance: Increase frequency requirement by 15%
  reduce_area: Reduce die area budget by 15%
  tighten_thermal: Reduce thermal limit by 10°C
  yield_requirement: Increase yield requirement by 5%
  add_feature: New feature requiring +8% power, +10% area
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
        choices=['tighten_power', 'increase_performance', 'reduce_area', 'tighten_thermal', 'yield_requirement', 'add_feature'],
        default=None,
        help='Type of requirement shift (default: random)',
    )

    parser.add_argument(
        '--process',
        type=str,
        choices=['7nm', '5nm'],
        default='7nm',
        help='Process technology node (default: 7nm)',
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
        default='advanced_results.json',
        help='Output JSON file for results (default: advanced_results.json)',
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output for each simulation run',
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

    # Convert shift type string to enum
    shift_type = None
    if args.shift_type:
        shift_type = ShiftType(args.shift_type)

    # Select process technology
    if args.process == '7nm':
        process = ProcessTechnology.create_7nm()
    elif args.process == '5nm':
        process = ProcessTechnology.create_5nm()
    else:
        process = None

    # Run simulations
    try:
        results = run_advanced_simulations(
            num_runs=args.runs,
            design_steps=args.design_steps,
            adaptation_steps=args.adapt_steps,
            checkpoint_frequency=args.checkpoint_freq,
            shift_type=shift_type,
            process=process,
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
