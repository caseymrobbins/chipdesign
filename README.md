# Chip Design Optimization Strategy Simulator

A simulation framework for testing two competing optimization strategies for engineering design under multiple constraints. This project tests the hypothesis that **optimizing for constraint headroom** produces more adaptable designs than **greedy performance maximization**.

## Core Hypothesis

When designing complex systems with multiple constraints, optimizing `log(min(constraint_headrooms))` produces designs that are more adaptable to requirement changes than optimizing performance directly while staying within constraints.

## Table of Contents

- [Quick Start](#quick-start)
- [Concept Overview](#concept-overview)
- [Design Space](#design-space)
- [The Two Agents](#the-two-agents)
- [Simulation Structure](#simulation-structure)
- [Metrics Explained](#metrics-explained)
- [Usage Examples](#usage-examples)
- [Output Format](#output-format)
- [Interpreting Results](#interpreting-results)
- [Advanced Usage](#advanced-usage)

## Quick Start

```bash
# Install dependencies
pip install numpy

# Quick test run (5 simulations with verbose output)
python run_experiments.py --runs 5 --design-steps 20 --adapt-steps 10 --verbose

# Standard experiment (100 runs, default parameters)
python run_experiments.py --runs 100 --output results.json

# View results
python -c "import json; print(json.dumps(json.load(open('results.json'))['statistics'], indent=2))"
```

## Concept Overview

### The Problem

In engineering design (chip design, system architecture, etc.), we face multiple competing constraints:
- Heat dissipation limits
- Power budget
- Physical area
- Signal timing
- Manufacturing tolerances
- And more...

Traditional approaches optimize for **maximum performance** while staying within constraints. But what happens when requirements change? Does the design have enough margin to adapt?

### The Hypothesis

This simulator tests whether optimizing for **constraint headroom** (keeping all constraints comfortably away from their limits) produces designs that survive and adapt better when requirements shift.

## Design Space

The simulation creates a realistic chip/system design environment with:

### Constraint Dimensions (8 total)

Each constraint has:
- **Floor**: Hard limit (cannot go below this value)
- **Current Value**: Current design value
- **Headroom**: `current_value - floor` (how much margin remains)

Default constraints:
1. **heat_dissipation**: Thermal management (floor: 20, initial: 100)
2. **power_budget**: Power consumption limit (floor: 50, initial: 200)
3. **area**: Physical chip area (floor: 10, initial: 100)
4. **signal_latency**: Signal propagation time (floor: 5, initial: 50)
5. **manufacturing_tolerance**: Fabrication precision (floor: 2, initial: 20)
6. **memory_bandwidth**: Memory throughput (floor: 100, initial: 500)
7. **voltage_margin**: Operating voltage margin (floor: 0.1, initial: 1.0)
8. **timing_slack**: Clock timing margin (floor: 1, initial: 10)

### Performance Metric

Separate from constraints, represents throughput/clock speed (higher is better).

### Actions

Design decisions that modify multiple dimensions simultaneously, creating realistic tradeoffs:

**Balanced Actions**: Moderate performance gains with reasonable tradeoffs
- `optimize_cache`: +8 performance, moderate impact on heat/power/area
- `increase_frequency`: +12 performance, higher heat/power cost
- `add_pipeline_stage`: +6 performance, affects latency and area

**Trap Actions**: High performance but crush one critical dimension
- `aggressive_voltage_boost`: +20 performance, BUT crushes voltage_margin to near-zero
- `dense_placement`: +15 performance, BUT destroys manufacturing_tolerance
- `max_parallelization`: +25 performance, BUT obliterates power_budget

**Safe Actions**: Lower performance but preserve/improve margins
- `efficiency_tuning`: +4 performance, improves multiple margins
- `conservative_optimization`: +5 performance, maintains good headroom

## The Two Agents

### Agent 1: Greedy Performance Maximizer

**Objective**: `max(performance)` subject to all constraints ≥ floors

**Strategy**: Always select the action with highest performance gain that maintains feasibility

**Characteristics**:
- Standard optimization approach
- May fall into "trap" actions that crush one dimension
- Maximizes short-term performance
- Representative of traditional design practices

### Agent 2: Log-Min-Headroom Optimizer

**Objective**: `max(log(min(headroom_1, headroom_2, ...headroom_n)))`

**Secondary objective**: Among equal log-min scores, prefer higher performance

**Strategy**: Optimize for preserving options in the most constrained dimension

**Characteristics**:
- Avoids actions that crush any single dimension to near-zero
- Maintains balanced headroom across all constraints
- Hypothetically more adaptable to requirement changes
- May sacrifice some short-term performance for long-term flexibility

## Simulation Structure

### Phase 1: Design Phase (default: 75 steps)

Both agents start from identical initial conditions and make sequential design decisions.

**Key observations**:
- Agent 1 typically achieves higher raw performance
- Agent 2 typically maintains higher minimum headroom
- Trap actions may look attractive to Agent 1

### Phase 2: Requirement Shift

After design, requirements suddenly change (simulating real-world requirement evolution):

**Shift Types**:
1. **Tighten Constraint**: One random constraint floor increases by 20%
2. **Performance Increase**: Performance target increases by 15%
3. **Add Constraint**: New constraint dimension appears

**Critical Question**: Does the design survive the shift without violating floors?

### Phase 3: Adaptation Phase (default: 25 steps)

Both agents attempt to adapt their designs to meet new requirements.

**Metrics**:
- Did the design survive the shift?
- How many steps to reach new requirements?
- Final performance after adaptation?

## Metrics Explained

### Performance
- **What**: Throughput/clock speed metric
- **Units**: Arbitrary (starts at 100)
- **Higher is better**: More performance = better design
- **Used by**: Agent 1 as primary objective

### Headroom
- **What**: Distance from constraint floor (`current_value - floor`)
- **Units**: Same as constraint dimension
- **Higher is better**: More headroom = more adaptation margin
- **Example**: If area=50 and area_floor=10, headroom=40

### Min Headroom
- **What**: Minimum headroom across ALL constraint dimensions
- **Units**: Mixed (depends on most constrained dimension)
- **Higher is better**: The "bottleneck" metric
- **Used by**: Agent 2 as primary objective (optimizes log of this)
- **Critical**: When min_headroom → 0, design is fragile

### Survival Rate
- **What**: Percentage of designs that remain feasible after requirement shift
- **Range**: 0% to 100%
- **Key metric**: Tests adaptability hypothesis
- **Expected**: Agent 2 should have higher survival rate

### Win Rate
- **What**: Percentage of runs where agent achieves better overall outcome
- **Considers**: Survival, performance, headroom
- **Overall success**: Winner based on multiple criteria

## Usage Examples

### Basic Usage

```bash
# Standard run with defaults
python run_experiments.py

# Specify number of runs
python run_experiments.py --runs 200

# Change output file
python run_experiments.py --output my_results.json
```

### Adjusting Simulation Parameters

```bash
# Longer design phase
python run_experiments.py --design-steps 150

# Shorter adaptation phase
python run_experiments.py --adapt-steps 15

# More frequent checkpoints
python run_experiments.py --checkpoint-freq 5
```

### Testing Specific Scenarios

```bash
# Test constraint tightening specifically
python run_experiments.py --runs 100 --shift-type tighten_constraint

# Test performance increase requirement
python run_experiments.py --runs 100 --shift-type performance_increase

# Test new constraint addition
python run_experiments.py --runs 100 --shift-type add_constraint
```

### Reproducibility and Debugging

```bash
# Reproducible results with fixed seed
python run_experiments.py --runs 100 --seed 42

# Detailed verbose output (for small runs)
python run_experiments.py --runs 5 --verbose

# Quiet mode (only final statistics)
python run_experiments.py --runs 100 --quiet
```

### Quick Test Runs

```bash
# Very fast test (5 runs, short phases)
python run_experiments.py --runs 5 --design-steps 20 --adapt-steps 10 --verbose

# Medium test (20 runs)
python run_experiments.py --runs 20 --design-steps 40 --adapt-steps 15
```

## Output Format

### JSON Structure

Results are saved to a JSON file with the following structure:

```json
{
  "parameters": {
    "num_runs": 100,
    "design_steps": 75,
    "adaptation_steps": 25,
    "checkpoint_frequency": 10,
    "shift_type": "random",
    "seed": null
  },
  "statistics": {
    "total_runs": 100,
    "survival": {
      "agent1_survived": 45,
      "agent1_survival_rate": 0.45,
      "agent2_survived": 78,
      "agent2_survival_rate": 0.78
    },
    "wins": {
      "agent1_wins": 22,
      "agent2_wins": 68,
      "ties": 10,
      "agent1_win_rate": 0.22,
      "agent2_win_rate": 0.68
    },
    "performance_design": { ... },
    "headroom_design": { ... },
    "shift_type_breakdown": { ... }
  },
  "results": [
    {
      "run_id": 0,
      "winner": "LogMinHeadroom",
      "agent1_survived_shift": true,
      "agent2_survived_shift": true,
      "checkpoints": [ ... ]
    },
    ...
  ]
}
```

### Console Output

During execution, you'll see:

```
================================================================================
RUNNING 100 SIMULATIONS
================================================================================
Design steps: 75
Adaptation steps: 25
...

Completed 10/100 runs...
Completed 20/100 runs...
...

================================================================================
AGGREGATE RESULTS (100 runs)
================================================================================

SURVIVAL RATES:
  GreedyPerformance:   45/100 (45.0%)
  LogMinHeadroom:      78/100 (78.0%)

WIN RATES:
  GreedyPerformance:   22/100 (22.0%)
  LogMinHeadroom:      68/100 (68.0%)
  Ties:                10/100

PERFORMANCE (Design Phase):
  GreedyPerformance:   450.23 ± 45.12
  LogMinHeadroom:      380.45 ± 38.67

...
```

## Interpreting Results

### What to Look For

1. **Survival Rate Difference**
   - Is Agent 2 (LogMinHeadroom) surviving at higher rates?
   - Confirms/refutes adaptability hypothesis

2. **Performance vs Adaptability Tradeoff**
   - Agent 1 should have higher performance in design phase
   - But Agent 2 should survive better after shifts
   - Classic tradeoff: immediate performance vs future flexibility

3. **Shift Type Sensitivity**
   - Which shift types are most challenging?
   - Do agents perform differently on different shift types?

4. **Headroom Statistics**
   - Agent 2 should maintain higher minimum headroom
   - This "safety margin" is the key to adaptability

### Success Criteria

The hypothesis is supported if:
- Agent 2 has significantly higher survival rate (e.g., 70% vs 45%)
- Agent 2 wins more often overall
- Performance difference in design phase is moderate (Agent 1 only 10-20% better)
- Headroom difference in design phase is substantial (Agent 2 has 2-3x higher min headroom)

### Example Interpretation

```
SURVIVAL RATES:
  GreedyPerformance:   45/100 (45.0%)
  LogMinHeadroom:      78/100 (78.0%)
```

**Interpretation**: Agent 2 survives 73% more often! Strong support for the hypothesis that maintaining headroom creates adaptable designs.

```
PERFORMANCE (Design Phase):
  GreedyPerformance:   450.23 ± 45.12
  LogMinHeadroom:      380.45 ± 38.67
```

**Interpretation**: Agent 1 achieves 18% higher performance in the design phase, but this comes at the cost of fragility.

## Advanced Usage

### Programmatic Usage

You can also use the simulator as a Python library:

```python
from chip_design_simulator import (
    run_multiple_simulations,
    Simulation,
    ShiftType,
)

# Run experiments programmatically
results = run_multiple_simulations(
    num_runs=100,
    design_steps=75,
    adaptation_steps=25,
    shift_type=ShiftType.TIGHTEN_CONSTRAINT,
    seed=42,
    output_file="custom_results.json",
    verbose=False,
)

# Access statistics
print(results['statistics']['survival'])

# Run a single simulation with custom parameters
sim = Simulation(
    design_steps=100,
    adaptation_steps=30,
    seed=123,
    verbose=True,
)
result = sim.run_single_simulation(run_id=0)
```

### Custom Actions

To add your own actions, modify `chip_design_simulator.py`:

```python
def initialize_actions(self):
    self.actions = [
        Action(
            "my_custom_action",
            performance_delta=15.0,
            dimension_deltas={
                'heat_dissipation': -5.0,
                'power_budget': -8.0,
                'custom_dimension': +10.0,
            },
            description="My custom design action"
        ),
        # ... more actions
    ]
```

### Custom Constraints

Add new constraint dimensions:

```python
def initialize_default_constraints(self):
    self.constraints = {
        'my_new_constraint': ConstraintDimension(
            'my_new_constraint',
            floor=10.0,
            current_value=50.0
        ),
        # ... existing constraints
    }
```

## Requirements

- Python 3.7+
- NumPy

## Installation

```bash
# Clone or download the repository
cd chipdesign

# Install dependencies
pip install numpy

# Make CLI script executable (optional)
chmod +x run_experiments.py
```

## Files

- `chip_design_simulator.py`: Core simulation engine
  - DesignSpace class: Manages constraints and state
  - Agent classes: GreedyPerformanceAgent and LogMinHeadroomAgent
  - Simulation class: Orchestrates experiments
  - Analysis functions: Statistics and reporting

- `run_experiments.py`: Command-line interface
  - Argument parsing
  - User-friendly execution
  - Help documentation

- `README.md`: This documentation

## Citation

If you use this simulator in your research, please reference the core hypothesis:

> "Optimizing log(min(constraint_headrooms)) produces more adaptable engineering designs than greedy performance maximization under multi-constraint optimization problems."

## License

MIT License - Feel free to use, modify, and distribute.

## Contributing

Contributions welcome! Areas for extension:
- Additional agent strategies
- More realistic constraint interactions
- Visualization tools for results
- Multi-objective optimization variants
- Real-world case studies

## Contact

For questions, issues, or suggestions, please open an issue on the repository.

---

**Happy Experimenting!** 🚀
