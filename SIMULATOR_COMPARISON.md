# Advanced vs Simple Simulator Comparison

This document explains the differences between the two simulation frameworks.

## Overview

**Simple Simulator** (`chip_design_simulator.py`)
- Abstract constraint model
- Direct constraint manipulation
- Good for initial hypothesis testing

**Advanced Simulator** (`advanced_chip_simulator.py`)
- Physics-based realistic chip design
- Derived constraints from design parameters
- Closer to actual VLSI design flows

## Key Differences

### 1. Design Representation

**Simple Simulator:**
```python
# Direct constraint values
constraints = {
    'heat_dissipation': ConstraintDimension(floor=20.0, current_value=100.0),
    'power_budget': ConstraintDimension(floor=50.0, current_value=200.0),
}
```

**Advanced Simulator:**
```python
# Design parameters (what engineers actually control)
params = DesignParameters(
    clock_freq_ghz=3.0,
    supply_voltage=0.8,
    num_cores=8,
    pipeline_stages=14,
)

# Constraints are DERIVED from physics
power = calculate_dynamic_power(params) + calculate_static_power(params)
temperature = calculate_temperature(power, area)
timing_slack = period - critical_path_delay(params, temperature)
```

### 2. Physics Models

**Simple Simulator:**
- Abstract actions with predefined constraint deltas
- No physics relationships
- Example: `Action("boost", performance=+20, heat=-10, power=-15)`

**Advanced Simulator:**
- **Power Model**: P = C·V²·f·α + P_leakage(T)
  - Dynamic power scales with V² and frequency
  - Leakage increases exponentially with temperature
  - Thermal feedback loop (power → temp → leakage → power)

- **Timing Model**: delay ∝ 1/(V-Vth) × f(T)
  - Lower voltage = slower circuits
  - Higher temperature = slower circuits
  - Determines maximum achievable frequency

- **Thermal Model**: T = T_ambient + P·R_thermal
  - Temperature depends on total power
  - Creates feedback with leakage power

- **Yield Model**: Y = e^(-defect_density × area)
  - Larger dies = lower yield
  - Poisson defect model

### 3. Design Actions

**Simple Simulator:**
```python
Action("increase_frequency",
    performance_delta=12.0,
    dimension_deltas={
        'heat_dissipation': -8.0,
        'power_budget': -10.0,
    })
```

**Advanced Simulator:**
```python
Action("increase_frequency_aggressive",
    apply_fn=lambda p: modify_params(p,
        clock_freq_ghz=p.clock_freq_ghz * 1.10))
# Physics automatically calculates:
# - Higher power (P ∝ f)
# - Higher temperature (T ∝ P)
# - May violate timing if frequency too high
# - Reduced timing slack
```

### 4. Constraint Interactions

**Simple Simulator:**
- Independent constraints
- No coupling between dimensions
- Actions explicitly specify all effects

**Advanced Simulator:**
- **Coupled constraints** via physics:
  - Increasing frequency → more power → higher temp → more leakage → even more power!
  - Lowering voltage → less power BUT slower timing → can't run as fast
  - Adding cores → more area → lower yield, more power
  - Temperature affects both leakage AND timing

### 5. Realistic Tradeoffs

**Advanced Simulator captures real chip design tradeoffs:**

1. **DVFS Tradeoff** (Dynamic Voltage-Frequency Scaling):
   - Increase V and f together: P ∝ V²·f means power explodes!
   - Classic trap: +10% voltage, +10% frequency = +21% power

2. **Thermal Runaway Risk**:
   - High power → high temp → high leakage → higher power → ...
   - Must maintain thermal margin

3. **Timing Closure Challenge**:
   - Want high frequency for performance
   - But must ensure critical_path_delay < clock_period
   - Lower voltage for power savings makes timing worse

4. **Area-Yield Tradeoff**:
   - Add more cores/cache for performance
   - But larger die area = lower manufacturing yield

### 6. Process Technology

**Advanced Simulator Only:**
- Support for different process nodes (7nm, 5nm)
- Each node has different characteristics:
  - Vdd_nominal, Vdd_min, Vdd_max
  - Maximum frequency
  - Leakage characteristics
  - Wire resistance

### 7. Constraint Types

**Simple Simulator (8 constraints):**
- All abstract, directly controlled
- heat_dissipation, power_budget, area, signal_latency, etc.

**Advanced Simulator (10 constraints):**
- All derived from physics
- `total_power_w`: P_dynamic + P_static
- `temperature_c`: From thermal model
- `timing_slack_ps`: Can we meet clock period?
- `ir_drop_mv`: Voltage droop in power grid
- `yield`: Manufacturing yield (0-1)
- `signal_integrity`: Signal quality metric
- `power_density_w_mm2`: Hotspot metric
- `area_mm2`: Die area
- `wire_delay_ps`: Interconnect delay

### 8. Requirement Shifts

**Simple Simulator:**
- Tighten constraint floor
- Performance increase
- Add new constraint

**Advanced Simulator:**
- Tighten power budget (realistic: customer spec change)
- Increase performance requirement (market demand)
- Reduce area budget (cost reduction)
- Tighten thermal limit (packaging constraint)
- Increase yield requirement (manufacturing target)
- Add new feature (+power, +area)

## When to Use Each

### Use Simple Simulator When:
- Initial hypothesis exploration
- Abstract design space studies
- Quick experiments (runs faster)
- Teaching constraint optimization concepts
- You want more control over exact constraint relationships

### Use Advanced Simulator When:
- Validating hypothesis with realistic physics
- Studying actual chip design tradeoffs
- Understanding power-thermal feedback
- Analyzing DVFS strategies
- Comparing process technologies
- Publishing results (more credible with realistic models)
- Training on real VLSI design scenarios

## Performance Comparison

**Simple Simulator:**
- Fast: ~0.1s per simulation
- Can run 1000s of runs quickly
- Good for statistical analysis

**Advanced Simulator:**
- Slower: ~0.5-1s per simulation
- Physics calculations add overhead
- Still fast enough for 100s of runs
- More expensive clone() operations

## Example Results

### Simple Simulator
```
Initial: performance=100, headroom=20
After optimization: performance=450, headroom=0.0
Power: abstract value 150/150
```

### Advanced Simulator
```
Initial: 3.0GHz, 0.8V, 8 cores, 29W power, 37°C
After optimization: 4.95GHz, 0.65V, 8 cores, 62W power, 50°C
Physics shows: lower voltage enables higher frequency due to timing!
Realistic power-frequency relationship visible
```

## Validation

The Advanced Simulator's physics models are calibrated to match real chip behavior:
- 8-core processor: ~30-80W (realistic TDP range)
- Temperature rise: ~0.4°C per Watt (typical thermal resistance)
- Timing: 15ps per pipeline stage at nominal (realistic for 7nm)
- Leakage: ~20-30% of total power (typical for modern processes)
- Yield: 85-90% for 150mm² die (realistic)

## Code Organization

Both simulators share:
- Agent base classes (Greedy, LogMinHeadroom)
- Simulation orchestration structure
- Three-phase execution (Design, Shift, Adaptation)
- JSON output format
- Analysis tools compatible with both

Differences:
- `DesignSpace` vs `AdvancedDesignSpace`
- `Action` vs `DesignAction` (with apply_fn)
- Simple constraints vs physics-based constraints

## Migration Path

To convert experiments from Simple to Advanced:
1. Replace `run_multiple_simulations` with `run_advanced_simulations`
2. Optionally specify `process` technology
3. Analysis tools work with both output formats
4. Expect different absolute numbers but similar relative behavior

## Conclusion

Both simulators test the same hypothesis:
> **Optimizing log(min(headroom)) produces more adaptable designs than greedy performance maximization**

The Advanced Simulator adds:
- Realistic chip physics
- Coupled constraint interactions
- Process technology effects
- More credible validation of the hypothesis

Choose based on your needs: Simple for speed and exploration, Advanced for realism and validation.
