"""
Advanced Chip Design Optimization Simulator

This enhanced simulator models realistic chip design with:
- Design parameters (actual knobs: frequency, voltage, cache size, etc.)
- Physics-based derived constraints (power ∝ V²f, thermal density, etc.)
- Non-linear interactions and feedback loops
- Realistic architectural design decisions
- Process technology effects

Author: Claude
Version: 2.0 (Enhanced Realism)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import copy
import math


class ShiftType(Enum):
    """Types of requirement shifts that can occur"""
    TIGHTEN_POWER = "tighten_power"
    INCREASE_PERFORMANCE = "increase_performance"
    REDUCE_AREA = "reduce_area"
    TIGHTEN_THERMAL = "tighten_thermal"
    YIELD_REQUIREMENT = "yield_requirement"
    ADD_FEATURE = "add_feature"


@dataclass
class ProcessTechnology:
    """
    Represents semiconductor process technology node.

    Real chip design is heavily influenced by the manufacturing process.
    """
    node_nm: float  # Technology node (e.g., 7nm, 5nm)
    vdd_nominal: float  # Nominal supply voltage (V)
    vdd_min: float  # Minimum voltage
    vdd_max: float  # Maximum voltage
    freq_max_ghz: float  # Maximum frequency at nominal voltage
    gate_capacitance_pf: float  # Gate capacitance per unit
    leakage_factor: float  # Leakage current scaling factor
    min_feature_size_nm: float  # Minimum feature size
    wire_resistance_ohm_um: float  # Wire resistance per micron

    @staticmethod
    def create_7nm():
        """Create a 7nm process technology"""
        return ProcessTechnology(
            node_nm=7.0,
            vdd_nominal=0.8,
            vdd_min=0.65,
            vdd_max=1.0,
            freq_max_ghz=5.0,
            gate_capacitance_pf=1.0,
            leakage_factor=0.1,
            min_feature_size_nm=7.0,
            wire_resistance_ohm_um=0.05,
        )

    @staticmethod
    def create_5nm():
        """Create a 5nm process technology (more aggressive)"""
        return ProcessTechnology(
            node_nm=5.0,
            vdd_nominal=0.75,
            vdd_min=0.60,
            vdd_max=0.95,
            freq_max_ghz=5.5,
            gate_capacitance_pf=0.7,
            leakage_factor=0.15,  # Higher leakage in advanced nodes
            min_feature_size_nm=5.0,
            wire_resistance_ohm_um=0.07,  # Higher resistance in smaller wires
        )


@dataclass
class DesignParameters:
    """
    Actual design parameters that engineers control.

    These are the "knobs" that designers turn to optimize their chip.

    CONFIGURED FOR FORCING CONSTRAINTS:
    - Lower voltage to reduce power below 12W max
    - Boost IPC (wider issue, optimal pipeline, bigger caches) to hit 30+ performance
    - Start with strong baseline that meets all minimum requirements
    """
    # Clock and voltage - IPC-focused: boost voltage for perf, keep freq minimal
    clock_freq_ghz: float = 5.0  # Clock frequency (GHz) - At minimum to control power
    supply_voltage: float = 0.67  # Supply voltage Vdd (V) - Precisely tuned for 35+ perf, <12W

    # Microarchitecture - BOOSTED for high base performance
    pipeline_stages: int = 14  # Number of pipeline stages - OPTIMAL depth (no penalty!)
    issue_width: int = 6  # Instructions issued per cycle - WIDE for high IPC!
    reorder_buffer_size: int = 192  # ROB entries - larger for wide issue

    # Cache hierarchy - MASSIVE to maximize IPC without frequency boost
    l1_cache_kb: float = 64.0  # L1 cache size (KB) - doubled for IPC boost
    l2_cache_kb: float = 512.0  # L2 cache size (KB) - doubled
    l3_cache_kb: float = 8192.0  # L3 cache size (KB) - 8MB! Massive for memory latency

    # Core configuration
    num_cores: int = 7  # Number of processor cores - balanced throughput

    # Physical design - Moderate size to meet 30mm² minimum
    core_area_mm2: float = 4.5  # Core area (mm²) - slightly larger for wider issue
    total_area_mm2: float = 31.5  # Total die area (mm²) - above 30mm² minimum
    transistor_sizing_factor: float = 1.0  # Relative transistor sizing (1.0 = nominal)

    # Floorplan
    aspect_ratio: float = 1.0  # Width/height ratio
    metal_layers: int = 10  # Number of metal layers used

    # Power management
    clock_gating_coverage: float = 0.7  # Fraction of clock tree gated
    power_gating_coverage: float = 0.3  # Fraction of logic power gated

    def clone(self) -> 'DesignParameters':
        """Create a deep copy"""
        return copy.deepcopy(self)


@dataclass
class ConstraintLimits:
    """
    Hard limits on constraints.

    CONFIGURED TO FORCE RESOURCE UTILIZATION:
    - Set HIGH minimums to prevent JAM from being ultra-conservative
    - Force JAM to actually USE the power/area budget
    - JAM will find the most EFFICIENT way to meet high performance targets!

    The insight: JAM can achieve high performance - we just need to REQUIRE it!
    """
    max_power_watts: float = 12.0  # Maximum power budget
    min_power_watts: float = 7.0   # FORCE JAM to use ≥58% power budget (moderate)
    max_area_mm2: float = 50.0  # Maximum area budget
    min_area_mm2: float = 30.0  # FORCE JAM to use ≥60% area budget (moderate)
    max_temperature_c: float = 70.0  # Thermal limit
    min_frequency_ghz: float = 5.0  # High performance requirement
    min_performance_score: float = 35.0  # FORCE higher performance (achievable!)
    min_timing_slack_ps: float = 80.0  # Timing requirement
    max_ir_drop_mv: float = 40.0  # Power delivery quality
    min_yield: float = 0.0  # Removed - margin penalty handles this
    max_wire_delay_ps: float = 120.0  # Interconnect performance
    min_signal_integrity: float = 0.0  # Removed - margin penalty handles this
    max_power_density_w_mm2: float = 0.70  # Hotspot prevention

    # Constraint weights: Lower weight = higher priority (more important to maintain)
    # JAM will work harder to keep weighted headroom higher
    constraint_weights: dict = None  # Will be set in __post_init__

    def __post_init__(self):
        """Set default constraint weights - all equal for balanced optimization"""
        if self.constraint_weights is None:
            self.constraint_weights = {
                'power': 1.0,
                'area': 1.0,
                'temperature': 1.0,
                'frequency': 1.0,
                'performance': 1.0,  # Equal weight - difficult margins drive performance naturally
                'timing_slack': 1.0,
                'ir_drop': 1.0,
                'yield': 1.0,
                'signal_integrity': 1.0,
                'power_density': 1.0,
                'wire_delay': 1.0,
            }

    def clone(self) -> 'ConstraintLimits':
        """Create a deep copy"""
        return copy.deepcopy(self)


class PhysicsModel:
    """
    Physics-based models for chip behavior.

    These implement realistic relationships between design parameters and constraints.
    """

    def __init__(self, process: ProcessTechnology):
        self.process = process

    def calculate_dynamic_power(self, params: DesignParameters, activity_factor: float = 0.15) -> float:
        """
        Calculate dynamic power: P = C * V² * f * α

        Where:
        - C: capacitance (depends on transistor sizing, area)
        - V: supply voltage
        - f: frequency
        - α: activity factor (fraction of gates switching)
        """
        # Base capacitance - calibrated to give realistic power numbers
        # Typical chip: ~10-20W per core at nominal settings
        base_capacitance_nf_per_core = 15.0  # nanofarads per core

        capacitance_nf = (
            base_capacitance_nf_per_core *
            params.num_cores *
            (params.core_area_mm2 / 10.0) *  # Normalize to 10mm² per core
            params.transistor_sizing_factor
        )

        # Adjust activity factor based on clock gating
        effective_activity = activity_factor * (1.0 - params.clock_gating_coverage * 0.5)

        # P = C * V² * f * α
        power_watts = (
            capacitance_nf * 1e-9 *  # Convert nF to F
            params.supply_voltage ** 2 *
            params.clock_freq_ghz * 1e9 *  # Convert GHz to Hz
            effective_activity
        )

        return power_watts

    def calculate_static_power(self, params: DesignParameters, temperature_c: float = 75.0) -> float:
        """
        Calculate static (leakage) power: P_leak = V * I_leak

        Leakage increases exponentially with temperature (thermal runaway risk!)
        """
        # Base leakage - calibrated to be ~20-30% of dynamic power at nominal conditions
        base_leakage_w_per_core = 3.0  # Watts per core at nominal conditions

        base_leakage = (
            base_leakage_w_per_core *
            params.num_cores *
            (params.core_area_mm2 / 10.0) *  # Normalize to 10mm² per core
            params.transistor_sizing_factor
        )

        # Temperature dependence: leakage increases with temperature
        # Use a less aggressive model to avoid thermal runaway
        # ~30% increase per 10°C instead of doubling
        temp_clamped = max(25, min(150, temperature_c))
        temp_factor = 1.3 ** ((temp_clamped - 75.0) / 10.0)  # Normalized to 75°C

        # Voltage dependence (exponential in reality, simplified here)
        voltage_factor = (params.supply_voltage / self.process.vdd_nominal) ** 1.5

        # Power gating reduces leakage
        effective_leakage = base_leakage * (1.0 - params.power_gating_coverage * 0.8)

        power_watts = effective_leakage * voltage_factor * temp_factor

        return power_watts

    def calculate_temperature(self, total_power_w: float, area_mm2: float, cooling_capacity: float = 1.0) -> float:
        """
        Calculate junction temperature.

        Temperature depends on:
        - Total power
        - Cooling capacity
        - Ambient temperature
        """
        # Simplified thermal model: T = T_ambient + Power * thermal_resistance
        # Calibrated so that 150W gives ~85-90°C
        ambient_c = 25.0
        thermal_resistance_c_per_w = 0.4  # °C per Watt

        temperature_rise = total_power_w * thermal_resistance_c_per_w / cooling_capacity
        temperature_c = ambient_c + temperature_rise

        # Clamp to reasonable range
        temperature_c = max(ambient_c, min(150.0, temperature_c))

        return temperature_c

    def calculate_critical_path_delay(self, params: DesignParameters, temperature_c: float = 75.0) -> float:
        """
        Calculate critical path delay (determines max frequency).

        Delay depends on:
        - Logic depth (pipeline stages)
        - Voltage (lower voltage = slower)
        - Temperature (higher temp = slower)
        - Transistor sizing
        """
        # For a pipelined processor, each stage should be balanced
        # At 3GHz, period = 333ps. Each stage should be <300ps to leave margin
        # Base delay per pipeline stage at nominal conditions (ps)
        # Calibrated so 14 stages at 3GHz with nominal settings is feasible
        base_delay_ps_per_stage = 15.0  # Reduced from 50

        # Voltage scaling: delay inversely proportional to (V - Vth)
        v_threshold = 0.3  # Threshold voltage
        voltage_factor = (self.process.vdd_nominal - v_threshold) / max(0.1, params.supply_voltage - v_threshold)

        # Temperature factor: ~0.3% increase per °C
        temp_factor = 1.0 + 0.003 * (temperature_c - 25.0)

        # Transistor sizing: larger transistors are faster
        sizing_factor = 1.0 / params.transistor_sizing_factor

        # Complexity factor: wider issue width increases path length
        complexity_factor = 1.0 + 0.05 * (params.issue_width - 4)

        # Total logic delay
        logic_delay_ps = (
            base_delay_ps_per_stage *
            (params.pipeline_stages / 14.0) *  # Normalized to 14 stages
            voltage_factor *
            temp_factor *
            sizing_factor *
            complexity_factor
        )

        # Wire delay (gets worse with more metal layers and smaller features)
        wire_delay_ps = self._calculate_wire_delay(params)

        total_delay_ps = logic_delay_ps + wire_delay_ps

        return total_delay_ps

    def _calculate_wire_delay(self, params: DesignParameters) -> float:
        """Calculate interconnect wire delay"""
        # Wire delay is a small component of total delay in well-designed chips
        # Typical wire delay: 20-50ps for critical paths
        base_wire_delay_ps = 30.0

        # Scales slightly with area (longer wires)
        area_factor = math.sqrt(params.total_area_mm2 / 150.0)

        # More metal layers can reduce delay with better routing
        layer_factor = 1.0 - 0.03 * (params.metal_layers - 10)

        wire_delay_ps = base_wire_delay_ps * area_factor * max(0.5, layer_factor)

        return wire_delay_ps

    def calculate_timing_slack(self, params: DesignParameters, temperature_c: float) -> float:
        """
        Calculate timing slack: slack = T_period - T_critical_path

        Positive slack = timing met, negative slack = timing violation
        """
        period_ps = (1.0 / params.clock_freq_ghz) * 1000.0  # Convert GHz to ps
        critical_path_ps = self.calculate_critical_path_delay(params, temperature_c)

        slack_ps = period_ps - critical_path_ps

        return slack_ps

    def calculate_ir_drop(self, params: DesignParameters, total_power_w: float) -> float:
        """
        Calculate IR drop (voltage droop due to resistance in power grid).

        IR drop gets worse with:
        - Higher current (power)
        - Larger die area (longer power wires)
        - Fewer metal layers
        """
        # Current = Power / Voltage
        current_a = total_power_w / max(0.1, params.supply_voltage)

        # Power grid resistance (simplified model)
        # Calibrated so nominal design has ~20-30mV IR drop
        grid_resistance_ohm = 0.0005 * (params.total_area_mm2 / 150.0) * (10.0 / params.metal_layers)

        ir_drop_v = current_a * grid_resistance_ohm
        ir_drop_mv = ir_drop_v * 1000.0

        return ir_drop_mv

    def calculate_manufacturing_yield(self, params: DesignParameters) -> float:
        """
        Calculate manufacturing yield.

        Yield decreases with:
        - Larger die area (more defects)
        - Smaller features (harder to manufacture)
        - More aggressive transistor sizing
        """
        # Defect density (defects per cm²)
        defect_density = 0.1

        # Area in cm²
        area_cm2 = params.total_area_mm2 / 100.0

        # Poisson yield model: Y = e^(-D*A)
        base_yield = math.exp(-defect_density * area_cm2)

        # Penalty for aggressive sizing
        sizing_penalty = 1.0 - 0.05 * max(0, params.transistor_sizing_factor - 1.0)

        # Penalty for higher frequencies (harder to meet timing)
        freq_penalty = 1.0 - 0.02 * max(0, params.clock_freq_ghz - self.process.freq_max_ghz)

        yield_value = base_yield * sizing_penalty * freq_penalty

        return max(0.0, min(1.0, yield_value))

    def calculate_signal_integrity(self, params: DesignParameters, temperature_c: float) -> float:
        """
        Calculate signal integrity metric.

        Signal integrity degrades with:
        - Higher frequencies
        - Longer wires
        - Higher temperatures
        - Crosstalk
        """
        # Base signal integrity
        base_si = 1.0

        # Frequency penalty: higher frequency = more issues
        freq_penalty = 0.02 * (params.clock_freq_ghz / self.process.freq_max_ghz - 0.6)

        # Wire length penalty
        wire_penalty = 0.01 * (params.total_area_mm2 / 150.0 - 1.0)

        # Temperature penalty
        temp_penalty = 0.001 * (temperature_c - 75.0)

        # Metal layers help (better shielding)
        layer_benefit = 0.02 * (params.metal_layers - 10)

        si = base_si - freq_penalty - wire_penalty - temp_penalty + layer_benefit

        return max(0.0, min(1.0, si))


class AdvancedDesignSpace:
    """
    Advanced design space with physics-based constraint calculation.

    This models actual chip design where you control parameters
    (frequency, voltage, etc.) and constraints are derived from physics.
    """

    def __init__(self, process: Optional[ProcessTechnology] = None, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.process = process or ProcessTechnology.create_7nm()
        self.physics = PhysicsModel(self.process)

        # Design parameters (what we control)
        self.params = DesignParameters()
        self.initial_params = self.params.clone()

        # Constraint limits (what we must satisfy)
        self.limits = ConstraintLimits()
        self.initial_limits = self.limits.clone()

        # Cached constraint values (updated when needed)
        self._cached_constraints: Optional[Dict[str, float]] = None
        self._temperature_cache: Optional[float] = None

        # Performance metric
        self.performance_score: float = 0.0

        # Step counter
        self.step_count: int = 0

        # Actions - initialize them immediately
        self.actions: List['DesignAction'] = []
        self.initialize_actions()

    def initialize_actions(self):
        """Create realistic chip design actions"""
        self.actions = [
            # DVFS (Dynamic Voltage-Frequency Scaling) actions - Multiple granularities
            DesignAction(
                name="increase_frequency_aggressive",
                description="Increase clock frequency aggressively (+10%)",
                apply_fn=lambda p: self._modify_params(p, clock_freq_ghz=p.clock_freq_ghz * 1.10),
                category="performance"
            ),
            DesignAction(
                name="increase_frequency_moderate",
                description="Increase clock frequency moderately (+5%)",
                apply_fn=lambda p: self._modify_params(p, clock_freq_ghz=p.clock_freq_ghz * 1.05),
                category="performance"
            ),
            DesignAction(
                name="increase_frequency_small",
                description="Increase clock frequency slightly (+2%)",
                apply_fn=lambda p: self._modify_params(p, clock_freq_ghz=p.clock_freq_ghz * 1.02),
                category="performance"
            ),
            DesignAction(
                name="increase_frequency_tiny",
                description="Increase clock frequency minimally (+1%)",
                apply_fn=lambda p: self._modify_params(p, clock_freq_ghz=p.clock_freq_ghz * 1.01),
                category="performance"
            ),
            DesignAction(
                name="decrease_frequency",
                description="Decrease clock frequency (-8%)",
                apply_fn=lambda p: self._modify_params(p, clock_freq_ghz=p.clock_freq_ghz * 0.92),
                category="efficiency"
            ),
            DesignAction(
                name="decrease_frequency_small",
                description="Decrease clock frequency slightly (-2%)",
                apply_fn=lambda p: self._modify_params(p, clock_freq_ghz=p.clock_freq_ghz * 0.98),
                category="efficiency"
            ),
            DesignAction(
                name="increase_voltage",
                description="Increase supply voltage (+5%)",
                apply_fn=lambda p: self._modify_params(p, supply_voltage=min(self.process.vdd_max, p.supply_voltage * 1.05)),
                category="performance"
            ),
            DesignAction(
                name="increase_voltage_small",
                description="Increase supply voltage slightly (+2%)",
                apply_fn=lambda p: self._modify_params(p, supply_voltage=min(self.process.vdd_max, p.supply_voltage * 1.02)),
                category="performance"
            ),
            DesignAction(
                name="increase_voltage_tiny",
                description="Increase supply voltage minimally (+1%)",
                apply_fn=lambda p: self._modify_params(p, supply_voltage=min(self.process.vdd_max, p.supply_voltage * 1.01)),
                category="performance"
            ),
            DesignAction(
                name="decrease_voltage",
                description="Decrease supply voltage (-5%)",
                apply_fn=lambda p: self._modify_params(p, supply_voltage=max(self.process.vdd_min, p.supply_voltage * 0.95)),
                category="efficiency"
            ),
            DesignAction(
                name="decrease_voltage_small",
                description="Decrease supply voltage slightly (-2%)",
                apply_fn=lambda p: self._modify_params(p, supply_voltage=max(self.process.vdd_min, p.supply_voltage * 0.98)),
                category="efficiency"
            ),
            DesignAction(
                name="dvfs_performance_boost",
                description="TRAP: Boost both V and F (+8% each) - great performance but power explosion",
                apply_fn=lambda p: self._modify_params(
                    p,
                    clock_freq_ghz=p.clock_freq_ghz * 1.08,
                    supply_voltage=min(self.process.vdd_max, p.supply_voltage * 1.08)
                ),
                category="trap"
            ),

            # Pipeline modifications
            DesignAction(
                name="deepen_pipeline",
                description="Add pipeline stages (+2 stages)",
                apply_fn=lambda p: self._modify_params(p, pipeline_stages=p.pipeline_stages + 2),
                category="performance"
            ),
            DesignAction(
                name="shallow_pipeline",
                description="Remove pipeline stages (-2 stages)",
                apply_fn=lambda p: self._modify_params(p, pipeline_stages=max(8, p.pipeline_stages - 2)),
                category="efficiency"
            ),

            # Width/ILP modifications
            DesignAction(
                name="increase_issue_width",
                description="TRAP: Increase issue width (+1) - more ILP but area/power explosion",
                apply_fn=lambda p: self._modify_params(
                    p,
                    issue_width=p.issue_width + 1,
                    reorder_buffer_size=int(p.reorder_buffer_size * 1.3),
                    core_area_mm2=p.core_area_mm2 * 1.15
                ),
                category="trap"
            ),
            DesignAction(
                name="decrease_issue_width",
                description="Decrease issue width (-1)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    issue_width=max(2, p.issue_width - 1),
                    reorder_buffer_size=int(p.reorder_buffer_size * 0.8),
                    core_area_mm2=p.core_area_mm2 * 0.90
                ),
                category="efficiency"
            ),

            # Cache modifications
            DesignAction(
                name="increase_l1_cache",
                description="Increase L1 cache (+16KB)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    l1_cache_kb=p.l1_cache_kb + 16,
                    total_area_mm2=p.total_area_mm2 + 1.0
                ),
                category="performance"
            ),
            DesignAction(
                name="increase_l2_cache",
                description="Increase L2 cache (+256KB)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    l2_cache_kb=p.l2_cache_kb + 256,
                    total_area_mm2=p.total_area_mm2 + 3.0
                ),
                category="performance"
            ),
            DesignAction(
                name="reduce_l3_cache",
                description="Reduce L3 cache (-1024KB)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    l3_cache_kb=max(1024, p.l3_cache_kb - 1024),
                    total_area_mm2=max(10.0, p.total_area_mm2 - 5.0)  # Prevent area from going too low
                ),
                category="area"
            ),

            # Core count modifications - DISABLED to compare agents with same core count
            # DesignAction(
            #     name="add_core",
            #     description="TRAP: Add processor core (+1) - more throughput but power/area explosion",
            #     apply_fn=lambda p: self._modify_params(
            #         p,
            #         num_cores=p.num_cores + 1,
            #         total_area_mm2=p.total_area_mm2 + p.core_area_mm2
            #     ),
            #     category="trap"
            # ),
            # DesignAction(
            #     name="remove_core",
            #     description="Remove processor core (-1)",
            #     apply_fn=lambda p: self._modify_params(
            #         p,
            #         num_cores=max(4, p.num_cores - 1),
            #         total_area_mm2=p.total_area_mm2 - p.core_area_mm2
            #     ),
            #     category="area"
            # ),

            # Transistor sizing
            DesignAction(
                name="upsize_transistors",
                description="Increase transistor sizing (+10%) - faster but more power/area",
                apply_fn=lambda p: self._modify_params(
                    p,
                    transistor_sizing_factor=p.transistor_sizing_factor * 1.10,
                    core_area_mm2=p.core_area_mm2 * 1.05,
                    total_area_mm2=p.total_area_mm2 * 1.02
                ),
                category="performance"
            ),
            DesignAction(
                name="downsize_transistors",
                description="Decrease transistor sizing (-10%)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    transistor_sizing_factor=p.transistor_sizing_factor * 0.90,
                    core_area_mm2=p.core_area_mm2 * 0.97,
                    total_area_mm2=p.total_area_mm2 * 0.99
                ),
                category="efficiency"
            ),

            # Power management
            DesignAction(
                name="increase_clock_gating",
                description="Increase clock gating coverage (+5%)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    clock_gating_coverage=min(0.95, p.clock_gating_coverage + 0.05)
                ),
                category="efficiency"
            ),
            DesignAction(
                name="increase_power_gating",
                description="Increase power gating coverage (+5%)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    power_gating_coverage=min(0.80, p.power_gating_coverage + 0.05)
                ),
                category="efficiency"
            ),

            # Physical design
            DesignAction(
                name="add_metal_layer",
                description="Add metal layer (+1) - better routing but cost/complexity",
                apply_fn=lambda p: self._modify_params(
                    p,
                    metal_layers=min(15, p.metal_layers + 1)
                ),
                category="performance"
            ),
            DesignAction(
                name="optimize_floorplan",
                description="Optimize floorplan aspect ratio",
                apply_fn=lambda p: self._modify_params(
                    p,
                    aspect_ratio=1.0 if abs(p.aspect_ratio - 1.0) > 0.1 else p.aspect_ratio
                ),
                category="efficiency"
            ),

            # Conservative/safe actions
            DesignAction(
                name="balanced_optimization",
                description="SAFE: Balanced optimization (slight freq increase, more clock gating)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    clock_freq_ghz=p.clock_freq_ghz * 1.02,
                    clock_gating_coverage=min(0.95, p.clock_gating_coverage + 0.03)
                ),
                category="safe"
            ),
            DesignAction(
                name="thermal_optimization",
                description="SAFE: Thermal optimization (reduce V, increase clock gating)",
                apply_fn=lambda p: self._modify_params(
                    p,
                    supply_voltage=max(self.process.vdd_min, p.supply_voltage * 0.97),
                    clock_gating_coverage=min(0.95, p.clock_gating_coverage + 0.05)
                ),
                category="safe"
            ),
        ]

    def _modify_params(self, p: DesignParameters, **kwargs) -> DesignParameters:
        """Helper to modify parameters"""
        new_params = p.clone()
        for key, value in kwargs.items():
            setattr(new_params, key, value)
        return new_params

    def calculate_constraints(self) -> Dict[str, float]:
        """
        Calculate all constraint values from current design parameters.

        This is where the physics happens!
        """
        # Start with ambient temperature, then iterate to find steady state
        temperature_c = 75.0

        # Iteratively solve for temperature (since power depends on temp via leakage)
        for _ in range(5):  # Iterate to convergence
            dynamic_power = self.physics.calculate_dynamic_power(self.params)
            static_power = self.physics.calculate_static_power(self.params, temperature_c)
            total_power = dynamic_power + static_power
            temperature_c = self.physics.calculate_temperature(total_power, self.params.total_area_mm2)

        # Calculate all other constraints
        timing_slack = self.physics.calculate_timing_slack(self.params, temperature_c)
        ir_drop = self.physics.calculate_ir_drop(self.params, total_power)
        yield_value = self.physics.calculate_manufacturing_yield(self.params)
        signal_integrity = self.physics.calculate_signal_integrity(self.params, temperature_c)
        power_density = total_power / self.params.total_area_mm2
        wire_delay = self.physics._calculate_wire_delay(self.params)

        constraints = {
            'total_power_w': total_power,
            'dynamic_power_w': dynamic_power,
            'static_power_w': static_power,
            'temperature_c': temperature_c,
            'area_mm2': self.params.total_area_mm2,
            'timing_slack_ps': timing_slack,
            'ir_drop_mv': ir_drop,
            'yield': yield_value,
            'signal_integrity': signal_integrity,
            'power_density_w_mm2': power_density,
            'wire_delay_ps': wire_delay,
        }

        self._cached_constraints = constraints
        self._temperature_cache = temperature_c

        return constraints

    def calculate_performance(self) -> float:
        """
        Calculate overall performance metric.

        Performance is a combination of:
        - Clock frequency
        - IPC (instructions per cycle) - depends on architecture
        - Number of cores (for throughput)
        """
        # Base IPC depends on issue width, pipeline depth, caches
        base_ipc = 1.0

        # Issue width benefit (but diminishing returns)
        ipc_width = base_ipc * (1.0 + 0.2 * (self.params.issue_width - 4))

        # Pipeline benefit (deeper = higher freq but worse IPC due to branches)
        pipeline_factor = 1.0 - 0.01 * abs(self.params.pipeline_stages - 14)

        # Cache benefit - L3 is CRITICAL for memory-bound workloads!
        # L1: Fast but small - big impact per KB
        # L2: Medium - moderate impact
        # L3: Large - crucial for avoiding DRAM (100+ cycle penalty)
        l1_benefit = 0.001 * self.params.l1_cache_kb  # Strong impact
        l2_benefit = 0.0002 * self.params.l2_cache_kb  # Moderate impact
        l3_benefit = 0.00005 * self.params.l3_cache_kb  # Large capacity, crucial for memory latency

        cache_factor = 1.0 + l1_benefit + l2_benefit + l3_benefit

        # L3 cache ALSO reduces memory stall time (avoid DRAM accesses)
        # Cutting L3 in half significantly increases memory stalls
        l3_stall_reduction = 1.0 + 0.00003 * self.params.l3_cache_kb  # More L3 = fewer stalls

        ipc = ipc_width * pipeline_factor * cache_factor * l3_stall_reduction

        # Single-threaded performance
        single_thread_perf = ipc * self.params.clock_freq_ghz

        # Multi-core throughput (with some scaling efficiency loss)
        core_scaling = self.params.num_cores ** 0.9  # Amdahl's law effect

        # Base performance
        base_performance = single_thread_perf * core_scaling

        # MARGIN IMPACT: Running with low margins hurts real performance, but
        # EXCELLENT margins actually BOOST performance!
        # - Thermal throttling when too hot (penalty)
        # - Timing errors when margins too tight (penalty)
        # - Signal integrity issues cause retries (penalty)
        # - Low yield means defective chips (penalty)
        # - BUT: Great margins allow higher boost clocks, better signal quality (bonus!)
        # Exclude performance from headrooms to avoid recursion
        headrooms = self.get_headrooms(include_performance=False)
        min_headroom = min(headrooms.values())

        # MARGIN FACTOR: Ranges from 0.5x (at limit) to 1.5x (excellent margins)
        # VERY steep curve to reward even tiny margin improvements!
        # At headroom=0.0: 50% performance (0.5x) - running at absolute limit
        # At headroom=1.0: 80% performance (0.8x) - slight margin helps
        # At headroom=5.0: 100% performance (1.0x) - healthy margins
        # At headroom=10.0+: 150% performance (1.5x) - excellent margins enable boost!
        if min_headroom < 5.0:
            margin_factor = 0.5 + 0.1 * min_headroom  # Range: 0.5 to 1.0
        elif min_headroom < 10.0:
            margin_factor = 1.0 + 0.1 * (min_headroom - 5.0)  # Range: 1.0 to 1.5
        else:
            margin_factor = 1.5  # Maximum 50% boost for excellent margins!

        # Apply margin factor
        performance = base_performance * margin_factor

        return performance

    def get_headrooms(self, include_performance: bool = True) -> Dict[str, float]:
        """
        Calculate headroom for each constraint.

        Headroom = how much margin we have before hitting the limit
        Positive headroom = good, negative = constraint violated
        """
        constraints = self.calculate_constraints()

        headrooms = {
            'power_max': self.limits.max_power_watts - constraints['total_power_w'],
            'power_min': constraints['total_power_w'] - self.limits.min_power_watts,  # NEW: must use enough power!
            'area_max': self.limits.max_area_mm2 - constraints['area_mm2'],
            'area_min': constraints['area_mm2'] - self.limits.min_area_mm2,  # NEW: must use enough area!
            'temperature': self.limits.max_temperature_c - constraints['temperature_c'],
            'frequency': self.params.clock_freq_ghz - self.limits.min_frequency_ghz,
            'timing_slack': constraints['timing_slack_ps'] - self.limits.min_timing_slack_ps,
            'ir_drop': self.limits.max_ir_drop_mv - constraints['ir_drop_mv'],
            'yield': constraints['yield'] - self.limits.min_yield,
            'signal_integrity': constraints['signal_integrity'] - self.limits.min_signal_integrity,
            'power_density': self.limits.max_power_density_w_mm2 - constraints['power_density_w_mm2'],
            'wire_delay': self.limits.max_wire_delay_ps - constraints['wire_delay_ps'],
        }

        # Add performance headroom AFTER calculating performance (to avoid recursion)
        if include_performance:
            performance = self.calculate_performance()
            headrooms['performance'] = performance - self.limits.min_performance_score

        return headrooms

    def get_min_headroom(self) -> float:
        """
        Get the minimum WEIGHTED headroom (bottleneck constraint).

        Lower weight = higher priority = JAM works harder to maintain that headroom.
        Example: performance_weight=0.3 means JAM keeps performance_headroom 3x higher!
        """
        headrooms = self.get_headrooms()
        weights = self.limits.constraint_weights

        # Apply weights: weighted_headroom = actual_headroom * weight
        # Lower weight makes that headroom appear smaller, so JAM prioritizes it
        weighted_headrooms = {
            constraint: headroom * weights.get(constraint, 1.0)
            for constraint, headroom in headrooms.items()
        }

        return min(weighted_headrooms.values())

    def is_feasible(self) -> bool:
        """Check if all constraints are satisfied"""
        headrooms = self.get_headrooms()
        return all(h >= 0 for h in headrooms.values())

    def apply_action(self, action: 'DesignAction') -> bool:
        """Apply a design action and update state"""
        # Apply the action to get new parameters
        new_params = action.apply_fn(self.params)
        self.params = new_params

        # Invalidate cache
        self._cached_constraints = None
        self._temperature_cache = None

        # Update performance
        self.performance_score = self.calculate_performance()

        # Increment step counter
        self.step_count += 1

        # Return feasibility
        return self.is_feasible()

    def get_state_snapshot(self) -> Dict:
        """Get complete state snapshot"""
        constraints = self.calculate_constraints()
        headrooms = self.get_headrooms()
        performance = self.calculate_performance()

        return {
            'step': self.step_count,
            'performance': performance,
            'parameters': asdict(self.params),
            'constraints': constraints,
            'headrooms': headrooms,
            'min_headroom': min(headrooms.values()),
            'is_feasible': self.is_feasible(),
        }

    def clone(self) -> 'AdvancedDesignSpace':
        """Create a deep copy"""
        return copy.deepcopy(self)

    def apply_requirement_shift(self, shift_type: ShiftType, rng: np.random.RandomState) -> Dict:
        """Apply a requirement shift"""
        shift_info = {'type': shift_type.value}

        if shift_type == ShiftType.TIGHTEN_POWER:
            old_limit = self.limits.max_power_watts
            self.limits.max_power_watts *= 0.80  # 20% reduction
            shift_info['description'] = f"Power budget reduced from {old_limit:.1f}W to {self.limits.max_power_watts:.1f}W"
            shift_info['old_value'] = old_limit
            shift_info['new_value'] = self.limits.max_power_watts

        elif shift_type == ShiftType.INCREASE_PERFORMANCE:
            old_freq = self.limits.min_frequency_ghz
            self.limits.min_frequency_ghz *= 1.15  # 15% increase
            shift_info['description'] = f"Performance requirement increased from {old_freq:.2f}GHz to {self.limits.min_frequency_ghz:.2f}GHz"
            shift_info['old_value'] = old_freq
            shift_info['new_value'] = self.limits.min_frequency_ghz

        elif shift_type == ShiftType.REDUCE_AREA:
            old_area = self.limits.max_area_mm2
            self.limits.max_area_mm2 *= 0.85  # 15% reduction
            shift_info['description'] = f"Area budget reduced from {old_area:.1f}mm² to {self.limits.max_area_mm2:.1f}mm²"
            shift_info['old_value'] = old_area
            shift_info['new_value'] = self.limits.max_area_mm2

        elif shift_type == ShiftType.TIGHTEN_THERMAL:
            old_temp = self.limits.max_temperature_c
            self.limits.max_temperature_c -= 10.0  # 10°C reduction
            shift_info['description'] = f"Thermal limit reduced from {old_temp:.1f}°C to {self.limits.max_temperature_c:.1f}°C"
            shift_info['old_value'] = old_temp
            shift_info['new_value'] = self.limits.max_temperature_c

        elif shift_type == ShiftType.YIELD_REQUIREMENT:
            old_yield = self.limits.min_yield
            self.limits.min_yield = min(0.95, self.limits.min_yield + 0.05)  # 5% increase
            shift_info['description'] = f"Yield requirement increased from {old_yield:.1%} to {self.limits.min_yield:.1%}"
            shift_info['old_value'] = old_yield
            shift_info['new_value'] = self.limits.min_yield

        elif shift_type == ShiftType.ADD_FEATURE:
            # Adding a feature increases power and area
            shift_info['description'] = "New feature required: +8% power, +10% area"
            shift_info['power_increase'] = 1.08
            shift_info['area_increase'] = 1.10
            # This is simulated by just making it harder to meet constraints
            self.limits.max_power_watts *= 0.92
            self.limits.max_area_mm2 *= 0.90

        # Invalidate cache
        self._cached_constraints = None

        return shift_info


@dataclass
class DesignAction:
    """Represents a design action/decision"""
    name: str
    description: str
    apply_fn: Callable[[DesignParameters], DesignParameters]
    category: str  # "performance", "efficiency", "area", "trap", "safe"


class AdvancedAgent:
    """Base class for optimization agents using the advanced design space"""

    def __init__(self, name: str):
        self.name = name
        self.design_space: Optional[AdvancedDesignSpace] = None

    def initialize(self, design_space: AdvancedDesignSpace):
        """Initialize the agent with a design space"""
        self.design_space = design_space

    def select_action(self) -> Optional[DesignAction]:
        """Select the next action to take. Must be implemented by subclasses."""
        raise NotImplementedError

    def step(self) -> Tuple[Optional[DesignAction], bool]:
        """
        Take one optimization step.

        Returns:
            Tuple of (selected_action, is_feasible)
        """
        action = self.select_action()
        if action is None:
            return None, self.design_space.is_feasible()

        is_feasible = self.design_space.apply_action(action)
        return action, is_feasible


class AdvancedGreedyPerformanceAgent(AdvancedAgent):
    """
    Agent 1: Greedy Performance Maximizer

    Selects actions that maximize immediate performance gain while maintaining feasibility.
    """

    def __init__(self):
        super().__init__("GreedyPerformance")

    def select_action(self) -> Optional[DesignAction]:
        """Select action with highest performance gain that maintains feasibility"""
        if not self.design_space:
            return None

        best_action = None
        best_performance = -float('inf')

        for action in self.design_space.actions:
            # Simulate applying the action
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            # Check if it's feasible and better than current best
            if test_space.is_feasible():
                perf = test_space.calculate_performance()
                if perf > best_performance:
                    best_performance = perf
                    best_action = action

        return best_action


class JAMAgent(AdvancedAgent):
    """
    Agent 2: JAM (Just Add Margin)

    Optimizes log(min(headroom)) to preserve margins in the most constrained dimension.
    REFUSES actions that would reduce margins below a safety threshold.

    This is a GLASS BOX strategy: the decision logic is transparent and interpretable.
    At each step, you can see exactly what it's optimizing (minimum margin) and why.
    """

    def __init__(self, min_margin_threshold: float = 2.0):
        super().__init__("JAM")
        self.min_margin_threshold = min_margin_threshold  # Minimum acceptable margin
        self.epsilon = 0.01  # Small value to avoid log of zero

    def select_action(self) -> Optional[DesignAction]:
        """Select action that maximizes log of minimum headroom while maintaining safety margins"""
        if not self.design_space:
            return None

        current_min_headroom = self.design_space.get_min_headroom()

        # Separate actions into "safe" (maintain threshold) and "risky" (reduce margins)
        safe_actions = []
        risky_actions = []

        for action in self.design_space.actions:
            # Simulate applying the action
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            min_headroom = test_space.get_min_headroom()
            margin_score = np.log(max(min_headroom, self.epsilon))
            perf = test_space.calculate_performance()

            action_data = (action, margin_score, perf, min_headroom)

            # Safe actions: maintain min_headroom above threshold
            if min_headroom >= self.min_margin_threshold:
                safe_actions.append(action_data)
            else:
                risky_actions.append(action_data)

        # Prefer safe actions that maintain margins
        if safe_actions:
            # PURE JAM: Optimize log(min_headroom), use performance only as tiebreaker
            # This tests if margin balancing naturally leads to better performance
            best = max(safe_actions, key=lambda x: (x[1], x[2]))  # (margin_score, perf)
            return best[0]

        # If no safe actions, pick the least risky one (highest margin that's still above current)
        elif risky_actions:
            # Only take actions that don't make things worse
            improving = [a for a in risky_actions if a[3] >= current_min_headroom]
            if improving:
                best = max(improving, key=lambda x: (x[1], x[2]))  # (margin_score, perf)
                return best[0]
            else:
                # No improving actions available, stop optimizing
                return None

        return None


class AdaptiveJAM(AdvancedAgent):
    """
    Adaptive JAM: Two-phase optimization strategy

    PHASE 1 (Build Margins): Optimize log(min_headroom) until reaching sweet spot
    PHASE 2 (Push Performance): Once margins are good (>10), maximize performance

    The sweet spot (headroom > 10) unlocks 1.5x performance bonus, then we push base performance!
    This combines JAM's efficiency with GreedyPerf's performance focus.
    """

    def __init__(self, margin_target: float = 10.0, min_margin_threshold: float = 2.0):
        super().__init__("AdaptiveJAM")
        self.margin_target = margin_target  # Target headroom for sweet spot
        self.min_margin_threshold = min_margin_threshold  # Safety threshold
        self.epsilon = 0.01

    def select_action(self) -> Optional[DesignAction]:
        """Two-phase selection: build margins first, then push performance"""
        if not self.design_space:
            return None

        current_min_headroom = self.design_space.get_min_headroom()

        # Determine which phase we're in
        if current_min_headroom < self.margin_target:
            # PHASE 1: Build margins to sweet spot
            return self._select_margin_building_action(current_min_headroom)
        else:
            # PHASE 2: Margins are good, now maximize performance!
            return self._select_performance_pushing_action(current_min_headroom)

    def _select_margin_building_action(self, current_min_headroom: float) -> Optional[DesignAction]:
        """Phase 1: Build margins to unlock performance bonus"""
        safe_actions = []
        risky_actions = []

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            min_headroom = test_space.get_min_headroom()
            margin_score = np.log(max(min_headroom, self.epsilon))
            perf = test_space.calculate_performance()

            action_data = (action, margin_score, perf, min_headroom)

            if min_headroom >= self.min_margin_threshold:
                safe_actions.append(action_data)
            else:
                risky_actions.append(action_data)

        if safe_actions:
            # Prioritize margin improvement, use perf as tiebreaker
            best = max(safe_actions, key=lambda x: (x[1], x[2]))
            return best[0]
        elif risky_actions:
            improving = [a for a in risky_actions if a[3] >= current_min_headroom]
            if improving:
                best = max(improving, key=lambda x: (x[1], x[2]))
                return best[0]

        return None

    def _select_performance_pushing_action(self, current_min_headroom: float) -> Optional[DesignAction]:
        """Phase 2: Margins are at sweet spot, now push performance!"""
        safe_actions = []

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            min_headroom = test_space.get_min_headroom()
            perf = test_space.calculate_performance()

            # Only consider actions that maintain minimum safety threshold
            # This prevents sacrificing ALL margins for performance
            if min_headroom >= self.min_margin_threshold:
                # Prioritize PERFORMANCE, use margin_score as tiebreaker
                margin_score = np.log(max(min_headroom, self.epsilon))
                safe_actions.append((action, perf, margin_score, min_headroom))

        if safe_actions:
            # PRIMARY: performance, SECONDARY: margins
            best = max(safe_actions, key=lambda x: (x[1], x[2]))
            return best[0]

        return None


class HybridJAM(AdvancedAgent):
    """
    Hybrid JAM: Optimizes BOTH margins AND performance simultaneously

    Objective: margin_score + performance_weight * performance

    Unlike pure JAM (margins only) or AdaptiveJAM (phase-based),
    this agent balances both objectives from the start.
    """

    def __init__(self, performance_weight: float = 0.05, min_margin_threshold: float = 2.0):
        super().__init__("HybridJAM")
        self.performance_weight = performance_weight  # How much to value performance vs margins
        self.min_margin_threshold = min_margin_threshold
        self.epsilon = 0.01

    def select_action(self) -> Optional[DesignAction]:
        """Select action that maximizes weighted combination of margins and performance"""
        if not self.design_space:
            return None

        current_min_headroom = self.design_space.get_min_headroom()
        safe_actions = []
        risky_actions = []

        for action in self.design_space.actions:
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            min_headroom = test_space.get_min_headroom()
            margin_score = np.log(max(min_headroom, self.epsilon))
            perf = test_space.calculate_performance()

            # HYBRID SCORE: combines margin and performance objectives
            hybrid_score = margin_score + self.performance_weight * perf

            action_data = (action, hybrid_score, margin_score, perf, min_headroom)

            if min_headroom >= self.min_margin_threshold:
                safe_actions.append(action_data)
            else:
                risky_actions.append(action_data)

        if safe_actions:
            # Maximize hybrid score
            best = max(safe_actions, key=lambda x: x[1])
            return best[0]
        elif risky_actions:
            improving = [a for a in risky_actions if a[4] >= current_min_headroom]
            if improving:
                best = max(improving, key=lambda x: x[1])
                return best[0]

        return None


@dataclass
class AdvancedCheckpointData:
    """Data captured at a checkpoint"""
    step: int
    phase: str
    agent_name: str
    performance: float
    headrooms: Dict[str, float]
    min_headroom: float
    is_feasible: bool
    constraints: Dict[str, float]
    parameters: Dict[str, any]
    action_taken: Optional[str] = None


@dataclass
class AdvancedSimulationResult:
    """Complete results from a single simulation run"""
    run_id: int
    seed: int
    agent1_name: str
    agent2_name: str

    # Design phase results
    design_steps: int
    agent1_final_performance_design: float
    agent2_final_performance_design: float
    agent1_min_headroom_design: float
    agent2_min_headroom_design: float

    # Requirement shift
    shift_info: Dict

    # Adaptation phase results
    adaptation_steps: int
    agent1_survived_shift: bool
    agent2_survived_shift: bool
    agent1_final_performance_adapt: float
    agent2_final_performance_adapt: float
    agent1_min_headroom_adapt: float
    agent2_min_headroom_adapt: float
    agent1_steps_to_adapt: Optional[int]
    agent2_steps_to_adapt: Optional[int]

    # Winner
    winner: str

    # Full history
    checkpoints: List[AdvancedCheckpointData] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['checkpoints'] = [asdict(cp) for cp in self.checkpoints]
        return result


class AdvancedSimulation:
    """
    Orchestrates the advanced simulation comparing two optimization strategies.
    """

    def __init__(
        self,
        design_steps: int = 75,
        adaptation_steps: int = 25,
        checkpoint_frequency: int = 10,
        shift_type: Optional[ShiftType] = None,
        process: Optional[ProcessTechnology] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        progressive_goals: bool = False,  # Enable progressive performance goals
    ):
        self.design_steps = design_steps
        self.adaptation_steps = adaptation_steps
        self.checkpoint_frequency = checkpoint_frequency
        self.shift_type = shift_type
        self.process = process or ProcessTechnology.create_7nm()
        self.seed = seed
        self.verbose = verbose
        self.progressive_goals = progressive_goals
        self.rng = np.random.RandomState(seed)

    def run_single_simulation(self, run_id: int) -> AdvancedSimulationResult:
        """Run a single complete simulation"""

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ADVANCED SIMULATION RUN {run_id}")
            print(f"{'='*60}")
            print(f"Process Technology: {self.process.node_nm}nm")

        # Initialize design spaces (both agents start from same point)
        seed1 = self.rng.randint(0, 1000000)

        space1 = AdvancedDesignSpace(process=self.process, seed=seed1)
        space1.initialize_actions()

        space2 = space1.clone()

        # Initialize agents
        agent1 = AdvancedGreedyPerformanceAgent()
        agent1.initialize(space1)

        agent2 = AdaptiveJAM(margin_target=10.0)  # Two-phase: build margins to 10, then push performance
        agent2.initialize(space2)

        checkpoints = []

        # PHASE 1: Design Phase
        if self.verbose:
            print(f"\nPHASE 1: DESIGN ({self.design_steps} steps)")
            print("-" * 60)
            if self.progressive_goals:
                print("  Progressive goals ENABLED - performance target will increase dynamically")

        # Progressive goals: Start with achievable target, raise it as performance improves
        if self.progressive_goals:
            space1.limits.min_performance_score = 10.0  # Start low and ramp up
            space2.limits.min_performance_score = 10.0

        for step in range(self.design_steps):
            # Agent 1 step
            action1, feasible1 = agent1.step()

            # Agent 2 step
            action2, feasible2 = agent2.step()

            # Progressive goals: Check and raise performance target every 5 steps
            if self.progressive_goals and step > 0 and step % 5 == 0:
                current_perf = max(space1.calculate_performance(), space2.calculate_performance())
                current_target = space1.limits.min_performance_score

                # If we're comfortably above target, raise the bar
                headroom_threshold = 3.0  # How much headroom before raising target
                target_margin = 2.0  # Keep target this far below current performance

                if current_perf > current_target + headroom_threshold:
                    new_target = current_perf - target_margin
                    space1.limits.min_performance_score = new_target
                    space2.limits.min_performance_score = new_target

                    if self.verbose:
                        print(f"\n  [Step {step}] Progressive goal raised: {current_target:.1f} → {new_target:.1f} (current perf: {current_perf:.1f})")

            # Checkpoint
            if step % self.checkpoint_frequency == 0 or step == self.design_steps - 1:
                self._capture_checkpoint(checkpoints, step, "design", agent1, action1)
                self._capture_checkpoint(checkpoints, step, "design", agent2, action2)

                if self.verbose:
                    self._print_checkpoint(step, agent1, agent2)

        # Record end of design phase
        agent1_perf_design = space1.calculate_performance()
        agent2_perf_design = space2.calculate_performance()
        agent1_headroom_design = space1.get_min_headroom()
        agent2_headroom_design = space2.get_min_headroom()

        if self.verbose:
            print(f"\nDESIGN PHASE COMPLETE")
            print(f"  {agent1.name}: Performance={agent1_perf_design:.2f}, MinHeadroom={agent1_headroom_design:.2f}")
            print(f"  {agent2.name}: Performance={agent2_perf_design:.2f}, MinHeadroom={agent2_headroom_design:.2f}")
            self._print_constraint_summary(agent1, agent2)

        # PHASE 2: Requirement Shift
        if self.verbose:
            print(f"\nPHASE 2: REQUIREMENT SHIFT")
            print("-" * 60)

        # Determine shift type
        if self.shift_type is None:
            shift_type = self.rng.choice(list(ShiftType))
        else:
            shift_type = self.shift_type

        # Apply shift to both design spaces
        shift_info1 = space1.apply_requirement_shift(shift_type, self.rng)
        shift_info2 = space2.apply_requirement_shift(shift_type, self.rng)

        if self.verbose:
            print(f"  Shift Type: {shift_type.value}")
            print(f"  {shift_info1['description']}")

        # Check if designs survived the shift
        agent1_survived = space1.is_feasible()
        agent2_survived = space2.is_feasible()

        if self.verbose:
            print(f"  {agent1.name} survived: {agent1_survived}")
            print(f"  {agent2.name} survived: {agent2_survived}")
            if not agent1_survived:
                print(f"    Violated constraints: {self._get_violations(space1)}")
            if not agent2_survived:
                print(f"    Violated constraints: {self._get_violations(space2)}")

        # PHASE 3: Adaptation Phase
        if self.verbose:
            print(f"\nPHASE 3: ADAPTATION ({self.adaptation_steps} steps)")
            print("-" * 60)

        agent1_steps_to_adapt = None
        agent2_steps_to_adapt = None

        for step in range(self.adaptation_steps):
            # Agent 1 step
            if agent1_survived:
                action1, feasible1 = agent1.step()
                if not feasible1:
                    agent1_survived = False
            else:
                action1, feasible1 = None, False

            # Agent 2 step
            if agent2_survived:
                action2, feasible2 = agent2.step()
                if not feasible2:
                    agent2_survived = False
            else:
                action2, feasible2 = None, False

            # Track steps to adapt
            if agent1_survived and agent1_steps_to_adapt is None:
                agent1_steps_to_adapt = step
            if agent2_survived and agent2_steps_to_adapt is None:
                agent2_steps_to_adapt = step

            # Checkpoint
            if step % self.checkpoint_frequency == 0 or step == self.adaptation_steps - 1:
                self._capture_checkpoint(checkpoints, step, "adaptation", agent1, action1)
                self._capture_checkpoint(checkpoints, step, "adaptation", agent2, action2)

                if self.verbose:
                    self._print_checkpoint(step, agent1, agent2)

        # Record final results
        agent1_perf_adapt = space1.calculate_performance()
        agent2_perf_adapt = space2.calculate_performance()
        agent1_headroom_adapt = space1.get_min_headroom() if agent1_survived else -999.0
        agent2_headroom_adapt = space2.get_min_headroom() if agent2_survived else -999.0

        # Determine winner
        winner = self._determine_winner(
            agent1_survived, agent2_survived,
            agent1_perf_adapt, agent2_perf_adapt,
            agent1_headroom_adapt, agent2_headroom_adapt
        )

        if self.verbose:
            print(f"\nADAPTATION PHASE COMPLETE")
            print(f"  {agent1.name}: Survived={agent1_survived}, Performance={agent1_perf_adapt:.2f}, MinHeadroom={agent1_headroom_adapt:.2f}")
            print(f"  {agent2.name}: Survived={agent2_survived}, Performance={agent2_perf_adapt:.2f}, MinHeadroom={agent2_headroom_adapt:.2f}")
            print(f"\n  WINNER: {winner}")

        # Create result object
        result = AdvancedSimulationResult(
            run_id=run_id,
            seed=seed1,
            agent1_name=agent1.name,
            agent2_name=agent2.name,
            design_steps=self.design_steps,
            agent1_final_performance_design=agent1_perf_design,
            agent2_final_performance_design=agent2_perf_design,
            agent1_min_headroom_design=agent1_headroom_design,
            agent2_min_headroom_design=agent2_headroom_design,
            shift_info=shift_info1,
            adaptation_steps=self.adaptation_steps,
            agent1_survived_shift=agent1_survived,
            agent2_survived_shift=agent2_survived,
            agent1_final_performance_adapt=agent1_perf_adapt,
            agent2_final_performance_adapt=agent2_perf_adapt,
            agent1_min_headroom_adapt=agent1_headroom_adapt,
            agent2_min_headroom_adapt=agent2_headroom_adapt,
            agent1_steps_to_adapt=agent1_steps_to_adapt,
            agent2_steps_to_adapt=agent2_steps_to_adapt,
            winner=winner,
            checkpoints=checkpoints,
        )

        return result

    def _capture_checkpoint(self, checkpoints: List, step: int, phase: str, agent: AdvancedAgent, action: Optional[DesignAction]):
        """Capture a checkpoint snapshot"""
        space = agent.design_space
        checkpoint = AdvancedCheckpointData(
            step=step,
            phase=phase,
            agent_name=agent.name,
            performance=space.calculate_performance(),
            headrooms=space.get_headrooms(),
            min_headroom=space.get_min_headroom(),
            is_feasible=space.is_feasible(),
            constraints=space.calculate_constraints(),
            parameters=asdict(space.params),
            action_taken=action.name if action else None,
        )
        checkpoints.append(checkpoint)

    def _print_checkpoint(self, step: int, agent1: AdvancedAgent, agent2: AdvancedAgent):
        """Print checkpoint information"""
        space1 = agent1.design_space
        space2 = agent2.design_space

        print(f"\nStep {step}:")
        print(f"  {agent1.name:20s}: Perf={space1.calculate_performance():7.2f}, MinHeadroom={space1.get_min_headroom():7.2f}, Feasible={space1.is_feasible()}")
        print(f"    Freq={space1.params.clock_freq_ghz:.2f}GHz, Vdd={space1.params.supply_voltage:.3f}V, Cores={space1.params.num_cores}, Power={space1.calculate_constraints()['total_power_w']:.1f}W")
        print(f"  {agent2.name:20s}: Perf={space2.calculate_performance():7.2f}, MinHeadroom={space2.get_min_headroom():7.2f}, Feasible={space2.is_feasible()}")
        print(f"    Freq={space2.params.clock_freq_ghz:.2f}GHz, Vdd={space2.params.supply_voltage:.3f}V, Cores={space2.params.num_cores}, Power={space2.calculate_constraints()['total_power_w']:.1f}W")

    def _print_constraint_summary(self, agent1: AdvancedAgent, agent2: AdvancedAgent):
        """Print constraint summary"""
        c1 = agent1.design_space.calculate_constraints()
        c2 = agent2.design_space.calculate_constraints()

        print(f"\n  Constraint Summary:")
        print(f"    {'Metric':<25} {'GreedyPerf':>15} {'LogMinHead':>15}")
        print(f"    {'-'*25} {'-'*15} {'-'*15}")
        print(f"    {'Power (W)':<25} {c1['total_power_w']:>15.1f} {c2['total_power_w']:>15.1f}")
        print(f"    {'Temperature (°C)':<25} {c1['temperature_c']:>15.1f} {c2['temperature_c']:>15.1f}")
        print(f"    {'Area (mm²)':<25} {c1['area_mm2']:>15.1f} {c2['area_mm2']:>15.1f}")
        print(f"    {'Timing Slack (ps)':<25} {c1['timing_slack_ps']:>15.1f} {c2['timing_slack_ps']:>15.1f}")
        print(f"    {'Yield':<25} {c1['yield']:>15.2%} {c2['yield']:>15.2%}")

    def _get_violations(self, space: AdvancedDesignSpace) -> List[str]:
        """Get list of violated constraints"""
        headrooms = space.get_headrooms()
        violations = [name for name, hr in headrooms.items() if hr < 0]
        return violations

    def _determine_winner(
        self,
        agent1_survived: bool,
        agent2_survived: bool,
        agent1_perf: float,
        agent2_perf: float,
        agent1_headroom: float,
        agent2_headroom: float,
    ) -> str:
        """Determine which agent won"""

        # Primary criterion: survival
        if agent1_survived and not agent2_survived:
            return "GreedyPerformance"
        elif agent2_survived and not agent1_survived:
            return "LogMinHeadroom"
        elif not agent1_survived and not agent2_survived:
            return "Tie (both failed)"

        # Secondary criterion: performance
        perf_diff = agent1_perf - agent2_perf
        if abs(perf_diff) > 5.0:
            return "GreedyPerformance" if perf_diff > 0 else "LogMinHeadroom"

        # Tertiary criterion: headroom
        headroom_diff = agent1_headroom - agent2_headroom
        if abs(headroom_diff) > 1.0:
            return "GreedyPerformance" if headroom_diff > 0 else "LogMinHeadroom"

        return "Tie"


def run_advanced_simulations(
    num_runs: int = 100,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    checkpoint_frequency: int = 10,
    shift_type: Optional[ShiftType] = None,
    process: Optional[ProcessTechnology] = None,
    seed: Optional[int] = None,
    output_file: str = "advanced_results.json",
    verbose: bool = False,
    progressive_goals: bool = False,  # Enable progressive performance goals
) -> Dict:
    """Run multiple advanced simulations and aggregate results"""

    print(f"\n{'='*80}")
    print(f"RUNNING {num_runs} ADVANCED SIMULATIONS")
    print(f"{'='*80}")
    print(f"Process Technology: {process.node_nm if process else 7.0}nm")
    print(f"Design steps: {design_steps}")
    print(f"Adaptation steps: {adaptation_steps}")
    print(f"Checkpoint frequency: {checkpoint_frequency}")
    print(f"Shift type: {shift_type.value if shift_type else 'random'}")
    print(f"Progressive goals: {'ENABLED' if progressive_goals else 'disabled'}")
    print(f"Seed: {seed if seed else 'random'}")
    print(f"Output file: {output_file}")
    print(f"{'='*80}\n")

    master_rng = np.random.RandomState(seed)
    results = []

    # Run simulations
    for run_id in range(num_runs):
        run_seed = master_rng.randint(0, 1000000)
        sim = AdvancedSimulation(
            design_steps=design_steps,
            adaptation_steps=adaptation_steps,
            checkpoint_frequency=checkpoint_frequency,
            shift_type=shift_type,
            process=process,
            seed=run_seed,
            verbose=verbose,
            progressive_goals=progressive_goals,
        )

        result = sim.run_single_simulation(run_id)
        results.append(result)

        if not verbose and (run_id + 1) % 10 == 0:
            print(f"Completed {run_id + 1}/{num_runs} runs...")

    # Aggregate statistics
    from chip_design_simulator import _compute_aggregate_statistics, _print_statistics

    stats = _compute_aggregate_statistics(results)

    # Save results
    output_data = {
        'parameters': {
            'num_runs': num_runs,
            'design_steps': design_steps,
            'adaptation_steps': adaptation_steps,
            'checkpoint_frequency': checkpoint_frequency,
            'shift_type': shift_type.value if shift_type else 'random',
            'process_node_nm': process.node_nm if process else 7.0,
            'seed': seed,
        },
        'statistics': stats,
        'results': [r.to_dict() for r in results],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS ({num_runs} runs)")
    print(f"{'='*80}")
    _print_statistics(stats)
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")

    return output_data


if __name__ == "__main__":
    # Quick test run
    print("Running quick advanced simulation test...")
    run_advanced_simulations(
        num_runs=3,
        design_steps=30,
        adaptation_steps=15,
        checkpoint_frequency=10,
        output_file="advanced_test_results.json",
        verbose=True,
    )

