"""
Chip Design Optimization Simulator

This module simulates two competing optimization strategies for engineering design:
1. Greedy Performance Maximizer: Directly optimizes performance while staying feasible
2. Log-Min-Headroom Optimizer: Optimizes for maintaining margin in the most constrained dimension

The simulation tests the hypothesis that optimizing for constraint headroom produces
more adaptable designs when requirements change.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import copy


class ShiftType(Enum):
    """Types of requirement shifts that can occur"""
    TIGHTEN_CONSTRAINT = "tighten_constraint"
    PERFORMANCE_INCREASE = "performance_increase"
    ADD_CONSTRAINT = "add_constraint"


@dataclass
class ConstraintDimension:
    """Represents a single constraint dimension in the design space"""
    name: str
    floor: float  # Hard limit (cannot go below this)
    current_value: float  # Current value
    initial_floor: float = field(init=False)

    def __post_init__(self):
        self.initial_floor = self.floor
        if self.current_value < self.floor:
            raise ValueError(f"{self.name}: current_value ({self.current_value}) cannot be below floor ({self.floor})")

    @property
    def headroom(self) -> float:
        """How much margin remains above the floor"""
        return self.current_value - self.floor

    def is_feasible(self) -> bool:
        """Check if constraint is satisfied"""
        return self.current_value >= self.floor

    def apply_delta(self, delta: float) -> None:
        """Apply a change to the current value"""
        self.current_value += delta

    def tighten_floor(self, percentage: float) -> None:
        """Tighten the floor constraint by a percentage"""
        reduction = self.floor * percentage
        self.floor += reduction


@dataclass
class Action:
    """
    Represents a design action that modifies multiple dimensions simultaneously.

    Actions create realistic tradeoffs - improving one dimension often costs another.
    """
    name: str
    performance_delta: float
    dimension_deltas: Dict[str, float]
    description: str = ""

    def __str__(self):
        return f"{self.name}: perf={self.performance_delta:+.2f}, dims={self.dimension_deltas}"


class DesignSpace:
    """
    Manages the design space with multiple constraint dimensions and available actions.

    The design space includes:
    - Multiple constraint dimensions (heat, power, area, latency, etc.)
    - A performance metric (separate from constraints)
    - A set of available actions that create tradeoffs
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.constraints: Dict[str, ConstraintDimension] = {}
        self.performance: float = 0.0
        self.initial_performance: float = 0.0
        self.actions: List[Action] = []
        self.step_count: int = 0

    def initialize_default_constraints(self):
        """Initialize a realistic chip design space with 8 constraint dimensions"""
        self.constraints = {
            'heat_dissipation': ConstraintDimension('heat_dissipation', floor=20.0, current_value=100.0),
            'power_budget': ConstraintDimension('power_budget', floor=50.0, current_value=200.0),
            'area': ConstraintDimension('area', floor=10.0, current_value=100.0),
            'signal_latency': ConstraintDimension('signal_latency', floor=5.0, current_value=50.0),
            'manufacturing_tolerance': ConstraintDimension('manufacturing_tolerance', floor=2.0, current_value=20.0),
            'memory_bandwidth': ConstraintDimension('memory_bandwidth', floor=100.0, current_value=500.0),
            'voltage_margin': ConstraintDimension('voltage_margin', floor=0.1, current_value=1.0),
            'timing_slack': ConstraintDimension('timing_slack', floor=1.0, current_value=10.0),
        }
        self.performance = 100.0
        self.initial_performance = 100.0

    def initialize_actions(self):
        """
        Create a diverse set of actions including:
        - Balanced actions (moderate tradeoffs)
        - Trap actions (great performance but crush one dimension)
        - Safe actions (preserve headroom but lower performance gains)
        """
        self.actions = [
            # BALANCED ACTIONS - reasonable tradeoffs
            Action(
                "optimize_cache",
                performance_delta=8.0,
                dimension_deltas={
                    'heat_dissipation': -3.0,
                    'power_budget': -5.0,
                    'area': -2.0,
                    'memory_bandwidth': +5.0,
                }
            ),
            Action(
                "increase_frequency",
                performance_delta=12.0,
                dimension_deltas={
                    'heat_dissipation': -8.0,
                    'power_budget': -10.0,
                    'timing_slack': -4.0,
                    'signal_latency': +2.0,
                }
            ),
            Action(
                "add_pipeline_stage",
                performance_delta=6.0,
                dimension_deltas={
                    'signal_latency': -3.0,
                    'area': -4.0,
                    'power_budget': -3.0,
                    'timing_slack': +3.0,
                }
            ),
            Action(
                "widen_datapath",
                performance_delta=10.0,
                dimension_deltas={
                    'area': -6.0,
                    'power_budget': -7.0,
                    'memory_bandwidth': +4.0,
                    'heat_dissipation': -4.0,
                }
            ),

            # TRAP ACTIONS - great performance but crush one dimension
            Action(
                "aggressive_voltage_boost",
                performance_delta=20.0,  # Great performance!
                dimension_deltas={
                    'voltage_margin': -0.7,  # But crushes voltage margin!
                    'heat_dissipation': -10.0,
                    'power_budget': -15.0,
                    'timing_slack': +5.0,
                },
                description="TRAP: High performance but crushes voltage margin"
            ),
            Action(
                "dense_placement",
                performance_delta=15.0,  # Great performance!
                dimension_deltas={
                    'area': -8.0,
                    'manufacturing_tolerance': -0.8,  # But crushes manufacturing tolerance!
                    'heat_dissipation': -12.0,
                    'timing_slack': -3.0,
                },
                description="TRAP: Good performance but crushes manufacturing tolerance"
            ),
            Action(
                "max_parallelization",
                performance_delta=25.0,  # Excellent performance!
                dimension_deltas={
                    'power_budget': -25.0,  # But crushes power budget!
                    'heat_dissipation': -20.0,
                    'area': -15.0,
                    'memory_bandwidth': +8.0,
                },
                description="TRAP: Excellent performance but crushes power budget"
            ),

            # SAFE ACTIONS - preserve headroom, lower performance gains
            Action(
                "efficiency_tuning",
                performance_delta=4.0,  # Lower performance
                dimension_deltas={
                    'heat_dissipation': +2.0,  # But improves constraints
                    'power_budget': +3.0,
                    'voltage_margin': +0.1,
                    'timing_slack': +1.0,
                },
                description="SAFE: Lower performance but improves margins"
            ),
            Action(
                "conservative_optimization",
                performance_delta=5.0,
                dimension_deltas={
                    'heat_dissipation': -1.0,
                    'power_budget': -2.0,
                    'manufacturing_tolerance': +0.2,
                    'timing_slack': +2.0,
                },
                description="SAFE: Modest performance with good margins"
            ),
            Action(
                "selective_buffering",
                performance_delta=7.0,
                dimension_deltas={
                    'signal_latency': +3.0,
                    'area': -3.0,
                    'power_budget': -4.0,
                    'timing_slack': +4.0,
                },
                description="SAFE: Good timing slack preservation"
            ),

            # MIXED ACTIONS - some interesting tradeoffs
            Action(
                "algorithm_improvement",
                performance_delta=9.0,
                dimension_deltas={
                    'memory_bandwidth': +6.0,
                    'signal_latency': +2.0,
                    'power_budget': -3.0,
                },
                description="Algorithmic efficiency gains"
            ),
            Action(
                "thermal_optimization",
                performance_delta=3.0,
                dimension_deltas={
                    'heat_dissipation': +8.0,
                    'power_budget': +2.0,
                    'area': -5.0,
                    'performance': -2.0,
                },
                description="Focus on thermal management"
            ),
        ]

    def apply_action(self, action: Action) -> bool:
        """
        Apply an action to the design space.

        Returns:
            True if action results in feasible design, False otherwise
        """
        # Apply deltas
        self.performance += action.performance_delta

        for dim_name, delta in action.dimension_deltas.items():
            if dim_name in self.constraints:
                self.constraints[dim_name].apply_delta(delta)

        self.step_count += 1

        # Check feasibility
        return self.is_feasible()

    def is_feasible(self) -> bool:
        """Check if all constraints are satisfied"""
        return all(c.is_feasible() for c in self.constraints.values())

    def get_headrooms(self) -> Dict[str, float]:
        """Get headroom for all constraints"""
        return {name: c.headroom for name, c in self.constraints.items()}

    def get_min_headroom(self) -> float:
        """Get the minimum headroom across all constraints"""
        headrooms = [c.headroom for c in self.constraints.values()]
        return min(headrooms) if headrooms else 0.0

    def get_state_snapshot(self) -> Dict:
        """Get a complete snapshot of the current state"""
        return {
            'performance': self.performance,
            'step_count': self.step_count,
            'constraints': {
                name: {
                    'current_value': c.current_value,
                    'floor': c.floor,
                    'headroom': c.headroom,
                }
                for name, c in self.constraints.items()
            },
            'min_headroom': self.get_min_headroom(),
            'is_feasible': self.is_feasible(),
        }

    def clone(self) -> 'DesignSpace':
        """Create a deep copy of the design space"""
        return copy.deepcopy(self)

    def apply_requirement_shift(self, shift_type: ShiftType, rng: np.random.RandomState) -> Dict:
        """
        Apply a requirement shift to test adaptability.

        Returns:
            Dictionary describing the shift that was applied
        """
        shift_info = {'type': shift_type.value}

        if shift_type == ShiftType.TIGHTEN_CONSTRAINT:
            # Pick a random constraint and tighten its floor by 20%
            constraint_name = rng.choice(list(self.constraints.keys()))
            constraint = self.constraints[constraint_name]
            old_floor = constraint.floor
            constraint.tighten_floor(0.20)
            shift_info['constraint'] = constraint_name
            shift_info['old_floor'] = old_floor
            shift_info['new_floor'] = constraint.floor
            shift_info['description'] = f"Tightened {constraint_name} floor from {old_floor:.2f} to {constraint.floor:.2f}"

        elif shift_type == ShiftType.PERFORMANCE_INCREASE:
            # Increase performance requirement
            old_perf = self.performance
            increase = old_perf * 0.15
            shift_info['old_performance'] = old_perf
            shift_info['required_increase'] = increase
            shift_info['new_target'] = old_perf + increase
            shift_info['description'] = f"Performance requirement increased by {increase:.2f}"

        elif shift_type == ShiftType.ADD_CONSTRAINT:
            # Add a new constraint dimension
            new_constraint_name = f"new_constraint_{len(self.constraints)}"
            # Set floor at 80% of initial value to make it challenging
            initial_value = 50.0
            floor = initial_value * 0.8
            self.constraints[new_constraint_name] = ConstraintDimension(
                new_constraint_name,
                floor=floor,
                current_value=initial_value
            )
            shift_info['constraint'] = new_constraint_name
            shift_info['floor'] = floor
            shift_info['initial_value'] = initial_value
            shift_info['description'] = f"Added new constraint {new_constraint_name} with floor={floor:.2f}"

        return shift_info


class Agent:
    """Base class for optimization agents"""

    def __init__(self, name: str):
        self.name = name
        self.design_space: Optional[DesignSpace] = None

    def initialize(self, design_space: DesignSpace):
        """Initialize the agent with a design space"""
        self.design_space = design_space

    def select_action(self) -> Optional[Action]:
        """Select the next action to take. Must be implemented by subclasses."""
        raise NotImplementedError

    def step(self) -> Tuple[Optional[Action], bool]:
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


class GreedyPerformanceAgent(Agent):
    """
    Agent 1: Greedy Performance Maximizer

    Objective: max(performance) subject to all constraints >= floors

    This agent always selects the action that maximizes immediate performance gain
    while maintaining feasibility. It represents the standard optimization approach.
    """

    def __init__(self):
        super().__init__("GreedyPerformance")

    def select_action(self) -> Optional[Action]:
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
            if test_space.is_feasible() and test_space.performance > best_performance:
                best_performance = test_space.performance
                best_action = action

        return best_action


class LogMinHeadroomAgent(Agent):
    """
    Agent 2: Log-Min-Headroom Optimizer

    Objective: max(log(min(headroom_1, headroom_2, ...headroom_n)))
    Secondary: among equal log-min scores, prefer higher performance

    This agent optimizes for preserving options in the most constrained dimension,
    hypothetically leading to more adaptable designs.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__("LogMinHeadroom")
        self.epsilon = epsilon  # Small value to avoid log(0)

    def select_action(self) -> Optional[Action]:
        """Select action that maximizes log of minimum headroom"""
        if not self.design_space:
            return None

        best_action = None
        best_score = -float('inf')
        best_performance = -float('inf')

        for action in self.design_space.actions:
            # Simulate applying the action
            test_space = self.design_space.clone()
            test_space.apply_action(action)

            if not test_space.is_feasible():
                continue

            # Calculate log(min_headroom) score
            min_headroom = test_space.get_min_headroom()
            score = np.log(max(min_headroom, self.epsilon))

            # Select based on score, with performance as tiebreaker
            if score > best_score or (abs(score - best_score) < 1e-9 and test_space.performance > best_performance):
                best_score = score
                best_performance = test_space.performance
                best_action = action

        return best_action


@dataclass
class CheckpointData:
    """Data captured at a checkpoint during simulation"""
    step: int
    phase: str
    agent_name: str
    performance: float
    headrooms: Dict[str, float]
    min_headroom: float
    is_feasible: bool
    action_taken: Optional[str] = None


@dataclass
class SimulationResult:
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

    # Overall winner
    winner: str

    # Full history
    checkpoints: List[CheckpointData] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert checkpoints to simpler format
        result['checkpoints'] = [asdict(cp) for cp in self.checkpoints]
        return result


class Simulation:
    """
    Orchestrates the complete simulation comparing two optimization strategies.

    Phases:
    1. Design Phase: Both agents optimize from same starting point
    2. Requirement Shift: Change constraints or requirements
    3. Adaptation Phase: Both agents adapt to new requirements
    """

    def __init__(
        self,
        design_steps: int = 75,
        adaptation_steps: int = 25,
        checkpoint_frequency: int = 10,
        shift_type: Optional[ShiftType] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.design_steps = design_steps
        self.adaptation_steps = adaptation_steps
        self.checkpoint_frequency = checkpoint_frequency
        self.shift_type = shift_type
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

    def run_single_simulation(self, run_id: int) -> SimulationResult:
        """Run a single complete simulation"""

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SIMULATION RUN {run_id}")
            print(f"{'='*60}")

        # Initialize design spaces (both agents start from same point)
        seed1 = self.rng.randint(0, 1000000)
        seed2 = seed1  # Same seed for same initial state

        space1 = DesignSpace(seed=seed1)
        space1.initialize_default_constraints()
        space1.initialize_actions()

        space2 = space1.clone()

        # Initialize agents
        agent1 = GreedyPerformanceAgent()
        agent1.initialize(space1)

        agent2 = LogMinHeadroomAgent()
        agent2.initialize(space2)

        checkpoints = []

        # PHASE 1: Design Phase
        if self.verbose:
            print(f"\nPHASE 1: DESIGN ({self.design_steps} steps)")
            print("-" * 60)

        for step in range(self.design_steps):
            # Agent 1 step
            action1, feasible1 = agent1.step()

            # Agent 2 step
            action2, feasible2 = agent2.step()

            # Checkpoint
            if step % self.checkpoint_frequency == 0 or step == self.design_steps - 1:
                self._capture_checkpoint(checkpoints, step, "design", agent1, action1)
                self._capture_checkpoint(checkpoints, step, "design", agent2, action2)

                if self.verbose:
                    self._print_checkpoint(step, agent1, agent2)

        # Record end of design phase
        agent1_perf_design = space1.performance
        agent2_perf_design = space2.performance
        agent1_headroom_design = space1.get_min_headroom()
        agent2_headroom_design = space2.get_min_headroom()

        if self.verbose:
            print(f"\nDESIGN PHASE COMPLETE")
            print(f"  {agent1.name}: Performance={agent1_perf_design:.2f}, MinHeadroom={agent1_headroom_design:.2f}")
            print(f"  {agent2.name}: Performance={agent2_perf_design:.2f}, MinHeadroom={agent2_headroom_design:.2f}")

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

            # Track steps to adapt (when they become feasible again if they weren't)
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
        agent1_perf_adapt = space1.performance
        agent2_perf_adapt = space2.performance
        agent1_headroom_adapt = space1.get_min_headroom() if agent1_survived else 0.0
        agent2_headroom_adapt = space2.get_min_headroom() if agent2_survived else 0.0

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
        result = SimulationResult(
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

    def _capture_checkpoint(self, checkpoints: List, step: int, phase: str, agent: Agent, action: Optional[Action]):
        """Capture a checkpoint snapshot"""
        checkpoint = CheckpointData(
            step=step,
            phase=phase,
            agent_name=agent.name,
            performance=agent.design_space.performance,
            headrooms=agent.design_space.get_headrooms(),
            min_headroom=agent.design_space.get_min_headroom(),
            is_feasible=agent.design_space.is_feasible(),
            action_taken=action.name if action else None,
        )
        checkpoints.append(checkpoint)

    def _print_checkpoint(self, step: int, agent1: Agent, agent2: Agent):
        """Print checkpoint information"""
        print(f"\nStep {step}:")
        print(f"  {agent1.name:20s}: Perf={agent1.design_space.performance:7.2f}, MinHeadroom={agent1.design_space.get_min_headroom():7.2f}, Feasible={agent1.design_space.is_feasible()}")
        print(f"  {agent2.name:20s}: Perf={agent2.design_space.performance:7.2f}, MinHeadroom={agent2.design_space.get_min_headroom():7.2f}, Feasible={agent2.design_space.is_feasible()}")

    def _determine_winner(
        self,
        agent1_survived: bool,
        agent2_survived: bool,
        agent1_perf: float,
        agent2_perf: float,
        agent1_headroom: float,
        agent2_headroom: float,
    ) -> str:
        """Determine which agent won based on multiple criteria"""

        # Primary criterion: survival
        if agent1_survived and not agent2_survived:
            return "GreedyPerformance"
        elif agent2_survived and not agent1_survived:
            return "LogMinHeadroom"
        elif not agent1_survived and not agent2_survived:
            return "Tie (both failed)"

        # Secondary criterion: performance
        perf_diff = agent1_perf - agent2_perf
        if abs(perf_diff) > 5.0:  # Significant difference
            return "GreedyPerformance" if perf_diff > 0 else "LogMinHeadroom"

        # Tertiary criterion: headroom
        headroom_diff = agent1_headroom - agent2_headroom
        if abs(headroom_diff) > 1.0:  # Significant difference
            return "GreedyPerformance" if headroom_diff > 0 else "LogMinHeadroom"

        return "Tie"


def run_multiple_simulations(
    num_runs: int = 100,
    design_steps: int = 75,
    adaptation_steps: int = 25,
    checkpoint_frequency: int = 10,
    shift_type: Optional[ShiftType] = None,
    seed: Optional[int] = None,
    output_file: str = "simulation_results.json",
    verbose: bool = False,
) -> Dict:
    """
    Run multiple simulation runs and aggregate results.

    Args:
        num_runs: Number of simulation runs to perform
        design_steps: Steps in the design phase
        adaptation_steps: Steps in the adaptation phase
        checkpoint_frequency: How often to capture checkpoints
        shift_type: Type of requirement shift (None for random)
        seed: Random seed for reproducibility
        output_file: Path to save results JSON
        verbose: Whether to print detailed output for each run

    Returns:
        Dictionary containing aggregate results and statistics
    """

    print(f"\n{'='*80}")
    print(f"RUNNING {num_runs} SIMULATIONS")
    print(f"{'='*80}")
    print(f"Design steps: {design_steps}")
    print(f"Adaptation steps: {adaptation_steps}")
    print(f"Checkpoint frequency: {checkpoint_frequency}")
    print(f"Shift type: {shift_type.value if shift_type else 'random'}")
    print(f"Seed: {seed if seed else 'random'}")
    print(f"Output file: {output_file}")
    print(f"{'='*80}\n")

    master_rng = np.random.RandomState(seed)
    results = []

    # Run simulations
    for run_id in range(num_runs):
        run_seed = master_rng.randint(0, 1000000)
        sim = Simulation(
            design_steps=design_steps,
            adaptation_steps=adaptation_steps,
            checkpoint_frequency=checkpoint_frequency,
            shift_type=shift_type,
            seed=run_seed,
            verbose=verbose,
        )

        result = sim.run_single_simulation(run_id)
        results.append(result)

        if not verbose and (run_id + 1) % 10 == 0:
            print(f"Completed {run_id + 1}/{num_runs} runs...")

    # Aggregate statistics
    stats = _compute_aggregate_statistics(results)

    # Save results
    output_data = {
        'parameters': {
            'num_runs': num_runs,
            'design_steps': design_steps,
            'adaptation_steps': adaptation_steps,
            'checkpoint_frequency': checkpoint_frequency,
            'shift_type': shift_type.value if shift_type else 'random',
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


def _compute_aggregate_statistics(results: List[SimulationResult]) -> Dict:
    """Compute aggregate statistics across all simulation runs"""

    total_runs = len(results)

    # Survival rates
    agent1_survived_count = sum(1 for r in results if r.agent1_survived_shift)
    agent2_survived_count = sum(1 for r in results if r.agent2_survived_shift)

    # Win counts
    agent1_wins = sum(1 for r in results if r.winner == "GreedyPerformance")
    agent2_wins = sum(1 for r in results if r.winner == "LogMinHeadroom")
    ties = sum(1 for r in results if "Tie" in r.winner)

    # Performance statistics (design phase)
    agent1_perf_design = [r.agent1_final_performance_design for r in results]
    agent2_perf_design = [r.agent2_final_performance_design for r in results]

    # Performance statistics (adaptation phase)
    agent1_perf_adapt = [r.agent1_final_performance_adapt for r in results if r.agent1_survived_shift]
    agent2_perf_adapt = [r.agent2_final_performance_adapt for r in results if r.agent2_survived_shift]

    # Headroom statistics
    agent1_headroom_design = [r.agent1_min_headroom_design for r in results]
    agent2_headroom_design = [r.agent2_min_headroom_design for r in results]

    agent1_headroom_adapt = [r.agent1_min_headroom_adapt for r in results if r.agent1_survived_shift]
    agent2_headroom_adapt = [r.agent2_min_headroom_adapt for r in results if r.agent2_survived_shift]

    # Adaptation speed
    agent1_adapt_steps = [r.agent1_steps_to_adapt for r in results if r.agent1_steps_to_adapt is not None]
    agent2_adapt_steps = [r.agent2_steps_to_adapt for r in results if r.agent2_steps_to_adapt is not None]

    # Shift type breakdown
    shift_types = {}
    for r in results:
        shift_type = r.shift_info['type']
        if shift_type not in shift_types:
            shift_types[shift_type] = {'count': 0, 'agent1_survived': 0, 'agent2_survived': 0}
        shift_types[shift_type]['count'] += 1
        if r.agent1_survived_shift:
            shift_types[shift_type]['agent1_survived'] += 1
        if r.agent2_survived_shift:
            shift_types[shift_type]['agent2_survived'] += 1

    return {
        'total_runs': total_runs,
        'survival': {
            'agent1_survived': agent1_survived_count,
            'agent1_survival_rate': agent1_survived_count / total_runs,
            'agent2_survived': agent2_survived_count,
            'agent2_survival_rate': agent2_survived_count / total_runs,
        },
        'wins': {
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'ties': ties,
            'agent1_win_rate': agent1_wins / total_runs,
            'agent2_win_rate': agent2_wins / total_runs,
        },
        'performance_design': {
            'agent1_mean': np.mean(agent1_perf_design),
            'agent1_std': np.std(agent1_perf_design),
            'agent2_mean': np.mean(agent2_perf_design),
            'agent2_std': np.std(agent2_perf_design),
        },
        'performance_adaptation': {
            'agent1_mean': np.mean(agent1_perf_adapt) if agent1_perf_adapt else 0.0,
            'agent1_std': np.std(agent1_perf_adapt) if agent1_perf_adapt else 0.0,
            'agent2_mean': np.mean(agent2_perf_adapt) if agent2_perf_adapt else 0.0,
            'agent2_std': np.std(agent2_perf_adapt) if agent2_perf_adapt else 0.0,
        },
        'headroom_design': {
            'agent1_mean': np.mean(agent1_headroom_design),
            'agent1_std': np.std(agent1_headroom_design),
            'agent2_mean': np.mean(agent2_headroom_design),
            'agent2_std': np.std(agent2_headroom_design),
        },
        'headroom_adaptation': {
            'agent1_mean': np.mean(agent1_headroom_adapt) if agent1_headroom_adapt else 0.0,
            'agent1_std': np.std(agent1_headroom_adapt) if agent1_headroom_adapt else 0.0,
            'agent2_mean': np.mean(agent2_headroom_adapt) if agent2_headroom_adapt else 0.0,
            'agent2_std': np.std(agent2_headroom_adapt) if agent2_headroom_adapt else 0.0,
        },
        'adaptation_speed': {
            'agent1_mean_steps': np.mean(agent1_adapt_steps) if agent1_adapt_steps else None,
            'agent2_mean_steps': np.mean(agent2_adapt_steps) if agent2_adapt_steps else None,
        },
        'shift_type_breakdown': shift_types,
    }


def _print_statistics(stats: Dict):
    """Print formatted statistics"""

    print("\nSURVIVAL RATES:")
    print(f"  GreedyPerformance:  {stats['survival']['agent1_survived']:4d}/{stats['total_runs']} ({stats['survival']['agent1_survival_rate']:.1%})")
    print(f"  LogMinHeadroom:     {stats['survival']['agent2_survived']:4d}/{stats['total_runs']} ({stats['survival']['agent2_survival_rate']:.1%})")

    print("\nWIN RATES:")
    print(f"  GreedyPerformance:  {stats['wins']['agent1_wins']:4d}/{stats['total_runs']} ({stats['wins']['agent1_win_rate']:.1%})")
    print(f"  LogMinHeadroom:     {stats['wins']['agent2_wins']:4d}/{stats['total_runs']} ({stats['wins']['agent2_win_rate']:.1%})")
    print(f"  Ties:               {stats['wins']['ties']:4d}/{stats['total_runs']}")

    print("\nPERFORMANCE (Design Phase):")
    print(f"  GreedyPerformance:  {stats['performance_design']['agent1_mean']:7.2f} ± {stats['performance_design']['agent1_std']:.2f}")
    print(f"  LogMinHeadroom:     {stats['performance_design']['agent2_mean']:7.2f} ± {stats['performance_design']['agent2_std']:.2f}")

    print("\nPERFORMANCE (Adaptation Phase):")
    print(f"  GreedyPerformance:  {stats['performance_adaptation']['agent1_mean']:7.2f} ± {stats['performance_adaptation']['agent1_std']:.2f}")
    print(f"  LogMinHeadroom:     {stats['performance_adaptation']['agent2_mean']:7.2f} ± {stats['performance_adaptation']['agent2_std']:.2f}")

    print("\nHEADROOM (Design Phase):")
    print(f"  GreedyPerformance:  {stats['headroom_design']['agent1_mean']:7.2f} ± {stats['headroom_design']['agent1_std']:.2f}")
    print(f"  LogMinHeadroom:     {stats['headroom_design']['agent2_mean']:7.2f} ± {stats['headroom_design']['agent2_std']:.2f}")

    print("\nHEADROOM (Adaptation Phase):")
    print(f"  GreedyPerformance:  {stats['headroom_adaptation']['agent1_mean']:7.2f} ± {stats['headroom_adaptation']['agent1_std']:.2f}")
    print(f"  LogMinHeadroom:     {stats['headroom_adaptation']['agent2_mean']:7.2f} ± {stats['headroom_adaptation']['agent2_std']:.2f}")

    print("\nSHIFT TYPE BREAKDOWN:")
    for shift_type, data in stats['shift_type_breakdown'].items():
        print(f"  {shift_type}:")
        print(f"    Count: {data['count']}")
        print(f"    GreedyPerformance survived: {data['agent1_survived']}/{data['count']} ({data['agent1_survived']/data['count']:.1%})")
        print(f"    LogMinHeadroom survived:    {data['agent2_survived']}/{data['count']} ({data['agent2_survived']/data['count']:.1%})")


if __name__ == "__main__":
    # Quick test run
    print("Running quick test simulation...")
    run_multiple_simulations(
        num_runs=5,
        design_steps=20,
        adaptation_steps=10,
        checkpoint_frequency=5,
        output_file="test_results.json",
        verbose=True,
    )
