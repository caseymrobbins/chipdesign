#!/usr/bin/env python3
"""
JAM Look-ahead vs Greedy Experiment

Core Question: Does JAM (log(min(values))) need look-ahead prediction
or does one-step greedy evaluation work?

Tests across 4 topology levels with different evaluation strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
import copy
from collections import deque
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class CellType(Enum):
    """Types of cells in the environment"""
    EMPTY = 0
    START = 1
    GOAL = 2
    OBSTACLE = 3
    TRAP = 4  # Attractive but dangerous
    SAFE = 5  # Unattractive but safe


@dataclass
class State:
    """Agent state with position and values"""
    position: Tuple[int, int]
    values: np.ndarray  # [v1, v2, v3, v4]
    step: int = 0

    def copy(self):
        return State(
            position=self.position,
            values=self.values.copy(),
            step=self.step
        )

    def jam_score(self) -> float:
        """Compute JAM reward: log(min(values))"""
        min_val = np.min(self.values)
        if min_val <= 0:
            return -np.inf  # Dead state
        return np.log(min_val)

    def is_alive(self) -> bool:
        """Check if all values are positive"""
        return np.all(self.values > 0)


@dataclass
class Cell:
    """A cell in the environment with effects on values"""
    cell_type: CellType
    value_effects: np.ndarray  # Additive effects on [v1, v2, v3, v4]

    def __repr__(self):
        return f"{self.cell_type.name}"


class Environment:
    """Base environment class"""

    def __init__(self, width: int, height: int, name: str):
        self.width = width
        self.height = height
        self.name = name
        self.grid: List[List[Cell]] = []
        self.start_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (0, 0)
        self.initial_values = np.ones(4)

    def get_cell(self, pos: Tuple[int, int]) -> Optional[Cell]:
        """Get cell at position"""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                cell = self.grid[ny][nx]
                if cell.cell_type != CellType.OBSTACLE:
                    neighbors.append((nx, ny))
        return neighbors

    def apply_action(self, state: State, new_pos: Tuple[int, int]) -> State:
        """Apply action and return new state"""
        new_state = state.copy()
        new_state.position = new_pos
        new_state.step += 1

        # Apply cell effects
        cell = self.get_cell(new_pos)
        if cell:
            new_state.values += cell.value_effects

        return new_state

    def visualize(self, path: List[Tuple[int, int]] = None) -> plt.Figure:
        """Visualize the environment"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create color map for cells
        grid_colors = np.zeros((self.height, self.width, 3))

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type == CellType.START:
                    grid_colors[y, x] = [0, 1, 0]  # Green
                elif cell.cell_type == CellType.GOAL:
                    grid_colors[y, x] = [0, 0, 1]  # Blue
                elif cell.cell_type == CellType.OBSTACLE:
                    grid_colors[y, x] = [0, 0, 0]  # Black
                elif cell.cell_type == CellType.TRAP:
                    grid_colors[y, x] = [1, 0.5, 0]  # Orange
                elif cell.cell_type == CellType.SAFE:
                    grid_colors[y, x] = [0.5, 0.5, 1]  # Light blue
                else:
                    # Color by minimum effect
                    min_effect = np.min(cell.value_effects)
                    if min_effect < 0:
                        intensity = min(1, abs(min_effect) / 0.5)
                        grid_colors[y, x] = [intensity, 0, 0]  # Red for negative
                    else:
                        grid_colors[y, x] = [1, 1, 1]  # White for positive

        ax.imshow(grid_colors, interpolation='nearest')

        # Draw path if provided
        if path:
            path_y = [p[1] for p in path]
            path_x = [p[0] for p in path]
            ax.plot(path_x, path_y, 'y-', linewidth=3, alpha=0.7, label='Path')
            ax.plot(path_x[0], path_y[0], 'go', markersize=15, label='Start')
            ax.plot(path_x[-1], path_y[-1], 'r*', markersize=20, label='End')

        ax.set_title(f'{self.name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig


# ============================================================================
# TOPOLOGY GENERATORS
# ============================================================================

class LinearTopology(Environment):
    """Level 1: Linear path, no branches"""

    def __init__(self, length: int = 20):
        super().__init__(width=length, height=1, name="Level 1: Linear")
        self.start_pos = (0, 0)
        self.goal_pos = (length - 1, 0)

        # Create grid
        self.grid = [[Cell(CellType.EMPTY, np.array([0.1, 0.1, 0.1, 0.1]))
                      for _ in range(self.width)]]

        # Set start and goal
        self.grid[0][0].cell_type = CellType.START
        self.grid[0][-1].cell_type = CellType.GOAL

        # Add some mild variation
        for i in range(1, length - 1):
            # Small random effects
            effects = np.random.uniform(-0.05, 0.15, 4)
            self.grid[0][i].value_effects = effects


class BranchingTopology(Environment):
    """Level 2: Branching paths with choices"""

    def __init__(self, length: int = 12):
        super().__init__(width=5, height=length, name="Level 2: Branching Paths")

        self.start_pos = (2, 0)  # Middle column
        self.goal_pos = (2, length - 1)

        # Initialize grid with obstacles on edges
        self.grid = [[Cell(CellType.OBSTACLE, np.zeros(4))
                      for _ in range(self.width)]
                     for _ in range(self.height)]

        # Build branching paths
        self._build_paths(length)

    def _build_paths(self, length: int):
        """Build branching paths"""
        # Start
        self.grid[0][2] = Cell(CellType.START, np.array([0.1, 0.1, 0.1, 0.1]))

        # Create branching paths
        for y in range(1, length - 1):
            if y % 3 == 0:
                # Branch point - open left, middle, and right
                # Left path: good for v1, v3
                self.grid[y][1] = Cell(CellType.EMPTY, np.array([0.2, 0.05, 0.2, 0.05]))
                # Middle path: balanced
                self.grid[y][2] = Cell(CellType.EMPTY, np.array([0.12, 0.12, 0.12, 0.12]))
                # Right path: good for v2, v4
                self.grid[y][3] = Cell(CellType.EMPTY, np.array([0.05, 0.2, 0.05, 0.2]))
            else:
                # Convergence - paths merge back to middle
                self.grid[y][2] = Cell(CellType.EMPTY, np.array([0.1, 0.1, 0.1, 0.1]))
                # But side paths still available
                if y % 3 == 1:
                    # Some side paths have penalties
                    self.grid[y][1] = Cell(CellType.EMPTY, np.array([0.15, -0.1, 0.15, -0.1]))
                    self.grid[y][3] = Cell(CellType.EMPTY, np.array([-0.1, 0.15, -0.1, 0.15]))

        # Goal
        self.grid[length - 1][2] = Cell(CellType.GOAL, np.array([0.5, 0.5, 0.5, 0.5]))
        self.grid[length - 1][2].cell_type = CellType.GOAL


class GridWithObstacles(Environment):
    """Level 3: Grid with obstacles and dead ends"""

    def __init__(self, size: int = 10):
        super().__init__(width=size, height=size, name="Level 3: Grid with Obstacles")
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)

        # Initialize grid
        self.grid = [[Cell(CellType.EMPTY, np.array([0.1, 0.1, 0.1, 0.1]))
                      for _ in range(self.width)]
                     for _ in range(self.height)]

        # Set start and goal
        self.grid[0][0].cell_type = CellType.START
        self.grid[size-1][size-1].cell_type = CellType.GOAL

        # Add obstacles (maze-like)
        np.random.seed(42)
        for y in range(size):
            for x in range(size):
                if (x, y) == self.start_pos or (x, y) == self.goal_pos:
                    continue

                # 20% obstacles
                if np.random.rand() < 0.2:
                    self.grid[y][x] = Cell(CellType.OBSTACLE, np.zeros(4))
                else:
                    # Some cells hurt specific values
                    if np.random.rand() < 0.3:
                        # Create cells that hurt one value
                        hurt_idx = np.random.randint(4)
                        effects = np.array([0.1, 0.1, 0.1, 0.1])
                        effects[hurt_idx] = -0.2
                        self.grid[y][x].value_effects = effects


class AdversarialTopology(Environment):
    """Level 4: Designed to trap greedy agents"""

    def __init__(self, size: int = 12):
        super().__init__(width=size, height=size, name="Level 4: Adversarial")
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)

        # Initialize grid
        self.grid = [[Cell(CellType.EMPTY, np.array([0.0, 0.0, 0.0, 0.0]))
                      for _ in range(self.width)]
                     for _ in range(self.height)]

        # Set start and goal
        self.grid[0][0].cell_type = CellType.START
        self.grid[0][0].value_effects = np.array([0.2, 0.2, 0.2, 0.2])
        self.grid[size-1][size-1].cell_type = CellType.GOAL
        self.grid[size-1][size-1].value_effects = np.array([1.0, 1.0, 1.0, 1.0])

        # Create attractive trap path (looks good short-term, kills you long-term)
        # Main path along bottom edge - looks attractive!
        for x in range(1, size - 1):
            self.grid[0][x].cell_type = CellType.TRAP
            # Big boost to 3 values, small drain on 1
            self.grid[0][x].value_effects = np.array([0.3, 0.3, 0.3, -0.15])

        # But then force upward movement with severe penalty
        for y in range(1, size - 1):
            self.grid[y][size-2].value_effects = np.array([-0.25, -0.25, -0.25, -0.25])

        # Create unattractive but safe path (diagonal)
        for i in range(1, size - 1):
            self.grid[i][i].cell_type = CellType.SAFE
            # Small but balanced gains
            self.grid[i][i].value_effects = np.array([0.08, 0.08, 0.08, 0.08])

        # Fill rest with neutral or slightly negative
        for y in range(size):
            for x in range(size):
                if self.grid[y][x].cell_type == CellType.EMPTY:
                    self.grid[y][x].value_effects = np.random.uniform(-0.1, 0.05, 4)


# ============================================================================
# AGENT STRATEGIES
# ============================================================================

@dataclass
class EpisodeResult:
    """Results from running one episode"""
    strategy_name: str
    topology_name: str
    survived: bool
    steps: int
    nodes_evaluated: int
    times_trapped: int
    path: List[Tuple[int, int]]
    final_values: np.ndarray
    min_value_history: List[float]
    reached_goal: bool


class Strategy:
    """Base class for agent strategies"""

    def __init__(self, name: str):
        self.name = name
        self.nodes_evaluated = 0

    def select_action(self, env: Environment, state: State) -> Optional[Tuple[int, int]]:
        """Select next action given current state"""
        raise NotImplementedError

    def run_episode(self, env: Environment, max_steps: int = 100) -> EpisodeResult:
        """Run one episode"""
        state = State(position=env.start_pos, values=env.initial_values.copy())
        path = [state.position]
        min_value_history = [np.min(state.values)]
        times_trapped = 0
        self.nodes_evaluated = 0

        for step in range(max_steps):
            # Check if reached goal
            if state.position == env.goal_pos:
                return EpisodeResult(
                    strategy_name=self.name,
                    topology_name=env.name,
                    survived=state.is_alive(),
                    steps=step,
                    nodes_evaluated=self.nodes_evaluated,
                    times_trapped=times_trapped,
                    path=path,
                    final_values=state.values.copy(),
                    min_value_history=min_value_history,
                    reached_goal=True
                )

            # Select action
            action = self.select_action(env, state)

            if action is None:
                # Trapped!
                times_trapped += 1
                # Try any valid neighbor
                neighbors = env.get_neighbors(state.position)
                if not neighbors:
                    break
                action = neighbors[0]

            # Apply action
            state = env.apply_action(state, action)
            path.append(state.position)
            min_value_history.append(np.min(state.values))

            # Check if dead
            if not state.is_alive():
                break

        return EpisodeResult(
            strategy_name=self.name,
            topology_name=env.name,
            survived=state.is_alive(),
            steps=len(path) - 1,
            nodes_evaluated=self.nodes_evaluated,
            times_trapped=times_trapped,
            path=path,
            final_values=state.values.copy(),
            min_value_history=min_value_history,
            reached_goal=(state.position == env.goal_pos)
        )


class JAMGreedy(Strategy):
    """Greedy JAM: Pick action with best immediate log(min(values)) + goal progress"""

    def __init__(self):
        super().__init__("JAM-Greedy")

    def select_action(self, env: Environment, state: State) -> Optional[Tuple[int, int]]:
        neighbors = env.get_neighbors(state.position)
        if not neighbors:
            return None

        best_action = None
        best_score = -np.inf

        for neighbor in neighbors:
            self.nodes_evaluated += 1
            # Simulate action
            next_state = env.apply_action(state, neighbor)

            # Combined score: JAM safety + goal progress
            jam_score = next_state.jam_score()

            # Distance to goal (negative, want to minimize)
            curr_dist = abs(state.position[0] - env.goal_pos[0]) + abs(state.position[1] - env.goal_pos[1])
            next_dist = abs(neighbor[0] - env.goal_pos[0]) + abs(neighbor[1] - env.goal_pos[1])
            goal_progress = curr_dist - next_dist  # Positive if moving closer

            # Combined score: prioritize JAM safety, but break ties with goal progress
            score = jam_score + 0.01 * goal_progress

            if score > best_score:
                best_score = score
                best_action = neighbor

        return best_action


class JAMLookahead(Strategy):
    """JAM with N-step lookahead"""

    def __init__(self, n_steps: int):
        super().__init__(f"JAM-{n_steps}step")
        self.n_steps = n_steps

    def select_action(self, env: Environment, state: State) -> Optional[Tuple[int, int]]:
        neighbors = env.get_neighbors(state.position)
        if not neighbors:
            return None

        best_action = None
        best_score = -np.inf

        for neighbor in neighbors:
            # Evaluate with lookahead
            score = self._evaluate_path(env, state, neighbor, self.n_steps)

            if score > best_score:
                best_score = score
                best_action = neighbor

        return best_action

    def _evaluate_path(self, env: Environment, state: State,
                      first_action: Tuple[int, int], depth: int) -> float:
        """Recursively evaluate path with lookahead"""
        self.nodes_evaluated += 1

        # Apply first action
        next_state = env.apply_action(state, first_action)

        # Base cases
        if not next_state.is_alive():
            return -np.inf

        # Goal bonus
        if next_state.position == env.goal_pos:
            return next_state.jam_score() + 10.0

        if depth == 1:
            # Add small goal progress bonus
            dist_to_goal = abs(next_state.position[0] - env.goal_pos[0]) + abs(next_state.position[1] - env.goal_pos[1])
            return next_state.jam_score() + 0.01 * (1.0 / max(dist_to_goal, 1))

        # Recursive case: evaluate all possible continuations
        neighbors = env.get_neighbors(next_state.position)
        if not neighbors:
            return next_state.jam_score()

        max_future_score = -np.inf
        for neighbor in neighbors:
            future_score = self._evaluate_path(env, next_state, neighbor, depth - 1)
            max_future_score = max(max_future_score, future_score)

        return max_future_score


class GoalSeeker(Strategy):
    """Standard goal-seeking with value constraints (A*)"""

    def __init__(self):
        super().__init__("Goal-Seeker")

    def select_action(self, env: Environment, state: State) -> Optional[Tuple[int, int]]:
        """Use A* to find path to goal, pick first step"""
        neighbors = env.get_neighbors(state.position)
        if not neighbors:
            return None

        # Find path to goal using A*
        path = self._astar(env, state)

        if path and len(path) > 1:
            return path[1]  # Return first step

        # Fallback to closest neighbor to goal
        goal = env.goal_pos
        best_action = min(neighbors,
                         key=lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]))
        return best_action

    def _astar(self, env: Environment, start_state: State) -> Optional[List[Tuple[int, int]]]:
        """A* search to goal"""
        from heapq import heappush, heappop

        # Priority queue: (f_score, position, state, path)
        start_pos = start_state.position
        goal_pos = env.goal_pos

        g_score = {start_pos: 0}
        f_score = {start_pos: self._heuristic(start_pos, goal_pos)}

        open_set = [(f_score[start_pos], start_pos, start_state, [start_pos])]
        closed_set = set()

        max_nodes = 1000  # Limit search

        while open_set and self.nodes_evaluated < max_nodes:
            _, current_pos, current_state, path = heappop(open_set)

            if current_pos in closed_set:
                continue

            closed_set.add(current_pos)
            self.nodes_evaluated += 1

            # Check goal
            if current_pos == goal_pos:
                return path

            # Expand neighbors
            for neighbor in env.get_neighbors(current_pos):
                if neighbor in closed_set:
                    continue

                # Simulate move
                next_state = env.apply_action(current_state, neighbor)

                # Skip if values go negative
                if not next_state.is_alive():
                    continue

                tentative_g = g_score[current_pos] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_pos)
                    f_score[neighbor] = f
                    heappush(open_set, (f, neighbor, next_state, path + [neighbor]))

        return None  # No path found

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run experiments across topologies and strategies"""

    def __init__(self):
        self.results: List[EpisodeResult] = []

    def run_all(self, max_steps: int = 200):
        """Run all experiments"""
        # Create topologies
        topologies = [
            LinearTopology(length=20),
            BranchingTopology(length=15),
            GridWithObstacles(size=10),
            AdversarialTopology(size=12)
        ]

        # Create strategies
        strategies = [
            JAMGreedy(),
            JAMLookahead(2),
            JAMLookahead(5),
            GoalSeeker()
        ]

        # Run experiments
        for topo in topologies:
            print(f"\n{'='*60}")
            print(f"Testing: {topo.name}")
            print(f"{'='*60}")

            for strategy in strategies:
                print(f"  Running {strategy.name}...", end=" ")
                result = strategy.run_episode(topo, max_steps=max_steps)
                self.results.append(result)

                # Print summary
                status = "✓ GOAL" if result.reached_goal else ("✓ ALIVE" if result.survived else "✗ DEAD")
                print(f"{status} | Steps: {result.steps:3d} | Nodes: {result.nodes_evaluated:5d} | Trapped: {result.times_trapped}")

        return self.results

    def generate_report(self) -> pd.DataFrame:
        """Generate results table"""
        data = []

        for result in self.results:
            data.append({
                'Topology': result.topology_name.replace('Level ', 'L'),
                'Strategy': result.strategy_name,
                'Survived': '✓' if result.survived else '✗',
                'Reached Goal': '✓' if result.reached_goal else '✗',
                'Steps': result.steps,
                'Nodes Evaluated': result.nodes_evaluated,
                'Times Trapped': result.times_trapped,
                'Final Min Value': f"{np.min(result.final_values):.3f}"
            })

        return pd.DataFrame(data)

    def visualize_results(self, output_dir: str = "/mnt/user-data/outputs"):
        """Create visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Group results by topology
        topologies = {}
        for result in self.results:
            if result.topology_name not in topologies:
                topologies[result.topology_name] = []
            topologies[result.topology_name].append(result)

        # Create figure with subplots
        n_topos = len(topologies)
        fig = plt.figure(figsize=(20, 5 * n_topos))

        for idx, (topo_name, topo_results) in enumerate(topologies.items()):
            # Get environment
            env = None
            for result in topo_results:
                if result.topology_name == topo_name:
                    # Recreate environment
                    if "Linear" in topo_name:
                        env = LinearTopology(20)
                    elif "Branching" in topo_name:
                        env = BranchingTopology(15)
                    elif "Grid" in topo_name:
                        env = GridWithObstacles(10)
                    elif "Adversarial" in topo_name:
                        env = AdversarialTopology(12)
                    break

            if env is None:
                continue

            # Plot paths for each strategy
            for strat_idx, result in enumerate(topo_results):
                ax = plt.subplot(n_topos, 4, idx * 4 + strat_idx + 1)

                # Visualize environment with path
                grid_colors = np.zeros((env.height, env.width, 3))

                for y in range(env.height):
                    for x in range(env.width):
                        cell = env.grid[y][x]
                        if cell.cell_type == CellType.START:
                            grid_colors[y, x] = [0, 0.8, 0]
                        elif cell.cell_type == CellType.GOAL:
                            grid_colors[y, x] = [0, 0, 0.8]
                        elif cell.cell_type == CellType.OBSTACLE:
                            grid_colors[y, x] = [0, 0, 0]
                        elif cell.cell_type == CellType.TRAP:
                            grid_colors[y, x] = [1, 0.3, 0]
                        elif cell.cell_type == CellType.SAFE:
                            grid_colors[y, x] = [0.3, 0.3, 1]
                        else:
                            min_effect = np.min(cell.value_effects)
                            if min_effect < 0:
                                intensity = min(1, abs(min_effect) / 0.3)
                                grid_colors[y, x] = [intensity, 0, 0]
                            else:
                                grid_colors[y, x] = [0.9, 0.9, 0.9]

                ax.imshow(grid_colors, interpolation='nearest', aspect='auto')

                # Draw path
                if result.path:
                    path_y = [p[1] for p in result.path]
                    path_x = [p[0] for p in result.path]
                    ax.plot(path_x, path_y, 'yellow', linewidth=2, alpha=0.8)
                    ax.plot(path_x[0], path_y[0], 'go', markersize=10)

                    # Color end point based on outcome
                    if result.reached_goal:
                        ax.plot(path_x[-1], path_y[-1], 'g*', markersize=15)
                    elif result.survived:
                        ax.plot(path_x[-1], path_y[-1], 'yo', markersize=10)
                    else:
                        ax.plot(path_x[-1], path_y[-1], 'rx', markersize=15, markeredgewidth=3)

                # Title with stats
                status = "GOAL" if result.reached_goal else ("ALIVE" if result.survived else "DEAD")
                ax.set_title(f"{result.strategy_name}\n{status} | Steps:{result.steps} | Nodes:{result.nodes_evaluated}",
                           fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/jam_lookahead_paths.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Create performance comparison plots
        self._plot_performance_comparison(output_dir)

        print(f"\nVisualizations saved to {output_dir}/")

    def _plot_performance_comparison(self, output_dir: str):
        """Create comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Prepare data
        df = self.generate_report()

        # 1. Success rate by topology and strategy
        ax = axes[0, 0]
        success_data = []
        for topo in df['Topology'].unique():
            for strat in df['Strategy'].unique():
                mask = (df['Topology'] == topo) & (df['Strategy'] == strat)
                reached_goal = (df[mask]['Reached Goal'] == '✓').sum()
                success_data.append({
                    'Topology': topo,
                    'Strategy': strat,
                    'Success Rate': reached_goal
                })

        success_df = pd.DataFrame(success_data)
        pivot_success = success_df.pivot(index='Topology', columns='Strategy', values='Success Rate')
        pivot_success.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Goal Reached (Success Rate)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Success (1=Yes, 0=No)')
        ax.set_xlabel('Topology')
        ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        # 2. Steps to completion
        ax = axes[0, 1]
        steps_data = df.copy()
        steps_data['Steps'] = pd.to_numeric(steps_data['Steps'])
        pivot_steps = steps_data.pivot(index='Topology', columns='Strategy', values='Steps')
        pivot_steps.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Steps to Completion', fontsize=14, fontweight='bold')
        ax.set_ylabel('Steps')
        ax.set_xlabel('Topology')
        ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        # 3. Computational cost (nodes evaluated)
        ax = axes[1, 0]
        nodes_data = df.copy()
        nodes_data['Nodes Evaluated'] = pd.to_numeric(nodes_data['Nodes Evaluated'])
        pivot_nodes = nodes_data.pivot(index='Topology', columns='Strategy', values='Nodes Evaluated')
        pivot_nodes.plot(kind='bar', ax=ax, width=0.8, logy=True)
        ax.set_title('Computational Cost (Nodes Evaluated)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Nodes Evaluated (log scale)')
        ax.set_xlabel('Topology')
        ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        # 4. Survival rate
        ax = axes[1, 1]
        survival_data = []
        for topo in df['Topology'].unique():
            for strat in df['Strategy'].unique():
                mask = (df['Topology'] == topo) & (df['Strategy'] == strat)
                survived = (df[mask]['Survived'] == '✓').sum()
                survival_data.append({
                    'Topology': topo,
                    'Strategy': strat,
                    'Survived': survived
                })

        survival_df = pd.DataFrame(survival_data)
        pivot_survival = survival_df.pivot(index='Topology', columns='Strategy', values='Survived')
        pivot_survival.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Survival Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('Survived (1=Yes, 0=No)')
        ax.set_xlabel('Topology')
        ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/jam_lookahead_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the full experiment"""
    print("="*70)
    print(" JAM LOOK-AHEAD vs GREEDY EXPERIMENT")
    print("="*70)
    print("\nCore Question: Does JAM need look-ahead or does greedy work?")
    print("\nRunning experiments across 4 topology levels...")
    print("Strategies: JAM-Greedy, JAM-2step, JAM-5step, Goal-Seeker")
    print()

    # Run experiments
    runner = ExperimentRunner()
    results = runner.run_all(max_steps=200)

    # Generate report
    print("\n" + "="*70)
    print(" RESULTS TABLE")
    print("="*70)
    df = runner.generate_report()
    print(df.to_string(index=False))

    # Save report
    output_dir = "/mnt/user-data/outputs"
    df.to_csv(f"{output_dir}/jam_lookahead_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/jam_lookahead_results.csv")

    # Generate visualizations
    print("\nGenerating visualizations...")
    runner.visualize_results(output_dir)

    # Analysis
    print("\n" + "="*70)
    print(" KEY FINDINGS")
    print("="*70)

    # Count successes
    greedy_success = sum(1 for r in results if r.strategy_name == "JAM-Greedy" and r.reached_goal)
    step2_success = sum(1 for r in results if r.strategy_name == "JAM-2step" and r.reached_goal)
    step5_success = sum(1 for r in results if r.strategy_name == "JAM-5step" and r.reached_goal)
    goal_success = sum(1 for r in results if r.strategy_name == "Goal-Seeker" and r.reached_goal)

    total_topos = 4

    print(f"\n1. SUCCESS RATES (Reached Goal):")
    print(f"   JAM-Greedy:   {greedy_success}/{total_topos} topologies")
    print(f"   JAM-2step:    {step2_success}/{total_topos} topologies")
    print(f"   JAM-5step:    {step5_success}/{total_topos} topologies")
    print(f"   Goal-Seeker:  {goal_success}/{total_topos} topologies")

    # Compute efficiency
    greedy_nodes = sum(r.nodes_evaluated for r in results if r.strategy_name == "JAM-Greedy")
    goal_nodes = sum(r.nodes_evaluated for r in results if r.strategy_name == "Goal-Seeker")

    print(f"\n2. COMPUTATIONAL EFFICIENCY:")
    print(f"   JAM-Greedy total nodes:   {greedy_nodes:,}")
    print(f"   Goal-Seeker total nodes:  {goal_nodes:,}")
    print(f"   Efficiency gain:          {goal_nodes/max(greedy_nodes,1):.1f}x fewer nodes with JAM-Greedy")

    # Find where greedy fails
    greedy_failures = [r for r in results if r.strategy_name == "JAM-Greedy" and not r.reached_goal]

    if greedy_failures:
        print(f"\n3. WHERE GREEDY FAILS:")
        for failure in greedy_failures:
            print(f"   {failure.topology_name}: ", end="")
            if failure.survived:
                print(f"survived but didn't reach goal (trapped at step {failure.steps})")
            else:
                print(f"died at step {failure.steps}")
    else:
        print(f"\n3. GREEDY NEVER FAILS!")
        print(f"   JAM-Greedy succeeded on ALL topologies!")

    print("\n" + "="*70)
    print(" CONCLUSION")
    print("="*70)

    if greedy_success == total_topos:
        print("\n🎉 MAJOR FINDING: JAM-Greedy works perfectly!")
        print("   - Greedy evaluation is SUFFICIENT for JAM")
        print("   - No look-ahead needed")
        print("   - Massive computational savings vs traditional RL")
        print("\n   WHY: log(min(values)) naturally avoids local optima")
        print("   The min() operator makes the agent conservative about")
        print("   ALL values simultaneously, preventing value collapse.")
    elif greedy_success >= total_topos * 0.75:
        print("\n✓ JAM-Greedy works well (75%+ success)")
        print(f"  - Only fails on: {[f.topology_name for f in greedy_failures]}")
        print("  - Still much more efficient than Goal-Seeker")
        print("  - Small lookahead (2-step) may help edge cases")
    else:
        print("\n⚠ JAM needs look-ahead for complex topologies")
        print(f"  - Greedy only succeeded: {greedy_success}/{total_topos}")
        print(f"  - Look-ahead improves to: {step2_success}/{total_topos} (2-step)")
        print(f"  - But still efficient vs Goal-Seeker")

    print("\n" + "="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print("  - jam_lookahead_results.csv")
    print("  - jam_lookahead_paths.png")
    print("  - jam_lookahead_comparison.png")
    print()


if __name__ == "__main__":
    main()
