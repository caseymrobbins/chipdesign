#!/usr/bin/env python3
"""
Generate robustness data for JamRobust to add to raw data file
"""

import json
from advanced_chip_simulator import (
    AdvancedDesignSpace,
    ProcessTechnology,
    ShiftType,
)
from test_softmin_jam import JamRobustAgent

def test_jamrobust_robustness(design_steps=50):
    """Test JamRobust robustness with same methodology as other agents"""

    print("Testing JamRobust robustness...")

    # Design the chip
    space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
    space.initialize_actions()
    agent = JamRobustAgent()
    agent.initialize(space)

    for _ in range(design_steps):
        agent.step()

    # Collect design metrics
    design_perf = space.calculate_performance()
    constraints = space.calculate_constraints()
    design_power = constraints['total_power_w']
    design_eff = design_perf / design_power if design_power > 0 else 0
    design_headroom = space.get_min_headroom()

    print(f"  Design: {design_perf:.2f} perf @ {design_power:.2f}W (eff={design_eff:.2f})")

    # Test robustness across stress types
    results = []

    stress_tests = [
        ("Power Cuts", ShiftType.TIGHTEN_POWER, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        ("Performance Demand", ShiftType.INCREASE_PERFORMANCE,
         [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]),
        ("Area Reduction", ShiftType.REDUCE_AREA, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        ("Thermal Stress", ShiftType.TIGHTEN_THERMAL,
         [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]),
    ]

    for stress_name, stress_type, levels in stress_tests:
        max_survived = 0

        for level in levels:
            # Create fresh space with same design
            test_space = AdvancedDesignSpace(process=ProcessTechnology.create_7nm(), seed=42)
            test_space.initialize_actions()
            test_agent = JamRobustAgent()
            test_agent.initialize(test_space)

            for _ in range(design_steps):
                test_agent.step()

            # Apply stress
            if stress_type == ShiftType.TIGHTEN_POWER:
                test_space.limits.max_power_watts *= (1.0 - level)
            elif stress_type == ShiftType.INCREASE_PERFORMANCE:
                test_space.limits.min_frequency_ghz *= (1.0 + level)
            elif stress_type == ShiftType.REDUCE_AREA:
                test_space.limits.max_area_mm2 *= (1.0 - level)
            elif stress_type == ShiftType.TIGHTEN_THERMAL:
                test_space.limits.max_temperature_c -= (level * 50)

            # Check if still feasible
            if test_space.is_feasible():
                max_survived = level
            else:
                break

        print(f"  {stress_name}: {max_survived*100:.0f}% tolerance")
        results.append({
            "stress_type": stress_name,
            "max_tolerance": max_survived * 100
        })

    # Create data entry in same format as existing data
    data_entry = {
        "name": "JamRobust",
        "design_performance": design_perf,
        "design_efficiency": design_eff,
        "design_power": design_power,
        "design_min_headroom": design_headroom,
        "robustness_breakdown": results
    }

    return data_entry

if __name__ == "__main__":
    jamrobust_data = test_jamrobust_robustness()

    print("\n" + "="*60)
    print("JamRobust Data Entry:")
    print("="*60)
    print(json.dumps(jamrobust_data, indent=2))

    # Save to separate file
    with open('jamrobust_robustness_data.json', 'w') as f:
        json.dump(jamrobust_data, f, indent=2)

    print("\n✓ Data saved to jamrobust_robustness_data.json")
