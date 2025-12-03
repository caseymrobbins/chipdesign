#!/usr/bin/env python3
"""Analyze results from bug fix test"""

import json
import numpy as np

with open('greedy_vs_intrinsic_data.json', 'r') as f:
    data = json.load(f)

# Collect results by agent
agents = {}
for run in data:
    for agent_data in run:
        name = agent_data['name']
        if name not in agents:
            agents[name] = {
                'design_perf': [],
                'design_power': [],
                'design_min_headroom': [],
                'survived': [],
                'final_perf': [],
            }

        agents[name]['design_perf'].append(agent_data['design_performance'])
        agents[name]['design_power'].append(agent_data['design_power'])
        agents[name]['design_min_headroom'].append(agent_data['design_min_headroom'])
        agents[name]['survived'].append(agent_data['survived'])
        if agent_data['survived']:
            agents[name]['final_perf'].append(agent_data['final_performance'])

# Print results
print("="*80)
print("BUG FIX RESULTS - Full 50 Run Comparison")
print("="*80)
print("\nDESIGN PHASE PERFORMANCE (40 steps):")
print("-" * 80)

for name in sorted(agents.keys()):
    perf = agents[name]['design_perf']
    power = agents[name]['design_power']
    headroom = agents[name]['design_min_headroom']

    print(f"\n{name}:")
    print(f"  Performance:  {np.mean(perf):7.2f} ± {np.std(perf):5.2f}")
    print(f"  Power:        {np.mean(power):7.2f} ± {np.std(power):5.2f} W")
    print(f"  Min Headroom: {np.mean(headroom):7.4f} ± {np.std(headroom):6.4f}")

print("\n" + "="*80)
print("ROBUSTNESS (Survival Rate):")
print("-" * 80)

for name in sorted(agents.keys()):
    survived = agents[name]['survived']
    survival_rate = sum(survived) / len(survived) * 100
    survived_count = sum(survived)

    print(f"{name:20s}: {survival_rate:5.1f}% ({survived_count}/50)")

print("\n" + "="*80)
print("FINAL PERFORMANCE (Survivors Only):")
print("-" * 80)

for name in sorted(agents.keys()):
    final = agents[name]['final_perf']
    if final:
        print(f"{name:20s}: {np.mean(final):7.2f} ± {np.std(final):5.2f} (n={len(final)})")
    else:
        print(f"{name:20s}: No survivors")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Compare performances
industrybest_perf = np.mean(agents['IndustryBest']['design_perf'])
jam_perf = np.mean(agents['JAM']['design_perf'])
jamadv_perf = np.mean(agents['JAMAdvanced']['design_perf'])

print(f"\n1. JAMAdvanced vs IndustryBest:")
improvement = ((jamadv_perf - industrybest_perf) / industrybest_perf) * 100
print(f"   JAMAdvanced: {jamadv_perf:.2f}")
print(f"   IndustryBest: {industrybest_perf:.2f}")
print(f"   Improvement: {improvement:+.1f}%")
if jamadv_perf > industrybest_perf:
    print(f"   ✓ JAMAdvanced BEATS IndustryBest!")
else:
    print(f"   ✗ JAMAdvanced below IndustryBest")

print(f"\n2. JAMAdvanced vs JAM:")
diff = jam_perf - jamadv_perf
print(f"   JAM: {jam_perf:.2f}")
print(f"   JAMAdvanced: {jamadv_perf:.2f}")
print(f"   Difference: {diff:.2f} ({diff/jam_perf*100:.1f}%)")

print(f"\n3. Bug Fix Impact:")
print(f"   Before bug fix: 36.62 performance")
print(f"   After bug fix:  {jamadv_perf:.2f} performance")
print(f"   Improvement: +{((jamadv_perf - 36.62) / 36.62) * 100:.1f}%")

print("\n" + "="*80)
