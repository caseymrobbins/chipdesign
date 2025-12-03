#!/usr/bin/env python3
"""Quick test to verify the bug fix in JAMAdvanced"""

from compare_greedy_vs_intrinsic import run_experiments

# Run just 3 iterations to quickly test the bug fix
results = run_experiments(
    num_runs=3,
    design_steps=40,
    adaptation_steps=25,
    seed=42,
    verbose=False,
)

print("\n" + "="*80)
print("BUG FIX TEST RESULTS")
print("="*80)

# Extract JAMAdvanced results (3rd agent in each run)
jamadv_perfs = []
print(f"\nJAMAdvanced Design Phase Performance:")
for run_idx, run_results in enumerate(results):
    for result in run_results:
        if 'JAMAdvanced' in result.name or 'SoftminJAM' in result.name:
            print(f"  Run {run_idx}: {result.design_performance:.2f}")
            jamadv_perfs.append(result.design_performance)
            break

if jamadv_perfs:
    avg = sum(jamadv_perfs) / len(jamadv_perfs)
    print(f"\nAverage: {avg:.2f}")
    print(f"\n{'='*80}")
    if avg > 36.62:
        improvement = ((avg - 36.62) / 36.62) * 100
        print(f"✓ BUG FIX SUCCESSFUL!")
        print(f"  JAMAdvanced: {avg:.2f} (was 36.62)")
        print(f"  Improvement: +{improvement:.1f}%")
    else:
        print(f"✗ Still stuck at {avg:.2f}")
    print("="*80)
else:
    print("\nERROR: Could not find JAMAdvanced results")
    print("="*80)
