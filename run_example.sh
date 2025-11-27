#!/bin/bash
# Example script to run a comprehensive simulation experiment
# This demonstrates the chip design optimization simulator

echo "========================================="
echo "Chip Design Optimization Simulator"
echo "Example Experiment Run"
echo "========================================="
echo ""
echo "This script will run 100 simulations comparing two optimization strategies:"
echo "  1. Greedy Performance Maximizer (maximize performance)"
echo "  2. Log-Min-Headroom Optimizer (preserve constraint margins)"
echo ""
echo "The experiment will:"
echo "  - Run 100 independent simulations"
echo "  - Each with 75 design steps"
echo "  - Followed by a random requirement shift"
echo "  - Then 25 adaptation steps"
echo "  - Results saved to example_results.json"
echo ""
read -p "Press Enter to start the experiment (or Ctrl+C to cancel)..."
echo ""

# Run the experiment
python run_experiments.py \
  --runs 100 \
  --design-steps 75 \
  --adapt-steps 25 \
  --checkpoint-freq 10 \
  --output example_results.json \
  --seed 42

echo ""
echo "========================================="
echo "Experiment Complete!"
echo "========================================="
echo ""
echo "Analyzing results..."
echo ""

# Analyze the results
python analyze_results.py example_results.json

echo ""
echo "========================================="
echo "Additional Analysis Options"
echo "========================================="
echo ""
echo "To view failure cases only:"
echo "  python analyze_results.py example_results.json --failures"
echo ""
echo "To export to CSV:"
echo "  python analyze_results.py example_results.json --csv example_results.csv"
echo ""
echo "To view individual runs:"
echo "  python analyze_results.py example_results.json --show-runs"
echo ""
echo "For more options, see:"
echo "  python run_experiments.py --help"
echo "  python analyze_results.py --help"
echo ""
