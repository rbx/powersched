import numpy as np
import subprocess
import itertools
import argparse
import os
import sys

def generate_weight_combinations(step=0.1, fixed_weight=None, fixed_value=None):
    weights = np.linspace(0, 1, num=int(1/step) + 1, endpoint=True)
    combinations = []

    if fixed_weight:
        # If a weight is fixed, we only need to vary the other two
        other_weights = [w for w in ['efficiency', 'price', 'idle'] if w != fixed_weight]
        for w in weights:
            remaining = round(1 - fixed_value - w, 2)
            if 0 <= remaining <= 1:
                combo = [0, 0, 0]
                combo[['efficiency', 'price', 'idle'].index(fixed_weight)] = fixed_value
                combo[['efficiency', 'price', 'idle'].index(other_weights[0])] = round(w, 2)
                combo[['efficiency', 'price', 'idle'].index(other_weights[1])] = round(remaining, 2)
                combinations.append(tuple(combo))
    else:
        # If no weight is fixed, generate all combinations
        for e, p in itertools.product(weights, repeat=2):
            i = round(1 - e - p, 2)
            if 0 <= i <= 1:
                combinations.append((round(e, 2), round(p, 2), round(i, 2)))

    return combinations

def run(efficiency_weight, price_weight, idle_weight, iter_limit_per_step, session):
    python_executable = sys.executable
    command = [
        python_executable, "train.py",
        "--efficiency-weight", f"{efficiency_weight:.2f}",
        "--price-weight", f"{price_weight:.2f}",
        "--idle-weight", f"{idle_weight:.2f}",
        "--iter-limit", f"{iter_limit_per_step}",
        "--session", f"{session}"
    ]
    print(f"executing: {command}")
    current_env = os.environ.copy()
    result = subprocess.run(command, capture_output=False, text=True, env=current_env)
    if result.returncode != 0:
        print(f"Error occurred: {result.stderr}")
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for weights")
    parser.add_argument("--step", type=float, default=0.1, help="Step size for weight combinations")
    parser.add_argument("--fix-weight", choices=['efficiency', 'price', 'idle'], help="Which weight to fix")
    parser.add_argument("--fix-value", type=float, help="Value for the fixed weight")
    parser.add_argument("--iter-limit-per-step", type=int, help="Max number of training iterations per step (1 iteration = {TIMESTEPS} steps)")
    parser.add_argument("--session", help="Session ID")
    args = parser.parse_args()

    if args.fix_weight and args.fix_value is None:
        parser.error("--fix-value must be provided when --fix-weight is set")

    combinations = generate_weight_combinations(step=args.step, fixed_weight=args.fix_weight, fixed_value=args.fix_value)

    for combo in combinations:
        efficiency_weight, price_weight, idle_weight = combo
        print(f"Running with weights: {efficiency_weight}, {price_weight}, {idle_weight}")
        run(efficiency_weight, price_weight, idle_weight, iter_limit_per_step=args.iter_limit_per_step, session=args.session)

if __name__ == "__main__":
    main()
