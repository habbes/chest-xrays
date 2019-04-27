import argparse
from baseline_core import run_baseline

if __name__ == '__main__':
    print("Baseline model with 1.0 for uncertain")
    output_dir = './models/baseline_u1'
    run_baseline(output_dir=output_dir, uncertainty_strategy='one')