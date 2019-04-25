import argparse
from baseline_core import run_baseline

if __name__ == '__main__':
    print("Baseline model with 0.0 for uncertain")
    output_dir = './models/baseline_u0'
    run_baseline(output_dir=output_dir, uncertainty_strategy='zero')