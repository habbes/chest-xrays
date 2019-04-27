import argparse
from baseline_core import run_baseline

if __name__ == '__main__':
    print("Baseline model with 1.0 for uncertain, and finetuned model")
    output_dir = './models/baseline_u1_fine'
    run_baseline(output_dir=output_dir, uncertainty_strategy='one', finetune=True, num_models=2)