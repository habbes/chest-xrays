import argparse
from baseline_core import run_baseline

if __name__ == '__main__':
    print("Baseline model with 1.0 for uncertain, and finetuned model")
    output_dir = './models/baseline_u1_fine'
    run_baseline(output_dir=output_dir, value_for_uncertain=1.0, finetune=True)