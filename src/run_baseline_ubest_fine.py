import argparse
from baseline_core import run_baseline

if __name__ == '__main__':
    print("Baseline model with best for uncertain, and finetuned model")
    output_dir = './models/baseline_ubest_fine_50'
    run_baseline(
        max_train_samples=None,
        output_dir=output_dir,
        uncertainty_strategy='best',
        finetune=True,
        num_models=2,
        epochs=2
    )