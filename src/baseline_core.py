import argparse
import os
import time
import sklearn.metrics as mt

from trainer import Trainer, plot_roc_auc, evaluate
from model import load_ensemble_from_dirs
from util import get_device, display_elapsed_time
from dataset import get_val_loader

def run_baseline(
    output_dir,
    num_models=3,
    max_train_samples=None,
    epochs=3,
    finetune=False,
    value_for_uncertain=1.0
    ):

    started = time.time()
    model_dirs = []
    for i in range(num_models):
        model_dir = os.path.join(output_dir, f'm{i + 1}')
        model_dirs.append(model_dir)

    for model_dir in model_dirs:
        print("=== Training model", model_dir, "===")
        trainer = Trainer(
            max_train_samples=max_train_samples,
            epochs=epochs,
            finetune=finetune,
            value_for_uncertain=value_for_uncertain,
            output_path=model_dir
        )
        trainer.train()
        y_true, y_pred = trainer.evaluate()
        plot_roc_auc(y_true, y_pred, save_to_file=True, output_path=model_dir)

        print("=== Completed training of", model_dir, "===")
        display_elapsed_time(started, "Total elapsed")
        print()
    
    ensemble = load_ensemble_from_dirs(model_dirs)
    results = evaluate(
        model=ensemble,
        dataloader=get_val_loader(),
        device=get_device()
    )
    labels = results['labels']
    preds = results['predictions']
    final_auc = mt.roc_auc_score(labels, preds)
    print("Ensemble Validation AUC Score", final_auc)
    plot_roc_auc(labels, preds, save_to_file=True, output_path=output_dir)
    display_elapsed_time(started, "Total time taken")
