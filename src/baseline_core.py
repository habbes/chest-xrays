import argparse
import os
import time
import sklearn.metrics as mt

from trainer import Trainer, plot_roc_auc, evaluate
from model import load_ensemble_from_dirs, get_model, get_feature_extractor, MultiSide
from util import get_device, display_elapsed_time
from dataset import get_val_loader

def run_baseline(
    output_dir,
    num_models=3,
    max_train_samples=None,
    epochs=3,
    finetune=False,
    uncertainty_strategy='best'
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
            uncertainty_strategy=uncertainty_strategy,
            output_path=model_dir,
            arch="resnet",
            layers=50
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

def run_multiside(
    output_dir
    ):
    max_batches=None
    started = time.time()
    sides = ['frontal', 'lateral']
    base_dir = os.path.join(output_dir, 'base')
    base_trainer = Trainer(
        max_train_samples=max_batches,
        epochs=1,
        finetune=True,
        uncertainty_strategy='best',
        output_dir=base_dir,
        arch="resnet"
    )
    print("Training base model")
    base_trainer.train()
    y_true, y_pred = base_trainer.evaluate()
    plot_roc_auc(y_true, y_pred, save_to_file=True, output_path=base_dir)

    print("=== Completed training of base model", base_dir, "===")
    display_elapsed_time(started, "Total elapsed")
    print("== Extracting features ==")
    base_model_ws = base_trainer.train_results['checkpoints'].checkpoints[0]['model']
    base_model = get_model(arch='resnet')
    base_model.load_state_dict(base_model_ws)
    print()

    print("Training frontal model")
    frontal_dir = os.path.join(output_dir, 'frontal')
    frontal_trainer = Trainer(
        max_train_samples=max_batches,
        epochs=2,
        model=get_feature_extractor(base_model),
        uncertainty_strategy='best',
        side='frontal',
        output_dir=frontal_dir
    )
    frontal_trainer.train()
    y_true, y_pred = frontal_trainer.evaluate()
    plot_roc_auc(y_true, y_pred, save_to_file=True, output_path=frontal_dir)
    print("=== Completed training frontal model", frontal_dir, "===")
    print()

    print("Training lateral model")
    lateral_dir = os.path.join(output_dir, 'lateral')
    lateral_trainer = Trainer(
        max_train_samples=max_batches,
        epochs=2,
        model=get_feature_extractor(base_model),
        uncertainty_strategy='best',
        side='lateral',
        output_dir=frontal_dir
    )
    lateral_trainer.train()
    y_true, y_pred = lateral_trainer.evaluate()
    plot_roc_auc(y_true, y_pred, save_to_file=True, output_path=lateral_dir)
    print("=== Completed training frontal model", lateral_dir, "===")
    print()

    frontal = load_ensemble_from_dirs([frontal_dir])
    lateral = load_ensemble_from_dirs([lateral_dir])

    multiside = MultiSide(frontal=frontal, lateral=lateral)
    
