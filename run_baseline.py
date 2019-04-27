import argparse
import time
import sklearn.metrics as mt

from trainer import Trainer, plot_roc_auc, evaluate
from model import load_ensemble_from_dirs
from util import get_device, display_elapsed_time
from dataset import get_val_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run baseline experiment")

    parser.add_argument("--max_train_samples", type=int, default=None, metavar="s",
        help="Max number of training batches to use for training")
    parser.add_argument("--epochs", type=int, default=3, metavar="e",
        help="Number of epochs (default: 3)")
    parser.add_argument("--finetune", type=bool, default=False, metavar="f",
        help="Whether to fine tune all weights during training, otherwise only weights in the final FC layer are trained (default: false)")
    parser.add_argument("--output_path", type=str, metavar="o",
        default="./models/baseline/m1,./models/baseline/m2,./models/baseline/m3",
        help="comma-separated directories where to save the trained models. Training is performed for each direcotry specified.")
    parser.add_argument("--uncertainty_strategy", type=str, metavar="u", default='best',
        help="Strategy for replacing uncertain values, one=replace with 1.0, zero=replace with 0.0, best=replace with 1.0 or 0.0 depending on disease")
    
    args = parser.parse_args()

    started = time.time()
    model_dirs = args.output_path.split(",")
    for model_dir in model_dirs:
        print("=== Training model", model_dir, "===")
        trainer = Trainer(
            max_train_samples=args.max_train_samples,
            epochs=args.epochs,
            finetune=args.finetune,
            uncertainty_strategy=args.uncertainty_strategy,
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
    plot_roc_auc(labels, preds, save_to_file=True, output_path="./models/baseline")
    display_elapsed_time(started, "Total time taken")
