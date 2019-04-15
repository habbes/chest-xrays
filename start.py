import argparse
import chexpert as chx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CheXpert classifier")

    parser.add_argument("--max_train_samples", type=int, default=None, metavar="S",
        help="Max number of training samples to use for training. (default is to use all training data)")
    parser.add_argument("--lr", type=float,
        help="learning rate (default: 0.0001)", default=0.0001)
    parser.add_argument("--epochs", type=int, default=3, metavar="E",
        help="Number of epochs (default: 3)")
    parser.add_argument("--finetune", type=bool, default=False, metavar="F",
        help="Whether to fine tune all weights during training, otherwise only weights in the final FC layer are trained (default: false)")
    parser.add_argument("--result_prefix", type=str, default="", metavar="P",
        help="prefix added to result output file names")
    parser.add_argument("--arch", type=str, default="densenet", metavar="A",
        help="model architecture to use, either 'densnet' or 'resnet'. Default is densenet")
    
    args = parser.parse_args()

    trainer = chx.Trainer(
        max_train_samples=args.max_train_samples,
        epochs= args.epochs,
        lr=args.lr,
        arch=args.arch,
        finetune=args.finetune
    )
    trainer.train()
    y_true, y_pred = trainer.evaluate()
    chx.plot_roc_auc(y_true, y_pred, save_to_file=True, prefix=args.result_prefix)