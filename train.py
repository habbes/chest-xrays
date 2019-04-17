import argparse
import lib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CheXpert classifier")

    parser.add_argument("--max_train_samples", type=int, default=None, metavar="s",
        help="Max number of training samples to use for training. (default is to use all training data)")
    parser.add_argument("--lr", type=float, metavar="l",
        help="learning rate (default: 0.0001)", default=0.0001)
    parser.add_argument("--epochs", type=int, default=3, metavar="e",
        help="Number of epochs (default: 3)")
    parser.add_argument("--finetune", type=bool, default=False, metavar="f",
        help="Whether to fine tune all weights during training, otherwise only weights in the final FC layer are trained (default: false)")
    parser.add_argument("--result_prefix", type=str, default="", metavar="p",
        help="prefix added to result output file names")
    parser.add_argument("--arch", type=str, default="densenet", metavar="a",
        help="model architecture to use, either 'densnet' or 'resnet'. Default is densenet")
    parser.add_argument("--output_path", type=str, default="./model/model.pth", metavar="o",
        help="file where to save trained model")
    
    args = parser.parse_args()

    trainer = lib.Trainer(
        max_train_samples=args.max_train_samples,
        epochs= args.epochs,
        lr=args.lr,
        arch=args.arch,
        finetune=args.finetune,
        output_path=args.output_path
    )
    trainer.train()
    y_true, y_pred = trainer.evaluate()
    lib.plot_roc_auc(y_true, y_pred, save_to_file=True, prefix=args.result_prefix)