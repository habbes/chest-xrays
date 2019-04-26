from predict import Predict
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CheXpert prediction")

    parser.add_argument('input_csv', type=str,
        help="path of the input csv file that contains paths of the images to run predictions on")
    parser.add_argument('output_csv', type=str,
        help="path where to save the csv file that contains predictions for each study")
    parser.add_argument('--model_dirs', type=str, metavar='m', default="./models/baseline_ubest_fine/m1,./models/baseline_ubest_fine/m2",
        help="comma-separated list of dirs containing models to use for inference (default= ./models/baseline_ubest_fine/m1,./models/baseline_best_fine_m2)")
    args = parser.parse_args()

    model = Predict(args.input_csv, model_dirs=args.model_dirs, output_csv=args.output_csv)
    model.predict()
    print("Done")