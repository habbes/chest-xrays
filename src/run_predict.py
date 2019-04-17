from predict import Predict
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CheXpert prediction")

    parser.add_argument('input_csv', type=str,
        help="path of the input csv file that contains paths of the images to run predictions on")
    parser.add_argument('output_csv', type=str,
        help="path where to save the csv file that contains predictions for each study")
    parser.add_argument('--model_file', type=str, metavar='m', default="./model/model.pth",
        help="path to the saved model (default= ./model/model.pth)")
    args = parser.parse_args()

    model = Predict(args.input_csv, model_path=args.model_path, output_csv=args.output_csv)
    model.predict()
    print("Done")