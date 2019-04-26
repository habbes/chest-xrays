import numpy as np
import pandas as pd
import torch
import os
import time

from dataset import LABELS, get_test_loader
from model import get_model, load_ensemble_from_dirs

def predict(model, dataloader, device):
    model.eval()
    all_preds = None
    all_paths = None
    for inputs, img_paths in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)
        np_preds = preds.detach().cpu().numpy()
        np_paths = np.array(img_paths)
        if all_preds is None:
            all_preds = np_preds
            all_paths = np_paths
        else:
            all_preds = np.vstack((all_preds, np_preds))
            all_paths = np.concatenate((all_paths, np_paths))
    return all_preds, all_paths

def aggregate_studies(preds, paths):
    results = {}
    for i, p in enumerate(paths):
        study = os.path.dirname(p)
        scores = preds[i]
        if study in results:
            results[study] = np.max(np.vstack((results[study], scores)), axis=0)
        else:
            results[study] = scores
    return results

def results_to_df(agg_results):
    rows = []
    for study, scores in agg_results.items():
        row = [study] + list(scores)
        rows.append(row)
    return pd.DataFrame(rows, columns=['Study'] + LABELS)

class Predict():
    def __init__(self, input_csv, model_dirs, output_csv='./results/results.csv'):
        model = load_ensemble_from_dirs(model_dirs)
        self.model = model
        self.model_dirs = model_dirs
        self.output_csv = output_csv
        self.input_csv = input_csv
        self.dataloader = get_test_loader(input_csv)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.raw_results = None
        self.result_df = None
    
    def predict(self):
        print(f"Running predictions on dataset '{self.input_csv}' using models '{self.model_dirs}'...")
        started = time.time()
        self.raw_results = predict(self.model, self.dataloader, self.device)
        duration = time.time() - started
        print(f"Prediction completed in {duration}s")
        print("Collecting results...")
        aggregated = aggregate_studies(*self.raw_results)
        self.result_df = results_to_df(aggregated)
        self.result_df.to_csv(self.output_csv, index=False)
        print(f"Results saved to {self.output_csv}")
        return self.result_df

        

