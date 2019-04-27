import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import col , column
from pyspark.sql.functions import when

import os.path as path

spark = SparkSession.builder \
    .appName("Chexpert") \
    .master("local[4]") \
    .config("spark.driver.memory","11G") \
    .config("spark.driver.maxResultSize", "8G") \
    .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:1.8.2") \
    .config("spark.jars.packages", "databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11") \
    .config("spark.kryoserializer.buffer.max", "500m") \
    .getOrCreate()

DATA_DIR = './data'
TRAIN_CSV = './data/CheXpert-v1.0-small/train.csv'
VALID_CSV = './data/CheXpert-v1.0-small/valid.csv'

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
BATCH_SIZE = 8
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

def data_path(file_path):
    return path.join(DATA_DIR, file_path)

def add_side_to_df(df):
    side_udf = udf(lambda p: 'lateral' if 'lateral' in p else 'frontal')
    df.withColumn('side', (side_udf(df['Path'])))
    return df

def filter_by_side(df, side):
    return df.query(f"side=='{side}'")

class TrainingDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, uncertainty_strategy='best', side=None, image_paths=False):
        self.spark_df = add_side_to_df(spark.read.csv(csv_file, header=True))
        if side is not None:
            self.spark_df = filter_by_side(self.spark_df, side)
        self.data_dir = data_dir
        self.transform = transform
        self.uncertainty_strategy = uncertainty_strategy
        self.image_paths = image_paths

        if self.uncertainty_strategy == 'best':
            ones_labels = ['Atelectasis', 'Edema']
            zeros_labels = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
            for colName in LABELS:
                self.spark_df = self.spark_df.withColumn(colName, col(colName).cast("float")).fillna(0.0)
                if colName in ones_labels:
                    self.spark_df = self.spark_df.withColumn(colName, when(self.spark_df[colName] == -1.0, 1.0).otherwise(self.spark_df[colName]))
                elif colName in zeros_labels:
                    self.spark_df = self.spark_df.withColumn(colName, when(self.spark_df[colName] == -1.0, 0.0).otherwise(self.spark_df[colName]))
        else:
            value_for_uncertain = 1.0 if self.uncertainty_strategy == 'one' else 0.0
            for colName in LABELS:
                self.spark_df = self.spark_df.withColumn(colName, col(colName).cast("float")).fillna(0.0)
                self.spark_df = self.spark_df.withColumn(colName, when(self.spark_df[colName] == -1.0, value_for_uncertain).otherwise(self.spark_df[colName]))
        print('inside training')
        print(self.spark_df.count())
        self.spark_df.show()
        self.df = self.spark_df.toPandas()
        
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_path = path.join(self.data_dir, self.df.iloc[idx]['Path'])
        img = Image.open(img_path).convert('RGB')

        labels = self.df.iloc[idx][LABELS]
        labels = labels.values
        labels = torch.from_numpy(labels)
        print(labels)
        if self.transform:
            img = self.transform(img)
        if self.image_paths:
            return img, labels, img_path
        else:
            return img, labels


class TestDataset(Dataset):
    def __init__(self, csv_file, transform=None, data_dir=None):
        self.df = spark.read.csv(csv_file, header=True)
        self.transform = transform
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['Path']
        if self.data_dir is not None:
            img_path = path.join(self.data_dir, img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

def get_transformer():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_dataset(train_or_val, uncertainty_strategy='best', side=None):
    csv_file = TRAIN_CSV if train_or_val == "train" else VALID_CSV
    return TrainingDataset(csv_file, DATA_DIR, get_transformer(), uncertainty_strategy=uncertainty_strategy, side=side)

def get_loader(dataset, shuffle=True, batch_size=BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

def get_train_loader(uncertainty_strategy='best', side=None):
    return get_loader(get_dataset('train', uncertainty_strategy=uncertainty_strategy, side=side))

def get_val_loader(side=None):
    return get_loader(get_dataset('val', side=side), shuffle=False)

def get_test_dataset(csv_file, data_dir=None):
    return TestDataset(csv_file, get_transformer(), data_dir=data_dir)

def get_test_loader(csv_file, data_dir=None):
    return get_loader(get_test_dataset(csv_file, data_dir=data_dir), shuffle=False)

def get_val_loader_for_multiside():
    return get_loader(
        TrainingDataset(VALID_CSV, DATA_DIR, get_transformer(), image_paths=True),
        shuffle=False, batch_size=1
    )

