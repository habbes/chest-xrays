# CheXpert

## Setup

Before running the code, you should get set up the dataset. The model exports the CheXpert dataset to be in the `data` directory
of the repo. The dataset should correspond to the small version of the dataset, and the name of the dataset folder should be `CheXpert-v1.0-small`,
i.e. you should end up with the following data layout:

```
data/
    CheXpert-v1.0-small/
        train/
        valid/
        train.csv
        valid.csv
```

If you already have the data on your machine and don't want to make a copy to the `data` folder mentioned above, you can create a symbolic link.
For example, on the compute node the data is located in `/data/chexpert/CheXpert-v1.0-small`. You can create the symbolic link by navigating
into the `data` folder of this repository then running:

```
ln -s  /data/chexpert/CheXpert-v1.0-small CheXpert-v1.0-small
```

To setup the environment, use conda to create the environment. Inside the project folder run following command:

```
conda env create -f environment.yml
```
Before running code, you will be required to activate conda environment:

```
conda activate pyspark_env
```

## Training the model

You can train the model by running the `start.py` script:

```
python src/run_train.py
```

The script accepts a number of parameters like learning rate, number of epochs, etc.
to allow you to tune the training process, use `-h` option to see all available options:

```
python src/run_train.py -h
```

### Saved model

The best model is saved in `model/model.pth` by default. You can change where the model is saved by setting
the `--output_path` option.

## Prediction

To run inference on a dataset, use the following command:

```
python src/run_predict.py <input_csv> <output_csv>
```
where:

- **input_csv**: is a single-column CSV file where each row is the path of an image to run predictions. If multiple images are in the same directory,
they will be considered to be in the same study. The column `Path` as the header.
Here's what the contents of a sample input csv file should look like:

```
Path
CheXpert-v1.0/valid/patient00000/study1/view1_frontal.jpg
CheXpert-v1.0/valid/patient00000/study1/view2_lateral.jpg
CheXpert-v1.0/valid/patient00000/study2/view1_frontal.jpg
```
- **output_csv**: this is the path where the results should be saved. The results are saved in a CSV file where each row contains the predicted
probabilities of a given **study** (not image).

Here's a sample output file:

```
Study,Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion
CheXpert-v1.0/valid/patient00000/study1,0.18182496720710062,0.18340450985343382,0.3042422429595377,0.5247564316322378,0.43194501864211576
CheXpert-v1.0/valid/patient00000/study2,0.5924145688620425,0.046450412719997725,0.6075448519014384,0.17052412368729153,0.06505159298527952
```

### Specifying model for prediction

By default, the program will look for the model in `model/model.pth`. You can specify
a different model to load using the `--model_path` option.


For more information about this command run:

```
python src/run_predict -h
```