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

**You should have Spark 2.4 installed and properly configured.**

## Saved reference model

Our best performing model, the one referenced in our report, is saved as a
series of `.pth` files in `models/report_baseline/m1` and `models/report_baseline/m2`.
These are loaded as an ensemble when doing inference.

## Training models

There are a number of models based on different strategies and architectures that were
implemented in this project. There are separate scripts to train these models, each
with a filename like `run_{modelname}.py`.

You would run the scripts by using the command `python run_{modelname}.py`. Some of
the scripts accept a number of arguments. You can find out more about available
arguments for a script adding a `--help` argument.

Here are examples of running the main models:

### Baseline model

To train our baseline, and best-performing model, run the following script:

```
python src/run_baseline_ubest_fine.py
```

This trains an ensemble of RestNet-18 pretrained networks with full network finetuning.

The models will be saved to the `models/baseline_ubest_fine` directory. Two models will
be trained, with two checkpoints saved for each. The models' checkpoints and results metrics
will be saved in `models/baseline_ubest_fine/m1` and `models/baseline_ubest_fine/m2`.
Charts of with overall scores of the ensemble will be saved in `models/baseline_ubest_fine` folder.

Here's a summary of the output folder structure:

```
models/
  baseline_ubest_fine/       -> contains plots of ensemble scores
    m1/                      -> contains metrics .pth snaphosts files of the first model
    m2/                      -> contains metrics and .pth snaphosts of the second model
```

### Specialized models for frontal and lateral images

Script to run:

```
python src/run_multiside.py
```

Output folder structure:

```
models/
  multiside/    -> contains plots of the final ensemble
    base/       -> contains .pth snapshots and metrics of the base model
    frontal/    -> contains .pth snapshots and metrics of the frontal model
    lateral     -> contains .pth snaphosts and metrics of the lateral model
```


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

By default, the program will look for the `.pth` files in `models/report_baseline/m1` and `models/baseline/m2` and use them as an ensemble model for doing predicitions. You can specify
a different model to load using the `--model_dirs` option. This should be a comma-separated list of directories containing `.pth` files.

For example:

```
python src/run_predict --model_dirs=models/mymodel/m1,models/mymodel/m2
```


For more information about this command run:

```
python src/run_predict -h
```