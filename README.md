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

## Training the model

You can train the model by running the `start.py` script:

```
python start.py
```

The script accepts a number of parameters like learning rate, number of epochs, etc.
to allow you to tune the training process, use `-h` option to see all available options:

```
python start.py -h
```