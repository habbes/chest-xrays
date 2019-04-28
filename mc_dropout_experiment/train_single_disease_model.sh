#!/bin/bash
# Train single disease models for 
# ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

python big_five_train.py 0 # Atelectasis
python big_five_train.py 1 # Cardiomegaly
#  and so on ...
