# Diagnosis of Lymphocytosis via Multiple Instance Learning
DLMI Data Challenge: https://www.kaggle.com/c/3md3070-dlmi/overview 
Team Name: Juju & Pedro
Julia LINHART & Pierre CLAVIER

MVA 2020/21 - ENS Paris-Saclay

## Overview:
This repository contains the PyTorch implementation of our solution to the above stated Data Challenge. Our models, methods and results are detailed and explained in the provided report. Our solution contains the following steps:
- Feature extraction at instance-level using a pretrained ResNet50 (see ```features_extrator.py```)
- the implementation of two different baseline MIL-model architectures - **CHOWDER** and **DeepMIL** - modified by integrating additional features (age, sex and lymph_count), as well as a variational version of DeepMIL using VAE for generalization purposes (see ```models_DLMI.py```)

The other python files consist in:
- ```functions_DLMI.py``` providing train and test functions for each model, as well as the preprocessing function for the additional features 
- ```main_DLMI.py``` that runs training of a given model and writes the predictions of the testset in a csv file, to export them in the suitable format to the data challenge platform

## Instructions to run the code and reproduce results:
The data folder needs to be placed in the smae directory as the python files and contain:
- the folders *trainset* and *testset* 
- the file *clinical_annotations.csv*
form the Challenge platform.

### Feature extraction
To extract features, please create a folder named *features* in both, trainset and testset folders. Then execute the following command line:

```python features_extractor --data_dir <data_folder_name>```

### Model training + Submission Results
First set the right step size and gamma parameter of the lr_scheduler in the *main_DLMI.py* file. Then run the following commandlines to reproduce the results for 
- **E-CHOWDER + AF model**:

```python main_DLMI.py --data_dir <data_folder_name> --model CHOWDER --batch_size 15 --num_epochs 100 --lr 0.001 --n_models 10 --reg_lambda 0.4 --lymph_count_features True --name Julia```

- **E-DeepMIL + AF model** (best submission scores):

```python main_DLMI.py --data_dir <data_folder_name> --model DeepMIL --batch_size 15 --num_epochs 50 --lr 0.001 --n_models 10 --lymph_count_features True --name Julia```

- **Variational E-DeepMIL + AF model**:

```python main_DLMI.py --data_dir <data_folder_name> --model auto_DeepMIL --batch_size 15 --num_epochs 100 --lr 0.001 --n_models 1 --lymph_count_features True --name Julia```


