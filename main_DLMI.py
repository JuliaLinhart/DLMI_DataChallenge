

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn
import sklearn.linear_model

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models_DLMI import CHOWDER, DeepMIL,auto_DeepMIL
from functions_DLMI import train_ch,test_ch,train_DMIL,test_DMIL , preprocess, train_auto_DMIL,test_auto_DMIL

from os import listdir
from os.path import isfile, join

import random


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="directory where data is stored")
parser.add_argument("--save_dir", default='results', type=Path,
                    help="directory where results are saved")
parser.add_argument("--num_epochs", default=100, type=int,
                    help="Number of epochs for training")
parser.add_argument("--batch_size", default=15, type=int,help="Mini-batch size")
parser.add_argument("--reg_lambda", default=0.1, type=float,
                    help="L2-regularization trade-off parameter for conv weights")
parser.add_argument("--lr", default=0.0002, type=float,help="Learning Rate")  # 0.0002
parser.add_argument("--weight_decay", default=0.0, type=float,help="weight decay for Adam optimizer")

parser.add_argument("--n_models", default=1, type=int,
                    help="number of chowder models for ensemble prediction")
parser.add_argument("--model", required=True, type=str,
                    help="chosen model (CHOWDER or DeepMIL, auto_DeepMIL)")
parser.add_argument("--lymph_count_features", default=False, type=bool,
                    help="add lymph count features")
parser.add_argument("--reg_kl", default=0.5, type=float,
                    help="KL-divergence regulation for VAE")


parser.add_argument("--name", required=True, type=str,
                    help="Julia or Pierre")

def get_features(filenames):
    """Load and aggregate the resnet features by the average.

    Args:
        filenames: list of filenames of length `num_patients` corresponding to resnet features

    Returns:
        features: np.array of mean resnet features, shape `(num_patients, 2048)`
    """
    # Load numpy arrays
    features = []
    for i,f in enumerate(filenames):
        patient_features = pd.read_csv(f,index_col=False).values
        patient_features = patient_features[:,1:] #REMOVE fist colomn of index
        features.append(patient_features)
    features = np.stack(features,axis=0)
    print("features extracted",features.shape)
    return features

if __name__ == "__main__":
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load the data
    assert args.data_dir.is_dir()

    train_dir = args.data_dir / "trainset"
    test_dir = args.data_dir / "testset"

    train_output_filename = train_dir / "trainset_true.csv"
    test_output_filename = args.data_dir /"testset" /"testset_data.csv"

    test_output = pd.read_csv(test_output_filename)
    train_output=pd.read_csv(train_output_filename )

    # list of all patients paths
    if args.name=='Julia':
        patient_filenames_train = [train_dir /"features" / Path(str(idx)) for idx in train_output["ID"]]

        patient_filenames_test = [test_dir  /"features" / Path(str(idx)) for idx in test_output["ID"]]
    else :
        patient_filenames_train = [train_dir /"features" / Path(str(idx)+")") for idx in train_output["ID"]]

        patient_filenames_test = [test_dir  /"features" / Path(str(idx)+")") for idx in test_output["ID"]]

    # Preprocess data
    train_output=preprocess(train_output)
    test_output=preprocess(test_output)

    additional_features_train=np.array(train_output[['LYMPH_COUNT','AGE','SEX']]).reshape(-1,3)
    additional_features_test=np.array(test_output[['LYMPH_COUNT','AGE','SEX']]).reshape(-1,3)

    # Get the labels
    labels_train = train_output["LABEL"].values

    # get extracted image features
    features_train = get_features(patient_filenames_train)
    features_test = get_features(patient_filenames_test)

    # convert to torch tensors
    features_train_torch = torch.Tensor(features_train)
    labels_train_torch = torch.Tensor(labels_train[:,None])
    features_test_torch = torch.Tensor(features_test)

    add_features_pytorch_train=torch.Tensor(additional_features_train)
    add_features_pytorch_test=torch.Tensor(additional_features_test)
    n_add_features = add_features_pytorch_train.size(1)

    # define data loaders for pytorch model and split training set into train and validation sets
    dataset_train = TensorDataset(features_train_torch,labels_train_torch,add_features_pytorch_train) # create your datset
    dataset_test = TensorDataset(features_test_torch,add_features_pytorch_test) # create your datset

    train_len = int(0.7*len(dataset_train))
    valid_len = len(dataset_train) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset_train,[train_len,valid_len])

    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=len(val_set),shuffle=False)
    test_loader = DataLoader(dataset_test,batch_size=len(dataset_test),shuffle=False)
    print()
    print('Data successfully loaded: train size = {}, val size {}, test size = {}'.format(len(train_set),len(val_set), len(dataset_test)))
    print()

    if args.model == 'CHOWDER':
        print('CHOWDER Model parameters: batch_size = {}, lr = {}, weight_decay {}, convolution l2-reg = {}, lymph_count features = {}'.format(args.batch_size,args.lr,args.weight_decay,args.reg_lambda,args.lymph_count_features))
    elif args.model == 'DeepMIL':
        print('DeepMIL Model parameters: batch_size = {}, lr = {}, weight_decay {}'.format(args.batch_size,args.lr,args.weight_decay))
    elif args.model == 'auto_DeepMIL':
        print('auto_DeepMIL Model parameters: batch_size = {}, lr = {}, weight_decay {}, kl_reg = {}'.format(args.batch_size,args.lr,args.weight_decay,args.reg_kl))
    print()
    print()

    # train and evaluate chowder-ensemble model
    ensemble_probs = np.zeros(len(dataset_test))
    val_ba = 0

    for i in range(args.n_models):
        #define model
        if args.model == 'CHOWDER':
            model = CHOWDER(lymph_count=args.lymph_count_features,num_add_features = n_add_features)
        elif args.model == 'DeepMIL':
            model = DeepMIL(lymph_count=args.lymph_count_features,num_add_features = n_add_features)
        elif args.model == 'auto_DeepMIL':
            model = auto_DeepMIL(lymph_count=args.lymph_count_features,num_add_features = n_add_features)
        else:
            print('model not defined')

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

        #train model
        print('Training model {}/{} ...'.format(i,args.n_models-1))
        print()
        if args.model == 'CHOWDER':
            best_model,metrics= train_ch(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs=args.num_epochs,reg_lambda=args.reg_lambda)
        elif args.model == 'DeepMIL':
            best_model,metrics = train_DMIL(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs=args.num_epochs)
        elif args.model == 'auto_DeepMIL':
            best_model,metrics = train_auto_DMIL(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs=args.num_epochs, kl_reg = args.reg_kl)
        else:
            print('model not defined')

        print('Model {} successfully trained'.format(i))

        # evaluate model on val set
        print('Inference on val set...')
        if args.model == 'CHOWDER':
           results_df, val_metrics = test_ch(best_model,val_loader,criterion,reg_lambda=args.reg_lambda)
        elif args.model == 'DeepMIL':
           results_df, val_metrics = test_DMIL(best_model,val_loader,criterion)

        elif args.model == 'auto_DeepMIL':
           results_df, val_metrics = test_auto_DMIL(best_model,val_loader,criterion,kl_reg=args.reg_kl)
        else:
           print('model not defined')
        val_ba+=val_metrics["balanced_accuracy"]

        # evaluate model on test set
        print('Inference on test set...')
        if args.model == 'CHOWDER':
           results_df, _ = test_ch(best_model,test_loader,criterion,reg_lambda=args.reg_lambda,test=True)
        elif args.model == 'DeepMIL':
           results_df, _ = test_DMIL(best_model,test_loader,criterion,test=True)

        elif args.model == 'auto_DeepMIL':
           results_df, _ = test_auto_DMIL(best_model,test_loader,criterion,kl_reg=args.reg_kl,test=True)
        else:
           print('model not defined')

        print('Model {} successfully evaluated'.format(i))
        print()
        probs = results_df['proba']
        ensemble_probs=ensemble_probs+probs
    preds_test = ensemble_probs/args.n_models
    val_ba = val_ba/args.n_models
    print("ensemble Val BA = {}".format(val_ba))

    #save train/val history of one model
    history = pd.DataFrame.from_dict(metrics)
    history.to_csv(args.save_dir / "{}_history.csv".format(args.model))
    print('Training history saved.')
    print()
    # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

    # -------------------------------------------------------------------------
    # Write the predictions in a csv file, to export them in the suitable format
    # to the data challenge platform
    ids_number_test = [str(idx) for idx in test_output["ID"]]
    test_output = pd.DataFrame({"ID": ids_number_test, "Predicted": np.round(preds_test)})
    test_output = test_output.astype({"ID": str, "Predicted": int})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv("preds_test_DeepMIL_AF_E10_3.csv")#args.save_dir /
    print('Results saved!')
