import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models_DLMI import CHOWDER, DeepMIL
from functions_DLMI import train_ch,test_ch,train_DMIL,test_DMIL

from os import listdir
from os.path import isfile, join


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="directory where data is stored")
parser.add_argument("--num_epochs", default=30, type=int,
                    help="Number of epochs for training")
parser.add_argument("--batch_size", default=10, type=int,help="Mini-batch size")
parser.add_argument("--reg_lambda", default=0.5, type=float,
                    help="L2-regularization trade-off parameter for conv weights")
parser.add_argument("--lr", default=0.0001, type=float,help="Learning Rate")
parser.add_argument("--weight_decay", default=0.0, type=float,help="weight decay for Adam optimizer")

parser.add_argument("--n_models", default=1, type=int,
                    help="number of chowder models for ensemble prediction")
parser.add_argument("--ann_lambda", default=0.5, type=float,
                    help="additional importance for annotated patients")
parser.add_argument("--model", required=True, type=str,
                    help="chosen model (CHOWDER or DeepMIL)")


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
        # if i%30==0:
            # print(patient_features,patient_features.shape)

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

    train_output = pd.read_csv(train_output_filename)

    test_output_filename = args.data_dir /"testset" /"testset_data.csv"

    test_output = pd.read_csv(test_output_filename)

    # list of all patients paths
    patient_filenames_train = [train_dir /"features" / Path(str(idx)) for idx in train_output["ID"]]

    patient_filenames_test = [test_dir  /"features" / Path(str(idx)) for idx in test_output["ID"]]

    #print(patient_filenames_train)

    # list of all features per patients
    #features_train_patients=[]
    #features_test_patients=[]

    #for i, patient_path in  enumerate(patient_filenames_train):

    #    features_train_patients.append( [patient_path  /"features"/ Path(str(f) +')' ) for f in listdir(patient_path) if isfile(join(patient_path, f))] )

    #for i, patient_path in  enumerate(patient_filenames_test):

    #    features_test_patients.append( [patient_path  /"features"/ Path(str(f) +')' ) for f in listdir(patient_path) if isfile(join(patient_path, f))] )


    features_train = get_features(patient_filenames_train)
    features_test = get_features(patient_filenames_test)

    # Get the labels
    labels_train = train_output["LABEL"].values
    print(labels_train)

    #assert len(filenames_train) == len(labels_train)
    # transform to tensor

    features_train_torch = torch.Tensor(features_train)
    labels_train_torch = torch.Tensor(labels_train[:,None])
    #(labels_train_torch)
    #is_annotated_train_torch = torch.Tensor(np.array(is_annotated_train)[:,None])
    features_test_torch = torch.Tensor(features_test)

     # define data loaders for pytorch model and split training set into train and validation sets
    dataset_train = TensorDataset(features_train_torch,labels_train_torch) #is_annotated_train_torch) # create your datset
    dataset_test = TensorDataset(features_test_torch) # create your datset

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
        print('CHOWDER Model parameters: batch_size = {}, lr = {}, weight_decay {}, convolution l2-reg = {}, annotation importance = {}'.format(args.batch_size,args.lr,args.weight_decay,args.reg_lambda,args.ann_lambda))
    elif args.model == 'DeepMIL':
        print('DeepMIL Model parameters: batch_size = {}, lr = {}, weight_decay {}, annotation importance = {}'.format(args.batch_size,args.lr,args.weight_decay))#,args.ann_lambda))
    print()
    # train and evaluate chowder-ensemble model
    ensemble_probs = np.zeros(len(dataset_test))

    for i in range(args.n_models):
        #define model
        if args.model == 'CHOWDER':
            model = CHOWDER()
        elif args.model == 'DeepMIL':
            model = DeepMIL()
        else:
            print('model not defined')

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

        #train model
        print('Training model {}/{} ...'.format(i,args.n_models-1))
        print()
        if args.model == 'CHOWDER':
            best_model,metrics= train_ch(model, train_loader, val_loader, criterion, optimizer, n_epochs=args.num_epochs,reg_lambda=args.reg_lambda,ann_lambda=args.ann_lambda)
        elif args.model == 'DeepMIL':
            best_model,metrics = train_DMIL(model, train_loader, val_loader, criterion, optimizer, n_epochs=args.num_epochs,ann_lambda=args.ann_lambda)
        else:
            print('model not defined')

        print('Model {} successfully trained'.format(i))

       # evaluate model on test set
        #print('Inference on test set...')
        #if args.model == 'CHOWDER':
         #   results_df, _ = test_ch(best_model,test_loader,criterion,reg_lambda=args.reg_lambda,ann_lambda=args.ann_lambda,test=True)
        #elif args.model == 'DeepMIL':
        #    results_df, _ = test_DMIL(best_model,test_loader,criterion,ann_lambda=args.ann_lambda,test=True)
        #else:
        #    print('model not defined')

       # print('Model {} successfully evaluated'.format(i))
       # print()
       # probs = results_df['proba']
       # ensemble_probs=ensemble_probs+probs
    #preds_test = ensemble_probs/args.n_models

    #save train/val history of one model
    #history = pd.DataFrame.from_dict(metrics)
    #history.to_csv(args.data_dir / "{}_history_ann.csv".format(args.model))
    #print('Training history saved.')
    #print()
    # Check that predictions are in [0, 1]
   # assert np.max(preds_test) <= 1.0
   # assert np.min(preds_test) >= 0.0

    # -------------------------------------------------------------------------
    # Write the predictions in a csv file, to export them in the suitable format
    # to the data challenge platform
    #ids_number_test = [i.split("ID_")[1] for i in ids_test]
    #test_output = pd.DataFrame({"ID": ids_number_test, "Target": preds_test})
    #test_output.set_index("ID", inplace=True)
    #test_output.to_csv(args.data_dir / "preds_test_deepMil_1_ann.csv")
    #print('Results saved!')
