
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn
import sklearn.linear_model

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models_DLMI import CHOWDER, DeepMIL
from functions_DLMI import train_ch,test_ch,train_DMIL,test_DMIL , preprocess

from os import listdir
from os.path import isfile, join


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="directory where data is stored")
parser.add_argument("--save_dir", default='results', type=Path,
                    help="directory where results are saved")
parser.add_argument("--num_epochs", default=100, type=int,
                    help="Number of epochs for training")
parser.add_argument("--batch_size", default=10, type=int,help="Mini-batch size")
parser.add_argument("--reg_lambda", default=0.1, type=float,
                    help="L2-regularization trade-off parameter for conv weights")
parser.add_argument("--lr", default=0.0002, type=float,help="Learning Rate")
parser.add_argument("--weight_decay", default=0.0, type=float,help="weight decay for Adam optimizer")

parser.add_argument("--n_models", default=50, type=int,
                    help="number of chowder models for ensemble prediction")
parser.add_argument("--ann_lambda", default=0.5, type=float,
                    help="additional importance for annotated patients")
parser.add_argument("--model", required=True, type=str,
                    help="chosen model (CHOWDER or DeepMIL)") 
parser.add_argument("--lymph_count_features", default=True, type=bool,
                    help="add lymph count features")
parser.add_argument("--lymph_count_weights", default=False, type=bool,
                    help="add lymph count probs as weights for BCE loss")

parser.add_argument("--name", required=True, type=str,
                    help="Julia or Pierre") 




def bonne_humeur():
    print("\n***************************************")
    print("Bonjour, \n                 ")  
    print("  ( )       | |( ) ( )    | ")
    print("   _  _   _ | | _  ____   | ")
    print("  | || | | || || |/_   |  | ")
    print("  | || |_| || || |/ (| |  | ")
    print("  | | \__,_||_||_|\__,_|  | ")
    print(" (__/                       ")
    print("                            ")
    print("Etonant de te retouver sur ce language barbar avec ce drole de jolie prenom...")

    print("                            ")
    print("***************************************")
    print(" ")
    print("        .....           ..... ")
    print("     ,ad8PPPP88b,     ,d88PPPP8ba, ")
    print("    d8P'      'Y8b, ,d8P'      'Y8b ")
    print("  dP'           '8a8'           `Yd ")
    print("  8(              '              )8 ")
    print("  I8                             8I  ")
    print("   Yb,                         ,dP  ")
    print("   '8a,                     ,a8'    ")
    print("      '8a,                 ,a8'     ")
    print("        'Yba             adP'           Signed EDC (cest pas moi o:) ")
    print("          `Y8a         a8P'  ")
    print("           `88,     ,88'  ")
    print("              '8b   d8'  ")
    print("               '8b d8'   ")
    print("                `888'    ")
    print("                          ")
    print(" ******************* CE CODE S'EXECUTE POUR L'INSTANT SANS ENCOMBRE ******************* \n")

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

def get_logreg_probs(features_train,features_test,labels_train):
    """Logistic Regression results for a training set of features

    Args:
        features_train: np array of training features
        labels_train: np array of corresponding labels

    Returns:
        binary classification probabilities for eachh training feature
    """
    estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
    estimator.fit(features_train, labels_train)
    probs_train = estimator.predict_proba(features_train)[:,1]
    probs_test = estimator.predict_proba(features_test)[:,1]
    return probs_train,probs_test

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

    if args.name=='Julia':

    # list of all patients paths
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

    # Get Classification probabilities based on Lymph_count features
    l_count_train = np.array(train_output['LYMPH_COUNT']).reshape(-1,1)
    l_count_test = np.array(test_output['LYMPH_COUNT']).reshape(-1,1)

    #l_count_probs_train,l_count_probs_test = get_logreg_probs(l_count_train,l_count_test,labels_train)  ### non juste les features pas le log des probs

    

    

    #bonne_humeur()

    # convert to torch tensors
    features_train_torch = torch.Tensor(features_train)
    labels_train_torch = torch.Tensor(labels_train[:,None])
    #l_count_probs_train_torch = torch.Tensor(l_count_probs_train[:,None,None])

    l_count_train_torch=torch.Tensor(l_count_train )
    l_count_test_torch=torch.Tensor(l_count_test ) 

    add_features_pytorch_train=torch.Tensor(additional_features_train)
    add_features_pytorch_test=torch.Tensor(additional_features_test)



    features_test_torch = torch.Tensor(features_test)
    #l_count_probs_test_torch = torch.Tensor(l_count_probs_test[:,None,None])


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
        print('CHOWDER Model parameters: batch_size = {}, lr = {}, weight_decay {}, convolution l2-reg = {}, lymph_count features = {}, lymph_count weights = {}'.format(args.batch_size,args.lr,args.weight_decay,args.reg_lambda,args.lymph_count_features, args.lymph_count_weights))
    elif args.model == 'DeepMIL':
        print('DeepMIL Model parameters: batch_size = {}, lr = {}, weight_decay {}'.format(args.batch_size,args.lr,args.weight_decay))#,args.ann_lambda))
    print()
    # train and evaluate chowder-ensemble model
    ensemble_probs = np.zeros(len(dataset_test))

    for i in range(args.n_models):
        #define model
        if args.model == 'CHOWDER':
            model = CHOWDER(lymph_count=args.lymph_count_features)
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
            best_model,metrics= train_ch(model, train_loader, val_loader, criterion, optimizer, n_epochs=args.num_epochs,reg_lambda=args.reg_lambda,lymph_count_weights=args.lymph_count_weights)
        elif args.model == 'DeepMIL':
            best_model,metrics = train_DMIL(model, train_loader, val_loader, criterion, optimizer, n_epochs=args.num_epochs,ann_lambda=args.ann_lambda)
        else:
            print('model not defined')

        print('Model {} successfully trained'.format(i))

        # evaluate model on test set
        print('Inference on test set...')
        if args.model == 'CHOWDER':
           results_df, _ = test_ch(best_model,test_loader,criterion,reg_lambda=args.reg_lambda,lymph_count_weights=args.lymph_count_weights,test=True)
        elif args.model == 'DeepMIL':
           results_df, _ = test_DMIL(best_model,test_loader,criterion,ann_lambda=args.ann_lambda,test=True)
        else:
           print('model not defined')

        print('Model {} successfully evaluated'.format(i))
        print()
        probs = results_df['proba']
        ensemble_probs=ensemble_probs+probs
    preds_test = ensemble_probs/args.n_models

    #save train/val history of one model
    #history = pd.DataFrame.from_dict(metrics)
    #history.to_csv(args.data_dir / "{}_history_ann.csv".format(args.model))
    #print('Training history saved.')
    #print()
    # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

    # -------------------------------------------------------------------------
    # Write the predictions in a csv file, to export them in the suitable format
    # to the data challenge platform
    ids_number_test = [str(idx) for idx in test_output["ID"]]
    test_output = pd.DataFrame({"ID": ids_number_test, "Predicted": np.round(preds_test)})
    test_output.astype({"ID": str, "Predicted": int})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(args.save_dir / "preds_test_LWChowder_E50.csv")
    print('Results saved!')
