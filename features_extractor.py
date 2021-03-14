import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from PIL import Image
import tqdm
from tqdm import tqdm


from os import listdir
from os.path import isfile, join
#from models import CHOWDER, DeepMIL
#from functions import train_ch,test_ch,train_DMIL,test_DMIL
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="directory where data is stored")

def get_features(filenames):


    # Load numpy arrays



    preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])



    feature_per_patient = np.zeros((2048,198)) # 198is the max number of images per patient
    for i,f in enumerate(filenames):
        patient_image = Image.open(f)
        patient_image = np.asarray(patient_image)

        # preprocess the image
        input_tensor = preprocess(patient_image)
        input_tensor=input_tensor.unsqueeze(0) # good dimentiont 1, 3, 224 ,224

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            newmodel.to('cuda')

        with torch.no_grad():
            output = newmodel(input_tensor)
        feature_per_patient[:,i] = output.reshape(2048)


    return feature_per_patient



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

    patient_filenames_train = [train_dir / Path(str(idx)) for idx in train_output["ID"]]
    patient_filenames_test = [test_dir / Path(str(idx)) for idx in test_output["ID"]]


    ### ============ extract train features =========== ###
    photos_patients_train=[]

    for patient_path in  tqdm(patient_filenames_train):

        photos_patients_train.append( [patient_path / Path(str(f)) for f in listdir(patient_path) if isfile(join(patient_path, f))] )

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

    newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    newmodel.eval()

    for i,idx in enumerate(train_output["ID"]): # for all patients
        print(idx)
        features =get_features(photos_patients_train[i]) # of size 2048 *194
        print(features.shape)
        df = pd.DataFrame(data=features)
        # output_path = args.data_dir / "testset\features\{})".format(str(idx))
        df.to_csv(train_dir / "features/{}".format(str(idx)))

    print("train features exctracted")

    ### ============ extract test features =========== ###

    photos_patients_test=[]

    for patient_path in  tqdm(patient_filenames_test):

        photos_patients_test.append( [patient_path / Path(str(f)) for f in listdir(patient_path) if isfile(join(patient_path, f))] )

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

    newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    newmodel.eval()

    for i,idx in enumerate(test_output["ID"]): # for all patients
        print(idx)
        features =get_features(photos_patients_test[i]) # of size 2048 *194
        print(features.shape)
        df = pd.DataFrame(data=features)
        # output_path = args.data_dir / "testset\features\{})".format(str(idx))
        df.to_csv(test_dir / "features/{}".format(str(idx)))

    print("test features exctracted")
