
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_auc_score

from datetime import datetime, date



def train_ch(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs,reg_lambda):
    """
    train the model

    Args:
        model: (nn.Module) the neural network
        train_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images
        optimizer: (torch.optim) an optimization algorithm
        n_epochs: (int) number of epochs performed during training

    Returns:
        best_model: (nn.Module) the trained neural network
    """
    best_model = deepcopy(model)
    val_best_ba = 0

    train_losses = []
    train_bas = []
    val_losses = []
    val_bas = []

    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            # Retrieve mini-batch
            x,target,features  = data[0] ,data[1],data[2]
            # Forward pass
            output = model(x,features)[:,0]
            # Loss computation
            loss = criterion(output,target)
            # L2 regularization for convolutional weights
            conv_weights = model.state_dict()['conv1d.weight']
            l2_reg = conv_weights.norm(2)
            loss+= reg_lambda*l2_reg
            # Backpropagation (gradient computation)
            loss.backward()
            # Parameter update
            optimizer.step()
            # Erase previous gradients
            optimizer.zero_grad()
        # learning rate step
        scheduler.step()

        _, train_metrics = test_ch(model, train_loader, criterion,reg_lambda)
        _, val_metrics = test_ch(model, val_loader, criterion,reg_lambda)

        print('Epoch %i/%i: train loss = %f, train BA = %f, val loss = %f, val BA = %f'
              % (epoch, n_epochs,train_metrics['mean_loss'],
                 train_metrics['balanced_accuracy'],
                 val_metrics['mean_loss'],
                 val_metrics['balanced_accuracy']))
        print()

        if val_metrics['balanced_accuracy'] >= val_best_ba:
            best_model = deepcopy(model)
            val_best_ba = val_metrics['balanced_accuracy']

        train_losses.append(train_metrics['mean_loss'])
        train_bas.append(train_metrics['balanced_accuracy'])
        val_losses.append(val_metrics['mean_loss'])
        val_bas.append(val_metrics['balanced_accuracy'])
    metrics = {'train_losses':train_losses,'train_bas':train_bas,
                'val_losses':val_losses,'val_bas':val_bas}

    return best_model, metrics


def test_ch(model, data_loader, criterion, reg_lambda, test=False):
    """
    Evaluate/ test model

    Args:
        model: (nn.Module) the neural network
        data_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images

    Returns:
        results_df: (DataFrame) the label predicted for every subject
        results_metrics: (dict) a set of metrics
    """
    model.eval()

    columns = ["index","proba", "predicted_label"]
    if not test:
        columns.append("true_label")
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            x = data[0]
            if not test:
                labels,features = data[1], data[2]
                outputs = model(x,features)[:,0]
                loss = criterion(outputs, labels)
                # L2 regularization for convolutional weights
                conv_weights = model.state_dict()['conv1d.weight']
                l2_reg = conv_weights.norm(2)
                loss+= reg_lambda*l2_reg

                total_loss += loss.item()
            else:
                features = data[1]
                outputs = model(x,features)[:,0]

            preds = np.round(outputs.detach())

            for k in range(data[0].size(0)):
                row = [k,outputs[k,0].item(),preds[k,0].item()]
                if not test:
                    row.append(labels[k,0].item())
                row_df = pd.DataFrame([row], columns=columns)
                results_df = pd.concat([results_df, row_df])
    if not test:
        results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values)
        results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)
    else:
        results_metrics = None
    results_df.reset_index(inplace=True, drop=True)
    return results_df, results_metrics

def train_DMIL(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs):
    """
    train the model

    Args:
        model: (nn.Module) the neural network
        train_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images
        optimizer: (torch.optim) an optimization algorithm
        n_epochs: (int) number of epochs performed during training

    Returns:
        best_model: (nn.Module) the trained neural network
    """
    best_model = deepcopy(model)
    val_best_ba = 0

    train_losses = []
    train_bas = []
    val_losses = []
    val_bas = []

    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            # Retrieve mini-batch
            x,target,features  = data[0] ,data[1],data[2]
            # Forward pass
            output = model(x,features)[:,0]
            # Loss computation
            loss = criterion(output,target)
            # Backpropagation (gradient computation)
            loss.backward()
            # Parameter update
            optimizer.step()
            # Erase previous gradients
            optimizer.zero_grad()
        # learning rate step
        scheduler.step()

        _, train_metrics = test_DMIL(model, train_loader, criterion)
        _, val_metrics = test_DMIL(model, val_loader, criterion)

        print('Epoch %i/%i: train loss = %f, train BA = %f, val loss = %f, val BA = %f'
              % (epoch, n_epochs,train_metrics['mean_loss'],
                 train_metrics['balanced_accuracy'],
                 val_metrics['mean_loss'],
                 val_metrics['balanced_accuracy']))
        print()

        if val_metrics['balanced_accuracy'] >= val_best_ba:
            best_model = deepcopy(model)
            val_best_ba = val_metrics['balanced_accuracy']

        train_losses.append(train_metrics['mean_loss'])
        train_bas.append(train_metrics['balanced_accuracy'])
        val_losses.append(val_metrics['mean_loss'])
        val_bas.append(val_metrics['balanced_accuracy'])
    metrics = {'train_losses':train_losses,'train_bas':train_bas,
                'val_losses':val_losses,'val_bas':val_bas}

    return best_model, metrics


def test_DMIL(model, data_loader, criterion,test=False):
    """
    Evaluate/ test model

    Args:
        model: (nn.Module) the neural network
        data_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images

    Returns:
        results_df: (DataFrame) the label predicted for every subject
        results_metrics: (dict) a set of metrics
    """
    model.eval()

    columns = ["index","proba", "predicted_label"]
    if not test:
        columns.append("true_label")
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            x = data[0]
            if not test:
                labels,features = data[1], data[2]
                outputs = model(x,features)[:,0]
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            else:
                features = data[1]
                outputs = model(x,features)[:,0]

            preds = np.round(outputs.detach())

            for k in range(data[0].size(0)):
                row = [k,outputs[k,0].item(),preds[k,0].item()]
                if not test:
                    row.append(labels[k,0].item())
                row_df = pd.DataFrame([row], columns=columns)
                results_df = pd.concat([results_df, row_df])
    if not test:
        results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values)
        results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)
    else:
        results_metrics = None
    results_df.reset_index(inplace=True, drop=True)
    return results_df, results_metrics

def compute_metrics(ground_truth, prediction):
    """Computes the accuracy, sensitivity, specificity and balanced accuracy and AUC"""
    tp = np.sum((prediction == 1) & (ground_truth == 1))
    tn = np.sum((prediction == 0) & (ground_truth == 0))
    fp = np.sum((prediction == 1) & (ground_truth == 0))
    fn = np.sum((prediction == 0) & (ground_truth == 1))

    metrics_dict = dict()
    metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    # Sensitivity
    if tp + fn != 0:
        metrics_dict['sensitivity'] = tp / (tp + fn)
    else:
        metrics_dict['sensitivity'] = 0.0

    # Specificity
    if fp + tn != 0:
        metrics_dict['specificity'] = tn / (fp + tn)
    else:
        metrics_dict['specificity'] = 0.0

    metrics_dict['balanced_accuracy'] = (metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2

    # auc
    metrics_dict['AUC'] = roc_auc_score(ground_truth,prediction)


    return metrics_dict

def preprocess(data):
    """
    preprocessing of features to get float and int
    """
    def sex(data):
        init=np.zeros(data.shape[0],dtype=int)
        init[data=="M"]=np.int(1)
        return init

    def age(born):
        age=np.zeros(born.shape[0])
        for i,dob in enumerate(born):
            #print("dob",dob)
            if dob.find("/")!=-1:


                dob = datetime.strptime(dob, "%m/%d/%Y").date()
                today = date.today()
                dob= today.year - dob.year - ((today.month,
                                        today.day) < (dob.month,
                                                        dob.day))

                age[i]=dob

            else :

                dob = datetime.strptime(dob, "%d-%m-%Y").date()
                today = date.today()
                dob= today.year - dob.year - ((today.month,
                                        today.day) < (dob.month,
                                                        dob.day))

                age[i]=dob
        return age

    data["AGE"]=age(data["DOB"])
    data["SEX"]=sex(data["GENDER"])
    # normalization of features
    data['LYMPH_COUNT']=(data['LYMPH_COUNT'] -  data['LYMPH_COUNT'].mean())/ data['LYMPH_COUNT'].std()
    data["AGE"]=(data["AGE"] -  data["AGE"].mean())/ data["AGE"].std()

    return data

########################### VAE ############################


def train_auto_DMIL(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs, kl_reg):
    """
    train the model

    Args:
        model: (nn.Module) the neural network
        train_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images
        optimizer: (torch.optim) an optimization algorithm
        n_epochs: (int) number of epochs performed during training

    Returns:
        best_model: (nn.Module) the trained neural network
    """
    best_model = deepcopy(model)
    val_best_ba = 0

    train_losses = []
    train_bas = []
    val_losses = []
    val_bas = []

    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            # Retrieve mini-batch
            x,target,features  = data[0] ,data[1],data[2]
            # Forward pass
            output,z,mu,logvar = model(x,features)
            output=output[:,0]
            # KullbackLeiblei divergence for VAE
            KLD=-kl_reg*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
            # Loss computation
            loss = criterion(output,target)  + KLD
            # Backpropagation (gradient computation)
            loss.backward()
            # Parameter update
            optimizer.step()
            # Erase previous gradients
            optimizer.zero_grad()
        # learning rate step
        scheduler.step()

        _, train_metrics = test_auto_DMIL(model, train_loader, criterion, kl_reg)
        _, val_metrics = test_auto_DMIL(model, val_loader, criterion, kl_reg)

        print('Epoch %i/%i: train loss = %f, train BA = %f, val loss = %f, val BA = %f'
              % (epoch, n_epochs,train_metrics['mean_loss'],
                 train_metrics['balanced_accuracy'],
                 val_metrics['mean_loss'],
                 val_metrics['balanced_accuracy']))
        print()

        if val_metrics['balanced_accuracy'] >= val_best_ba:
            best_model = deepcopy(model)
            val_best_ba = val_metrics['balanced_accuracy']

        train_losses.append(train_metrics['mean_loss'])
        train_bas.append(train_metrics['balanced_accuracy'])
        val_losses.append(val_metrics['mean_loss'])
        val_bas.append(val_metrics['balanced_accuracy'])
    metrics = {'train_losses':train_losses,'train_bas':train_bas,
                'val_losses':val_losses,'val_bas':val_bas}

    return best_model, metrics


def test_auto_DMIL(model, data_loader, criterion, kl_reg, test=False):
    """
    Evaluate/ test model

    Args:
        model: (nn.Module) the neural network
        data_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images

    Returns:
        results_df: (DataFrame) the label predicted for every subject
        results_metrics: (dict) a set of metrics
    """
    model.eval()

    columns = ["index","proba", "predicted_label"]
    if not test:
        columns.append("true_label")
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            x = data[0]
            if not test:
                labels,features = data[1], data[2]
                outputs,z,mu,logvar= model(x,features)
                outputs=outputs[:,0]
                # KullbackLeiblei divergence for VAE
                KLD=-kl_reg*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
                loss = criterion(outputs, labels) +KLD
                total_loss += loss.item()
            else:
                features = data[1]
                outputs,z,mu,logvar = model(x,features)
                outputs=outputs[:,0]

            preds = np.round(outputs.detach())

            for k in range(data[0].size(0)):
                row = [k,outputs[k,0].item(),preds[k,0].item()]
                if not test:
                    row.append(labels[k,0].item())
                row_df = pd.DataFrame([row], columns=columns)
                results_df = pd.concat([results_df, row_df])
    if not test:
        results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values)
        results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)
    else:
        results_metrics = None
    results_df.reset_index(inplace=True, drop=True)
    return results_df, results_metrics
