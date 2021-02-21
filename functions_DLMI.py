import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_auc_score



def train_ch(model, train_loader, val_loader, criterion, optimizer, n_epochs,reg_lambda,ann_lambda):
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
            #x,target ,is_ann = data[0] ,data[1]#, data[2]
            x,target  = data[0] ,data[1]

            #print(x.shape)
            # Forward pass
            output = model(x)[:,0]
            # if i%6==0:
            #     print(output,target)
            # Loss computation
            loss = criterion(output,target)
            # addinitonal loss to add importance to annotations
            #####loss_ann = criterion(is_ann*output,target*is_ann)
            # L2 regularization for convolutional weights
            conv_weights = model.state_dict()['conv1d.weight']
            l2_reg = conv_weights.norm(2)
            loss+= reg_lambda*l2_reg #+### ann_lambda*loss_ann
            # Backpropagation (gradient computation)
            loss.backward()
            # Parameter update
            optimizer.step()
            # Erase previous gradients
            optimizer.zero_grad()

        _, train_metrics = test_ch(model, train_loader, criterion,reg_lambda,ann_lambda)
        _, val_metrics = test_ch(model, val_loader, criterion,reg_lambda,ann_lambda)

        print('Epoch %i/%i: train loss = %f, train BA = %f, train AUC = %f, val loss = %f, val BA = %f, val AUC = %f'
              % (epoch, n_epochs,train_metrics['mean_loss'],
                 train_metrics['balanced_accuracy'],
                 train_metrics['AUC'],val_metrics['mean_loss'],
                 val_metrics['balanced_accuracy'],
                 val_metrics['AUC']))
        print()

        if val_metrics['balanced_accuracy'] > val_best_ba:
            best_model = deepcopy(model)
            val_best_ba = val_metrics['balanced_accuracy']

        train_losses.append(train_metrics['mean_loss'])
        train_bas.append(train_metrics['balanced_accuracy'])
        val_losses.append(val_metrics['mean_loss'])
        val_bas.append(val_metrics['balanced_accuracy'])
    metrics = {'train_losses':train_losses,'train_bas':train_bas,
                'val_losses':val_losses,'val_bas':val_bas}

    return best_model, metrics


def test_ch(model, data_loader, criterion, reg_lambda, ann_lambda,test=False):
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

            outputs = model(x)[:,0]
            if not test:
                #labels, is_ann = data[1], data[2]
                labels = data[1]#, data[2]
                loss = criterion(outputs, labels)
                #loss_ann = criterion(outputs*is_ann,labels*is_ann)
                # L2 regularization for convolutional weights
                conv_weights = model.state_dict()['conv1d.weight']
                l2_reg = conv_weights.norm(2)
                loss+= reg_lambda*l2_reg #+ ann_lambda*loss_ann

                total_loss += loss.item()

            preds = np.round(outputs.detach())
            # print(outputs)

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

def train_DMIL(model, train_loader, val_loader, criterion, optimizer, n_epochs,ann_lambda):
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
    val_best_auc = 0

    train_losses = []
    train_aucs = []
    val_losses = []
    val_aucs = []

    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            # Retrieve mini-batch
            x,target = data[0] ,data[1]
            # Forward pass
            output = model(x)[:,0]
            # Loss computation
            loss = criterion(output,target)
            # # addinitonal loss to add importance to annotations
            # loss_ann = criterion(is_ann*output,target*is_ann)

            # loss+= ann_lambda*loss_ann
            # Backpropagation (gradient computation)
            loss.backward()
            # Parameter update
            optimizer.step()
            # Erase previous gradients
            optimizer.zero_grad()

        _, train_metrics = test_DMIL(model, train_loader, criterion,ann_lambda)
        _, val_metrics = test_DMIL(model, val_loader, criterion,ann_lambda)

        print('Epoch %i/%i: train loss = %f, train BA = %f, train AUC = %f, val loss = %f, val BA = %f, val AUC = %f'
              % (epoch, n_epochs,train_metrics['mean_loss'],
                 train_metrics['balanced_accuracy'],
                 train_metrics['AUC'],val_metrics['mean_loss'],
                 val_metrics['balanced_accuracy'],
                 val_metrics['AUC']))
        print()

        if val_metrics['AUC'] > val_best_auc:
            best_model = deepcopy(model)
            val_best_auc = val_metrics['AUC']

        train_losses.append(train_metrics['mean_loss'])
        train_aucs.append(train_metrics['AUC'])
        val_losses.append(val_metrics['mean_loss'])
        val_aucs.append(val_metrics['AUC'])
    metrics = {'train_losses':train_losses,'train_aucs':train_aucs,
                'val_losses':val_losses,'val_aucs':val_aucs}

    return best_model, metrics


def test_DMIL(model, data_loader, criterion, ann_lambda,test=False):
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
            outputs = model(x)[:,0]
            if not test:
                labels = data[1]
                loss = criterion(outputs, labels)
                # loss_ann = criterion(outputs*is_ann,labels*is_ann)

                # loss+= ann_lambda*loss_ann

                total_loss += loss.item()

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
