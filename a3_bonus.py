##### Imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn as nn
from torch import optim
import random
from torch.utils.data import Dataset, DataLoader
import csv 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import a3_model

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

####  Network with some fixed parameters: 
#### TRAINSIZE = 500
#### TESTSIZE = 100
#### BATCHSIZE = 10
def main(featurefile, hidden_size, non_linearity):

    #Load the data
    df = a3_model.get_csv(featurefile)

    #prepare data for training
    train_df = df.loc[df['train_or_test'] == 'train']
    train_list = a3_model.get_train_samples(train_df,500)
    train_set = a3_model.MyDataset(train_list)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

    #prepare data for testing
    test_df = df.loc[df['train_or_test'] == 'test']
    test_list = a3_model.get_test_samples(test_df,100)
    test_set = a3_model.MyDataset(test_list)
    test_loader = DataLoader(test_set, shuffle=False)


    # FNN
    class FeedForward(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(FeedForward, self).__init__()
            self.input_size = input_size
            self.linear0 = nn.Linear(in_features=input_size, out_features=1, bias=True)
            if hidden_size != 0:
                self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
                self.linear2 = nn.Linear(in_features=hidden_size, out_features=1)
            self.tanh =  nn.Tanh()
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            if hidden_size != 0:
                out = self.linear1(x)
                if non_linearity == 'relu':
                    out = self.relu(out)
                if non_linearity == 'tanh':
                    out=self.tanh(out)
                out = self.linear2(out)
            else: 
                out = self.linear0(x)
            out = self.sigmoid(out)
            return out

    dimension = len(test_list[0][0]) #features of two documents = 2*number of features per doc in original featurefile 
    model = FeedForward(dimension, hidden_size)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion =  nn.BCELoss()
    epochs = 10

    ##### TRAINING
    model.train()
    for _ in range(epochs):
        losses = []
        for _, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            output = model(x)
            loss = criterion(output, y.float())
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

    ##### EVALUATION 
    model.eval()
    actual_label = []
    pred_eval = []
    with torch.no_grad():
        for _, input_data in enumerate(test_loader):
            x , y = input_data
            y = y.tolist()[0]
            output = (model(x).tolist()[0])[0]
            output_evaluated = 0
            if output >= 0.5:
                output_evaluated = 1
            pred_eval.append(output_evaluated)
            actual_label.append(y)
        precicion=(precision_score(actual_label, pred_eval, average="weighted"))
        recall=recall_score(actual_label, pred_eval, average="weighted")
        return precicion, recall 

# PLOTTING
def get_plot(featurefile=None, hidden_size_range=None, steplength=None, outputpng=None):
    precicion_relu = []
    precicion_tanh = []
    precicion_none = []
    recall_relu = []
    recall_tanh = []
    recall_none = []

    size = []
    hidden_size = 0

    while hidden_size <= args.hidden_size_range:
        #precicion and recall per hidden layer size using ReLu
        p_r, r_r = main(args.featurefile, hidden_size, 'relu')
        precicion_relu.append(p_r)
        recall_relu.append(r_r)
        #precicion and recall per hidden layer size using Tanh
        p_t, r_t = main(args.featurefile, hidden_size, 'tanh')
        precicion_tanh.append(p_t)
        recall_tanh.append(r_t)
        #precicion and recall per hidden layer size without nonlinearity
        p_n, r_n = main(args.featurefile, hidden_size, 'None')
        precicion_none.append(p_n)
        recall_none.append(r_n)
        size.append(hidden_size)
        hidden_size = hidden_size + args.steplength

    #plot the precicion recall 
    plt.plot(size, precicion_relu, marker='.', label='ReLU_precicion')
    plt.plot(size, recall_relu, marker='.', label='ReLU_recall')
    plt.plot(size, precicion_tanh, marker='.', label='Tanh_precicion')
    plt.plot(size, recall_tanh, marker='.', label='Tanh_recall')
    plt.plot(size, precicion_none, marker='.', label='None_precicion')
    plt.plot(size, recall_none, marker='.', label='None_recall')
    #label the axes
    plt.xlabel('Size of hidden layer')
    plt.ylabel('Precision/Recall')
    # show the legend
    plt.legend()
    # save the plot
    plt.savefig("{}.png".format(args.outputpng))
    # show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot precicion and recall")
    parser.add_argument("featurefile", 
                        type=str, 
                        help="The file containing the table of instances and features.")

    parser.add_argument("hidden_size_range",
                        default=100,
                        type=int,
                        help="Range of the hidden layer sizes")

    parser.add_argument("steplength",
                        default=10,
                        type=int,
                        help="steplength of hiddensize change")

    parser.add_argument("outputpng",
                        default='outputpng',
                        type=str,
                        help="steplength of hiddensize change")
    
    args = parser.parse_args()
    
    get_plot(
            featurefile=args.featurefile, 
            hidden_size_range=args.hidden_size_range,
            steplength=args.steplength,
            outputpng=args.outputpng
            )