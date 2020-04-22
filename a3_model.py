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
from sklearn.metrics import classification_report

##### Classes and helper functions
currentDirectory = os.path.dirname(os.path.realpath(__file__))
os.chdir(currentDirectory)

def get_csv(csv_file):
    data = pd.read_csv(csv_file)
    return data

def get_train_samples(dataframe, trainsize):
    same_author_samples=[]
    not_same_author_samples=[]
    samples = []
    j=0
    while j < trainsize:
        #get a random sample
        train_length = dataframe.shape[0]
        i1=random.randrange(0, train_length, 1)
        i2=random.randrange(0, train_length, 1)
        doc1=dataframe.loc[dataframe.index[i1]]
        doc2=dataframe.loc[dataframe.index[i2]]
        author1=doc1['author']
        author2=doc2['author']
        values_doc1=(doc1.values.tolist()[2:])
        values_doc2=(doc2.values.tolist()[2:])
        sample = values_doc1 + values_doc2 # [values_doc1, values_doc2]
        #determine whether it's the same author or not
        if author1 == author2 and len(same_author_samples) < trainsize/2:
            if sample not in same_author_samples:
                same_author_samples.append([sample, 1])
                j += 1
        elif author1 != author2 and len(not_same_author_samples) < trainsize/2:
            if sample not in not_same_author_samples:
                not_same_author_samples.append([sample, 0])
                j += 1
    #join the samples
    samples = samples + same_author_samples + not_same_author_samples
    random.shuffle(samples)
    return samples

def get_test_samples(dataframe, testsize):
    samples = []
    j=0
    while j < testsize:
        #get a random sample
        train_length = dataframe.shape[0]
        i1=random.randrange(0, train_length, 1)
        i2=random.randrange(0, train_length, 1)
        doc1=dataframe.loc[dataframe.index[i1]]
        doc2=dataframe.loc[dataframe.index[i2]]
        author1=doc1['author']
        author2=doc2['author']
        values_doc1=(doc1.values.tolist()[2:])
        values_doc2=(doc2.values.tolist()[2:])
        sample = values_doc1 + values_doc2 # [values_doc1, values_doc2]
        #determine whether it's the same author or not
        if author1 == author2:
            if sample not in samples:
                samples.append([sample, 1])
                j += 1
        elif author1 != author2:
            if sample not in samples:
                samples.append([sample, 0])
                j += 1
    random.shuffle(samples)
    return samples

#PREPARING DATA
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, ind):
        x = np.array(self.data[ind][0], dtype=np.float32)
        y = self.data[ind][1]
        y = np.array([y])
        return x, y

#### Actual Network
def main(featurefile=None, 
            trainsize=False,
            testsize=False,
            batchsize=False,
            hidden_size=False,
            non_linearity=False):

    #Load the data
    df = get_csv(args.featurefile)

    #prepare data for training
    train_df = df.loc[df['train_or_test'] == 'train']
    train_list = get_train_samples(train_df,args.number_of_train_samples)
    train_set = MyDataset(train_list)
    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True)

    #prepare data for testing
    test_df = df.loc[df['train_or_test'] == 'test']
    test_list = get_test_samples(test_df,args.number_of_test_samples)
    test_set = MyDataset(test_list)
    test_loader = DataLoader(test_set, shuffle=False)


    # FNN
    if hidden_size == 0:
        print("\n---FNN without hidden layer---")
    else:
        print("\n---FNN with hidden layer of size {} and nonlinearity {} ---".format(args.hidden_size, args.nonlinearity))
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
    model = FeedForward(dimension, args.hidden_size)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion =  nn.BCELoss()
    epochs = 10

    ##### TRAINING
    print("\n Training with batch size {} ...".format(args.batchsize))
    model.train()
    for _ in range(epochs):
        losses = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            output = model(x)
            loss = criterion(output, y.float())
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

    ##### EVALUATION 
    print("\n Evaluating...")
    model.eval()
    with open('FeedForward_prediction.csv', 'w') as f:
        columns = ['Documents', 'Label', 'Prediction', 'rounded_Prediction', 'Difference']
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator = '\n')
        writer.writeheader()
        counter = 0
        actual_label = []
        pred_eval = []
        with torch.no_grad():
            for batch_num, input_data in enumerate(test_loader):
                ID = batch_num
                x , y = input_data
                y = y.tolist()[0]
                output = (model(x).tolist()[0])[0]
                output_evaluated = 0
                if output >= 0.5:
                    output_evaluated = 1
                pred_eval.append(output_evaluated)
                actual_label.append(y)
                difference = y[0] - output_evaluated
                if difference != 0:
                    counter += 1
                writer.writerow({columns[0]: ID, columns[1]: y[0], columns[2]: output, columns[3]:output_evaluated, columns[4]: difference})
            print('\n Classification report: ')
            print(classification_report(actual_label, pred_eval))
        print('\n Number of tested samples:' + str(ID+1))
        print('\n Number of wrong predictions:' + str(counter))
        print('\n Percentage of wrong predictions:' + str(counter*100/(ID+1))) #i.e. 100%-accuracy

##### OPTIONS
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", 
                        type=str, 
                        help="The file containing the table of instances and features.")

    parser.add_argument("-trainsize",
                        "--number_of_train_samples",
                        default=100,
                        type=int,
                        required=False,
                        help="Number of samples for training")
    
    parser.add_argument("-testsize",
                        "--number_of_test_samples",
                        default=100,
                        type=int,
                        required=False,
                        help="Number of samples for testing")
    
    parser.add_argument("-batchsize",
                        "--batchsize",
                        default=1,
                        type=int,
                        required=False,
                        help="Size of training batches per epoch")

    parser.add_argument("-hs",
                        "--hidden_size",
                        default=0,
                        type=int,
                        required=False,
                        help="Size of the hidden layer if wanted")

    parser.add_argument("-nl",
                        "--nonlinearity",
                        default=None,
                        type=str,
                        required=False,
                        help="Choice of nonlinearity for hidden layer either >>relu<< or >>tanh<<")
    
    args = parser.parse_args()
    
    print("Reading {}...".format(args.featurefile))
    
    main(
            featurefile=args.featurefile, 
            trainsize=args.number_of_train_samples,
            testsize=args.number_of_test_samples,
            batchsize=args.batchsize,
            hidden_size=args.hidden_size,
            non_linearity=args.nonlinearity
            )