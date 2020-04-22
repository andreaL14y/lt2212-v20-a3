import os
import sys
import argparse
import numpy as np
import pandas as pd
######
from os import walk
from os.path import join
import re
from sklearn.model_selection import train_test_split

import a2
##################################################################

currentDirectory = os.path.dirname(os.path.realpath(__file__))
os.chdir(currentDirectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    ##### README: enron_sample and a2.py in the same directory as this script
    def get_files(folder):
        f = []
        for (root, dirnames, _) in walk(folder): 
            for dirname in dirnames:
                subfolder = os.path.join(root, dirname)
                for (_, _, filenames) in walk(subfolder):
                    for filename in filenames:
                        f.append(os.path.join(subfolder, filename))
        return f

    def get_content(path):
        with open(path, 'r') as article:
            content = article.read().lower()
        return content #article as text 
    
    def load_Data(folder):
        samples = []
        labels = []
        all_paths = get_files(folder)
        for path in all_paths:
            content = get_content(path)
            author = (list(reversed(path.split(os.sep)))[1])
            samples.append(content)
            labels.append(author)
        return samples, labels
    
    samples, labels = load_Data(args.inputdir)
    X = a2.extract_features(samples)
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    X_reduced = a2.reduce_dim(X, args.dims)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, labels, test_size=0.01*args.testsize)

    train_label = pd.DataFrame(['train'] * len(X_train), columns =['train_or_test'])
    test_label = pd.DataFrame(['test'] * len(X_test), columns =['train_or_test'])

    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train, columns =['author'])
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test, columns =['author'])

    train_table = pd.concat([train_label, y_train, X_train], axis=1)
    test_table = pd.concat([test_label, y_test, X_test], axis=1)

    get_table = pd.concat([train_table, test_table], axis=0)
    
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    get_table.to_csv(args.outputfile, index=False)
    print("Done!")