import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
random.seed(42)

####### NEW IMPORTS
#for part 1
import os
import pandas as pd
import re
import math
#for part 2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
#for part 3
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X

##### HELPERFUNCTIONS: 

#get words of one article
def get_words(article):
    content = article.lower()
    words = re.findall(r'\w+', content) 
    filtered = [x for x in words if x.isalpha() and len(x)>3]
    return filtered

#get words of all articles
def get_all_words(samples):
    words = []
    for article in samples:
        article_words = get_words(article)
        words.extend(article_words)
    return words

#count words in a list of words
def get_word_counts(words):
    word_counts = {}
    for word in words:
        if not word in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    unique_words = set(words)
    return word_counts, unique_words

#count only words that occur in the list of reference words
def selected_words_count(words, reference_unique_words):
    word_counts = {}
    unique_words = set(words).intersection(reference_unique_words) #intersection of words and reference words
    for word in unique_words:
        word_counts[word] = words.count(word)
    return word_counts

#delete infrequent words here all words that occur in total less than 3 times
def del_infrequent_words(samples):
    all_words = get_all_words(samples)
    word_counts, _ = get_word_counts(all_words)
    new_unique_words_lst = []
    for key, value in word_counts.items():
        if value > 20:
            new_unique_words_lst.append(key)
    new_unique_words = set(new_unique_words_lst)
    return new_unique_words

##### END OF HELPER FUNCTIONS

def extract_features(samples):
    all_files = samples
    #print("Number of samples: " + str(len(all_files)))
    reference_words = del_infrequent_words(all_files)
    #print("Number of reference words: " + str(len(reference_words)))
    all_words = np.full((len(all_files), len(reference_words)), 0) #initialize numpy array
    reference_words_dict = { word:i for i,word in enumerate(reference_words) } #initialize dictionary to make iteration possible
    i=0
    #create input of array per article 
    for article in all_files:
        words = get_words(article)
        word_counts = selected_words_count(words, reference_words)
        for word, count in word_counts.items():
            index = reference_words_dict[word]
            all_words[i][index] = count
        i = i + 1
    return all_words


##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X_dr[0]) # I exchanged X[0] by X_dr[0]
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    pca = PCA(n_components=n)
    X_pca=pca.fit_transform(X)
    return X_pca
    

##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = GaussianNB() #Classifier 1: naive_bayes
    elif clf_id == 2:
        clf = SVC() #Classifier 2: SVC
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X,y):
    #shuffle split using scikit-learn package, 80/20 train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
    return X_train, X_test, y_train, y_test

#ALTERNATIVE SHUFFLE_SPLIT IF WE SHOULD NOT USE train_test_split
"""def shuffle_split(X,y):
    B = X.shape[0]
    M = random.sample(range(0,X.shape[0]), math.floor(0.8*X.shape[0])) #list of random numbers, 80% of the length of X
    #Initialize lists
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    #Put randomly chosen 80% of X's content in the training Lists & remaining 20% in testing Lists
    for m in range(b):
        if m in M:
            X_train.append(X[m])
            y_train.append(y[m])
        else:
            X_test.append(X[m])
            y_test.append(y[m])
    #Convert lists to arrays        
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test"""


def train_classifer(clf, X, y):
    clf_trained = clf.fit(X, y)
    return clf_trained


def evalute_classifier(clf, X, y):
    clf = train_classifer(clf, X, y)
    clf_evaluated = clf.predict(X)
    classification_rep = classification_report(clf_evaluated, y)
    print(classification_rep)


######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)  #shouldn't it be label_names instead of label here?


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            ) 