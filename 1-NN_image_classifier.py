from hashlib import sha1
import pickle
from pyexpat.model import XML_CQUANT_REP
import numpy as np
import matplotlib.pyplot as plt
from random import random
import random
from scipy.spatial.distance import cityblock

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict_tr1 = unpickle('G:/Ohjelmointi/DATAML100/datasets/cifar-10-batches-py/data_batch_1')
datadict_tr2 = unpickle('G:/Ohjelmointi/DATAML100/datasets/cifar-10-batches-py/data_batch_2')
datadict_tr3 = unpickle('G:/Ohjelmointi/DATAML100/datasets/cifar-10-batches-py/data_batch_3')
datadict_tr4 = unpickle('G:/Ohjelmointi/DATAML100/datasets/cifar-10-batches-py/data_batch_4')
datadict_tr5 = unpickle('G:/Ohjelmointi/DATAML100/datasets/cifar-10-batches-py/data_batch_5')
datadict = unpickle('G:/Ohjelmointi/DATAML100/datasets/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

X = X.astype(np.int64)

X_tr = np.concatenate((datadict_tr1["data"], datadict_tr2["data"], datadict_tr3["data"], datadict_tr4["data"], datadict_tr5["data"]))
Y_tr = np.concatenate((datadict_tr1["labels"], datadict_tr2["labels"], datadict_tr3["labels"], datadict_tr4["labels"], datadict_tr5["labels"]))

X_tr = X_tr.astype(np.int64)

def class_acc(pred,gt):
    #lazy solution, one for loop and just check if they are same.
    #retur accuracy in %
    mistakes = 0
    size = len(pred)
    for i in range(size):
        if pred[i] != gt[i]:
            mistakes = mistakes + 1
    accuracy = (size - mistakes)/size
    return accuracy # 1 = 100% etc.

def cifar10_classifier_random():
    r = random.randint(0,9)
    return(r)

def random_class_evaluation_script(X):
    #classifies randomly
    size = len(X)
    Z = []
    for i in range(size):
        Z.append(cifar10_classifier_random())
    #tests randomly classified stuff and prints it.
    Y_test = Y[0:size]
    print(len(Z))
    print(len(Y_test))
    acc = class_acc(Z,Y_test)
    return acc

def cifar10_classifier_1nn(x,trdata,trlabels):
    y = [] #labels
    for picture in x:
        distances = []
        #for this X lets go throught all training data.
        for j in range(trdata.shape[0]):
            distance = cityblock(picture,trdata[j])
            distances.append(distance)
        y.append(trlabels[np.argmin(distances)])#only one.
    return y

def main():
    """
    Runs random_class_evaluation_script to test how accurate random
    classfication is. Then runs cifar10_classifier_1nn to classifie using
    1nn classification. Accuracies are printed for comparison.
    """
    #lets run random classificaiton and 1nn classfication and get their accuracies
    acc_random = random_class_evaluation_script(X)
    acc_yy = class_acc(Y_tr,Y_tr) #to check that class acc works
    labels_1nn = cifar10_classifier_1nn(X,X_tr,Y_tr)
    acc_1nn = class_acc(labels_1nn,Y)
    print("Classifications done")
    print("class_acc function test (if 1.0 -> class_acc is OK):", acc_yy)
    print("Random classification accuracy:", acc_random)
    print("1-NN classification accuracy:", acc_1nn)
main()
