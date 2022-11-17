#Elmeri Lehtiniemi
#Dataml100, ex5 Neural Network

import tensorflow as tf
from tensorflow.keras import models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import pickle 
import numpy as np
from tensorflow.keras.utils import to_categorical
from random import random
import random

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
Y_0 = datadict["labels"]

X_tr = np.concatenate((datadict_tr1["data"], datadict_tr2["data"], datadict_tr3["data"], datadict_tr4["data"], datadict_tr5["data"]))
Y_tr0 = np.concatenate((datadict_tr1["labels"], ["labels"], datadict_tr3["labels"], datadict_tr4["labels"], datadict_tr5["labels"]))
X_tr = X_tr / 255
X = X / 255
#make the labels one-hot format (1 0 0 ..) etc.
Y_tr = to_categorical(Y_tr0,10)
Y = to_categorical(Y_0,10)


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
    Y_test = Y_0[0:size]
    acc = class_acc(Z,Y_test)
    return acc
#X -> now training X!

#lets make the neural network
#now the epochs and number of layers
#is based on experimentation, and most likely could be way better.
#same goes for learning rate.
model = Sequential()
model.add(Dense(200, input_dim=3072, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
opt = keras.optimizers.SGD(lr=0.5)
model.compile(optimizer=opt, loss='mse', metrics=['mse'])
#fit the data
model.fit(X_tr, Y_tr, epochs=150, verbose=1)
#predict test data
pred_X = model.predict(X)
#now lets take the actual labels that is most likely
classified = []
for pred in pred_X:
    most_likely_class_index = np.argmax(pred)
    classified.append(most_likely_class_index)
#prints
acc_random = random_class_evaluation_script(X)
print("Random accuracy: ", acc_random)
clas = np.array(classified)
acc = class_acc(clas, Y_0)
print("Neural Network Accuracy: ", acc)


