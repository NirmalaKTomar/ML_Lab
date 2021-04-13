#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import math
import pandas as pd

# using iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
X.shape,y.shape

a = np.random.randn(len(y))<float(2/3)
X_train = X[a]
y_train = y[a]
X_test = X[~a]
y_test = y[~a]

X_train.shape,y_train.shape,X_test.shape,y_test.shape

def calc_dist(a1,a2,d):
    e_len = 0 
    for i in range(d):
        e_len = e_len + (a1[i]-a2[i])**2
    e_len = math.sqrt(e_len)
    return e_len

def kNearestNeighbors(trainX,trainy,instance,k):
    dists = []
    n = len(instance)
    for i in range(len(trainy)):
        dist = calc_dist(trainX[i],instance,n)
        dists.append((trainX[i],trainy[i],dist))
    dists = sorted(dists, key = lambda x:x[2])
    neighbors = dists[:k]
    return neighbors

def getClass(neighbors):
    classes = dict()
    for neighbor in neighbors:
        classes[neighbor[1]] = classes.get(neighbor[1],0) +1
    classes = sorted(classes.items(), key = lambda x:x[1],reverse = True)
    return classes[0][0]

prediction = []
k=3
for x in X_test:
    neighbors = kNearestNeighbors(X_train,y_train,x,k)
    res = getClass(neighbors)
    prediction.append(res)

#printing correct predictions
acc = 0
for x in range(len(y_test)):
    if y_test[x] == prediction[x]:
        acc += 1

(acc/float(len(y_test)))*100.0