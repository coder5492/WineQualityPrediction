#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Tue Dec 11 09:43:03 2018
"""

@author: sangeeth
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pylab as pl
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

#Importing the data
data = pd.read_csv(url, sep=";")

#Declaring X and Y
X = data.iloc[:,:-1]
Y = data.iloc[:, 11:12]

#Standardising the data
sc = preprocessing.StandardScaler()
X  = sc.fit(X).transform(X)

#Splitting the observations to test set and train set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#Predicting the output using actual variables
bayes_real_model = GaussianNB()
bayes_real_model.fit(X_train,Y_train)
real_prediction = bayes_real_model.predict(X_test)

#Calculation of Accuracy Score and Confusion Matrix
from sklearn.metrics import accuracy_score
real_score = accuracy_score(Y_test,real_prediction)
print("The accuracy with Real data is " + str(real_score))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,real_prediction).trace())

#Predicting with Different models with different PCA components
for i in range(1,10):
    model = PCA(n_components = i)
    model.fit(X)
    X_PCA = model.transform(X)
    X_train_PCA, X_test_PCA, Y_train_PCA, Y_test_PCA = train_test_split(X_PCA, Y, test_size=0.33, random_state=42)
    bayes_PCA_model = GaussianNB()
    bayes_PCA_model.fit(X_train_PCA,Y_train)
    PCA_prediction = bayes_PCA_model.predict(X_test_PCA)
    PCA_score = accuracy_score(Y_test,PCA_prediction)
    print("The accuracy with PCA data with  " + str(i) + " components is "  + str(PCA_score))
    print(confusion_matrix(Y_test,PCA_prediction).trace())


