#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:39:36 2018

@author: admin
"""

from sklearn import datasets


#1 Prepare data
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)

#2 Train 

"""
# Replace these two lines to change classifier
from sklearn import tree   #DecisionTree
my_classifier = tree.DecisionTreeClassifier()
"""

from sklearn.neighbors import KNeighborsClassifier  #K neighbors 
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)


#3 Test
predictions = my_classifier.predict(X_test)
print(predictions)

 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

