#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:20:57 2018

@author: admin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:39:36 2018

@author: admin
"""

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)


import random

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            #label = random.choice(self.y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions


    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0 
        for i in range (1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
    
    
            

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

from sklearn.neighbors import KNeighborsClassifier  #K neighbors 
my_classifier = KNeighborsClassifier()
"""

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)


#3 Test
predictions = my_classifier.predict(X_test)
print(predictions)

 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

