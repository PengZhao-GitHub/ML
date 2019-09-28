
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree



#1. Collect data

iris = load_iris()

print(iris.feature_names)
print(iris.target_names)

print(iris.data[0])
print(iris.target[0])

for i in range(len(iris.target)):
    print('Example %d: Label %s, features %s'% (i, iris.target[i], iris.data[i]))

print('there are %d records'% len(iris.target))

# Split data into training data and testing data

test_idx = [0,50,100,149]

#Training data

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)


#Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


#2. Train classifier

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#3. Predict label for new flower

print(test_target)
print(clf.predict(test_data))


#4 Visualize the tree

from sklearn.externals.six import StringIO
#import pydot
import pydotplus


dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file = dot_data,
                     feature_names = iris.feature_names,
                     class_names = iris.target_names,
                     filled = True,
                     rounded = True,
                     impurity = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf')






    
