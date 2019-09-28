from sklearn import tree

#1. Collect training data
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]

#2. Train
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#3. Predict
print(clf.predict([[200,1]]))


#4. Visulize

from sklearn.externals.six import StringIO
#import pydot
import pydotplus


dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file = dot_data,
                     feature_names = ['Weight','Surface'],
                     class_names = ['Apple', 'Orange'],
                     filled = True,
                     rounded = True,
                     impurity = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('fruit.pdf')
