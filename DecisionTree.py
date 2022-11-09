import csv
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
import graphviz

iris = load_iris()

x = iris.data
y = iris.target

clf = DecisionTreeClassifier()
clf.fit(x , y)

x_test = [[5,3,1,0.5],[6,4,5,2]]
clf.predict(x_test)

tree.plot_tree(clf) 
dot_data = tree.export_graphviz(clf, out_file=None, 
...                      feature_names=iris.feature_names,  
...                      class_names=iris.target_names,  
...                      filled=True, rounded=True,  
...                      special_characters=True) 

graph = graphviz.Source(dot_data)
graph

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])
