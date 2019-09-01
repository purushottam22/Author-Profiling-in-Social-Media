import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

dataset = np.loadtxt("csv_file.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:7]
Y = dataset[:,7]
print(X)

print("K nearest neighbour accuracy score")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)
preds1 = neigh.predict(X)
np.savetxt('results.csv', preds1, delimiter='\n')   # X is an array


print("SVC accuracy score")
clf = SVC(gamma='auto')
clf.fit(X, Y) 
preds = clf.predict(X)
print(accuracy_score(Y, preds))



print("Decision tree accuracy score")
clf1 = DecisionTreeClassifier(random_state=0)
iris = load_iris()
print(cross_val_score(clf1, iris.data, iris.target, cv=3))



                         



