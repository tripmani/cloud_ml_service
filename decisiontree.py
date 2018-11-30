from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

pred = clf.predict(X_test)
pred = numpy.around(pred).astype(int)

score = accuracy_score(Y_test,pred)
print(score)