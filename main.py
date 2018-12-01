# system import
import os

# third parties import
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model, tree
from sklearn.naive_bayes import GaussianNB
import numpy

data = datasets.load_iris()
X = data.data
Y = data.target
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2)
app = Flask(__name__)

'''
def model(model):
	model.fit(X_train, Y_train)
	predcition = model.predict(X_test)
	prediction_int = numpy.around(prediction).astype(int)
	score = accuracy_score(Y_test, prediction_int)
	return score
'''

@app.route('/')
def hello():
    return render_template('index.html')



@app.route('/dtree', methods=['GET'])
def decisiontree():
    dtree = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    pred = clf.predict(X_test)
    pred = numpy.around(pred).astype(int)

    score = accuracy_score(Y_test,pred)
    return str("Accuracy for Decision Tree : ") + str(score)

@app.route('/lr', methods=['GET'])
def linear():
    reger = linear_model.LinearRegression()
    reger.fit(X_train,Y_train)

    pred = reger.predict(X_test)
    pred = numpy.around(pred).astype(int)

    score = accuracy_score(Y_test,pred)
    return str("Accuracy for Linear Regression : ") + str(score)

@app.route('/nb', methods=['GET'])
def naivebayes():
    gnb = GaussianNB()
    gnb.fit(X_train,Y_train)

    pred = gnb.predict(X_test)
    pred = numpy.around(pred).astype(int)

    score = accuracy_score(Y_test,pred)
    return str("Accuracy for Naive Bayes : ") + str(score)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8084))
    app.run("127.0.0.1", port)
