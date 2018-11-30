import os
from flask import Flask, request, render_template
from sklearn import datasets

from sklearn import linear_model
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2)
app = Flask(__name__)


str1 = '<!doctype html><html lang=''><head><style>body { background: url("/static/cc.png") ;}</style></head><body>'
str2 = '</body><html>'

@app.route('/')
def hello():

    x=5
    x+=5
    return render_template('index.html')




@app.route('/decisiontree', methods=['GET'])
def decisiontree():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    pred = clf.predict(X_test)
    pred = numpy.around(pred).astype(int)

    score = accuracy_score(Y_test,pred)

    return render_template(str1 + str(score) + str2)

@app.route('/linear', methods=['GET'])
def linear():
    reger = linear_model.LinearRegression()
    reger.fit(X_train,Y_train)

    pred = reger.predict(X_test)
    pred = numpy.around(pred).astype(int)

    score = accuracy_score(Y_test,pred)
    return render_template(str1 + str(score) + str2)

@app.route('/naivebayes', methods=['GET'])
def naivebayes():
    gnb = GaussianNB()
    gnb.fit(X_train,Y_train)

    pred = gnb.predict(X_test)
    pred = numpy.around(pred).astype(int)

    score = accuracy_score(Y_test,pred)
    return render_template(str1 + str(score) + str2)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8084))
    app.run("127.0.0.1", port)