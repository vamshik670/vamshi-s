from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

predict = Flask(__name__)


@predict.route('/')
def Home():
    return render_template('Home.html')


@predict.route('/templates')
def create():
    return render_template('create.html')


@predict.route('/', methods=['POST'])
def result():
    df = pd.read_csv("processed.cleveland.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 13].values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, train_size=0.8,
                                                                        random_state=42)
    linear = svm.SVC(kernel='linear', C=10, degree=3, probability=True, gamma='auto').fit(X_train, y_train)

    joblib.dump(linear, 'NB_spam_model.pkl')
    NB_spam_model = open('NB_spam_model.pkl', 'rb')
    if request.method == "POST":
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']

        example_measures = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    #example_measures = np.array([[67.0, 1.0, 4.0, 160.0, 286.0, 0.0, 2.0, 108.0, 1.0, 1.5, 2.0, 3.0, 3.0]])
        example_measures = example_measures.reshape(len(example_measures), -1)
        prediction = linear.predict(example_measures)
    return render_template('create.html', prediction=prediction)


@predict.route('/result1/', methods=['POST'])
def result1():

    df = pd.read_csv("processed.cleveland.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 13].values

    for index, item in enumerate(y):
        if not (item == 0.0):
            y[index] = 1
            # print(y)

    pca = PCA(n_components=2, whiten=True).fit(X)

    X_new = pca.transform(X)

    X_train, X_test, y_train, y_test =model_selection.train_test_split(X_new, y, test_size=0.3, random_state=1)
    clf = svm.SVC()
    clf = clf.fit(X_train, y_train)
    acc = clf.score(X_train, y_train)
    accuracy1 = round(acc*100,2)

    joblib.dump(clf, 'NB_spam_model.pkl')
    NB_spam_model = open('NB_spam_model.pkl', 'rb')
    linear = joblib.load(NB_spam_model)

    return render_template('Home.html', accuracy1=accuracy1)


@predict.route('/result2/', methods=['POST'])
def result2():
    try:

        df = pd.read_csv("cleveland2.csv")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 13].values

        for index, item in enumerate(y):
            if not (item == 0.0):
                y[index] = 1
                # print(y)


        from sklearn.metrics import accuracy_score

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                          max_depth=3, min_samples_leaf=5)
        clf_gini.fit(X_train, y_train)

        clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                             max_depth=3, min_samples_leaf=5)
        clf_entropy.fit(X_train, y_train)
        y_pred = clf_gini.predict(X_test)
        y_pred
        y_pred_en = clf_entropy.predict(X_test)
        y_pred_en

        accuracy2 = round(accuracy_score(y_test, y_pred)*100, 2)

        return render_template('Home.html', accuracy2=accuracy2)

    except Exception as e:
        return (str(e))


@predict.route('/result3/', methods=['POST'])
def result3():

    df = pd.read_csv("processed.cleveland.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 13].values

    for index, item in enumerate(y):
        if not (item == 0.0):
            y[index] = 1
            # print(y)

    pca = PCA(n_components=2, whiten=True).fit(X)

    X_new = pca.transform(X)

    X_train, X_test, y_train, y_test =model_selection.train_test_split(X_new, y, test_size=0.3, random_state=1)
    clf = svm.SVC()
    clf = clf.fit(X_train, y_train)
    acc = clf.score(X_train, y_train)
    accuracy3 = round(acc*100,2)

    return render_template('Home.html', accuracy3=accuracy3)


@predict.route('/result4/', methods=['POST'])
def result4():
    try:

        df = pd.read_csv("cleveland2.csv")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 13].values

        for index, item in enumerate(y):
            if not (item == 0.0):
                y[index] = 1
                # print(y)

        pca = PCA(n_components=2, whiten=True).fit(X)

        X_new = pca.transform(X)
        X_train, X_test, y_train, y_test =model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
        clf = MultinomialNB()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = clf.score(y_test, y_pred)

        accuracy4 = round(acc*100,2)

        #precision = classification_report(y_test, y_pred)

        return render_template('Home.html', accuracy4=accuracy4)

    except Exception as e:
        return (str(e))


if __name__ == '__main__':
    predict.run(debug=True)
