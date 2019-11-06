#!/usr/bin/python
# -*- coding: utf-8 -*-
import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

to_classify = ['data/BreastTissue.csv', 'data/contact-lenses.arff', 'data/diabetes.arff', 'data/ecoli.arff',
               'data/glass.arff', 'data/ionosphere.arff', 'data/divorce.arff', 'data/vote.arff', 'data/zoo.arff',
               'data/segment.arff']

treeaccuracy = list()
knnaccuracy = list()
svmaccuracy = list()

for i in to_classify:
    if i in ['data/BreastTissue.csv']:
        df = pd.read_csv(i, header=None)
        data = df.to_numpy()
    else:
        arffile = arff.load(open(i))
        data = np.array(arffile['data'])

    if i == 'data/vote.arff':
        data[data == None] = np.nan
        imp = SimpleImputer(strategy='most_frequent')
        imp.fit(data)
        data = imp.transform(data)

    X = np.array(data[:, 0:data.shape[1] - 1])
    X = X.astype(np.float64)
    Y = np.array(data[:, data.shape[1] - 1])
    #    Y = Y.astype(np.float64)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    print("Base de datos ", i)
    print("Árbol de decision")
    clf = DecisionTreeClassifier(random_state=42, max_depth=7, min_samples_split=5)
    accuracy1 = cross_val_score(clf, X, Y, cv=10,
                                scoring='accuracy')

    print("accuracy Score = ", accuracy1.mean())
    treeaccuracy.append(accuracy1.mean())

    print("K vecinos")
    clf1 = KNeighborsClassifier(n_neighbors=7, p=1)
    accuracy2 = cross_val_score(clf1, X, Y, cv=10,
                                scoring='accuracy')

    print("accuracy Score = ", accuracy2.mean())
    knnaccuracy.append(accuracy2.mean())

    print("SVM")
    clf2 = svm.SVC(kernel='poly', C=15, random_state=42)
    accuracy3 = cross_val_score(clf2, X, Y, cv=10,
                                scoring='accuracy')

    print("accuracy Score = ", accuracy3.mean())
    svmaccuracy.append(accuracy3.mean())

print("Tree accuracy")
for i in treeaccuracy:
    print(i)

print("KNN accuracy")
for i in knnaccuracy:
    print(i)

print("SVM accuracy")
for i in svmaccuracy:
    print(i)
