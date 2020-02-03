#!/usr/bin/python
# -*- coding: utf-8 -*-
import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
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

    X = np.array(data[:, :-1])
    X = X.astype(np.float64)
    Y = np.array(data[:, -1])

    treeclassifier = DecisionTreeClassifier()
    meow = linear_model.SGDClassifier()
    # knnclassifier = KNeighborsClassifier()
    # svmclassifier = svm.SVC(kernel='linear')
    clf = AdaBoostClassifier(meow, algorithm='SAMME'
                                             '')

    accuracy = cross_val_score(clf, X, Y, cv=10, n_jobs=-1)

    print(accuracy.mean().round(4))
