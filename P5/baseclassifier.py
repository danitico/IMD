import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

to_classify = ['avila-tr.csv', 'BreastTissue.csv', 'contact-lenses.arff', 'ecoli.arff', 'glass.arff',
               'movement_libras.csv', 'PhishingData.arff', 'segment.arff', 'statlog.csv', 'zoo.arff']

for i in to_classify:
    if i in ['BreastTissue.csv', 'movement_libras.csv', 'statlog.csv', 'avila-tr.csv']:
        df = pd.read_csv(i, header=None)
        data = df.to_numpy()
    else:
        arffile = arff.load(open(i))
        data = np.array(arffile['data'])

    X = np.array(data[:, :-1])
    X = X.astype(np.float64)
    Y = np.array(data[:, -1])

    treeclassifier = DecisionTreeClassifier()

    accuracy = cross_val_score(treeclassifier, X, Y, cv=10, n_jobs=-1, scoring='accuracy')
    print(accuracy.mean().round(4))
