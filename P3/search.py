from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import svm
import numpy as np
import pandas as pd
import arff
import warnings

warnings.filterwarnings('ignore')

classify = ['data/BreastTissue.csv', 'data/contact-lenses.arff', 'data/diabetes.arff', 'data/ecoli.arff',
               'data/glass.arff', 'data/ionosphere.arff', 'data/divorce.arff', 'data/vote.arff', 'data/zoo.arff',
               'data/segment.arff']

clf = DecisionTreeClassifier()
clf1 = KNeighborsClassifier()
clf2 = svm.SVC()

grid_values = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4, 5, 6],
    'min_samples_leaf':  [1, 2, 3, 4, 5]
}

grid_values1 = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'p': [1, 2]
}

grid_values2 = {
    'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1, 2, 3, 4, 5, 6, 7, 8]
}


for i in classify:
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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

#    grid_decisionTree = GridSearchCV(clf, param_grid=grid_values, scoring='accuracy', n_jobs=-1)
#    grid_decisionTree.fit(X_train, y_train)
#    print(accuracy_score(y_test, grid_decisionTree.predict(X_test)))

#    gridKnn = GridSearchCV(clf1, param_grid=grid_values1, scoring='accuracy', n_jobs=-1)
#    gridKnn.fit(X_train, y_train)
#    print(accuracy_score(y_test, gridKnn.predict(X_test)))

    gridsvm = GridSearchCV(clf2, param_grid=grid_values2, scoring='accuracy', n_jobs=-1)
    gridsvm.fit(X_train, y_train)
    print(accuracy_score(y_test, gridsvm.predict(X_test)))


