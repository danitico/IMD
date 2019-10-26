import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import warnings


# warnings.filterwarnings('ignore')

to_classify = ['data/BreastTissue.csv', 'data/contact-lenses.arff', 'data/diabetes.arff', 'data/ecoli.arff',
               'data/glass.arff', 'data/ionosphere.arff', 'data/divorce.arff', 'data/vote.arff', 'data/zoo.arff',
               'segment.arff']

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

    data[data == None] = np.nan
    X = np.array(data[:, 0:data.shape[1]-1])
    X = X.astype(np.float64)
    Y = np.array(data[:, data.shape[1]-1])
#    Y = Y.astype(np.float64)

    print("Base de datos ", i)
    print("Árbol de decision")
    clf = DecisionTreeClassifier(random_state=42)
    accuracy1 = cross_val_score(clf, X, Y, cv=10,
                             scoring='accuracy')

    print("accuracy Score = ", accuracy1.mean())
    treeaccuracy.append(accuracy1.mean())

    print("K vecinos")
    clf1 = KNeighborsClassifier()
    accuracy2 = cross_val_score(clf1, X, Y, cv=10,
                             scoring='accuracy')

    print("accuracy Score = ", accuracy2.mean())
    knnaccuracy.append(accuracy2.mean())

    # print("SVM")
    # clf2 = svm.SVC(kernel='linear', C=1, random_state=42)
    # accuracy3 = cross_val_score(clf2, X, Y, cv=10,
    #                          scoring='accuracy')
    #
    # print("accuracy Score = ", accuracy3.mean())
    # knnaccuracy.append(accuracy3.mean())

print(accuracy1)
print(accuracy2)
print(accuracy3)


