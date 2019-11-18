import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', header=0)
df1 = pd.read_csv('results1.csv', header=0)

data = df.to_numpy()
data1 = df1.to_numpy()

rankings = np.zeros(shape=(data.shape[0], data.shape[1]))
rankings1 = np.copy(rankings)

for i in range(rankings.shape[0]):
    rankings[i, :] = rankdata(data[i, :])
    rankings1[i, :] = rankdata(data1[i, :])

files = ['BreastTissue.csv', 'contact-lenses.arff', 'diabetes.arff', 'ecoli.arff',
               'glass.arff', 'ionosphere.arff', 'divorce.arff', 'vote.arff', 'zoo.arff',
               'segment.arff']

final = np.copy(rankings.mean(axis=0))
final = np.append(final, rankings1.mean(axis=0))
final = final.reshape((6, 1))


new_df = pd.DataFrame(final, index=['DTree_1', 'KNN_1', 'SVM_1', 'Dtree_2', 'KNN_2', 'SVM_2'], columns=['Ranking'])
# new_df1 = pd.DataFrame(rankings1, index=files, columns=['DecisionTree', 'KNN', 'SVM'])

new_df.plot.bar()
# new_df1.plot.line()

plt.show()


