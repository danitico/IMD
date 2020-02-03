import numpy as np
import pandas as pd
from scipy.stats import rankdata
from Orange.evaluation import graph_ranks, compute_CD
import matplotlib.pyplot as plt

# Para 4 clasificadores en nemenyi
q = 2.569
classifiers = 4
datasets = 10

cd = q*np.sqrt((classifiers*(classifiers + 1))/(6*datasets))
print(cd)

df = pd.read_csv('decisiontree.csv', header=0, index_col=0)
df1 = pd.read_csv('multiclass.csv', header=0, index_col=0)

data = df.to_numpy()
data = np.append(data, df1.to_numpy(), axis=1)

ranking = np.zeros(shape=data.shape)

for i in range(ranking.shape[0]):
    ranking[i, :] = rankdata(data[i, :])

final = np.copy(ranking.mean(axis=0))

names = ['DecisionTree', 'OVO', 'OVR', 'ECOC']

graph_ranks(list(final), names, cd)

plt.show()


