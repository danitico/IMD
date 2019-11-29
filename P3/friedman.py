import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', header=0)
df1 = pd.read_csv('results1.csv', header=0)
df2 = pd.read_csv('results2.csv', header=0)

data = df.to_numpy()
data = np.append(data, df1.to_numpy(), axis=1)
data = np.append(data, df2.to_numpy(), axis=1)

rankings = np.zeros(shape=(data.shape[0], data.shape[1]))

for i in range(rankings.shape[0]):
    rankings[i, :] = rankdata(data[i, :])


# files = ['BreastTissue.csv', 'contact-lenses.arff', 'diabetes.arff', 'ecoli.arff','glass.arff',
#          'ionosphere.arff', 'divorce.arff', 'vote.arff', 'zoo.arff', 'segment.arff']

final = np.copy(rankings.mean(axis=0))
final = final.reshape((9, 1))


new_df = pd.DataFrame(final, index=['DTree_1', 'KNN_1', 'SVM_1', 'Dtree_2', 'KNN_2', 'SVM_2', 'Dtree_3', 'KNN_3', 'SVM_3'], columns=['Ranking'])
new_df = new_df.sort_values(by='Ranking', ascending=False)

new_df.plot.barh()
plt.xlim(left=1)
plt.show()


