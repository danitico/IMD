from scipy.stats import friedmanchisquare, f
import pandas as pd

df = pd.read_csv('bagging.csv', header=0, index_col=0)
bagging = df.to_numpy()

treebagging = bagging[:, 0]
knnbagging = bagging[:, 1]
svmbagging = bagging[:, 2]

df1 = pd.read_csv('adaboost.csv', header=0, index_col=0)
adaboost = df1.to_numpy()
adaboost = adaboost.reshape((10,))

df2 = pd.read_csv('gradientboosting.csv', header=0, index_col=0)
gradientboosting = df2.to_numpy()
gradientboosting = gradientboosting.reshape((10,))

friedman_stat, pvalue = friedmanchisquare(treebagging, knnbagging, svmbagging, adaboost, gradientboosting)

datasets = 10
algorithms = 5

davenport_stat = ((datasets-1)*friedman_stat) / (datasets*(algorithms-1) - friedman_stat)
print(davenport_stat)
print(f.ppf(q=1-0.05, dfn=algorithms-1, dfd=(algorithms-1)*(datasets-1)))

if davenport_stat > f.ppf(q=1-0.05, dfn=algorithms-1, dfd=(algorithms-1)*(datasets-1)):
    print("Hay diferencias entre los algoritmos")
else:
    print("No hay diferencias significativas")
