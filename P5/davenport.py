from scipy.stats import friedmanchisquare, f
import pandas as pd

df = pd.read_csv('decisiontree.csv', header=0, index_col=0)
treeclassifier = df.to_numpy().reshape((10,))


df1 = pd.read_csv('multiclass.csv', header=0, index_col=0)
multiclass = df1.to_numpy()

ovo = multiclass[:, 0]
ovr = multiclass[:, 1]
ecoc = multiclass[:, 2]

friedman_stat, pvalue = friedmanchisquare(treeclassifier, ovo, ovr, ecoc)

datasets = 10
algorithms = 4

davenport_stat = ((datasets-1)*friedman_stat) / (datasets*(algorithms-1) - friedman_stat)
print(davenport_stat)
print(f.ppf(q=1-0.05, dfn=algorithms-1, dfd=(algorithms-1)*(datasets-1)))

if davenport_stat > f.ppf(q=1-0.05, dfn=algorithms-1, dfd=(algorithms-1)*(datasets-1)):
    print("Hay diferencias entre los algoritmos")
else:
    print("No hay diferencias significativas")
