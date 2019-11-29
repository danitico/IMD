from scipy.stats import friedmanchisquare, f
import pandas as pd

df = pd.read_csv('results.csv', header=0)
measures = df.to_numpy()

measure1 = measures[:, 0]
measure2 = measures[:, 1]
measure3 = measures[:, 2]

friedman_stat, pvalue = friedmanchisquare(measure1, measure2, measure3)

datasets = measures.shape[0]
algorithms = measures.shape[1]

davenport_stat = ((datasets-1)*friedman_stat) / (datasets*(algorithms-1) - friedman_stat)

if davenport_stat > f.ppf(q=1-0.05, dfn=algorithms-1, dfd=(algorithms-1)*(datasets-1)):
    print("Hay diferencias entre los algoritmos")
else:
    print("Se consideran iguales")
