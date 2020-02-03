from scipy.stats import wilcoxon
import pandas as pd

base = pd.read_csv('decisiontree.csv', header=0, index_col=0)
metodos = pd.read_csv('multiclass.csv', header=0, index_col=0)


T, p = wilcoxon(metodos['OneVsRest'].values, metodos['ECOC'].values)

print("El valor de T es ", T)
print("El valor de p es ", p)

if p > 0.05:
    print("Es de la misma distribución. Se suponen iguales")
else:
    print("Diferentes distribuciones. Se suponen diferentes")

