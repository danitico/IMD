from scipy.stats import wilcoxon
import pandas as pd

data = pd.read_csv('results.csv', header=0)

T, p = wilcoxon(data['Svm'], data['Knn'])

print("El valor de T es ", T)
print("El valor de p es ", p)

if p > 0.05:
    print("Es de la misma distribución. Se suponen iguales")
else:
    print("Diferentes distribuciones. Se suponen diferentes")

