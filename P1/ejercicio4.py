import numpy as np
from scipy import stats
import sys

try:
    rows = int(input("Introduzca el número de filas: "))
    cols = int(input("Introduzca el número de columnas: "))
except ValueError:
    print("No es un número")
    sys.exit(-1)

matrix = np.zeros(shape=(rows, cols))

for i in range(0, rows):
    for j in range(0, cols):
        try:
            print("matrix[", i, "][", j, "] = ", end='')
            matrix[i, j] = int(input())
        except ValueError:
            print("No es un numero")
            sys.exit(-1)

print(matrix)

print("La moda de la matriz es", stats.mode(matrix, axis=None)[0][0])
print("La media de la matriz es ", np.mean(matrix))
