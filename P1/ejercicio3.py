import numpy as np

try:
    rows = int(input("Introduzca el número de filas: "))
    cols = int(input("Introduzca el número de columnas: "))
except ValueError:
    print("No es un número")
    exit(-1)

matrix = np.zeros(shape=(rows, cols))

for i in range(0, rows):
    for j in range(0, cols):
        try:
            print("matrix[", i, "][", j, "] = ", end='')
            matrix[i, j] = float(input())
        except ValueError:
            print("No es un numero")
            exit(-1)

print(matrix)

for i in range(0, rows):
    print("El máximo de la fila ", i , " es ", np.max(matrix[i,:]))

for j in range(0, cols):
    print("El máximo de la columna ", j , " es ", np.max(matrix[:,j]))

print("El determinante de la matriz es ", np.round(np.linalg.det(matrix), 4))

print("El rango de la matriz es ", np.linalg.matrix_rank(matrix))
