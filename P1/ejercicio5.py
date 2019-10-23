#
# El formato del fichero será el siguiente:
# En la primera línea, separado por un espacio
# se pondrá el número de filas y columnas
# Cada fila del fichero será una fila de la matriz
# Cada elemento estará separado por un espacio
#
import numpy as np
import sys


if len(sys.argv) != 2:
    print("Se ha llamado mal al programa")
    sys.exit(-1)

try:
    f = open(sys.argv[1], 'rt')
except IOError:
    print("No se puede leer el fichero")
    sys.exit(-1)

lines = f.read().splitlines()

shape = lines[0].split(' ')
shape = [int(i) for i in shape]

matrix = np.zeros(shape=(shape[0], shape[1]))

for i in range(0, len(lines)-1):
    lines[i+1] = lines[i+1].split(' ')
    lines[i+1] = [int(j) for j in lines[i+1]]

    matrix[i, :] = np.array(lines[i+1])

print("La matriz es:")
print(matrix)

if np.linalg.det(matrix) == 0:
    print("La matriz no es invertible")
else:
    print("La inversa de la matriz es:")
    inversa = np.linalg.inv(matrix)
    print(inversa)

    #La matriz original multiplicada por su inversa tiene
    #que dar la matriz identidad

    print("Si la inversa se ha calculado correctamente")
    print("La proxima matriz es la identidad de orden ", shape[0])
    print(np.matmul(matrix, inversa).round(0))
