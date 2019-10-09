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

matrix = np.zeros(shape=(shape[0], shape[1]+1))

for i in range(0, len(lines)-1):
    lines[i+1] = lines[i+1].split(' ')
    lines[i+1] = [float(j) for j in lines[i+1]]

    matrix[i,:] = np.array(lines[i+1])

dependentValues = matrix[:, shape[1]]
coeficients = matrix[:, 0:shape[1]]

print(np.linalg.matrix_rank(matrix))
print(np.linalg.matrix_rank(coeficients))
if np.linalg.matrix_rank(matrix) == np.linalg.matrix_rank(coeficients) == shape[1]:
    results = []
    for i in range(0, shape[0]):
        newmatrix = coeficients.copy()

        newmatrix[:, i] = dependentValues
        results.append(np.linalg.det(newmatrix)/np.linalg.det(coeficients))

    results = np.array(results)

    for i in range(0, len(results)):
        print("El valor de la incognita ", i, " es ", results[i].round(3))
else:
    print("No es un sistema compatible determinado")