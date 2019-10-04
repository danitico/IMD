import numpy as np
import sys

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
    lines[i+1] = [int(j) for j in lines[i+1]]

    matrix[i,:] = np.array(lines[i+1])

dependentValues = matrix[:, shape[1]]
coeficients = matrix[:, 0:shape[1]]

results = []
for i in range(0, shape[0]):
    newmatrix = coeficients.copy()

    newmatrix[:, i] = dependentValues
    results.append(np.linalg.det(newmatrix)/np.linalg.det(coeficients))

results = np.array(results)
print(results.round(4))