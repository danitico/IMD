import numpy as np
import sys

try:
    rows = int(input("Introduzca el número de filas: "))
    cols = int(input("Introduzca el número de columnas: "))
except ValueError:
    print("No es un número")
    sys.exit(-1)


matrix = np.random.rand(rows, cols)

print("Los valores de la matriz son")
print(matrix)

print("El máximo valor de la matriz es ", np.max(matrix))
print("El mínimo valor de la matriz es ", np.min(matrix))

opcion = 0

while True:
    print("1. Ángulos formados entre dos filas")
    print("2. Ángulos formados entre dos columas")

    try:
        opcion = int(input("Eliga una de las opciones: "))
    except ValueError:
        print("No es un número")
        sys.exit(-1)

    if opcion in [1, 2]:
        break
    else:
        print("Opción Incorrecta\n")

if opcion is 1:
    try:
        row1 = int(input("Introduzca la primera fila: "))
        row2 = int(input("Introduzca la segunda fila: "))
    except ValueError:
        print("No es un número")
        sys.exit(-1)

    if row1 not in range(0, rows) or row2 not in range(0, rows):
        print("Una de las filas no existe")
        sys.exit(-1)

    rowdata1 = matrix[row1, :]
    rowdata2 = matrix[row2, :]

    dotproduct = np.dot(rowdata1, rowdata2)
    mod1 = np.sqrt(np.dot(rowdata1, rowdata1))
    mod2 = np.sqrt(np.dot(rowdata2, rowdata2))

    angle = np.arccos(dotproduct/(mod1*mod2))

    print("El ángulo entre los dos vectores en grados es ", np.degrees(angle))
else:
    try:
        col1 = int(input("Introduzca la primera columna: "))
        col2 = int(input("Introduzca la segunda columna: "))
    except ValueError:
        print("No es un número")
        sys.exit(-1)

    if col1 not in range(0, cols) or col2 not in range(0, cols):
        print("Una de las columnas no existe")
        sys.exit(-1)

    coldata1 = matrix[:, col1]
    coldata2 = matrix[:, col2]

    dotproduct = np.dot(coldata1, coldata2)
    mod1 = np.sqrt(np.dot(coldata1, coldata1))
    mod2 = np.sqrt(np.dot(coldata2, coldata2))

    angle = np.arccos(dotproduct/(mod1*mod2))

    print("El ángulo entre los dos vectores en grados es ", np.degrees(angle))