import numpy as np

escagno = int(input("¿Cuántos escaños se reparten? "))
numPartidos = int(input("¿Cuántos partidos se presentan a las elecciones? "))
listaPartidos = []


for _ in range(0, numPartidos):
    partido = dict()
    print("Introduzca el partido")
    partido["nombre"] = input("Introduzca el nombre del partido: ")
    partido["votos"] = int(input("Introduzca el número de votos: "))
    partido["escagno"] = 0

    listaPartidos.append(partido)
    print("\n")


matrix = np.zeros(shape=(numPartidos, escagno))

for i in range(0, numPartidos):
    for j in range(0, escagno):
        matrix[i][j] = listaPartidos[i]["votos"] / (j+1)

escagnosrepartidos = 0

while escagnosrepartidos < escagno:
    location = np.where(matrix == np.max(matrix))

    matrix[location[0][0]][location[1][0]] = -1
    listaPartidos[location[0][0]]["escagno"] += 1

    escagnosrepartidos += 1

for i in range(0, numPartidos):
    print(listaPartidos[i]["nombre"], "con ", listaPartidos[i]["votos"], " votos ha conseguido ",
          listaPartidos[i]["escagno"], " escaño/escaños.")
