import numpy as np


class Datos:
    TiposDeAtributos = ('Continuo', 'Nominal')

    # NOTA: No confundir TiposDeAtributos con tipoAtributos

    def __init__(self, nombreFichero):

        # Abrimos el fichero en formato READ
        file = open(nombreFichero, "r")
        # Linea 1 = numero de elementos
        line1 = file.readline()
        self.numeroElementos = int(line1)

        # Linea 2 = nombre atributos y clase
        line2 = file.readline()
        # Quitamos el salto de linea
        line2 = line2[:-1]
        self.nombreAtributos = line2.split(',')
        # numero de atributos en variable
        self.numeroAtributos = len(self.nombreAtributos)

        # Linea 3 tipo atributos y clase
        line3 = file.readline()
        # Quitamos el salto de linea
        line3 = line3[:-1]
        self.tipoAtributos = line3.split(',')

        for atr in self.tipoAtributos:
            if atr not in self.TiposDeAtributos:
                raise ValueError('Error en los tipos de los atributos')

        # Nominal atributos
        self.nominalAtributos = []

        for atr in self.tipoAtributos:
            if atr == self.TiposDeAtributos[1]:
                self.nominalAtributos.append(True)
            else:
                self.nominalAtributos.append(False)

        # Diccionarios
        self.diccionarios = []

        colecciones = []

        for atr in range(0, self.numeroAtributos):
            temp_set = set()
            colecciones.append(temp_set)

        for linea in range(0, self.numeroElementos):

            temp_line = file.readline()

            if temp_line[-1] == "\n":
                temp_line = temp_line[:-1]

            temp_list = temp_line.split(",")

            for atr in range(0, self.numeroAtributos):
                colecciones[atr].add(temp_list[atr])

        for atr in range(0, self.numeroAtributos):
            colecciones[atr] = sorted(colecciones[atr])

        for atr in range(0, self.numeroAtributos):
            temp_dict = {}
            if self.nominalAtributos[atr]:
                temp_colection = colecciones[atr]
                counter = 0
                for elem in temp_colection:
                    temp_dict[elem] = counter
                    counter += 1
                self.diccionarios.append(temp_dict)
            else:
                self.diccionarios.append(temp_dict)

        # Datos
        self.datos = np.zeros(
            shape=(self.numeroElementos, self.numeroAtributos))

        # Volvemos al inicio del fichero
        file.seek(0, 0)

        # Apuntamos al inicio de los datos otra vez
        for x in range(0, 3):
            file.readline()

        # Este bucle rellena self.datos usando diccionarios
        for elem in range(0, self.numeroElementos):
            temp_list = list()

            temp_line = file.readline()

            if temp_line[-1] == "\n":
                temp_line = temp_line[:-1]

            temp_line = temp_line.split(',')

            for atr in range(0, self.numeroAtributos):
                if self.nominalAtributos[atr]:
                    temp_list.append(self.diccionarios[atr].get(temp_line[atr]))
                else:
                    temp_list.append(temp_line[atr])

            self.datos[elem] = temp_list

        file.close()

    def extraeDatos(self, idx):

        submatriz = np.zeros(
            shape=(len(idx), self.numeroAtributos))

        for elements in range(0, len(idx)):
            submatriz[elements] = self.datos[idx[elements]]

        return submatriz


