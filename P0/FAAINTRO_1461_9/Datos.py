import numpy as np


class Datos:

    TiposDeAtributos = ('Continuo', 'Nominal')

    # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y diccionarios
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
        coleccionesSorted = []

        for atr in range(0, self.numeroAtributos):
            tempSet = set()
            colecciones.append(tempSet)

        for linea in range(0, self.numeroElementos):
            tempLine = file.readline()
            tempLine = tempLine[:-1]
            tempList = tempLine.split(",")

            for atr in range(0, self.numeroAtributos):
                colecciones[atr].add(tempList[atr])

        for atr in range(0, self.numeroAtributos):
            colecciones[atr] = sorted(colecciones[atr])

        for atr in range(0, self.numeroAtributos):
            tempDict = {}
            if(self.nominalAtributos[atr]):
                tempColection = colecciones[atr]
                counter = 0
                for elem in tempColection:
                    tempDict[elem] = counter
                    counter += 1
                self.diccionarios.append(tempDict)
            else:
                self.diccionarios.append(tempDict)

        # Datos
        # ¿Los queremos de tipo int o double?
        self.datos = np.zeros(
            shape=(self.numeroElementos, self.numeroAtributos), dtype=int)

        # Volvemos al inicio del fichero
        file.close()

        file = open(nombreFichero, "r")

        # Apuntamos al inicio de los datos otra vez
        for x in range(0, 3):
            file.readline()

        # Este bucle rellena self.datos usando diccionarios
        for elem in range(0, self.numeroElementos):
            tempList = list()

            tempLine = file.readline()
            tempLine = tempLine[:-1]
            tempLine = tempLine.split(',')

            for atr in range(0, self.numeroAtributos):
                if self.nominalAtributos[atr]:
                    tempList.append(self.diccionarios[atr].get(tempLine[atr]))
                else:
                    tempList.append(tempLine[atr])

            self.datos[elem] = tempList

            # Para comprobar matrices y comparar con los diccionarios
            #print(tempList)

        file.close()

    # Para comprobar matrices y comparar con los diccionarios
        #print(self.datos)
        #for atr in range(0, self.numeroAtributos):
        #    print(atr, "# DICCIONARIO:")
        #    print(self.diccionarios[atr])

    # TODO: implementar en la practica 1

    def extraeDatos(self, idx):
        pass
