import random
from abc import ABCMeta, abstractmethod


class Particion:

    # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
    def __init__(self):
        self.indicesTrain = []
        self.indicesTest = []


#####################################################################################################

class EstrategiaParticionado:
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones,
    # listaParticiones. Se pasan en el constructor

    @abstractmethod
    def creaParticiones(self, datos, seed=None):
        pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
    # Atributos
    nombre_estrategia = ""
    numero_particiones = 0  # deprecated
    particiones = []
    porcentaje_particion = 0

    # Constructor
    def __init__(self, porcentaje_particion=50):
        self.nombre_estrategia = "Validación Simple"
        self.numero_particiones = 2
        self.porcentaje_particion = porcentaje_particion

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos, seed=None):
        (num_elem, num_atr) = datos.shape

        random.seed(seed)
        lista_indices = list(range(num_elem))
        random.shuffle(lista_indices)
        lista_particiones = []
        particion = Particion()
        punto_division = int(num_elem * self.porcentaje_particion / 100)
        particion.indicesTrain = lista_indices[:punto_division]

        particion.indicesTest = lista_indices[punto_division:]

        lista_particiones.append(particion)

        return lista_particiones


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):
    # Atributos
    nombre_estrategia = ""
    numero_particiones = 0
    particiones = []

    # Constructor
    def __init__(self, numero_particiones=2):
        self.nombre_estrategia = "Validación Cruzada"
        self.numero_particiones = numero_particiones

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos, seed=None):

        (num_elem, num_atr) = datos.shape

        random.seed(seed)
        lista_indices = list(range(num_elem))
        random.shuffle(lista_indices)

        lista_particiones = []
        puntos_division = []

        # puntos_division contiene los puntos donde empiezan los grupos
        # ej: en un grupo de 100 elementos dividos en cuatro grupos puntos_division = [0, 25, 50, 75]
        for puntos in range(0, self.numero_particiones):
            puntos_division.append(int(num_elem / self.numero_particiones * puntos))

        lista_indices_temp = []

        for p in range(0, self.numero_particiones - 1):
            lista_indices_partida = lista_indices[puntos_division[p]:puntos_division[p + 1]]
            lista_indices_temp.append(lista_indices_partida)

        # ultima lista de indices partida:
        lista_indices_partida = lista_indices[puntos_division[self.numero_particiones - 1]:]
        lista_indices_temp.append(lista_indices_partida)

        for num_particiones in range(0, self.numero_particiones):
            lista_editable = lista_indices_temp.copy()
            particion = Particion()
            particion.indicesTest = lista_editable[num_particiones]
            lista_editable.remove(lista_indices_temp[num_particiones])

            for l in lista_editable:
                particion.indicesTrain += l

            lista_particiones.append(particion)

        return lista_particiones
