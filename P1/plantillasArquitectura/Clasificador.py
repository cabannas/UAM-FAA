from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import norm
import math


class Clasificador:
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    def error(self, datos, pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error

        num_errores = 0

        (num_elem, num_atr) = datos.shape

        for elem in range(0, num_elem):
            clase_datos = datos[elem, num_atr - 1]

            clase_pred = pred[elem]

            if clase_datos != clase_pred:
                num_errores += 1

        return num_errores / num_elem

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    def validacion(self, particionado, dataset, clasificador, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones - Para validacion
        # cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i y obtenemos el error
        # en la particion de test i - Para validacion simple (hold-out): entrenamos el clasificador con la particion
        # de train y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero
        # especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.

        # Creamos las particiones
        lista_particiones = particionado.creaParticiones(dataset.datos, seed)

        # Si la longitud de particiones es 1 sabemos que es Validación simple
        if len(lista_particiones) == 1:

            particion = lista_particiones[0]
            # Entrenamos el clasificador con los datos de train (la primera parte de la particion)
            clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain), dataset.nominalAtributos,
                                       dataset.diccionarios)

            # Obtenemos los resultados de clasificación
            tabla_resultados = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest),
                                                      dataset.nominalAtributos, dataset.diccionarios)
            # Calculamos el error con datos test
            error_a_devolver = self.error(dataset.extraeDatos(particion.indicesTest), tabla_resultados)

            # La desviación tipica es 0 en validacion simple
            desviacion = 0

        # Si la longitud es distinto de 1 sabemos que es Validación cruzada
        else:

            errores = np.array([])

            # Creamos un bucle que iterarará según el número de particiones
            for particion in lista_particiones:

                # Entrenamos el clasificador con los datos de train (la primera parte de la particion)
                clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain), dataset.nominalAtributos,
                                           dataset.diccionarios)

                # Obtenemos los resultados de clasificación
                tabla_resultados = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest),
                                                          dataset.nominalAtributos, dataset.diccionarios)
                # Calculamos el error con datos test
                error_parcial = self.error(dataset.extraeDatos(particion.indicesTest), tabla_resultados)
                errores = np.append(errores, [error_parcial])
                error_a_devolver = errores.mean()

                # Calculamos la desviación típica
                desviacion = errores.std()

        return error_a_devolver, desviacion

    ##############################################################################


class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace=False):
        self.verosimilitud = []
        self.priores = []
        self.laplace = laplace

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        (num_elem, num_atr) = datostrain.shape

        num_clases = len(diccionario[-1])

        # PRIORES

        tabla_reps = []

        for clases in range(0, num_clases):
            valor_reps = 0

            tabla_reps.append(valor_reps)

        # guardamos el número de veces que se repite una clase
        for elem in range(0, num_elem):
            valor_clase = datostrain[elem, num_atr - 1]

            tabla_reps[int(valor_clase)] += 1

        # calculamos el valor del prior
        # numero de repeticiones de la clase / el total de elementos
        for clases in range(0, num_clases):
            num_reps = tabla_reps[clases]

            valor_prior = num_reps / num_elem

            self.priores.append(valor_prior)

        # VEROSIMILITUD (TABLAS)

        # para cada atributo creamos una tabla de atributos
        for atr in range(0, num_atr - 1):
            tabla_atributo = []

            # un diccionario para cada clase por cada atributo
            for clases in range(0, num_clases):
                dict_clase = {}

                # Tenemos que inicializar los diccionarios segun el número de atributos
                for v in diccionario[atr].values():
                    dict_clase[v] = 0

                tabla_atributo.append(dict_clase)

            # Comprobamos si el atributo es continuo o nominal

            # if datostrain.tipoAtributos[atr] == "Nominal":
            if atributosDiscretos[atr]:

                # recorremos la matriz observando el atributo específico y la clase
                for elem in range(0, num_elem):
                    valor = datostrain[elem, atr]
                    valor_clase = datostrain[elem, num_atr - 1]

                    dict_in_use = tabla_atributo[int(valor_clase)]

                    # añadimos los valores a los diccionarios
                    # si existe el valor en el diccionario lo incrementamos
                    if valor in dict_in_use:
                        valor_to_increment = dict_in_use.get(valor)
                        valor_to_increment += 1
                        dict_in_use[valor] = valor_to_increment
                    # si no existe lo añadimos
                    else:
                        dict_in_use[valor] = 1

                # si debemos usar la correcion de laplace
                if self.laplace:

                    aplicar_laplace = False

                    # observamos si existen 0
                    for clases in range(0, len(tabla_atributo)):
                        dict_in_use = tabla_atributo[clases]
                        for valores in range(0, len(dict_in_use)):
                            if dict_in_use.get(valores) == 0:
                                aplicar_laplace = True

                    # si existen 0, aplicamos laplace
                    if aplicar_laplace:

                        for clases in range(0, len(tabla_atributo)):
                            dict_in_use = tabla_atributo[clases]
                            for valores in range(0, len(dict_in_use)):
                                valor_to_increment = dict_in_use.get(valores)
                                valor_to_increment += 1
                                dict_in_use[valores] = valor_to_increment

            # elif datostrain.tipoAtributos[atr] == "Continuo":
            else:
                tabla_valores = []

                # un array de valores para cada clase
                for clases in range(0, num_clases):
                    array_valores = []

                    tabla_valores.append(array_valores)

                # recorremos la matriz observando el atributo específico y la clase
                # guardamos el valor continuo en un array de valores para cada clase
                for elem in range(0, num_elem):
                    valor = datostrain[elem, atr]
                    valor_clase = datostrain[elem, num_atr - 1]

                    array_valores_in_use = tabla_valores[int(valor_clase)]

                    array_valores_in_use.append(valor)

                # una vez tenemos las tablas llenas de cada clase calculamos la media y varianza
                # en funcion de la clase

                # un array de valores para cada clase
                for clases in range(0, num_clases):

                    dict_in_use = tabla_atributo[clases]
                    tabla_in_use = tabla_valores[clases]

                    mean_clase = np.mean(tabla_in_use)
                    var_clase = np.var(tabla_in_use)

                    dict_in_use["mean"] = mean_clase
                    dict_in_use["var"] = var_clase

            # metemos la tabla de atributo en la tabla de tablas
            self.verosimilitud.append(tabla_atributo)

    def clasifica(self, datostest, atributosDiscretos, diccionario):

        (num_elem, num_atr) = datostest.shape

        num_clases = len(diccionario[-1])

        tabla_de_resultados = np.array([])

        # para cada fila vamos a realizar naive bayes
        for elem in range(0, num_elem):

            dict_bayes_clase = {}

            # vamos a sacar la verosimilitud por clase
            for clases in range(0, num_clases):

                tabla_vero_atr = []

                # para cada atributo dentro de la fila, accedemos a su tabla de verosimilitud
                # y calculamos su probabilidad en funcion de la clase
                for atr in range(0, num_atr - 1):

                    # Elementos Discretos / Nominales
                    if atributosDiscretos[atr]:

                        tabla_atr = self.verosimilitud[atr]
                        dict_clase = tabla_atr[clases]
                        atributo = datostest[elem, atr]

                        if int(atributo) in dict_clase:
                            numerador = dict_clase[int(atributo)]
                        else:
                            numerador = 0

                        denominador = 0  # elemento neutro de la suma

                        for k in dict_clase.keys():
                            denominador += dict_clase[k]

                        if denominador == 0:
                            verosimilitud_atr = 0
                        else:
                            verosimilitud_atr = numerador / denominador

                    # Elementos Continuos
                    else:
                        tabla_atr_cont = self.verosimilitud[atr]
                        dict_clase = tabla_atr_cont[clases]
                        atributo = datostest[elem, atr]

                        verosimilitud_atr = norm.cdf((atributo - dict_clase["mean"]) / math.sqrt(dict_clase["var"]))

                    tabla_vero_atr.append(verosimilitud_atr)

                # por ultimo, multiplicamos toda la tabla_vero_atr y el prior de la clase
                resultado_multiplicacion = 1  # elemento neutro de la multiplicacion

                for probs in range(0, len(tabla_vero_atr)):
                    resultado_multiplicacion *= tabla_vero_atr[probs]

                resultado_multiplicacion *= self.priores[clases]

                dict_bayes_clase[clases] = resultado_multiplicacion

            # Ahora vemos cual es el máximo de todos los resultados de la clase
            # lo que queremos es la clave del dict, no el valor
            clase_maxima = max(dict_bayes_clase, key=dict_bayes_clase.get)

            # guardamos en la tabla de resultados
            tabla_de_resultados = np.append(tabla_de_resultados, clase_maxima)

        return tabla_de_resultados
