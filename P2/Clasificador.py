from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import norm
import math
import random


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

    def calculaMediasDesv(self, datostrain):

        (num_elem, num_atr) = datostrain.shape

        lista_total_elem_atr = []

        medias = []
        desvtips = []

        # Creamos una lista de elementos por atributo para luego aplicarles las operaciones mean y std
        # Para ello tiene que ser de tipo numpy.array
        for atr in range(0, num_atr - 1):
            lista_elem_por_atr = np.array([])

            lista_total_elem_atr.append(lista_elem_por_atr)

        # Con este bucle rellenamos las listas de atributos
        for elem in range(0, num_elem):

            for atr in range(0, num_atr - 1):
                lista_elem_por_atr = lista_total_elem_atr[atr]

                elemento = datostrain[elem, atr]

                lista_elem_por_atr = np.append(lista_elem_por_atr, [elemento])

                lista_total_elem_atr[atr] = lista_elem_por_atr

        # Aplicamos mean y std a cada lista y guardamos el resultado en self.medias y self.desvtips
        for atr in range(0, num_atr - 1):
            lista_elem_por_atr = lista_total_elem_atr[atr]

            media_del_atr = lista_elem_por_atr.mean()
            desv_del_atr = lista_elem_por_atr.std()

            medias.append(media_del_atr)
            desvtips.append(desv_del_atr)

        return medias, desvtips

    def normalizaDatos(self, datos, medias, desvtips, atributosDiscretos):

        (num_elem, num_atr) = datos.shape

        for elem in range(0, num_elem):

            for atr in range(0, num_atr - 1):

                # Solo para valores continuos
                if not atributosDiscretos[atr]:
                    dato = datos[elem, atr]

                    # Normalización
                    if desvtips[atr] == 0:
                        desvtips[atr] = 0.0000001
                    dato_normalizado = (dato - medias[atr]) / desvtips[atr]

                    datos[elem, atr] = dato_normalizado

        return datos

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


class ClasificadorVecinosProximos(Clasificador):

    def __init__(self, vecinos=1, normalizar=True):
        self.vecinos = vecinos
        self.datos_entrenados = np.array([])
        self.normalizar = normalizar

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):

        if self.normalizar:

            # Primero calculamos las medias y desviaciones típicas
            (medias, desvtips) = self.calculaMediasDesv(datosTrain)

            # Luego Normalizamos los datos y los guardamos en nuestra matriz numpy
            self.datos_entrenados = self.normalizaDatos(datosTrain, medias, desvtips, atributosDiscretos)

        else:

            self.datos_entrenados = datosTrain

    def clasifica(self, datosTest, atributosDiscretos, diccionario):

        (num_elem, num_atr) = datosTest.shape
        (num_elem_train, num_atr_train) = self.datos_entrenados.shape

        if self.normalizar:

            # Primero calculamos las medias y desviaciones típicas
            (medias, desvtips) = self.calculaMediasDesv(datosTest)

            # Luego Normalizamos los datos y los guardamos en nuestra matriz numpy
            datos_testeo = self.normalizaDatos(datosTest, medias, desvtips, atributosDiscretos)

        else:

            datos_testeo = datosTest

        lista_resultados = []

        for elem in range(0, num_elem):

            lista_distancias = []
            lista_distancias_dict = []

            # Calculamos la distancia del elemento test test con cada elemento de train
            for elem_train in range(0, num_elem_train):

                dict_dist_clase = {}

                distancia = 0

                for atr in range(0, num_atr - 1):

                    # Si es nominal
                    if atributosDiscretos[atr]:

                        if datos_testeo[elem, atr] != self.datos_entrenados[elem_train, atr]:
                            distancia += 1

                    # Si es continuo
                    else:

                        distancia_particular = (datos_testeo[elem, atr] - self.datos_entrenados[elem_train, atr]) ** 2
                        distancia += distancia_particular

                distancia_final = math.sqrt(distancia)
                dict_dist_clase["distancia"] = distancia_final
                dict_dist_clase["clase"] = self.datos_entrenados[elem_train, num_atr_train - 1]

                lista_distancias_dict.append(dict_dist_clase)
                lista_distancias.append(distancia_final)

            # Ordenamos de mayor a menor las distancias
            lista_distancias.sort()

            dict_clase_result = {}

            # miramos las clases de cada elemento según el número de vecinos
            for k in range(0, self.vecinos):

                distancia = lista_distancias[k]

                for dicci in lista_distancias_dict:

                    if distancia in dicci.values():
                        clase = dicci.get("clase")

                # Ya tenemos la clase, la guardamos en el diccionario, si ya existe le sumamos un 1
                if clase in dict_clase_result.keys():

                    valor = dict_clase_result[clase]
                    valor += 1
                    dict_clase_result[clase] = valor

                else:
                    dict_clase_result[clase] = 1

            # Obtenemos la clase máxima
            clase_maxima = max(dict_clase_result, key=dict_clase_result.get)

            # guardamos la clase en el resultado
            lista_resultados.append(clase_maxima)

        return np.array(lista_resultados)


class ClasificadorRegresionLogistica(Clasificador):

    def __init__(self, aprendizaje=1, epocas=1):
        self.apredinzaje = aprendizaje
        self.epocas = epocas
        self.frontera = []

    def calculo_sigmoidal(self, frontera, ejemplo):

        # 1/(1+math.exp(-(w*x).sum()))

        expo = frontera[0]

        for atr in range(0, len(ejemplo) - 1):
            expo += frontera[atr + 1] * ejemplo[atr]

        expo = expo * -1

        e_con_expo = math.e ** expo

        dividendo = 1 + e_con_expo

        res = 1 / dividendo

        return res

    def producto_a_ejemplo(self, frontera, ejemplo):

        # Clase 1
        if ejemplo[len(ejemplo) - 1] == 0:
            t_de_ejemplo = 1
        # Clase 2
        else:
            t_de_ejemplo = 0

        return self.apredinzaje * (self.calculo_sigmoidal(frontera, ejemplo) - t_de_ejemplo)

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):

        self.frontera.clear()

        (num_elem, num_atr) = datosTrain.shape

        # Primero calculamos las medias y desviaciones típicas
        (medias, desvtips) = self.calculaMediasDesv(datosTrain)

        # Luego Normalizamos los datos y los guardamos en nuestra matriz numpy
        datos_train = self.normalizaDatos(datosTrain, medias, desvtips, atributosDiscretos)

        frontera_ini = []

        # Creamos nuestra frontera inicial a partir de números aleatorios dentro del intervalo [-0.5,0,5]
        for atr in range(0, num_atr):
            frontera_ini.append(random.uniform(-0.5, 0.5))

        # Entrenamos según el número de épocas
        for n_epocas in range(0, self.epocas):

            for elem in range(0, num_elem):

                producto = self.producto_a_ejemplo(frontera_ini, datos_train[elem])
                x_post = [producto]

                for atr in range(0, num_atr - 1):
                    x_post.append(producto * datos_train[elem, atr])

                for atr in range(0, num_atr):
                    frontera_ini[atr] = frontera_ini[atr] - x_post[atr]

        for atr in range(0, num_atr):
            self.frontera.append(frontera_ini[atr])

    def clasifica(self, datosTest, atributosDiscretos, diccionario):

        (num_elem, num_atr) = datosTest.shape

        # Primero calculamos las medias y desviaciones típicas
        (medias, desvtips) = self.calculaMediasDesv(datosTest)

        # Luego Normalizamos los datos y los guardamos en nuestra matriz numpy
        datos_test = self.normalizaDatos(datosTest, medias, desvtips, atributosDiscretos)

        lista_resultados = []

        for elem in range(0, num_elem):

            valor_sigmo = self.calculo_sigmoidal(self.frontera, datos_test[elem])

            # Si el valor es mayor a 0.5 clasifica Clase 1
            if valor_sigmo >= 0.5:

                lista_resultados.append(0)

            # Si el valor es menor de 0.5 clasifica Clase 2
            else:

                lista_resultados.append(1)

        return np.array(lista_resultados)
