from abc import ABCMeta, abstractmethod, ABC
import numpy as np
from scipy.stats import norm
import math
import random
import copy

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


class ClasificadorAlgoritmoGenetico(Clasificador):

    def __init__(self, tam_poblacion=100, num_epocas=100, num_reglas_individuo=1, prob_cruce=85, prob_mutacion=10,
                 prob_elitismo=5, no_mejora=50):
        self.tam_poblacion = tam_poblacion
        self.num_epocas = num_epocas
        self.num_reglas_individuo = num_reglas_individuo
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.prob_elitismo = prob_elitismo
        self.no_mejora = no_mejora
        self.best_buddy = []
        self.mejores_fitness = []
        self.medios_fitness = []

    def generar_poblacion_inicial(self, diccionario):

        # Necesitamos saber cuantos atributos y el num de posibilidades de cada atributo

        # Numero de atributos (sin contar con la clase)
        numero_atributos = len(diccionario) - 1

        # Creamos una lista con las longitudes de cada atributo
        lista_long_atrs = []

        # Rellenamos la lista con su correspondiente informacion
        for atr in range(0, numero_atributos):
            lista_long_atrs.append(len(diccionario[atr]))

        # Creamos una lista de individuos
        poblacion = []

        # Ahora generamos tantos individuos como numero de poblacion especificado
        for num_individuo in range(0, self.tam_poblacion):

            # Creamos una lista de reglas para cada individuo (es el propio individuo)
            individuo = []

            # Creamos un numero de reglas aleatorio para el individuo
            num_reglas_aleatorio = random.randint(1, self.num_reglas_individuo)

            # Creamos una regla por cada numero de reglas del individuo
            for num_regla in range(0, num_reglas_aleatorio):

                # Creamos una lista de atributos del diccionario para cada regla (es la propia regla)
                regla = []

                # Rellenamos esa lista por cada atributo que exista en diccionario
                for num_atr in range(0, numero_atributos):

                    # Generamos una lista por cada atributo (es el propio atributo)
                    atributo = []

                    # Para cada valor posible dentro del atributo
                    for num_valores in range(0, len(diccionario[num_atr])):
                        # Generamos un bit aleatorio (0 ó 1)
                        valor = random.getrandbits(1)

                        # Lo guardamos en el atributo
                        atributo.append(valor)

                    # Una vez el atributo está completo lo guardamos en la regla
                    regla.append(atributo)

                # Una vez la regla está completa le asignamos una clase vacia (-1)
                clase = random.getrandbits(1)
                regla.append(clase)

                # Ahora la regla está completa y la guardamos en el individuo
                individuo.append(regla)

            # Una vez el individuo está completo lo guardamos en la población
            poblacion.append(individuo)

        return poblacion

    def calcular_fitness(self, poblacion, datosTrain):

        # Obtenemos el numero de elementos y de atributos de datos train
        (num_elem, num_atr) = datosTrain.shape

        fitness_poblacion = {}

        # Para cada individuo
        for num_individuo in range(0, len(poblacion)):

            # Seleccionamos el individuo dentro de la poblacion
            individuo = poblacion[num_individuo]

            # Este es el fitness del individuo
            fitness_acumulativo = 0

            # Vamos a ver su fitness para cada dato
            for elem in range(0, num_elem):

                # Lista de clasificaciones del individuo
                lista_clasificaciones_elem = []

                # Necesitamos sacar el fitness individual de cada regla
                for num_regla in range(0, len(individuo)):

                    # Seleccionamos la regla dentro del individuo
                    regla = individuo[num_regla]

                    # Creamos una variable para ver que clasifica
                    clasifica = -1

                    longitud_regla = len(regla)

                    # Tenemos que iterar los atributos
                    for atr in range(0, num_atr - 1):

                        # Cogemos el dato
                        dato = datosTrain[elem, atr]

                        # Cogemos el atributo
                        atributo = regla[atr]  # Es una lista

                        # Si no acierta marcamos el flag a false
                        if atributo[int(dato)] != 1:
                            break
                        if atr == (longitud_regla - 2):  # Tiene que ser -2 porque el atr tambien
                            clasifica = regla[-1]

                    lista_clasificaciones_elem.append(clasifica)

                # Creamos un diccionario con las posibilidades
                resultado_clasifica = {0: 0, 1: 0}

                # Ahora vemos a que clasifica el individuo
                for resultado in lista_clasificaciones_elem:
                    if resultado != -1:  # No nos importan los -1
                        resultado_clasifica[resultado] += 1

                if resultado_clasifica[0] > resultado_clasifica[1]:
                    resultado_dato = 0
                elif resultado_clasifica[0] < resultado_clasifica[1]:
                    resultado_dato = 1
                else:
                    resultado_dato = -1

                # Ahora tenemos que ver si clasifica bien comparando el dato con el resultado clasificado
                clase = datosTrain[elem, num_atr - 1]

                if resultado_dato == clase:
                    fitness_acumulativo += 1

            # Calculamos el fitness total del individuo
            fitness = fitness_acumulativo / num_elem

            fitness_poblacion[num_individuo] = fitness

        # Guardamos en una lista de tuplas (indice, fitness) ordenadas segun su fitness
        fitness_sorted = []
        for i in reversed(sorted(fitness_poblacion.items(), key=lambda kv: (kv[1], kv[0]))):
            fitness_sorted.append(i)

        # Devolvemos la lista de fitness ya ordenada
        return fitness_sorted

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):

        # Definimos la variable epoca_actual
        epoca_actual = 0

        # Definimos la variable epoca_sin_mejora
        epoca_sin_mejora = 0

        # Definimos la variable mejor_fitness
        mejor_fitness = 0

        # Generamos una poblacion inicial
        poblacion = self.generar_poblacion_inicial(diccionario)

        # Calculamos el fitness (ya ordenado) de la poblacion
        fitness_poblacion = self.calcular_fitness(poblacion, datosTrain)

        # Mientras no lleguemos al fin de las epocas o no lleguemos al maximo numero de epocas sin mejorar
        while epoca_actual != self.num_epocas and epoca_sin_mejora != self.no_mejora:

            # ELITISMO --------------------------------------------------

            # Calculamos el numero de elitistas
            num_elitistas = self.tam_poblacion * (self.prob_elitismo / 100)

            lista_indices_elite = []

            # Guardamos los mejores en funcion del fitness y el elitismo (una lista mas escueta)
            for num_elite in range(0, int(num_elitistas)):
                # metemos los indices de la elite en una lista en una lista
                dict_indv_fitness = fitness_poblacion[num_elite]
                lista_indices_elite.append(dict_indv_fitness[0])

            lista_elite = []
            # Ahora guardamos los individuos cuyos indices estén en la lista
            for element in range(0, len(lista_indices_elite)):
                individuo = poblacion[lista_indices_elite[element]]
                individuo_copia = copy.deepcopy(individuo)
                lista_elite.append(individuo_copia)

            # Generamos una lista de indices aleatorios para seleccionar una nueva poblacion con repeticion a partir
            # de la poblacion anterior
            lista_indices_nueva_poblacion = []
            for num_individuos in range(0, self.tam_poblacion):
                indice_aleatorio = random.randint(1, self.tam_poblacion - 1)
                lista_indices_nueva_poblacion.append(indice_aleatorio)

            nueva_poblacion = []
            # Generamos una poblacion nueva a partir de los índices anteriores
            for num_indice in range(0, len(lista_indices_nueva_poblacion)):
                indice = lista_indices_nueva_poblacion[num_indice]
                nueva_poblacion.append(poblacion[indice])

            # CRUCE --------------------------------------------------

            # Tenemos que seleccionar pares de padres
            for padres in range(1, self.tam_poblacion, 2):

                # Vemos si ocurre un cruce entre los padres en funcion de la probabilidad de cruce
                # Generamos un numero aleatorio entre 1 y 100, y lo comparamos con la probabilidad de cruce
                hay_cruce = random.randint(1, 100)

                # Si hay cruce
                if hay_cruce <= self.prob_cruce:

                    padre1 = nueva_poblacion[padres - 1]
                    padre2 = nueva_poblacion[padres]

                    los_dos_hijos = []

                    for num_hijos in range(0, 2):

                        # Definimos el numero de reglas del hijo
                        tam_max_regla_padre = 0

                        if len(padre1) > len(padre2):
                            tam_max_regla_padre = len(padre1)
                        else:
                            tam_max_regla_padre = len(padre2)

                        num_reglas_hijo = random.randint(1, tam_max_regla_padre)

                        hijo = []

                        # Para cada regla del hijo
                        for num_regla in range(0, num_reglas_hijo):

                            # Cogemos una regla aleatorio del primer padre y una del segundo
                            regla_padre_1 = padre1[random.randint(0, len(padre1) - 1)]
                            regla_padre_2 = padre2[random.randint(0, len(padre2) - 1)]

                            regla_hijo = []

                            # Para cada atributo dentro de la regla:
                            for num_atributos in range(0, len(regla_padre_1) - 1):  # Todas las reglas tienen la misma
                                # longitud

                                atributo_padre1 = regla_padre_1[num_atributos]
                                atributo_padre2 = regla_padre_2[num_atributos]

                                atributo_hijo = []

                                for valor_in_atributo in range(0, len(atributo_padre1)):  # Todos los atributos
                                    # tienen la misma longitud

                                    # Generamos dos valores aleatorios para realizar el cruce entre 0 y 1
                                    elegir_padre = random.getrandbits(1)

                                    if elegir_padre == 0:
                                        atributo_hijo.append(atributo_padre1[valor_in_atributo])
                                    else:
                                        atributo_hijo.append(atributo_padre2[valor_in_atributo])

                                regla_hijo.append(atributo_hijo)

                            # Ahora elegimos la clase de uno de los padres
                            elegir_padre = random.getrandbits(1)

                            if elegir_padre == 0:
                                regla_hijo.append(regla_padre_1[-1])
                            else:
                                regla_hijo.append(regla_padre_2[-1])

                            hijo.append(regla_hijo)

                        los_dos_hijos.append(hijo)

                    # Ahora sustituimos a los padres por los hijos
                    nueva_poblacion[padres - 1] = los_dos_hijos[0]
                    nueva_poblacion[padres] = los_dos_hijos[1]

            # MUTACION --------------------------------------------------

            # Para todos los individuos de la población
            for num_individuo in range (0, self.tam_poblacion):

                # Vemos si ocurre una mutacion en el individuo en funcion de la probabilidad de mutacion
                # Generamos un numero aleatorio entre 1 y 100, y lo comparamos con la probabilidad de mutacion
                hay_mutacion = random.randint(1, 100)

                # Si hay mutacion
                if hay_mutacion <= self.prob_mutacion:

                    individuo_mutado = nueva_poblacion[num_individuo]

                    # Tenemos que elegir de forma aleatoria la regla del individuo en la que hay mutacion
                    num_regla_mutacion = random.randint(0, len(individuo_mutado)-1)

                    regla_mutada = individuo_mutado[num_regla_mutacion]

                    # Ahora tenemos que ver a que atributo de la regla afecta la mutacion
                    num_atr_mutacion = random.randint(0, len(regla_mutada)-1)

                    # Si el atributo a mutar es la clase
                    if num_atr_mutacion == len(regla_mutada)-1:

                        # Realizamos el bitflip propio de la mutacion
                        if regla_mutada[num_atr_mutacion] == 0:
                            regla_mutada[num_atr_mutacion] = 1
                        else:
                            regla_mutada[num_atr_mutacion] = 0

                    else:
                        atr_mutado = regla_mutada[num_atr_mutacion]

                        # Por ultimo vemos que valor del atributo se muta:
                        num_valor_mutacion = random.randint(0, len(atr_mutado)-1)

                        # Realizamos el bitflip propio de la mutacion
                        if atr_mutado[num_valor_mutacion] == 0:
                            atr_mutado[num_valor_mutacion] = 1
                        else:
                            atr_mutado[num_valor_mutacion] = 0

            # SUSTRACCION DE PEORES Y ADICIÓN DE LA ELITE EN LA NUEVA POBLACIÓN ----------------------------------------

            # Calculamos el fitness (ya ordenado) de la poblacion
            fitness_poblacion = self.calcular_fitness(nueva_poblacion, datosTrain)

            # Obtenemos el numero de individuos con peor fitness igual al numero de elitistas
            la_banda_y_yo = []

            for peores in range(1, len(lista_indices_elite)+1):
                tupla_fitness = fitness_poblacion[-peores]
                la_banda_y_yo.append(tupla_fitness[0])

            # Ahora sustituimos a los mejores por los peores
            for sustitutos in range(0, len(lista_indices_elite)):

                nueva_poblacion[la_banda_y_yo[sustitutos]] = lista_elite[sustitutos]

            # CHECKEO FITNESS Y PASO A NUEVA POBLACIÓN --------------------------------------------------

            # Calculamos el fitness (ya ordenado) de la poblacion
            fitness_poblacion = self.calcular_fitness(nueva_poblacion, datosTrain)

            # Checkeamos el mejor fitness para ver si hay mejor en esta poblacion
            mejor_fitness_tupla = fitness_poblacion[0]
            mejor_fitness_actual = mejor_fitness_tupla[1]

            self.mejores_fitness.append(mejor_fitness_actual)

            calcula_fitness_medio = 0

            for num_fitness in range (0, len(fitness_poblacion)):
                fitness_tupla = fitness_poblacion[num_fitness]
                calcula_fitness_medio += fitness_tupla[1]

            fitness_medio = calcula_fitness_medio / len(fitness_poblacion)

            self.medios_fitness.append(fitness_medio)

            if mejor_fitness_actual > mejor_fitness:

                mejor_fitness = mejor_fitness_actual
                epoca_sin_mejora = 0
            else:
                epoca_sin_mejora += 1

            # Pasamos de generacion
            poblacion = nueva_poblacion
            epoca_actual += 1

        # Ahora debemos coger al mejor de los individuos
        fitness_poblacion = self.calcular_fitness(poblacion, datosTrain)
        mejor_fitness_tupla = fitness_poblacion[0]
        mejor_fitness_indice = mejor_fitness_tupla[0]
        self.best_buddy = poblacion[mejor_fitness_indice]

        return self.best_buddy

    def clasifica(self, datosTest, atributosDiscretos, diccionario):

        # Obtenemos el numero de elementos y de atributos de datos train
        (num_elem, num_atr) = datosTest.shape

        lista_resultados = []

        # Vamos a ver su fitness para cada dato
        for elem in range(0, num_elem):

            # Creamos una lista de clasificacion del individuo segun sus reglas
            lista_clasificaciones_reglas = []

            # Necesitamos sacar el resultado individual de cada regla
            for num_regla in range(0, len(self.best_buddy)):
                # Seleccionamos la regla dentro del individuo
                regla = self.best_buddy[num_regla]

                # Creamos una variable para ver que clasifica
                clasifica = -1

                # Tenemos que iterar los atributos
                for atr in range(0, num_atr - 1):

                    # Cogemos el dato
                    dato = datosTest[elem, atr]

                    # Cogemos el atributo
                    atributo = regla[atr]  # Es una lista

                    # Si no acierta marcamos el flag a false
                    if atributo[int(dato)] != 1:
                        break
                    if atr == (len(regla) - 2):  # Tiene que ser -2 porque el atr tambien
                        clasifica = regla[-1]
                    lista_clasificaciones_reglas.append(clasifica)

            # Creamos un diccionario con las posibilidades
            resultado_clasifica = {0: 0, 1: 0}

            # Ahora vemos a que clasifica el individuo
            for resultado in lista_clasificaciones_reglas:
                if resultado != -1:  # No nos importan los -1
                    resultado_clasifica[resultado] += 1

            if resultado_clasifica[0] > resultado_clasifica[1]:
                resultado_dato = 0
            elif resultado_clasifica[0] < resultado_clasifica[1]:
                resultado_dato = 1
            else:
                resultado_dato = -1

            lista_resultados.append(resultado_dato)

        return lista_resultados

    def mejor_individuo(self):

        for num_reglas in range(0, len(self.best_buddy)):
            print(self.best_buddy[num_reglas])

        return self.best_buddy



