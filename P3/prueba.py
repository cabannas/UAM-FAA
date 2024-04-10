from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random

import Clasificador
import EstrategiaParticionado
from Datos import Datos

if __name__ == '__main__':
    # Elegimos el conjunto de datos
    # dataset = Datos('./ConjuntoDatos/online_shoppers.data')
    dataset = Datos('./DatasetEjemplo/ejemplo2.data')
    # dataset = Datos('./ConjuntoDatos/prueba.data')

    # Elegimos la estrategia de particionado
    estrategia = EstrategiaParticionado.ValidacionSimple(porcentaje_particion=50)
    # estrategia = EstrategiaParticionado.ValidacionCruzada(numero_particiones=4)

    # Elegimos el clasificados (actualmente solo tenemos naive bayes
    # clasificador = Clasificador.ClasificadorNaiveBayes(laplace=False)
    # clasificador = Clasificador.ClasificadorVecinosProximos(vecinos=3)
    # clasificador = Clasificador.ClasificadorRegresionLogistica(aprendizaje=0.1, epocas=10)
    clasificador = Clasificador.ClasificadorAlgoritmoGenetico(tam_poblacion=100, num_epocas=100, num_reglas_individuo=5,
                                                              prob_cruce=85, prob_mutacion=10, prob_elitismo=5,
                                                              no_mejora=100)

    # Calculamos el error y la desviación tipica
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)

    print(error_media)
    print(error_std)

    clasificador.mejor_individuo()
