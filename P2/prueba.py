from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from plotModel import plotModel
import matplotlib.pyplot as plt

import Clasificador
import EstrategiaParticionado
from Datos import Datos

if __name__ == '__main__':
    # Elegimos el conjunto de datos
    # dataset = Datos('./ConjuntoDatos/online_shoppers.data')
    dataset = Datos('./ConjuntoDatos/wdbc.data')
    # dataset = Datos('./ConjuntoDatos/prueba.data')

    # Elegimos la estrategia de particionado
    # estrategia = EstrategiaParticionado.ValidacionSimple(porcentaje_particion=50)
    estrategia = EstrategiaParticionado.ValidacionCruzada(numero_particiones=4)

    # Elegimos el clasificados (actualmente solo tenemos naive bayes
    # clasificador = Clasificador.ClasificadorNaiveBayes(laplace=False)
    # clasificador = Clasificador.ClasificadorVecinosProximos(vecinos=3)
    clasificador = Clasificador.ClasificadorRegresionLogistica(aprendizaje=0.1, epocas=10)

    # Calculamos el error y la desviación tipica
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)

    print(error_media)
    print(error_std)

    # Vecinos Próximos
    # dataset = Datos('./ConjuntoDatos/example1.data')
    # valores_train = dataset.datos[:, :-1]
    # valores_test = dataset.datos[:, -1]
    # clasificador_knn_sklearn = KNeighborsClassifier(n_neighbors=1, p=2)  # p = 2 -> Distancia euclidea
    # score = cross_val_score(clasificador_knn_sklearn, valores_train, valores_test, cv=4)
    # media = score.mean()
    # invertido = 1 - media
    # desv = score.std()

    # print("para sklearn")
    # print(invertido)
    # print(desv)

    # Regresion Logística
    # valores_train = dataset.datos[:, :-1]
    # valores_test = dataset.datos[:, -1]
    # clasificador_rl_sklearn = LogisticRegression(max_iter=10)
    # score = cross_val_score(clasificador_knn_sklearn, valores_train, valores_test, cv=4)
    # media = score.mean()
    # invertido = 1 - media
    # desv = score.std()


    print("=============A PARTIR DE AQUI =============")

    # Elegimos el conjunto de datos
    dataset = Datos('./ConjuntoDatos/example1.data')

    # Elegimos la estrategia de particionado
    # estrategia = EstrategiaParticionado.ValidacionSimple(porcentaje_particion=50)
    estrategia = EstrategiaParticionado.ValidacionCruzada(numero_particiones=4)

    # Elegimos el clasificador
    # clasificador = Clasificador.ClasificadorVecinosProximos(vecinos=3, normalizar=True)
    clasificador = Clasificador.ClasificadorRegresionLogistica(aprendizaje=0.5, epocas=10)

    # Calculamos el error y la desviación tipica
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)

    print(error_media)

    print(error_std)

    print(estrategia.nombre_estrategia)
    print(estrategia.numero_particiones)
    print(estrategia.particiones)


    clasificador.entrenamiento()
    print(estrategia.particiones[0])

    # Plot Model
    indices_train = estrategia.particiones[-1].indicesTrain

    plotModel(dataset.datos[indices_train, 0], dataset.datos[indices_train, 1], dataset.datos[indices_train, -1] != 0,
              clasificador, "Numero de vecinos = 1", dataset.diccionarios)

    plotModel(dataset.datos[datosTest, 0], dataset.datos[datosTest, 1], dataset.datos[datosTest, -1] != 0, knn, "KNN",
              dataset.diccionarios);

    plt.plot(dataset.datos[dataset.datos[:, -1] == 0, 0], dataset.datos[dataset.datos[:, -1] == 0, 1], 'bo')
    plt.plot(dataset.datos[dataset.datos[:, -1] == 1, 0], dataset.datos[dataset.datos[:, -1] == 1, 1], 'ro')
    plt.show()
