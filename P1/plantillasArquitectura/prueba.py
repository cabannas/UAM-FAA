import Clasificador
import EstrategiaParticionado
from Datos import Datos
from sklearn import preprocessing as pp
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Elegimos el conjunto de datos
    # dataset = Datos('./datos/tic-tac-toe.data')
    # dataset = Datos('./datos/german.data')
    dataset = Datos('./datos/lenses.data')

    # Elegimos la estrategia de particionado
    estrategia = EstrategiaParticionado.ValidacionSimple()
    # estrategia = EstrategiaParticionado.ValidacionCruzada(numero_particiones = 4)

    # Elegimos el clasificados (actualmente solo tenemos naive bayes
    clasificador = Clasificador.ClasificadorNaiveBayes(laplace=False)

    # Calculamos el error y la desviación tipica
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)

    print(error_media)
    print(error_std)

    g = GaussianNB()
    data = dataset.datos[:, :-1]
    data_class = dataset.datos[:, -1]

    data_train, data_test, data_class_train, data_class_test = train_test_split(data, data_class,
                                                                                test_size=0.5)
    g.fit(data_train, data_class_train)
    score = g.score(data_test, data_class_test)
    invertido = 1 - score
    print(invertido)

    score = cross_val_score(g, data, data_class, cv=4)
    media = score.mean()
    invertido = 1 - media
    desv = score.std()

    print(invertido, desv)

    # Elegimos el conjunto de datos
    dataset2 = Datos('./datos/tic-tac-toe.data')
    # dataset2 = Datos('./datos/german.data')
    # dataset2 = Datos('./datos/lenses.data')

    # Elegimos la estrategia de particionado
    # estrategia2 = EstrategiaParticionado.ValidacionSimple()
    estrategia2 = EstrategiaParticionado.ValidacionCruzada(numero_particiones=4)

    # Elegimos el clasificados (actualmente solo tenemos naive bayes
    clasificador2 = Clasificador.ClasificadorNaiveBayes(laplace=True)

    # Calculamos el error y la desviación tipica
    error_media2, error_std2 = clasificador2.validacion(estrategia2, dataset2, clasificador2)

    print(error_media)
    print(error_std)

    m = MultinomialNB(alpha=1.0)
    data2 = dataset2.datos[:, :-1]
    data_class2 = dataset2.datos[:, -1]
    #m.fit(data2, data_class2)
    score = cross_val_score(m, data2, data_class2, cv=4)
    media = score.mean()
    invertido = 1 - media
    desv = score.std()

    print(invertido, desv)
