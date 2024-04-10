
if __name__ == '__main__':

    lista1 = []

    sublista11 = [1 , 2 , 3]
    sublista12 = [4 , 5 , 6]

    subsublista = ["a", "b", "c"]

    lista1.append(sublista11)
    lista1.append(sublista12)

    sublista12.append(subsublista)

    lista2 = []

    precopia = lista1[1]
    copia = precopia.copy()

    lista2.append(copia)

    otra_precopia = lista1[1]

    otra_precopia[1] = "XXX"
    otra_precopia[3] = ["X", "Y", "Z"]

    print(lista1)
    print(lista2)
