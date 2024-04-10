from Datos import Datos

if __name__ == '__main__':

    dataset = Datos('./DatasetEjemplo/ejemplo1.data')

    print("Diccionarios:")
    print(dataset.diccionarios)
    print("Longitud de diccionarios (num atr + clase):")
    print(len(dataset.diccionarios))
    print("Primer diccionario:")
    print(dataset.diccionarios[0])
    print("Longitud del primer diccionario (num posibilidades)")
    print(len(dataset.diccionarios[0]))

    # Bucle que imprime todos los diccionarios con sus respectivas longitudes:
    print("==========================")

    for diccionario in range (0, len(dataset.diccionarios)-1):
        print("Diccionario " + str(diccionario))
        print(dataset.diccionarios[diccionario])
        print("Longitud del del diccionario " + str(diccionario))
        print(len(dataset.diccionarios[diccionario]))
