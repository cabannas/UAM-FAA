print("Numero de individuos:")
print(self.tam_poblacion)
print("Numero de reglas por individuo")
print(self.num_reglas_individuo)
print("Longitud reglas")
long_regla = 0
for atrs in lista_long_atrs:
    long_regla += atrs
print(long_regla)

for num_individuo in range(0, len(poblacion)):
    individuo = poblacion[num_individuo]

    print("Individuo " + str(num_individuo) + ":")

    for num_regla in range(0, len(individuo)):
        regla = individuo[num_regla]
        for num_atributo in range(0, len(regla) - 1):  # No printeamos la clase
            atributo = regla[num_atributo]
            for num_valores in range(0, len(atributo)):
                valor = atributo[num_valores]
                print(valor, end="")

        print(" - Clase: ", end="")
        print(regla[-1])
    print("")





# otro printer
    def imprimir_poblacion_pero_bien(self, poblacion):

        for individuos in range(0, len(poblacion)):
            print("Individuo " + str(individuos))
            individuo = poblacion[individuos]
            for reglas in range(0, len(individuo)):
                print("\t" + str(individuo[reglas]))