import numpy as np

class Datos:

  TiposDeAtributos=('Continuo','Nominal')
  tipoAtributos = []
  nombreAtributos = []
  nominalAtributos = []
  diccionarios = []


  # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y diccionarios
  # NOTA: No confundir TiposDeAtributos con tipoAtributos
  def __init__(self, nombreFichero):
      f = fopen(nombreFichero, 'r')
      lines = f.readlines()

      n_datos = int(lines[0]) # Primera linea -> numero de datos
      self.nombreAtributos = lines[1].split(',') # Segunda linea -> nombre de tipoAtributos
      tipos = lines[2].split(',')

      countTipos = 0
      for tipo in tipos:
          if tipo == self.TiposDeAtributos[0]:
              tipoAtributos[countTipos] = tipo

          elif tipo == self.TiposDeAtributos[1]:
              tipoAtributos[countTipos] = tipo

          else:
              # Lanzar expcecion ValueError








  # TODO: implementar en la practica 1
  def extraeDatos(self, idx):
      pass
