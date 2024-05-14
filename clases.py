import json
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 
from scipy.optimize import curve_fit
from funciones import * 
import logging




class Imagen:
    def __init__(self, uid:int, nombre:str, ruta:str, parametros:str):
        self.uid = uid
        self.nombre = nombre
        self.ruta = ruta
        self.imagen = os.path.join(ruta, nombre)
        ruta_json = "datos.json"
        self.parametros= parametros
        self.verificacion = verificar_uid(ruta_json, self.uid, self.nombre, self.ruta, self.parametros)  # Orden de los parámetros corregido
    

    



class Preproceso_imagen:
    def __init__(self, imagen):
        self.leida= cv.imread(imagen)
        # self.recortada=recortar_imagen(self.leida, x=720, y= 200, ancho=800, alto = 300 )  
        self.hsv = cv.cvtColor(self.leida, cv.COLOR_BGR2HSV)
        self.cortes =corte(self.leida) 
        # self.maximo= promedio_vectores_vectorizado(self.hsv, self.cortes)



ruta= "imagenes"
imag1= Imagen(3,"img1.jpg",ruta, "esto es un parametro" )

imagen1_procesada = Preproceso_imagen(imag1.imagen)

corte_s =imagen1_procesada.cortes
print(corte_s)

# # cv.imshow("img1.jpg", imagen1_procesada.recortada)
# # cv.imshow("img1.jpg", imagen1_procesada.imagen_hsv)
# cv.imshow("img1.jpg", imagen1_procesada.hsv)
# cv.waitKey()
# cv.destroyAllWindows
