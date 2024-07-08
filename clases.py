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
        self.hsv = Solo_v(imagen)



        
        


ruta= "imagenes"
imag1= Imagen(3,"img1.jpg",ruta, "esto es un parametro" )

imagen1_procesada = Preproceso_imagen(imag1.imagen)

corte_s =imagen1_procesada.cortes
# print (corte_s)
valores_cortes = np.zeros((np.shape(corte_s)[0],300)) #armar espacios vacios pero con el shape correspondiente


for j,i in enumerate(corte_s):
    # print(i)
    # print("hola",(np.shape(valores_cortes)))   
    valores_cortes[j,:] = promedio_vectorizado(imagen1_procesada.hsv, i)

# print(np.shape(corte_s))
# print(valores_cortes[j,:])
 
# plt.plot(np.transpose(valores_cortes))
# plt.show()
# # cv.imshow("img1.jpg", imagen1_procesada.recortada)
# # cv.imshow("img1.jpg", imagen1_procesada.imagen_hsv)
# cv.imshow("img1.jpg", imagen1_procesada.hsv)
# cv.waitKey()
# cv.destroyAllWindows
