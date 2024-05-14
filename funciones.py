import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 
from scipy.optimize import curve_fit
import skimage as skm
import logging
import json

def verificar_uid(ruta_json, uid_buscado, nombre_archivo, ruta_archivo, parametros):
    try:
        with open(ruta_json, 'r') as archivo_json:
            datos = json.load(archivo_json)
    except FileNotFoundError:
        datos = []

    encontrado = False

 
    for entrada in datos:
        if entrada['uid'] == uid_buscado:
            encontrado = True
            print(f"Existe {nombre_archivo}")

            
            break

    if not encontrado:
        nuevo_datos = {
            'archivo': ruta_archivo,
            'uid': uid_buscado,
            'ruta': nombre_archivo,
            'imagen': os.path.join(ruta_archivo, nombre_archivo),
            'parametros':parametros,
        }
        datos.append(nuevo_datos)
        with open(ruta_json, 'w') as archivo_json:
            json.dump(datos, archivo_json, indent=4)
            print(f"Se crea json para {nombre_archivo}")

        
        return nuevo_datos
    return entrada



def diferencia_imagenes (imagen1,nombre_1, imagen2,nombre_2):
    peso_a = 1
    peso_b = 2
    nombre_salida_1 = f"{os.path.splitext(nombre_1)[0]}"
    nombre_salida_2 = f"{os.path.splitext(nombre_2)[0]}"
    nombre_salida = f"{nombre_salida_1}-{nombre_salida_2}.jpg"
    diferencia = cv.addWeighted(imagen1, peso_a, -imagen2, peso_b, .5)
    cv.imwrite(nombre_salida , diferencia)
    return diferencia, nombre_salida

def otsu (imagen, nombre):
    h, s, v = cv.split(imagen)
    ret3,th3 = cv.threshold(v ,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    nombre_salida = f"{os.path.splitext(nombre)[0]}-otsu.jpg"
    logging.info(f"Se proceso  correctamente {nombre_salida}")
    return th3, nombre_salida

def convertir_gris (imagen, nombre):
    nombre_salida = f"{os.path.splitext(nombre)[0]}-gris.jpg"
    gris =  cv.cvtColor(imagen, cv.COLOR_RGB2GRAY)
    return gris, nombre_salida



def imagen_erode_dilate(imagen1, nombre):
    kernel = np.ones((5, 5), np.uint8)
    dilate_erode = cv.dilate(imagen1, kernel, iterations=1)
    dilate_erode_erode = cv.erode(dilate_erode, kernel, iterations=1)
    nombre_dilate_erode = f"{os.path.splitext(nombre)[0]}-dilate-erode.jpg"
    print(f"Dilate-erode terminado  de {nombre_dilate_erode} ")
    erode_dilate = cv.erode(imagen1, kernel, iterations=1)
    imagen_convertida = cv.dilate(erode_dilate, kernel, iterations=1)
    nombre_imagen = f"{os.path.splitext(nombre)[0]}-erode-dilate.jpg"
    logging.info(f"Erode-dileta terminada de {nombre_imagen}")

    return imagen_convertida, nombre_imagen




def recortar_imagen(imagen_leida, x:int, y:int, ancho:int, alto:int):
        imagen_recortada = imagen_leida[y:y+alto, x:x+ancho]
        
        return imagen_recortada
#Hay que modificar que v recibe esta funcion.

def armado_rutas (directorio):
    imagenes = []

    directorio_recorrido = os.listdir(directorio)


    for archivo in directorio_recorrido:
        if archivo is not None:
            imagen = os.path.join(directorio, archivo)
            nombre_salida, _ = os.path.splitext(archivo) 
            imagen_leida= cv.imread(imagen)
            print(f"salio todo piola con {nombre_salida}")
            imagenes.append((imagen_leida, nombre_salida  ))
        else:
            print("Ale la cago") 
    return imagenes
#Devuelve un array con las imagenes


def pasar_hsv(imagen, nombre_imagen):
    imagen_convertida = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    nombre_salida_hsv = f"{os.path.splitext(nombre_imagen)[0]}_hsv.jpg"
    print (f"Proceso hsv de {nombre_salida_hsv} COMPLETO")
    return imagen_convertida, nombre_salida_hsv


def encontrar_puntos_maximos(imagen_hsv, lineas):
    h, s, v = cv.split(imagen_hsv)
   
    v_maximos = []
    for x in lineas:
        columna_v = v[:, x]
        indice_max_v = np.argmax(columna_v) #devuelve el  indice del valor maximo 
        max_v = v[indice_max_v, x] 
        v_maximos.append((x, indice_max_v, max_v))
        # print (v_maximos)
        #Ojo! Revisar para ver si se puede encontrar mejor manera
    return v_maximos



def funcion_ajuste(x, A, sigma, offset,centro):
    return A/sigma * np.exp(-1/2*((x-centro)/sigma)**2) + offset


def encontrar_recta (array_ordenadas):
    x = []
    y = []
    for xmax, index,  vmax in array_ordenadas :
        x.append(xmax)
        y.append(index)
    print ("x:",x)
    print ("y:",y)
    z = np.polyfit(x, y, 1)
    return z

def ajustar_angulo (valores_maximos, imagen):
    z = encontrar_recta(valores_maximos)

    x = [point[0] for point in valores_maximos]
    y = [point[2] for point in valores_maximos]

    pendiente =z[0]

    angulo = np.degrees(np.arctan(pendiente))
    print ("el angulo es :",angulo)
    x_pred = np.linspace(min(x), max(x), 100)
    y_pred = z[0] * x_pred + z[1]
    imagen_rotada =skm.transform.rotate(imagen,angle =  angulo ,resize=False, mode='constant')
    return imagen_rotada, x_pred, y_pred


def graficar_datos_imagen(imagen, linea):
    h, s, v = cv.split(imagen)
    valores_y = []
    indices = []
    for y in linea:
        valor_y = v[:, y]
        indices.extend(list(range(len(valor_y))))
        valores_y.extend(valor_y)
    return indices, valores_y
    

def corte (imagen):
    ancho = np.arange(1,imagen.shape [1])
    corte = int(ancho.shape[0]/50)
    lineas = ancho[::corte]
  
    return lineas


def encontrar_limite(imagen, lineas):
    valores_y = []
    cambio_fase = []

    for linea in lineas:
        if not all(len(row) > linea for row in imagen):
            continue  # Salta la línea si alguna fila no tiene suficientes elementos

        valor_anterior = None
        for y, row in enumerate(imagen):
            pixel = row[linea]
            valor_y = pixel
            valores_y.append(valor_y)

            if valor_anterior is not None and valor_y != valor_anterior:
                cambio_fase.append((linea, valor_y, y))  # Guarda valor Y y su índice

            valor_anterior = valor_y

    return valores_y, cambio_fase

def limite_binario (imagen, linea):
    y = np.array(imagen[:,linea])
    bin= (imagen[:,linea])!= 0
    cero_gap = np.count_nonzero(bin)
    gap= (bin.size)-cero_gap
    return gap



def limitar_valores_vectorial(valores):
    valor_maximo = np.max(valores)
   
    limite = valor_maximo*0.97

    mascara = valores > limite
    
    valores_limitados = valores.copy()
    valores_limitados[mascara] = limite

    
    return valores_limitados

def promedio_vectores_vectorizado(imagen, linea):
    y = np.array(imagen[:, :, 2], dtype=np.float64)
    
    if linea -2 & linea +2 == 0: 
        columnas = [linea, linea + 1, linea - 1, linea + 2, linea - 2]
    
        mask = np.isin(np.arange(y.shape[1]), columnas)
    
        columnas_seleccionadas = y[:, mask]
    
    columna_y = columnas_seleccionadas.mean(axis=1)
    
    return columna_y






