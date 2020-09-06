#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fernando Carrillo A01194204
Tarea 2: Recocido simulado
Programar el algoritmo de recocido simulado y resolver el problema del vendedor viajero
El algoritmo puede tener cualquier criterio de terminacion (tiempo, iteraciones,
temperatura cercana a cero, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import math

n_ciudades = 12
# Crea las coordenadas de las ciudades aleatoriamente
#coordenadas = np.random.randint(1, 50, size=(n_ciudades, 2))
# Coordenadas fijas para probar ajustes de parametros
coordenadas = [[1.2, 30], [42, 33], [22, 21], [25, 32],
                [11.5, 23], [24, 49], [5, 9], [19, 8],
                [20, 30], [32, 23], [13, 7], [18, 28]]
# Matriz de distancias euclidiana entre ciudades
distancias = distance_matrix(coordenadas, coordenadas)

temp = 0.1  # temperatura inicial
alpha = 0.7  # constante entre 0 < a < 1 para enfriamiento
beta = 1.5  # constante b > 1 para calentamiento 
ruta_actual = list(range(n_ciudades))  # solucion inicial
np.random.shuffle(ruta_actual)
L = 100


### FUNCIONES AUXILIARES ###

# Calcula la distancia total de recorrer la ruta
def Evalua(ruta):
    suma = 0
    conexiones = list(zip(ruta, ruta[1:]+[ruta[0]]))
    for a, b in conexiones:
        suma += distancias[a, b]
    return np.round(suma, 5)

# Genera un vecino de la ruta actual
def GeneraVecino(ruta, temp):
    ruta_nueva = list(ruta)
    for i in range(len(ruta_nueva) - 1):
        p = np.random.rand()
        if p < 0.4 + temp / 100:
            ruta_nueva[i], ruta_nueva[i+1] = ruta_nueva[i+1], ruta_nueva[i]
    return ruta_nueva

# Despliega grafica la ruta para recorrer ciudades
def GraficaRuta(coordenadas, n_ciudades, ruta):
    data = np.array(coordenadas)
    x, y = data.T

    plt.scatter(x, y)
    for i in range(n_ciudades):
        plt.annotate(i, (x[i], y[i]))

    conexiones = list(zip(ruta, ruta[1:]+[ruta[0]]))
    for conexion in conexiones:
        a = coordenadas[conexion[0]]
        b = coordenadas[conexion[1]]
        x_values = [a[0], b[0]]
        y_values = [a[1], b[1]]
        plt.plot(x_values, y_values)

    plt.show()

# Despliega la curva de mejora del algoritmo
def GraficaMejora(x, y):
    data = np.array(list(zip(x, y)))
    x_val, y_val = data.T
    plt.scatter(x_val, y_val)

    for i in range(len(data)-1):
        a = data[i]
        b = data[i+1]
        x_values = [a[0], b[0]]
        y_values = [a[1], b[1]]
        plt.plot(x_values, y_values)

    plt.show()

# Calcula una cadena de Markov con longitud L, y regresa el porcentaje de cambios de un
# estado a otro dada una ruta inicial y una temperatura inicial
def CadenaMarkov(ruta, temp, L):
    aceptados = 0
    for _ in range(L):
        ruta_nueva = GeneraVecino(ruta, temp)
        eval_actual = Evalua(ruta)
        eval_nueva = Evalua(ruta_nueva)
        if eval_nueva < eval_actual:
            ruta = ruta_nueva
            aceptados += 1
        else:
            p = np.random.rand()
            if p < math.exp(-(eval_nueva - eval_actual) / temp):
                ruta = ruta_nueva
                aceptados += 1
    return aceptados / L

# Incrementa la temperatura inicial hasta que la probabilidad de cambio de un 
# estado a otro sea mayor al valor minimo especificado (0.6).
def InicializaTemp(ruta, temp, L):
    r_min = 0.7  # porcentaje de aceptacion minimo
    r_a = 0
    while r_a < r_min:
        r_a = CadenaMarkov(ruta, temp, L)
        temp = beta * temp  # incrementa temperatura
    return temp

### FUNCIONES AUXILIARES ###


### SOLUCION ###

temp = InicializaTemp(ruta_actual, temp, L)
print("Temp. inicial: ", temp)

# Variables de condiciones de paro
iter_sin_mejora = 0  # condicion para parar. No hay mejora si la evaluacion anterior y la actual son las mismas
max_iter_sin_mejora = 15
eval_anterior = 0
temp_minima = 0.01

# Variables para graficas
iteracion = 0
iteraciones = []
evaluaciones = []

# Busca optimo mientras la temperatura siga arriba de la minima, y no se hayan repetido 
# las evaluaciones mas de N veces
while (temp > temp_minima) and (iter_sin_mejora < max_iter_sin_mejora):
    for _ in range(L):
        ruta_nueva = GeneraVecino(ruta_actual, temp)
        eval_actual = Evalua(ruta_actual)
        eval_nueva = Evalua(ruta_nueva)
        if eval_nueva < eval_actual:
            ruta_actual = ruta_nueva
            temp = eval_nueva / eval_actual * temp
        else:
            p = np.random.rand()
            if p < math.exp(-(eval_nueva - eval_actual) / temp):
                ruta_actual = ruta_nueva
                temp = eval_nueva / eval_actual * temp
                
    temp = alpha * temp  # enfria temperatura
    
    # Compara la evaluacion anterior y la actual para ver si hay diferencia
    eval_actual = Evalua(ruta_actual)
    if eval_anterior == eval_actual:
        iter_sin_mejora += 1
    else:
        iter_sin_mejora = 0
    eval_anterior = eval_actual

    # Actualiza datos para la grafica de mejora
    iteraciones.append(iteracion)
    iteracion += 1
    evaluaciones.append(eval_actual)
        
print("Ruta optima: ", ruta_actual, " - Distancia: ", Evalua(ruta_actual))
GraficaRuta(coordenadas, n_ciudades, ruta_actual)
GraficaMejora(iteraciones, evaluaciones)

### SOLUCION ###