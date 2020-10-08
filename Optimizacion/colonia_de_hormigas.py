#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:15 2020
Fernando Carrillo A01194204
Colonia de hormigas

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from functools import partial
import math


N_CITIES = 8
ITERACIONES = 12
N_HORMIGAS = 6
APLHA = 1 # importancia de las fermonas
BETA = 1 # importancia de la heuristica

# Crea las coordenadas de las ciudades
coordenadas = np.random.randint(1, 20, size=(N_CITIES, 2))
# Matriz de distancias euclidiana entre ciudades
dm = distance_matrix(coordenadas, coordenadas)

# Inicializacion de fermonas
tau = np.ones((N_CITIES, N_CITIES))
# Heuristica de distancia, 1 / distancia entre ciudades, para que prefieran distancias cortas
eta = np.asarray([1/dm[i,j] for i in range(N_CITIES) for j in range(N_CITIES)]).reshape(N_CITIES, N_CITIES)
eta = np.where(eta==np.Inf, 0, eta)  # reemplaza los inf por 0

# Genera solucion para una hormiga
def correr_hormiga(ciudad_actual, n_cities, tau, eta, alpha, beta):
    permitidos = list(range(n_cities))
    permitidos.remove(ciudad_actual)
    ruta = [ciudad_actual]
    
    while permitidos:
        # Calcula las probabilidades de pasar a una ciudad
        p = np.zeros(n_cities)
        suma_p = 0
        for sig in permitidos:
            a = min(ciudad_actual, sig)
            b = max(ciudad_actual, sig)
            p[sig] = np.power(tau[a, b], alpha) * np.power(eta[a, b], beta)
            suma_p += p[sig]
        p = p/suma_p  # Normaliza las probabilidades

        # Selecciona la siguiente ciudad
        r = np.random.rand()    
        suma = 0
        for ciud in permitidos:
            prob = p[ciud]
            suma += prob
            if r <= suma:
                ciudad_actual = ciud
                ruta.append(ciudad_actual)
                permitidos.remove(ciudad_actual)
                break
    
    return ruta


def actualizar_fermonas(soluciones, evaluaciones, tau, Q, ro):
    tau = ro * tau  #evaporacion
    for idx, solucion in enumerate(soluciones):
        conexiones = zip(solucion, solucion[1:]+[solucion[0]])
        for i, j in conexiones:
            minimo = min(i, j)
            maximo = max(i, j)
            tau[minimo, maximo] = tau[minimo, maximo] + Q/evaluaciones[idx]
    
    return tau

# Evalua todas las soluciones y calcula sus distancias
def evaluar_todo(soluciones, matriz_distancias):
    distancia_parcial = partial(distancia, dm=matriz_distancias)
    evaluaciones = list(map(distancia_parcial, soluciones))
    return evaluaciones

# Calcula la distancia de recorrer una ruta
def distancia(solucion, dm):
    conexiones = zip(solucion, solucion[1:]+[solucion[0]])
    L = 0
    for i,j in conexiones:
        L += dm[i, j]
    return L

# Despliega grafica con puntos
def plot_ants(soluciones, cities, mejor, ultimo_mejor):
    n_soluciones = len(soluciones) + 1

    cx = cities[:, 0]
    cy = cities[:, 1]
    fig, axes = plt.subplots(math.ceil(n_soluciones/2), 2, figsize=(10, 15))
    for i, ax in enumerate(axes.flatten()):
        if i == 0:
            ax.scatter(cx, cy, c='r')
            ax.set_title("Mejor solucion")
            ax.plot(cx[mejor + [mejor[0]]], cy[mejor + [mejor[0]]], c='r')
            ax.set_aspect('auto')
        elif i == 1:
            ax.scatter(cx, cy, c='r')
            ax.set_title("Ultimo mejor solucion")
            ax.plot(cx[ultimo_mejor + [ultimo_mejor[0]]], cy[ultimo_mejor + [ultimo_mejor[0]]], c='g')
            ax.set_aspect('auto')
        else:
            sol = soluciones[i-2]
            ax.scatter(cx, cy, c='r')
            ax.set_title("Hormiga" + str(i))
            ax.plot(cx[sol + [sol[0]]], cy[sol + [sol[0]]], c='b')
            ax.set_aspect('auto')

    plt.show()


mejor_evaluacion = np.Inf
mejor_solucion = None
for cont in range(ITERACIONES):
    # Calcula las ciudades iniciales para cada hormiga
    ciudades = np.random.randint(0, N_CITIES, N_HORMIGAS)
    
    # Cada hormiga recorre las ciudades
    func = partial(correr_hormiga, n_cities=N_CITIES, alpha=APLHA, beta=BETA, tau=tau, eta=eta)
    resultados = list(map(func, ciudades))

    # Se calcula la heuristica para cada hormiga
    evaluaciones = evaluar_todo(resultados, dm)

    mejor_evaluacion_iteracion = min(evaluaciones)
    mejor_solucion_iteracion = resultados[evaluaciones.index(mejor_evaluacion_iteracion)]

    # Actualiza las fermonas
    #print(tau, end='\n'*3)
    tau = actualizar_fermonas(resultados, evaluaciones, tau, Q=10, ro=0.9)

    if mejor_evaluacion > mejor_evaluacion_iteracion:
        mejor_evaluacion = mejor_evaluacion_iteracion
        mejor_solucion = mejor_solucion_iteracion

    plot_ants(resultados, coordenadas, mejor_solucion, mejor_solucion_iteracion)

