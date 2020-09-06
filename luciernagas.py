#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:15 2020
Fernando Carrillo A01194204
Algoritmo de luciernagas
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Evaluaciones para f(x), 0 <= x <= 9
evaluaciones = [-3.271085, 2.994633, 14.999999, 34.999999,
                62.999999, 98.999999, 143, 195, 255, 323]

num_luciernagas = 15
num_coeficientes = 3

### FUNCIONES AUXILIARES ###

# Calcula la distancia euclidiana entre dos vectores
def DistanciaEntre(v1, v2):
    temp = v1 - v2
    temp = np.power(temp, 2)
    return np.sqrt(np.sum(temp))
    
# Calcula la nueva posicion de la luciernaga l1 respecto a otra l2
def NuevaPosicion(l1, l2, beta, gamma):
    alpha = np.random.rand()  # importancia del azar [0, 1]
    e = np.random.uniform(-0.5, 0.5, len(l1))
    intensidad = (beta * math.exp(-gamma * pow(DistanciaEntre(l1, l2), 2)))

    return l1 + (intensidad * (l2 - l1)) + (alpha * e)

# Funcion de intensidad evaluada con una luciernaga para valores de x -> 0 a 9 
def Intensidad(luciernaga):
    global evaluaciones
    a1, a2, a3 = luciernaga
    sq_error_sum = 0

    for i in range(len(evaluaciones)):
        parte2 = pow(a2, math.exp(-a3 * i) / 2)
        if math.isnan(parte2): parte2 = 0
        eval = a1 * pow(i, 2) - parte2
        sq_error_sum += pow(evaluaciones[i] - eval, 2)
    
    return 1 / (sq_error_sum / len(evaluaciones))

# Calcula la luciernaga con la mejor evaluacion de una lista de luciernagas
def Mejor(luciernagas):
    max = -math.inf
    max_idx = 0
    for idx, luc in enumerate(luciernagas):
        intensidad = Intensidad(luc)
        if intensidad > max:
            max = intensidad
            max_idx = idx
    return luciernagas[max_idx]

# Grafica mejora
def Grafica(data, iteraciones):
    x = [i for i in range(iteraciones)]
    plt.scatter(x, data)
    plt.show()

### FUNCIONES AUXILIARES ###

### SOLUCION ###

luciernagas = np.random.uniform(-10, 10, size=(num_luciernagas, num_coeficientes))
beta = 1
gamma = 0.6

intensidades = [Intensidad(luciernaga) for luciernaga in luciernagas]
mejores = []

mejor_global = Mejor(luciernagas)
num_iteraciones = 300
for _ in range(num_iteraciones):
    for i, l1 in enumerate(luciernagas):
        for j, l2 in enumerate(luciernagas):
            if i != j and intensidades[i] < intensidades[j]:
                luciernagas[i] = NuevaPosicion(l1, l2, beta, gamma) 
                intensidades[i] = Intensidad(l1)  # actualiza intensidad con nueva posicion

    mejor_global = Mejor(luciernagas)
    mejores.append(1 / Intensidad(mejor_global))

print(mejor_global, 1 / Intensidad(mejor_global))
Grafica(mejores, num_iteraciones)

### SOLUCION ###