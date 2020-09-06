#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:15 2020
Fernando Carrillo A01194204
Enjambre de particulas
Encontras los coeficientes de la funcion a1 * x^2 - a2^(e-a3*x)/2 para que se cumplan las
siguientes evaluaciones
[-3.271085, 2.994633, 14.999999, 34.999999, 62.999999, 98.999999, 143, 195, 255, 323]
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Evaluaciones para f(x), 0 <= x <= 9
evaluaciones = [-3.271085, 2.994633, 14.999999, 34.999999,
                62.999999, 98.999999, 143, 195, 255, 323]

num_particulas = 20
num_coeficientes = 3

### FUNCIONES AUXILIARES ###

# Calcula la nueva velocidad de una particula
def NuevaVelocidad(vel, pos, alpha, beta, mejor_local, mejor_global):
    global num_coeficientes
    inercia = np.random.uniform(0.5, 0.9, num_coeficientes)

    e1 = np.random.rand(num_coeficientes)  # relevancia mejor global
    compA = alpha * e1 * (mejor_global - pos)

    e2 = np.random.rand(num_coeficientes)  # relevancia mejor local
    compB = beta * e2 * (mejor_local - pos)

    return (inercia * vel) + compA + compB
    
# Calcula la nueva posicion de la particula
def NuevaPosicion(pos, vel):
    return pos + vel

# Funcion a1 * x^2 - a2^(e-a3*x)/2 evaluada con una particula para valores de x -> 0 a 9 
def Evalua(particula):
    global evaluaciones
    a1, a2, a3 = particula
    sq_error_sum = 0

    for i in range(len(evaluaciones)):
        compA2 = pow(a2, math.exp(-a3 * i) / 2)
        if math.isnan(compA2):
            compA2 = 0

        eval = a1 * pow(i, 2) - compA2
        sq_error_sum += pow(evaluaciones[i] - eval, 2)
    
    return sq_error_sum / len(evaluaciones)

# Calcula la particula con la mejor evaluacion de una lista de particulas
def Mejor(particulas):
    min = math.inf
    min_idx = 0
    for idx, par in enumerate(particulas):
        eval = Evalua(par)
        if eval < min:
            min = eval
            min_idx = idx
    return particulas[min_idx]

# Grafica mejora
def Grafica(data):
    x = [i for i in range(len(data))]
    plt.scatter(x, data)
    plt.show()

### FUNCIONES AUXILIARES ###

### SOLUCION ###

particulas = np.random.uniform(-10, 10, size=(num_particulas, num_coeficientes))
velocidades = np.zeros((num_particulas, num_coeficientes))
mejores_locales = particulas
alpha = 2
beta = 2

mejores = []  # para graficar

mejor_global = Mejor(mejores_locales)
num_iteraciones = 100
for _ in range(num_iteraciones):
    for idx, particula in enumerate(particulas):
        n_vel = NuevaVelocidad(velocidades[idx], particula, alpha, beta, mejores_locales[idx], mejor_global)
        particula = NuevaPosicion(particula, n_vel)
        n_eval = Evalua(particula)

        # Actualiza mejor local
        if n_eval < Evalua(mejores_locales[idx]):
            mejores_locales[idx] = particula

    mejor_global = Mejor(mejores_locales)
    mejores.append(Evalua(mejor_global))

print(mejor_global, Evalua(mejor_global))
Grafica(mejores)

### SOLUCION ###