#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fernando Carrillo A01194204
Algortimos geneticos
Resolver el problema de la mochila con los siguientes datos. 
El peso máximo es 165 kg. Los parámetros para el algoritmo quedan a discreción del alumno.
"""

from deap import base, creator, tools
from deap import algorithms
import random
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Datos [w, p]
datos = [[23, 92], [31, 57], [29, 49], [44, 68], [53, 60],
        [38, 43], [63, 67], [85, 847], [89, 87], [82, 73]]

### HELPER FUNCTIONS ###
# ind is an array containing bits that indicate if the objet datos[i] should
# be added to the backpack. The function evaluates the profit of the included
# items and calculated the punishment if the weight limit is exceeded.
def func_eval(ind):
    w_sum = 0
    p_sum = 0
    for idx, bit in enumerate(ind):
        if (bit):
            w_sum += datos[idx][0]
            p_sum += datos[idx][1]
    # Peso maximo 165
    if w_sum > 165:
        p_sum -= punishment(w_sum)

    return p_sum,

# Prints Weight and Profit values for individual
def decode(ind):
    w_sum = 0
    p_sum = 0
    for idx, bit in enumerate(ind):
        if (bit):
            w_sum += datos[idx][0]
            p_sum += datos[idx][1]
    # Peso maximo 165
    if w_sum > 165:
        p_sum -= punishment(w_sum)

    print("Weight: ", w_sum, " Profit: ", p_sum)

# Calculate punishment value for exceeding weight
def punishment(weight):
    return (weight - 165) * 13

# Display a dataframe
def display(df, title, color='r'):
    df = df.reset_index(drop=True)

    df_means = df.groupby(['algorithm', 'gen']).agg({'max': {'mean', 'std'}})

    X = df['gen'].unique()
    means = df_means['max']['mean'].values
    deviations = df_means['max']['std'].values
    plt.plot(X, means, color=color)
    plt.plot(X, means - deviations, color=color, linestyle='dashed')
    plt.plot(X, means + deviations, color=color, linestyle='dashed')
    plt.xlabel("Generacion")
    plt.ylabel("Valor")
    plt.title(title)

    plt.show()

# Display all 3 dataframes
def display_all(df):
    df = df.reset_index(drop=True)
    
    X = df['gen'].unique()

    df_simple = df.loc[df['algorithm'] == 'eaSimple']
    df_means = df_simple.groupby(['algorithm', 'gen']).agg({'max': {'mean', 'std'}})
    means = df_means['max']['mean'].values
    deviations = df_means['max']['std'].values
    plt.plot(X, means, color='r')
    red_patch = mpatches.Patch(color='red', label='Simple')

    df_muPLus = df.loc[df['algorithm'] == 'eaMuPlusLambda']
    df_means = df_muPLus.groupby(['algorithm', 'gen']).agg({'max': {'mean', 'std'}})
    means = df_means['max']['mean'].values
    deviations = df_means['max']['std'].values
    plt.plot(X, means, color='g')
    green_patch = mpatches.Patch(color='green', label='MuPlusLambda')

    df_muComma = df.loc[df['algorithm'] == 'eaMuCommaLambda']
    df_means = df_muComma.groupby(['algorithm', 'gen']).agg({'max': {'mean', 'std'}})
    means = df_means['max']['mean'].values
    deviations = df_means['max']['std'].values
    plt.plot(X, means, color='b')
    blue_patch = mpatches.Patch(color='blue', label='MuCommaLambda')
    
    plt.xlabel("Generacion")
    plt.ylabel("Valor")
    plt.legend(handles=[red_patch, green_patch, blue_patch])

    plt.show()
    

### SOLUTION ###
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # max p
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("select", tools.selRoulette)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("evaluate", func_eval)

toolbox.register("attribute", random.randint, a=0, b=1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len(datos))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=10)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("max", np.max)
stats.register("mean", np.mean)
stats.register("median", np.median)
stats.register("std", np.std)

num_iter = 10
num_gen = 100
df = pd.DataFrame()

# Simple
hof_simple = tools.HallOfFame(1)
for i in range(num_iter):
    res_Simple, log = algorithms.eaSimple(pop, toolbox, 1, 0.5, num_gen, stats=stats, verbose=False, halloffame=hof_simple)
    df2 = pd.DataFrame(log)
    df2['batch'] = i
    df2['algorithm'] = 'eaSimple'
    for idx, row in df2.iterrows():
        if df2.iloc[idx]['max'] < df2.iloc[idx-1]['max']:
            df2.loc[idx, 'max'] = df2.iloc[idx-1]['max']
    df = df.append(df2)


# MuPlusLambda
hof_muPlus = tools.HallOfFame(1)
for i in range(num_iter):
    res_MuPlusLambda, log = algorithms.eaMuPlusLambda(pop, toolbox, 10, 20, 0.7, 0.3, num_gen, stats=stats, verbose=False, halloffame=hof_muPlus)
    df2 = pd.DataFrame(log)
    df2['batch'] = i
    df2['algorithm'] = 'eaMuPlusLambda'
    for idx, row in df2.iterrows():
        if df2.iloc[idx]['max'] < df2.iloc[idx-1]['max']:
            df2.loc[idx, 'max'] = df2.iloc[idx-1]['max']
    df = df.append(df2)


# MuCommaLambda
hof_muComma = tools.HallOfFame(1)
for i in range(num_iter):
    res_MuCommaLambda, log = algorithms.eaMuCommaLambda(pop, toolbox, 10, 20, 0.7, 0.3, num_gen, stats=stats, verbose=False, halloffame=hof_muComma)
    df2 = pd.DataFrame(log)
    df2['batch'] = i
    df2['algorithm'] = 'eaMuCommaLambda'
    for idx, row in df2.iterrows():
        if df2.iloc[idx]['max'] < df2.iloc[idx-1]['max']:
            df2.loc[idx, 'max'] = df2.iloc[idx-1]['max']
    df = df.append(df2)


### RESULTS ###
df_simple = df.loc[df['algorithm'] == 'eaSimple']
display(df_simple, 'eaSimple', color='r')
print("Best for Simple: ", hof_simple)
decode(hof_simple[0])

df_muPlus = df.loc[df['algorithm'] == 'eaMuPlusLambda']
display(df_muPlus, 'eaMuPlusLambda', color='g')
print("Best for MuPlusLambda: ", hof_muPlus)
decode(hof_muPlus[0])

df_muComma = df.loc[df['algorithm'] == 'eaMuCommaLambda']
display(df_muComma, 'eaMuCommaLambda', color='b')
print("Best for MuCommaLambda: ", hof_muComma)
decode(hof_muComma[0])

display_all(df)