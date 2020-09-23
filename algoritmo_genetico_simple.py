from deap import base, creator, tools
from deap import algorithms
import random
import numpy as np
import pandas as pd
import multiprocessing

def func_eval(ind):
    numero_binario = "".join(map(str, ind))
    return int(numero_binario, base=2),

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("select", tools.selRoulette)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("evaluate", func_eval)

toolbox.register("attribute", random.randint, a=0, b=1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("max", np.max)
stats.register("median", np.median)
stats.register("mean", np.mean)
stats.register("std", np.std)

pop = toolbox.population(n=10)

fitness = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitness):
    ind.fitness.values = fit

CX = 1.0
mut = 0.5

log = tools.Logbook()

for gen in range(0, 10):
    records = stats.compile(pop)

    offsprings = toolbox.select(pop, len(pop))
    
    for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
        if random.random() < CX:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    for child in offsprings:
        if random.random() < mut:
            toolbox.mutate(child)
            del child.fitness.values
    
    invalid_fitness = [ind for ind in offsprings if not ind.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_fitness)

    log.record(gen=gen, evals=len(invalid_fitness), **records)
    print(log.stream)

    for ind, fit in zip(invalid_fitness, fitness):
        ind.fitness.values = fit

    pop[:] = offsprings

print(pop)