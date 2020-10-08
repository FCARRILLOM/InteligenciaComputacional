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

pop = toolbox.population(n=10)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("max", np.max)
stats.register("median", np.median)
stats.register("mean", np.mean)
stats.register("std", np.std)

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop, log = algorithms.eaSimple(pop, toolbox, 1.0, 0.5, 10)
    df = pd.DataFrame(log)

    print(df)
