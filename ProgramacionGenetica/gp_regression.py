from deap import base, creator, algorithms, tools, gp
import operator
import random
import numpy as np
from sklearn.metrics import mean_squared_error


def eval_func(ind, inputs, targets):
    func_eval = toolbox.compile(expr=ind)
    predictions = list(map(func_eval, inputs))
    return mean_squared_error(targets, predictions),

def func(x):
    return 3*x-5**2

x = [x for x in range(10)]
y = list(map(func, x))

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.sub, 2)
pset.renameArguments(ARG0='x')
pset.addEphemeralConstant('R', lambda: random.randint(0, 10))

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=5)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('mutate', gp.mutNodeReplacement, pset=pset)
toolbox.register('evaluate', eval_func, inputs=x, targets=y)
toolbox.register('compile', gp.compile, pset=pset)

toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("max", np.max)
stats.register("mean", np.mean)
stats.register("std", np.std)

hof = tools.HallOfFame(5)

pop = toolbox.population(n=20)

resultados, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=len(pop), lambda_=len(pop), cxpb=0.5, mutpb=0.1, 
                ngen=100, stats=stats, halloffame=hof)

print(resultados)

for ind in hof:
    print(ind)
    print(toolbox.evaluate(ind, inputs=x, targets=y))