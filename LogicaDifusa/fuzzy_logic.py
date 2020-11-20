#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fernando Carrillo A01194204
Logica difusa
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# Antecedentes
vel_i = ctrl.Antecedent(np.linspace(1, 5), 'vel_i')
miu = ctrl.Antecedent(np.linspace(0.1, 0.3), 'miu')
# Consecuencia
x_f = ctrl.Consequent(np.linspace(0.17, 12.75), 'x_f')


# Funciones de membresia
# Velocidad inicial
vel_i['muy_lento'] = fuzz.trimf(vel_i.universe, [1, 1, 2])
vel_i['lento'] = fuzz.trimf(vel_i.universe, [1, 2, 3])
vel_i['normal'] = fuzz.trimf(vel_i.universe, [2, 3, 4])
vel_i['rapido'] = fuzz.trimf(vel_i.universe, [3, 4, 5])
vel_i['muy_rapido'] = fuzz.trimf(vel_i.universe, [4, 5, 5])
#vel_i.view()
#plt.show()

# Coeficiente de friccion
miu['poca'] = fuzz.trimf(miu.universe, [0.1, 0.1, 0.2])
miu['normal'] = fuzz.trimf(miu.universe, [0.1, 0.2, 0.3])
miu['mucha'] = fuzz.trimf(miu.universe, [0.2, 0.3, 0.3])
#miu.view()
#plt.show()

# Posicion final
x_f.automf(5, names=['cerca', 'medio_cerca', 'normal', 'medio_lejos', 'lejos'])
#x_f.view()
#plt.show()


# Reglas
rule1 = ctrl.Rule(vel_i['muy_lento'], x_f['cerca'], 'Muy poca vel_i')
rule2 = ctrl.Rule(vel_i['lento'], x_f['medio_cerca'], 'Poca vel_i')
rule3 = ctrl.Rule(vel_i['normal'], x_f['normal'], 'Normal vel_i')
rule4 = ctrl.Rule(vel_i['rapido'], x_f['medio_lejos'], 'Mucha vel_i')
rule5 = ctrl.Rule(vel_i['muy_rapido'], x_f['lejos'], 'Muchisima vel_i')

rule6 = ctrl.Rule(vel_i['muy_lento'] & miu['mucha'], x_f['cerca'], 'Muy poca vel_i con mucha ficcion')
rule7 = ctrl.Rule(vel_i['normal'] & miu['normal'], x_f['medio_cerca'], 'Normal vel_i con algo de ficcion')
rule8 = ctrl.Rule(vel_i['muy_rapido'] & miu['poca'], x_f['lejos'], 'Mucha vel_i con poca ficcion')
rule9 = ctrl.Rule(vel_i['muy_rapido'] & miu['mucha'], x_f['normal'], 'Mucha vel_i con mucha ficcion')
rule10 = ctrl.Rule(vel_i['rapido'] & miu['mucha'], x_f['medio_cerca'], 'vel_i con mucha ficcion')

# Simulacion
pos_final_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5,
    rule6, rule7, rule8, rule9, rule10])
pos_final = ctrl.ControlSystemSimulation(pos_final_ctrl)

"""
pos_final.input['vel_i'] = 1.0
pos_final.input['miu'] = 0.3

pos_final.compute()

print(pos_final.output['x_f'])
x_f.view(sim=pos_final)
plt.show()
"""

def plotXf(miu, simulation):
    x = np.linspace(1,5,100)  # velocidades iniciales
    y = (x*x) / (2*miu*9.8)  # posiciones finales
    
    y_pred = []
    for i in x:
        simulation.input['vel_i'] = i
        simulation.input['miu'] = miu
        simulation.compute()
        y_pred.append(simulation.output['x_f'])

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel("Velocidad inicial")
    plt.ylabel("Posicion final")
    plt.title("Miu: {}".format(miu))

    # plot the function
    plt.plot(x, y, 'r', label="Original")
    plt.plot(x, y_pred, 'b', label="Prediccion")

    # show the plot
    plt.show()

plotXf(0.1, pos_final)
plotXf(0.2, pos_final)
plotXf(0.3, pos_final)