#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fernando Carrillo A01194204
Bin packing 2D

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw

### PROBLEM VALUES
containerSize = (10, 10)
objects = [(5, 3), (3, 4), (1, 1), (2, 2), (2, 2), (4, 5), (3, 4), (4, 2), (5, 3), (2, 6),
        (4, 3), (2, 4), (2, 1), (2, 3), (6, 2), (3, 3), (4, 4), (6, 6), (6, 7), (3, 5),
        (3, 3), (2, 2), (3, 5), (4, 7), (8, 7), (5, 2), (3, 1), (1, 4), (2, 5), (5, 6),
        (3, 3), (4, 4), (2, 1), (3, 2), (4, 3), (1, 1), (3, 2), (5, 7), (5, 6), (5, 2),
        (3, 3), (4, 3), (2, 3), (1, 2), (6, 3), (2, 2), (3, 2), (1, 2), (5, 3), (2, 2),
        (4, 5), (3, 4), (4, 2), (5, 3), (2, 6), (4, 3), (2, 4), (2, 1), (2, 3), (6, 2),
        (3, 3), (4, 4), (6, 6), (6, 7), (3, 5), (3, 3), (2, 2), (3, 5), (4, 7), (4, 4),
        (5, 2), (3, 1), (1, 4), (2, 5), (5, 6), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3),
        (1, 1), (3, 2), (5, 7), (5, 6), (5, 2), (3, 3), (4, 3), (2, 3), (1, 2), (6, 3),
        (2, 2), (3, 2), (1, 2), (5, 3), (2, 2), (5, 3), (3, 4), (1, 1), (2, 2), (2, 2)]

#objects = [(5, 3), (3, 4), (1, 1), (2, 2), (2, 2), (10, 8)]

### AUX FUNCTIONS
# Fill in containers with objets starting by lower left corner
# 0 - width, 1 - height
def FillContainers(container, objects):
    containersUsed = 1
    widthLeft = containerSize[0]
    heightLeft = containerSize[1]
    rowHeight = 0

    spaceWasted = 0
    objsInRow = []

    for obj in objects:
        # Adding object to current row
        if obj[0] <= widthLeft and obj[1] <= heightLeft:
            widthLeft -= obj[0]
            rowHeight = max(rowHeight, obj[1])
            objsInRow.append(obj)
        # Adding new row
        elif obj[0] > widthLeft and obj[1] <= (heightLeft - rowHeight):
            spaceWasted += CalcSpaceWasted(containerSize[0], rowHeight, objsInRow)
            objsInRow = [obj]

            widthLeft = containerSize[0] - obj[0]
            heightLeft -= rowHeight
            rowHeight = obj[1]
        # Adding new container
        else:
            spaceWasted += CalcSpaceWasted(containerSize[0], rowHeight, objsInRow)
            spaceWasted += (heightLeft-rowHeight) * containerSize[0]  # space left in container
            objsInRow = [obj]

            widthLeft = containerSize[0] - obj[0]
            heightLeft = containerSize[1]
            rowHeight = obj[1]
            containersUsed += 1
    spaceWasted += CalcSpaceWasted(containerSize[0], rowHeight, objsInRow)  # last obj in [objects]
    
    return (spaceWasted, containersUsed)

# Calculates the space wasted in a row
def CalcSpaceWasted(width, height, objects):
    spaceWasted = 0
    widthLeft = width
    for obj in objects:
        spaceWasted += obj[0] * (height - obj[1])
        widthLeft -= obj[0]
    spaceWasted += widthLeft * height
    return spaceWasted

# Evaluation for solution that evaluates the order of the objects
def Evaluate(container, objects):
    # 0 - spaceWasted, 1 - containersUsed
    return FillContainers(container, objects)[0]

# Generates a new list with objects arranged in a different order
def GenerateNeighbour(temp, objects):
    newOrder = list(objects)
    for i in range(len(objects) - 1):
        p = np.random.rand()
        if p < 0.5 + temp / 100:
            newOrder[i], newOrder[i+1] = newOrder[i+1], newOrder[i]
    return newOrder

# Generate Markov chain for Lk iterations
def MarkovChain(temp, Lk, objects):
    global containerSize
    accepted = 0
    for _ in range(Lk):
        newOrder = GenerateNeighbour(temp, objects)
        currentEval = Evaluate(containerSize, objects)
        newEval = Evaluate(containerSize, newOrder)
        if newEval < currentEval:
            objects = newOrder
            accepted += 1
        else:
            p = np.random.rand()
            if p < math.exp(-(newEval - currentEval) / temp):
                objects = newOrder
                accepted += 1
    return accepted / Lk

# Initialize temperature
def InitTemp(temp, objects):
    beta = 1.5  # constant b > 1 for temp heating rate
    Lk = 100  #

    r_min = 0.7  # Minimum acceptance percentage
    r_a = 0
    while r_a < r_min:
        r_a = MarkovChain(temp, Lk, objects)
        temp = beta * temp
    return temp

# Graph better solution
def ShowLineGraph(x, y):
    data = np.array(list(zip(x, y)))
    x_val, y_val = data.T
    plt.scatter(x_val, y_val)

    # Draw lines
    for i in range(len(data)-1):
        a = data[i]
        b = data[i+1]
        x_values = [a[0], b[0]]
        y_values = [a[1], b[1]]
        plt.plot(x_values, y_values)
    plt.xlabel("Iteracion")
    plt.ylabel("Espacio desperdiciado")
    plt.show()

# Shows 500 x 500 grid with objects inside containers
# Max. 25 containers shown
def ShowContainers(containerSize, objects, id):
    # Resize for image
    containerSize = tuple([10*x for x in containerSize])
    objects = tuple([(10*x[0], 10*x[1]) for x in objects])

    im = Image.new('RGB', (501, 501), (128, 128, 128))
    draw = ImageDraw.Draw(im)
    draw.rectangle((containerSize[0], containerSize[1], 0, 0), fill=(0, 0, 0), outline=(255, 255, 255))

    widthLeft = containerSize[0]
    heightLeft = containerSize[1]
    rowHeight = 0

    # Current container position
    containerX = 0
    containerY = 0

    # Current object position (0, 0) top left
    currX = 0
    currY = 0

    for obj in objects:
        # Adding object to current row
        if obj[0] <= widthLeft and obj[1] <= heightLeft:
            # Draw object in same row
            draw.rectangle((currX, currY, currX+obj[0], currY+obj[1]), fill=(255, 255, 255), outline=(255, 0, 0))
            currX += obj[0]

            widthLeft -= obj[0]
            rowHeight = max(rowHeight, obj[1])
        # Adding new row
        elif obj[0] > widthLeft and obj[1] <= (heightLeft - rowHeight):
            # Draw object in new row
            currY += rowHeight
            currX = containerX
            draw.rectangle((currX, currY, currX+obj[0], currY+obj[1]), fill=(255, 255, 255), outline=(255, 0, 0))
            currX += obj[0]

            widthLeft = containerSize[0] - obj[0]
            heightLeft -= rowHeight
            rowHeight = obj[1]
        # Adding new container
        else:
            # Draw new container
            containerX += containerSize[0]
            if containerX >= 500:
                containerX = 0
                containerY += containerSize[1]
            draw.rectangle((containerX, containerY, containerX+containerSize[0], containerY+containerSize[1]),
                            fill=(0, 0, 0), outline=(255, 255, 255))
            # Draw new object
            currY = containerY
            currX = containerX
            draw.rectangle((currX, currY, currX+obj[0], currY+obj[1]), fill=(255, 255, 255), outline=(255, 0, 0))
            currX += obj[0]

            widthLeft = containerSize[0] - obj[0]
            heightLeft = containerSize[1]
            rowHeight = obj[1]

    im.save('./containers/container' + str(id) + '.jpg', quality=95)

"""       
#objects = sorted(objects, key=lambda x:x[1])
ShowContainers(containerSize, objects, 1)
spaceWasted, numContainers = FillContainers(containerSize, objects)
print("NO. OF CONTAINERS: ", numContainers)
print("SPACE WASTED: ", spaceWasted)
"""

### SIMULATED ANNEALING
Lk = 100
alpha = 0.8 # constant 0 < a < 1 for temp cooling rate
temp = 0.1  # initial temperature
temp = InitTemp(temp, objects)
print("STARTING TEMP.: ", temp)
currSolution = objects
ShowContainers(containerSize, currSolution, 1)

# Stop conditions
minTemp = 0.01
maxSameEval = 20  # max number of iterations with same evaluation
sameEvalCounter = 0
lastEval = 0

# Graph variables
itertations = 0
evaluations = []

# Main loop
while (temp > minTemp) and (sameEvalCounter < maxSameEval):
    for _ in range(Lk):
        newSolution = GenerateNeighbour(temp, currSolution)
        currEval = Evaluate(containerSize, currSolution)
        newEval = Evaluate(containerSize, newSolution)
        if newEval <= currEval:
            currSolution = newSolution
            temp = newEval / currEval * temp
        else:
            p = np.random.rand()
            if p < math.exp(-(newEval - currEval) / temp):
                currSolution = newSolution
                temp = newEval / currEval * temp
        
    temp = alpha * temp

    # Compare with previous evaluation
    currEval = Evaluate(containerSize, currSolution)
    if currEval == lastEval:
        sameEvalCounter += 1
    else:
        sameEvalCounter = 0
    lastEval = currEval

    # Save data for graph
    itertations += 1
    evaluations.append(currEval)

# Results
ShowContainers(containerSize, currSolution, 2)
spaceWasted, numContainers = FillContainers(containerSize, currSolution)
print("NO. OF CONTAINERS: ", numContainers)
print("SPACE WASTED: ", spaceWasted)
print("OPTIMAL ORDER: ", currSolution)
x = [i for i in range(len(evaluations))]
ShowLineGraph(x, evaluations)
