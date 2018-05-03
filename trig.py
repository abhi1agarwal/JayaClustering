import pandas as pd

import matplotlib.pyplot as plt
import constants
from matplotlib import style
from fitness import fmeasure

style.use('ggplot')
import numpy as np
from readdata import get_file_contents
from fitness import find_fitness
from fitness import get_final_fitness_trig
from fitness import get_pop_distri
from fitness import get_pop_distri_label
from readdata import output
import random
from plotting import plot2D
from plotting import plot3D


def apply_jaya(dataset, K):
    x_dataset = dataset.drop(['class'], axis=1)
    y_dataset = dataset['class']
    global y_dataset_glob
    y_dataset_glob=y_dataset
    #print(x_dataset)
    if constants.URL.__contains__('glass') or constants.URL.__contains__('parkinsons'):
        x_dataset = x_dataset.drop(['A'], axis=1)
    max_x = np.max(np.array(x_dataset), axis=0)
    min_x = np.min(np.array(x_dataset), axis=0)

    print(max_x - min_x)
    # print(np.max(np.array(x_dataset),axis=0))
    # print(np.min(np.array(x_dataset), axis=0))
    features = np.array(x_dataset).shape[1]
    # population constains sets of clusters that are probable candidate solutions
    population = []
    xx = x_dataset
    xx = np.array(xx)
    #print(random.randint(0,xx.size/features))
    #print(xx[5])

    for i in range(0, constants.POP_COUNT):
        cur = []
        clusters = K
        for j in range(0, clusters):
            clust = []
            if constants.URL.__contains__('segmentation') or constants.URL.__contains__('glass')   :
                clust=xx[random.randint(1,xx.size/features-1)]+random.uniform(-1,1)
            else:
                for k in range(0, features):
                    newval = min_x[k] + (random.random() * (max_x[k] - min_x[k]))
                    # print(newval)
                    clust.append(newval)
            #print(clust)
            cur.append(clust)

        # one cluster has been made
        # one cluster list has been made that is one candidate solution
        population.append(np.array(cur))
    # print(population)

    # population[0] = [[5.9016129 ,2.7483871, 4.39354839,1.43387097],[5.006    ,  3.418    ,  1.464 ,0.244],[6.85     ,  3.07368421 ,5.74210526, 2.07105263]]
    # if (constants.DEBUG):
    # 	print(population)
    x_dataset = np.array(x_dataset)

    fitness = []
    final_fitness = np.zeros(constants.POP_COUNT)
    for iterations in range(0, constants.ITERATIONS):
        print("iteration : ", iterations, " started ...")
        fitness = find_fitness(x_dataset, population, K)
        final_fitness = get_final_fitness_trig(fitness)
        # print("iteration number ::",iterations,"\n","fitness :: ",fitness)
        # print(population)
        # print("\n\n")
        rat = np.zeros(features)
        for i in range(0, features):
            rat[i] = random.random()
        rat = np.array([list(rat)] * K)

        rat1 = np.zeros(features)
        for i in range(0, features):
            rat1[i] = random.random()
        rat1 = np.array([list(rat1)] * K)
        best = 0
        best_index = 0
        worst = 1e9
        worst_index = 0
        for i in range(0, len(population)):
            if best < final_fitness[i]:
                best = final_fitness[i]
                best_index = i
            if worst > final_fitness[i]:
                worst = final_fitness[i]
                worst_index = i

        lis = []
        for i in range(0, len(population)):
            lis.append((final_fitness[i], i))

        ############

        '''window = int(len(population) * .05)  # type: int
        
        lis.sort()
        #  print(lis)
        for i in range(0,window):
            newone = (int)(window * random.random())
            newind = population[lis[len(population) - newone-1][1]] +random.random()
            fit = find_fitness(x_dataset, [newind], K)
            ff = get_final_fitness_trig(fit)
            if final_fitness[lis[i][1]] < ff :
                population[lis[i][1]] = newind
        #fitness = find_fitness(x_dataset, population, K)
        #final_fitness = get_final_fitness_trig(fitness)
        #liss = []
       # for i in range(0, len(population)):
        #    liss.append((final_fitness[i], i))
        #liss.sort()
        #print(liss) '''
        #################

        lis.sort(reverse=True)
        window = int(len(population) * .1)  # type: int
        newone = (int)(window * random.random())
        best_index = lis[newone][1]
        newone = (int)(window * random.random())
        worst_index = lis[len(population) - 1 - newone][1]


        #print(best_index,final_fitness[best_index],worst_index,final_fitness[worst_index])
        # pop_distri2 = get_pop_distri(get_file_contents(constants.URL).drop(['class'], axis=1), [population[best_index]],
        # constants.K)
        # plotkaro(population[best_index],pop_distri2)

        for i, x in zip(range(0, constants.POP_COUNT), population):
            if i != best_index and i != worst_index:
                new_individual = population[i] + (population[best_index] - population[i]) * rat + (
                            population[i] - population[worst_index]) * rat1
                fit = find_fitness(x_dataset, [new_individual], K)
                ff = get_final_fitness_trig(fit)
                if ff[0] > final_fitness[i]:
                    population[i] = new_individual

        print("Iteration :", iterations, " ending...")

        best_index = 0
        mx = -(1e20)

        for i in range(0, len(population)):
            if final_fitness[i] > mx:
                mx = final_fitness[i]
                best_index = i

        print(final_fitness[best_index])
        print(final_fitness, "\n\n")

    print(fitness)
    return population, final_fitness


got_back, final_fitness = apply_jaya(get_file_contents(constants.URL), constants.K)

print(final_fitness)
print(got_back)

best_index = 0
mx = -(1e20)

for i in range(0, len(got_back)):
    if final_fitness[i] > mx:
        mx = final_fitness[i]
        best_index = i

print("Best candidate :: ", got_back[best_index])

file = get_file_contents(constants.URL).drop(['class'], axis=1)
if(constants.URL.__contains__('glass') or constants.URL.__contains__('parkinsons')):
    file = file.drop(['A'],axis=1)

pop_distri = get_pop_distri(file, [got_back[best_index]],constants.K)

print(pop_distri)

labelled = get_pop_distri_label(file, got_back[best_index],constants.K)



best_fitness = final_fitness[best_index]

# if(constants.URL.__contains__('sonar')) :
#     best_fitness = output(best_fitness)

# print("BEST FITNESS ::",output(best_fitness))
print("BEST FITNESS ::",(best_fitness))

print("labelled :: \n", labelled)


##############################

####   FMEASURE
if(constants.URL.__contains__('ionosphere') or constants.URL.__contains__('iris') or constants.URL.__contains__('wine')or constants.URL.__contains__('parkinsons') or constants.URL.__contains__('sonar')or constants.URL.__contains__('segmentation')or constants.URL.__contains__('glass')):
    # classdata = pd.read_csv(constants.cURL)
    classdata = np.array(y_dataset_glob)
    #print(classdata)
    fmeasure(labelled,classdata)
##############################

if constants.URL.__contains__('random2'):
    plot2D(got_back[best_index], pop_distri)

if constants.URL.__contains__('random3'):
    plot3D(got_back[best_index], pop_distri)

