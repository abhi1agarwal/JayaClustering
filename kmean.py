import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import fitness
from readdata import output
import constants
import matplotlib.pyplot as plt
from readdata import get_file_contents
from fitness import find_fitness
from fitness import get_final_fitness
from fitness import get_pop_distri
from fitness import get_pop_distri_label
from plotting import plot2D
from plotting import plot3D
from fitness import fmeasure

def dist(a, b):
    return np.linalg.norm(a - b)

def KMeansAlgo(k,data):
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids

dataset = pd.read_csv(constants.URL)
print(dataset)
x_dataset = dataset.drop(['class'],axis=1)
y_dataset = dataset['class']

print(np.array(y_dataset))
if constants.URL.__contains__('glass') or constants.URL.__contains__('parkinsons'):
    x_dataset = x_dataset.drop(['A'], axis=1)
#print(x_dataset)
centroid = KMeansAlgo(constants.K,x_dataset)
print(centroid)
fit = fitness.find_fitness(x_dataset,[centroid],constants.K)
#print(fit)
bestfitness =fitness.get_final_fitness(fit)
if(constants.URL.__contains__('sonar')):
    bestfitness = output(bestfitness)
print(output(bestfitness))



#print("Best candidate :: ",[centroid])
file = get_file_contents(constants.URL).drop(['class'], axis=1)
if(constants.URL.__contains__('glass') or constants.URL.__contains__('parkinsons')):
    file = file.drop(['A'],axis=1)
pop_distri  = get_pop_distri(file, [centroid], constants.K)
#print(pop_distri)

labelled = fitness.get_pop_distri_label(file,centroid,constants.K)

print("labelled :: \n",labelled)

##############################

####   FMEASURE
if(constants.URL.__contains__('ionosphere') or constants.URL.__contains__('iris') or constants.URL.__contains__('wine')or constants.URL.__contains__('parkinsons') or constants.URL.__contains__('sonar')or constants.URL.__contains__('segmentation')or constants.URL.__contains__('glass')):
    # classdata = pd.read_csv(constants.cURL)
    classdata = np.array(y_dataset)
    #print(classdata)
    fmeasure(labelled,classdata)
##############################


if  constants.URL.__contains__('random2'):
    plot2D(centroid,pop_distri)

if  constants.URL.__contains__('random3'):
    plot3D(centroid,pop_distri)
#data = np.array(data)
#print(dist(centroids[0],data[0]),dist(centroids[1],data[0]),dist(centroids[2],data[0]))
#print(dist(centroids[0],data[55]),dist(centroids[1],data[55]),dist(centroids[2],data[55]))
#print(dist(centroids[0],data[127]),dist(centroids[1],data[127]),dist(centroids[2],data[127]))



