import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import fitness


def dist(a, b):
    return np.linalg.norm(a - b)

def KMeansAlgo(k,data):
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids

datairis = pd.read_csv('datasets/iris.data')
datairis = datairis.drop(datairis.columns[[4]], axis=1)
centroid = KMeansAlgo(constants.K,datairis)
print(centroid)
#fit = fitness.find_fitness(datairis,[centroid],3)
#print(fit)

#data = np.array(data)
#print(dist(centroids[0],data[0]),dist(centroids[1],data[0]),dist(centroids[2],data[0]))
#print(dist(centroids[0],data[55]),dist(centroids[1],data[55]),dist(centroids[2],data[55]))
#print(dist(centroids[0],data[127]),dist(centroids[1],data[127]),dist(centroids[2],data[127]))



