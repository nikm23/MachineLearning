import argparse
import numpy as np
import math
import random
from csv import reader

def euclidean_distance(X, Y):
    d = 0.0
    for i in range(len(X)-1):
        d += np.square(X[i] - Y[i])
    return np.sqrt(d)

def manhattan_distance(X, Y):
    d = 0.0
    for i in range(len(X)-1):
        d += abs(X[i] - Y[i])
    return d

def read_data(dataset_path):
    data = list()
    f = open(dataset_path, 'r')
    file = reader(f)
    for line in file:
        data.append(line)
    data.pop(0)
    return data

def normalize(data):
    temp = np.array(data)
    mean_list = np.mean(temp, axis=0)
    std_list = np.std(temp, axis=0)
    for j in range(len(data)):
        for i in range(len(data[j])-1):
            data[j][i] = (data[j][i]-mean_list[i])/std_list[i]
    return

def get_centroids(clusters, data):
    centroids = []
    for cluster in clusters:
        cluster_rows = []
        for i in cluster:
            cluster_rows.append(data[i])
        centroids.append([sum(col) / float(len(col)) for col in zip(*cluster_rows)])
    return centroids

def get_distance(distance_metric, centroid, data):
    if distance_metric == "Manhattan":
        return manhattan_distance(centroid,data)
    else:
        return euclidean_distance(centroid,data)

def get_nearest_centroid(distance_metric, data, centroids):
    min_distance= math.inf
    cluster_idx= None
    for i in range(len(centroids)):
        centroid = centroids[i]
        distance = get_distance(distance_metric, centroid, data)
        if distance < min_distance:
            min_distance = distance
            cluster_idx = i
    return cluster_idx

def get_clusters(data, centroids, distance_metric):
    clusters = [[] for i in range(len(centroids))]
    len_data = len(data)
    new_clusters = [[] for i in range(len(centroids))]
    while new_clusters != clusters or len(clusters[0]) == 0:
        if len(new_clusters[0]) != 0:
            clusters = new_clusters.copy()
        new_clusters = [[] for i in range(len(centroids))]
        for i in range(len_data):
            cluster_idx = get_nearest_centroid(distance_metric, data[i], centroids)
            new_clusters[cluster_idx].append(i)
        centroids = get_centroids(new_clusters, data)
    return clusters

def evaluate_clusters(clusters, data):
    for cluster in clusters:
        p = 0
        n = 0
        for i in cluster:
            row = data[i]
            if row[len(row) - 1] == 1.0:
                p += 1
            else:
                n += 1
        print("Cluster with positive cases percentage : " + str(100*p/(p+n)) + " and negative cases percentage: " + str(100*n/(p+n)))
        print("Cluster with positive cases : " + str(p) + " and negative cases: " + str(n))
    return


def kmeans(dataset_path, k, distance):
    data = read_data(dataset_path)
    data = [[float(float(j)) for j in i] for i in data]
    normalize(data)
    centroids = random.sample(range(0, len(data)), k)
    centroids = [data[i] for i in centroids]
    clusters = get_clusters(data, centroids, distance)
    evaluate_clusters(clusters, data)
    return

argsparser = argparse.ArgumentParser()
argsparser.add_argument('--dataset', required=True)
argsparser.add_argument('--k', required=True)
argsparser.add_argument('--distance', default="Euclidean")
args = vars(argsparser.parse_args())
kmeans(args['dataset'], int(args['k']), args['distance'])