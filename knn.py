import numpy as np
import argparse
from csv import reader

def euclidean_distance(X, Y):
    d = 0.0
    for i in range(len(X)-1):
        d += np.square(X[i] - Y[i])
    return np.sqrt(d)

def read_data(dataset_path):
    data = list()
    f = open(dataset_path, 'r')
    file = reader(f)
    for line in file:
        data.append(line)
    data.pop(0)
    return data

def find_neighbors(data, k, test):
    len_train = len(data)
    distances = dict()
    for i in range(len_train):
        distances[i] = (euclidean_distance(test, data[i]))
    sort_d = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    neighbors = []
    sort_l = list(sort_d)
    for i in range(k):
        neighbors.append(sort_l[i])
    return neighbors

def get_label(neighbors, data):
    label = {}
    for x in neighbors:
        ans = data[x][-1]
        if ans in label:
            label[ans] += 1
        else:
            label[ans] = 1
    max = -1
    ans = None
    for key in label:
        if label[key] > max:
            max = label[key]
            ans = key
    return ans

def normalize(data):
    temp = np.array(data)
    mean_list = np.mean(temp, axis=0)
    std_list = np.std(temp, axis=0)
    for j in range(len(data)):
        for i in range(len(data[j])-1):
            data[j][i] = (data[j][i]-mean_list[i])/std_list[i]

def calculate_accuracy(predictions, test):
    correct = 0
    for i in range(len(test)):
        if predictions[i] == test[i][-1]:
            correct += 1
    return correct / float(len(test)) * 100.0

def knn(dataset_path, k):
    data = read_data(dataset_path)
    data = [[float(float(j)) for j in i] for i in data]
    normalize(data)
    train = data[:4*len(data) // 5]
    test = data[x:]
    predictions = list()
    for row in test:
        neighbors = find_neighbors(train, k, row)
        predictions.append(get_label(neighbors, train))
    accuracy = calculate_accuracy(predictions, test)
    return accuracy


argsparser = argparse.ArgumentParser()
argsparser.add_argument('-dataset', required=True)
argsparser.add_argument('-k', required=True)
args = vars(argsparser.parse_args())
acc = knn(args['dataset'], int(args['k']))
print("Accuracy is :",acc)
