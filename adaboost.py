import matplotlib.pyplot as plt
import copy
import argparse
from csv import reader
import math

global header

def predict(w_list, h_list, test_data):
    h = [0] * len(test_data)
    for itr in range(len(w_list)):
        split_point, best_dimension = h_list[itr]
        h_t = [(-1 if row[best_dimension] < split_point else 1) for row in test_data]
        for i in range(len(test_data)):
            h[i] = h[i] + h_t[i] * w_list[itr]
    error = 0
    for itr in range(len(test_data)):
        if (1 if h[itr] > 0 else -1) != test_data[itr][-1]:
            error = error + 1
    return error / len(test_data)

def invoke_weak_learner(Dvector, org_train_data):
    f_prime = float("inf")
    j_prime = None
    theta_prime = None
    temp_row = []
    train_data = copy.deepcopy(org_train_data)
    len_tr = len(train_data)
    for i in range(len(header) - 1):
        max = 0
        for j in range(len_tr):
            if max < train_data[j][i]:
                max = train_data[j][i]
        temp_row.append(max+1)
    train_data.append(temp_row)
    for itr in range(len_tr):
        train_data[itr].append(Dvector[itr])
    for j in range(len(header)-1):
        train_data = sorted(train_data, key=lambda train_data:train_data[j])
        f = sum([((train_data[i][-2]*train_data[i][-1]) if train_data[i][-2] == 1 else 0) for i in range(len(org_train_data))])
        if f < f_prime:
            f_prime = f
            theta_prime = train_data[0][j]-1
            j_prime = j
        for idx in range(len(train_data)-1):
            f = f - train_data[idx][-2]*train_data[idx][-1]
            if f < f_prime and train_data[idx][j] != train_data[idx+1][j]:
                f_prime = f
                theta_prime = 0.5 * (train_data[idx][j] + train_data[idx+1][j])
                j_prime = j
    return j_prime, theta_prime


def predict_model(train_data, test_data, threshold):
    len_train = len(train_data)
    Dvector = [(1 / len_train) for i in range(len_train)]
    h_list = []
    w_list = []
    for itr in range(threshold):
        best_dimension, split_point = invoke_weak_learner(Dvector, train_data)
        h_list.append((split_point, best_dimension))
        output = []
        for idx in range(len(train_data)):
            output.append(-1 if train_data[idx][best_dimension] < split_point else 1)
        error = sum([(Dvector[i] if train_data[i][-1] != output[i] else 0) for i in range(len_train)])
        w_t = 0.5 * math.log(1/error - 1)
        w_list.append(w_t)
        den = 0
        for idx in range(len_train):
            den += Dvector[idx] * math.exp(-1 * w_t * train_data[idx][-1] * output[idx])
        for idx in range(len_train):
            num = Dvector[idx] * math.exp(-1 * w_t * train_data[idx][-1] * output[idx])
            Dvector[idx] = num/den
    error = predict(w_list, h_list, test_data)
    return error

def read_data(dataset_path):
    global header
    data = list()
    f = open(dataset_path, 'r')
    file = reader(f)
    for line in file:
        data.append(line)
    header = data.pop(0)
    for i in range(len(data)):
        data[i] = list(map(float, data[i]))
        if data[i][-1] == 0:
            data[i][-1] = -1
    return data

def split_data(data):
    data_copy = copy.deepcopy(data)
    data_split = list()
    fold_size = int(len(data_copy) / 10)
    for i in range(10):
        index = 0
        fold = list()
        while len(fold) < fold_size:
            fold.append(data_copy.pop(index))
        data_split.append(fold)
    return data_split

def classify_kfolds(data, threshold):
    data_set_list = split_data(data)
    index = 0
    errors = []
    while index < len(data_set_list):
        train_data = data_set_list[0:index] + data_set_list[index + 1:]
        train_data = [item for sublist in train_data for item in sublist]
        test_data = data_set_list[index]
        errors.append(predict_model(train_data, test_data, threshold))
        index = index+1
    return errors

def classify_erm(data, threshold):
    result = predict_model(data, data, threshold)
    return result

def adaboost(dataset_path, mode):
    data = read_data(dataset_path)
    threshold = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 120, 180, 240]
    errors = []
    if (mode == 'erm'):
        for t in threshold:
            errors.append(classify_erm(data, t))
    else:
        for t in threshold:
            errors_list = classify_kfolds(data, t)
            errors.append(sum(errors_list)/len(errors_list))
    print("Errors ", errors)
    plt.plot(threshold, errors)
    plt.xlabel("Number of iterations")
    plt.ylabel("Errors")
    plt.show()
    return

argsparser = argparse.ArgumentParser()
argsparser.add_argument('-dataset', required=True)
argsparser.add_argument('-mode', required=True)
args = vars(argsparser.parse_args())
adaboost(args['dataset'], args['mode'])