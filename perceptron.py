import numpy as np
import argparse
from csv import reader

global threshold

def predict(row, weights):
    dot_product = np.dot(row, weights)
    return 1.0 if dot_product >= 0.0 else -1.0

def train_weights(train_data):
    global threshold
    for i in range(len(train_data)):
        train_data[i] = list(map(float, train_data[i]))
    weight_matrix = np.zeros(len(train_data[0]))
    flag = True
    count = 0
    while flag and count < threshold:
        count += 1
        flag = False
        for row in train_data:
            input = row[:-1]
            input.insert(0, 1)
            output = row[-1]
            output = -1 if output == 0 else 1
            prediction = predict(input, weight_matrix)
            if prediction * output <= 0:
                weight_matrix[:] += [output * input[i] for i in range(len(input))]
                flag = True
    return weight_matrix

def predict_model(train_data, test_data):
    weights = train_weights(train_data)
    errors = 0
    for i in range(len(test_data)):
        test_data[i] = list(map(float, test_data[i]))
    for row in test_data:
        input = row[:-1]
        input.insert(0, 1)
        prediction = predict(input, weights)
        output = -1 if row[-1] == 0 else 1
        if prediction != output:
            errors = errors + 1
    return errors/len(test_data), weights

def split_data(data):
    data_split = list()
    fold_size = int(len(data) / 10)
    for i in range(10):
        index = 0
        fold = list()
        while len(fold) < fold_size:
            fold.append(data.pop(index))
        data_split.append(fold)
    return data_split

def read_data(dataset_path):
    data = list()
    f = open(dataset_path, 'r')
    file = reader(f)
    for line in file:
        data.append(line)
    data.pop(0)
    return data

def classify_kfolds(data):
    data_set_list = split_data(data)
    index = 0
    fold_errors = []
    fold_weight_matrix = []
    while index < len(data_set_list):
        train_data = data_set_list[0:index] + data_set_list[index + 1:]
        train_data = [item for sublist in train_data for item in sublist]
        test_data = data_set_list[index]
        errors, weights = predict_model(train_data, test_data)
        fold_errors.append(errors)
        fold_weight_matrix.append(weights)
        index = index+1
    return fold_errors, fold_weight_matrix

def classify_erm(data):
    errors, weight_matrix = predict_model(data, data)
    return errors, weight_matrix

def classifier(dataset_path, mode):
    global threshold
    data = read_data(dataset_path)
    threshold = 400
    errors = None
    weight_matrix = None
    if(mode=='erm'):
        errors, weight_matrix = classify_erm(data)
    else:
        errors, weight_matrix = classify_kfolds(data)
    print("Errors is/are: ", errors)
    print("Weight matrix is/are :", weight_matrix)
    print("Mean error:", sum(errors)/len(errors))
    return

argsparser = argparse.ArgumentParser()
argsparser.add_argument('-dataset', required=True)
argsparser.add_argument('-mode', required=True)
args = vars(argsparser.parse_args())
classifier(args['dataset'], args['mode'])