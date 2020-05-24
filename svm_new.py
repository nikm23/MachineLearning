import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
X1 = np.c_[np.ones((X0.shape[0])), X0] # add one to the x-values to incorporate bias

positive_x =[]
negative_x =[]

for i,label in enumerate(y):
  if label == 0:
    negative_x.append(X1[i])
  else:
    positive_x.append(X1[i])
data_dict = {-1:np.array(negative_x), 1:np.array(positive_x)}

max_fval = float('-inf')
for y_i in data_dict:
  if np.amax(data_dict[y_i]) > max_fval:
    max_fval=np.amax(data_dict[y_i])

def test(X, Y, w):
  corr = 0
  total = 0
  for xi, yi in zip(X, Y):
    if (yi ==  np.sign(np.dot(w, xi) )):
      corr+=1
    total += 1
  print("Total data points: ",total)
  print("Correct labels: ", corr)
  print("Accuracy:",(corr/total) * 100)
  return

def train(X, Y, stepsize):
  iteration = 10000
  w = np.zeros(len(X[0]))
  w[0] = 45
  w_sum = np.zeros(len(X[0]))
  for epoch in range(1, iteration):
    stepsize *= 0.1
    w_sum = w + w_sum
    i = int(np.random.uniform(0,len(X)-1))
    if (Y[i] * np.dot(w, X[i]) < 1):
      temp = w + (X[i]*Y[i])
    else:
      temp = w
    w[0] = temp[0]
    w[1:] = temp[1:] - 2*(stepsize*w[1:])
  return [element*(1/iteration) for element in w_sum]

def draw(X, W):
  xfit = np.linspace(0, 10)
  plt.scatter(X[:, 1], X[:, 2], marker='o', c=y, s=25, edgecolor='k')
  yfit = (-W[0] - (W[1] * xfit)) / W[2]
  yprev = ((-2*W[0]-(W[1] * xfit)) / W[2])
  ynext = ((-(W[1] * xfit)) / W[2])
  plt.plot(xfit, yfit, '-k')
  plt.fill_between(xfit, yprev, ynext, edgecolor='none', color='#AAAAAA', alpha=0.4)
  plt.show()
  return

for i in range(len(y)):
  if y[i] == 0:
    y[i] = -1
trainX = X1[len(X1) // 5:]
trainY = y[len(y) // 5:]
testX = X1[:len(X1) // 5]
testY = y[:len(y) // 5]
#step_size = [max_fval * 0.1, max_fval * 0.01, max_fval * 0.001, max_fval*0.0001]
w = train(trainX, trainY, max_fval)
draw(X1,w)
test(testX,testY,w)
