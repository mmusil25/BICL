"""
This network trains fully connected neural network
on the IRIS dataset using no special ML libraries.

This network is not my original work but has been borrowed as an educational resource.

Code Source:

https://github.com/rianrajagede/iris-python/blob/master/Python/iris_plain_mlp.py
"""
from typing import List
import secrets

"""
SECTION 1 : Load and setup data for training
"""

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
 
 
secrets.SystemRandom().seed(123)

# Load dataset
with open('./iris/iris.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None)  # skip header
    dataset = list(csvreader)

# Change string value to numeric
for row in dataset:
    row[4] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
    row[:4] = [float(row[j]) for j in range(len(row))]

# Split x and y (feature and target)
secrets.SystemRandom().shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 3 neuron, activation using sigmoid
output layer : 3 neuron, represents the class of Iris
optimizer = gradient descent
loss function = Square ROot Error
learning rate = 0.005
epoch = 400
best result = 96.67%
"""


def multi_variate_sigmoid(x):  # Here x is a np array
    return (1 + np.exp(np.sum(x))) ** (-1)


def dendritic_boundary(x):  # Here x is a single valued real number
    alpha_L, alpha_U = 0.5, 0.5
    b_U, b_L = 0, 1
    numerator = (1 + np.exp(alpha_L * (x - b_L))) ** (alpha_L ** (-1))
    denominator = (1 + np.exp(alpha_U * (x - b_U))) ** (alpha_U ** (-1))
    return np.log(numerator / denominator) + b_L


def dendritic_transfer(x):  # Here x is a numpy array
    a_d, c_d, b_d = 1, 0.5, 0.5
    arg1 = c_d * multi_variate_sigmoid(np.multiply(a_d, np.subtract(x, b_d)) + np.sum(x))
    return dendritic_boundary(arg1)


def dendritic_layer(input_set, weights, dendrites, output_features, Den_view, batch_size=None, bias=None):
    
    """
    batch_size is not used in this implementation.

    """

    # if bias is not None:
    #     bias_np = self.bias.detach().numpy()
    soma_input = np.zeros(dendrites)
    #output = np.zeros((int(batch_size), int(output_features)))
    output = np.zeros(int(output_features))
    #for i in range(batch_size):
    #print(['weights.shape', weights.shape])
    for n in range(output_features):
        for d in range(dendrites):
            # soma_input[d] = np.dot(input_set[d:d + Den_view], weights[n, d].reshape(49, 1))
            soma_input[d] = np.dot(input_set[(Den_view * d): (Den_view * (d+1))], weights[n][d][:])
        output[n] = dendritic_transfer(soma_input)
            # if self.bias is not None:
            #     output[i] += bias_np
    return output

def matrix_mul_bias(A, B, bias):  # Matrix multiplication (for Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C


def vec_mat_bias(A, B, bias):  # Vector (A) x matrix (B) multiplication
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C


def mat_vec(A, B):  # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C


def sigmoid(A, deriv=False):
    if deriv:  # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Define parameter
alfa = 0.005
epoch = 400
neuron = [4, 6, 6]  # number of neuron each layer

# Dendritic layer's weight dimensions in order
Output_features = 3  # Number of neurons
dendrites = 2
Den_view = 3

f = open("IRIS_trial7.txt", "w+")
f.write(" alpha: %.4f, epoch: %d \n" % (alfa, epoch))
f.write(" neuron[0]: %d, neuron[1]: %d, neuron[2]: %d \n" % (neuron[0], neuron[1], neuron[2]))
f.write(" Dendritic layer dimensions (in order) \n: "
        "Output_features: %d, dendrites: %d, Den_view: %d \n" % (Output_features, dendrites, Den_view))
f.write("###### Begin Training Output ###### \n")
# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]

# dendrite_weights = np.subtract(2*np.random.normal(0, 1, [neuron[3], dendrites, int(neuron[2]/dendrites)]),
#                               np.full([neuron[3], dendrites, int(neuron[2]/dendrites)], -1))
dendrite_weights = np.subtract(np.random.normal(0, 1, [Output_features, dendrites, Den_view]),
                               np.full([Output_features, dendrites, Den_view], 0))
#print(['dendrite_weights.shape' , dendrite_weights.shape])

bias = [0 for i in range(neuron[1])]
bias_2 = [0 for i in range(neuron[2])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2*secrets.SystemRandom().gauss(0, 1) - 1

for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2*secrets.SystemRandom().gauss(0, 1) - 1

cost_for_graph = []
for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X):  # Update for each data; SGD
        print(["x", x])
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        print(["h_1", h_1])
        X_1 = sigmoid(np.clip(h_1, -500, 500))
        print(["X_1", X_1])
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        print(["h_2", h_2])
        X_2 = sigmoid(np.clip(h_2, -500, 500))
        print(["X_2", X_2])
        X_3 = dendritic_layer(X_2, dendrite_weights, dendrites, Output_features, Den_view)
        print(["X_3", X_3])
       # sys.exit()
        #  dendritic_layer(input_set, weights, dendrites, output_features, Den_view, batch_size=None, bias=None)
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(3):
            eror += 0.5 * (target[i] - X_3[i]) ** 2
        cost_total += eror

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_3: List[int] = []
        for j in range(Output_features):
            delta_3.append(-1 * (target[j] - X_3[j]) * X_3[j] * (1 - X_3[j]))

        for i in range(Output_features):
            for j in range(dendrites):
                for k in range(Den_view):
                     
                    dendrite_weights[i][j][k] -= alfa * (delta_3[i] * (X_2[k]+X_2[k+1])/2) # Averaging the outputs from X_2 to fit to dendritic_layer
                    # bias_2[j] -= alfa * delta_3[j]

        # TODO: Fix the line below here. I should make sure it's a good calculation.
        # Also resize dendrite_weights for mult
        #print(["dendrite_weights[0:3]:", dendrite_weights[0:3], "dendrite_weights[0:3].shape:", dendrite_weights[0:3].shape])
        #print(["delta_3:", delta_3, "delta_3.shape:", len(delta_3)])
        delta_2 = mat_vec(np.reshape(dendrite_weights[0:3], [6, 3]), delta_3)
        #print(["delta_2", delta_2])
        #print(["X_2", X_2])
        for i in range(neuron[1]):
            delta_2[i] = delta_2[i]*(X_2[i] * (1 - X_2[i]))
        # print(delta_3)
        delta_3np_half = np.array(delta_3)
        delta_3np_half = np.divide(delta_3np_half, np.full(delta_3np_half.shape, 2))

        # TODO: Revise this potentially ineffective way of matching delta_3 to bias_2
        delta_3_copy = np.concatenate((delta_3np_half, delta_3np_half), axis=0)

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                bias_2[j] -= alfa * delta_3_copy[j]

        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_3)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1 - X_1[j]))

        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -= alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]

    cost_total /= len(train_X)
    cost_for_graph.append(cost_total)
    interval = 50
    if (e % interval == 0):
        print("Epoch ", e/interval, " out of ", epoch/interval)
        print("Epoch cost: ", cost_total)

        f.write("Epoch " + str(e/interval) + " out of " + str(epoch/interval) + "\n")
        f.write("Epoch cost: %.5f \n" % cost_total)

cost_for_graph = np.array(cost_for_graph)
print(["cost_for_graph.shape: ", cost_for_graph.shape])
# Plot error over time
x_axis = np.arange(epoch)
#for i in range(epoch):
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_axis, cost_for_graph, 'r')
# axis(xmin, xmax, ymin, ymax)
plt.axis([0, epoch, 0, 0.40])
plt.xlabel('Training iteration')
plt.ylabel('Error')
ax.set_title('Dendritic Network\n'
             ' (learning rate: %(val1).4f, Epochs: %(val2)d)'
             % {'val1': alfa, 'val2': epoch})
plt.show()





"""
SECTION 3 : Testing
"""

res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

# Get prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x: x[1])[0])

# Print prediction
print("Predictions: ", preds)
f.write("Predictions: \n")
f.write("[ ")
for i in range(len(preds)):
    f.write("%d, " % preds[i])
f.write(" ] \n")
# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print("Network Accuracy: ", acc / len(preds) * 100, "%")
f.write("Network Accuracy: " + str(acc / len(preds) * 100) + "%")
