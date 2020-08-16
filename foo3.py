import numpy as np
from initializers.initializers import Ones
from activations.activations import LeakyReLU

def der_relu(inputs):
    op = np.zeros(shape=np.shape(inputs))
    for i in range(len(inputs)):
        for j in range(len(inputs[0])):
            if inputs[i][j] > 0:
                op[i][j] = 1
            else:
                op[i][j] = 0.2 * inputs[i][j]
    return np.array(op)

def der_mse(output,targets):
    return output - targets

def get_mse_loss(outputs,targets):
    return 0.5*(1/len(outputs))*np.sum(np.square(targets-outputs))

initializer = Ones()
activation = LeakyReLU(0.1);
learning_rate = 0.1
inputs = np.array([[x] for x in range(1,33)])
m_input = max(inputs)[0]
targets = np.array([[0.4*x[0] + 5*x[0] + 7.3] for x in inputs])
m_target = max(targets)[0]
inputs = inputs/m_input
targets = targets/m_target
layers_units = np.array([2,2,1])
weights = []
biases = []
prev_shape = inputs.shape[1]
for units in layers_units:
    weight = initializer(units, prev_shape)
    bias = initializer(units)
    prev_shape = units
    weights.append(weight)
    biases.append(bias)

losses = []
for _ in range(1000):
    #forward pass
    outputs = []
    prev_output = inputs
    for i in range(len(weights)):
        output = activation(np.dot(prev_output,weights[i])+biases[i])
        prev_output = output
        outputs.append(output)

    mse = get_mse_loss(outputs[-1],targets)
    losses.append(mse)
    weights_updated = []
    biases_updated = []
    for i in reversed(range(len(weights))):
        if(i==(len(weights)-1)): #it's the output layer
            t1 = der_mse(outputs[-1],targets)
            t2 = der_relu(np.dot(outputs[i-1],weights[i])+biases[i])
            t3 = outputs[i-1]
        else:
            t1 = t2*np.dot(t1,weights[i+1].T)###########################################
            if(i==0):
                t2 = der_relu(np.dot(inputs,weights[i])+biases[i])
                t3 = inputs
            else:
                t2 = der_relu(np.dot(outputs[i-1],weights[i])+biases[i])
                t3 = outputs[i-1]

        t4 = np.ones(shape=biases[i].shape)
        gradient_w = np.sum(t1 * t2 * t3, axis=0)
        gradient_b = np.sum(t1 * t2 * t4, axis=0)
        weight_u = (weights[i].T - learning_rate * (1 / len(inputs)) * gradient_w).T
        bias_u = (biases[i] - learning_rate * (1 / len(inputs)) * gradient_b)
        if (np.shape(weight_u) != weights[i].shape):
            weight_u = (weights[i] - learning_rate * (1 / len(inputs)) * gradient_w)
            print(f"weights[{i}] and updated shape dont match")

        if (np.shape(bias_u) != biases[i].shape):
            print(f"biases[{i}] and updated shape dont match")
        weights_updated.append(weight_u)
        biases_updated.append(bias_u)

    weights = list(reversed(weights_updated))
    biases = list(reversed(biases_updated))

import matplotlib.pyplot as plt

plt.plot(list(range(len(losses))), losses)
plt.show()

# VISUALIZATION

plt.plot(inputs,targets)
plt.show()

plt.plot(inputs,outputs[-1])
plt.show()