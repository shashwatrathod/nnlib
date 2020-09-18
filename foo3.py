import numpy as np
from initializers.initializers import Zeros, RandomNormal, Random
from activations.activations import LeakyReLU

def der_relu(inputs):
    op = np.zeros(shape=np.shape(inputs))
    for i in range(len(inputs)):
        for j in range(len(inputs[0])):
            if inputs[i][j] > 0:
                op[i][j] = 1
            else:
                op[i][j] = 0.01 * inputs[i][j]
    return np.array(op)

def der_mse(output,targets):
    return output - targets

def get_mse_loss(outputs,targets):
    return 0.5*(1/len(outputs))*np.sum(np.square(targets-outputs))

initializer = Zeros()
initializer_w = RandomNormal(mean=0, std_dev=0.02)
activation = LeakyReLU(0.01)
learning_rate = 0.1
inputs = np.array([[x] for x in range(1,33)])
m_input = max(inputs)[0]
targets = np.array([[0.4*x[0] + 5*x[0] + 7.3] for x in inputs])
m_target = max(targets)[0]
inputs = inputs/m_input
targets = targets/m_target
layers_units = np.array([4,5,1])
weights = []
biases = []
prev_shape = inputs.shape[1]

for units in layers_units:
    weight = initializer_w(prev_shape,units)
    bias = initializer(units)
    prev_shape = units
    weights.append(weight)
    biases.append(bias)

losses = []
for _ in range(200):
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
        if(i==(len(weights)-1)):
            t1 = der_mse(outputs[-1],targets)
            t2 = der_relu(np.dot(outputs[i-1],weights[i])+biases[i])
            t3 = outputs[i-1]
        else:
            t1 = np.dot(t1*t2,weights[i+1].T)
            if(i==0):
                t2 = der_relu(np.dot(inputs,weights[i])+biases[i])
                t3 = inputs
            else:
                t2 = der_relu(np.dot(outputs[i-1],weights[i])+biases[i])
                t3 = outputs[i-1]

        t4 = np.ones(shape=biases[i].shape)
        gradient_w = np.dot(t3.T,(t1*t2))
        gradient_b = np.sum(t1 * t2, axis=0,keepdims=True)
        weight_u = (weights[i] - learning_rate * (1 / len(inputs)) * gradient_w)
        bias_u = (biases[i] - learning_rate * (1 / len(inputs)) * gradient_b)

        assert (weight_u.shape == weights[i].shape)
        assert (bias_u.shape == biases[i].shape)
        weights_updated.append(weight_u)
        biases_updated.append(bias_u)

    weights = list(reversed(weights_updated))
    biases = list(reversed(biases_updated))

import matplotlib.pyplot as plt

plt.subplot(131)
plt.plot(list(range(len(losses))), losses)

# VISUALIZATION
plt.subplot(132)
plt.plot(inputs,targets)

plt.subplot(133)
prev_output = inputs
for i in range(len(weights)):
    output = activation(np.dot(prev_output, weights[i]) + biases[i])
    prev_output = output
plt.plot(inputs,prev_output)
plt.show()
