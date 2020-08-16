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
    return op


initializer = Ones()
activation = LeakyReLU(0.2)
inputs = [[x] for x in range(100)]
m_input = max(inputs)[0]
inputs = np.divide(inputs, m_input)
targets = [[3 * x[0] + 10] for x in inputs]
m_target = max(targets)[0]
targets = np.divide(targets, m_target)
weights_1 = initializer(2, 1)
weights_3 = initializer(1, 2)
biases_1 = initializer(2)
biases_3 = initializer(1)
losses = []

for _ in range(1000):
    abc = np.dot(inputs, weights_1) + biases_1
    output_1 = activation(abc)
    output_3 = activation(np.dot(output_1, weights_3) + biases_3)

    mse = 0.5 * (np.power(targets - output_3, 2))
    total_mse = np.sum(mse)

    dCdO3 = np.array(output_3) - np.array(targets)
    dO3dN3 = np.array(der_relu(np.dot(output_1, weights_3) + biases_3))
    dN3dW = output_1
    dCdW =  np.sum(dCdO3*dO3dN3*dN3dW,axis=0)
    weights_3_u = (weights_3.T - 0.1*(1/len(inputs))*dCdW).T
    dn3db = np.ones(shape=biases_3.shape)
    dCdb = np.sum(dCdO3*dO3dN3*dn3db,axis=0)
    biases_3_u = biases_3 - 0.1*(1/len(inputs))*dCdb


    dCdO1 = dCdO3*dO3dN3*weights_3.T
    dO1dN1 = np.array(der_relu(np.dot(inputs, weights_1) + biases_1))
    dN1dW = inputs
    dCdW = np.sum(dCdO1*dO1dN1*dN1dW,axis=0)
    weights_1_u = (weights_1 - 0.1*(1/len(inputs))*dCdW)
    dN1db = np.ones(shape=biases_1.shape)
    dCdb = np.sum(dCdO1*dO1dN1*dN1db,axis=0)
    biases_1_u = biases_1 - 0.1*(1/len(inputs))*dCdb

    biases_1 = biases_1_u
    biases_3 = biases_3_u
    weights_1 = weights_1_u
    weights_3 = weights_3_u

    losses.append(total_mse)

import matplotlib.pyplot as plt

plt.plot(list(range(len(losses))), losses)
plt.show()

plt.plot(inputs,targets)
plt.plot(inputs,output_3)
plt.show()
