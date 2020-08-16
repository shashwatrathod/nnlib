import numpy as np
import math
import timeit

##Gradient descent
def sigmoid(inputs):
    inputs = np.array(inputs)
    return 1 / (1 + np.exp(np.multiply(-1, inputs)))

inputs = [[0.05,0.1]]
targets = [[0.01,0.99]]
weights_layer1 = np.array([[0.15,0.2],[0.25,0.3]])
biases_layer1 = np.array([[0.35,0.35]])
weights_layer2 = np.array([[0.4,0.45],[0.5,0.55]])
biases_layer2 = np.array([[0.6,0.6]])
weights = []
weights.append(weights_layer1)
weights.append(weights_layer2)
losses = []
for _ in range(1000):
    output_layer1 = sigmoid(np.dot(inputs,weights_layer1)+biases_layer1)


    output_layer2 = sigmoid(np.dot(output_layer1,weights_layer2)+biases_layer2)


    mse = 0.5*(np.power(targets-output_layer2,2))

    total_mse = np.sum(mse)
    losses.append(total_mse)

    t1 = np.array(output_layer2)-np.array(targets)
    t2 = np.array(output_layer2)*(1-np.array(output_layer2))
    t3 = output_layer1

    gradient_w = t3.T*(t1*t2)
    gradient_b = (t1*t2)
    '''
    CONCLUSIONS
    t1*t2 is propagated back so we can write a function for a layer to return the same for that particular layer
    in calculation for one of the term of the hidden layers, the weights of the next layer are needed
    mid term for any layer is gonna be the derivative of the activation function..need a way to feed generalized i/p regardless of the type of activation
    '''

    weights_layer2_updated = weights_layer2 - 0.5*gradient_w
    biases_layer2_updated = biases_layer2 - 0.5*gradient_b

    t1 = np.array(output_layer2)-np.array(targets)
    t2 = np.array(output_layer2)*(1-np.array(output_layer2))
    t3 = np.array(weights_layer2)

    t4 = np.sum(t3*t1*t2,axis=1)

    t5 = np.array(output_layer1)*(1-np.array(output_layer1))
    t6 = np.array(inputs)

    gradient_w = t6.T*(t4*t5)
    gradient_b = (t4*t5)
    weights_layer1_updated = weights_layer1 - 0.5*gradient_w
    biases_layer1_updated = biases_layer1 - 0.5*gradient_b

    weights_layer2 = weights_layer2_updated
    weights_layer1 = weights_layer1_updated
    biases_layer1 = biases_layer1_updated
    biases_layer2 = biases_layer2_updated

import matplotlib.pyplot as plt
plt.plot(list(range(len(losses))),losses)
plt.show()