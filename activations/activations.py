import numpy as np
from abc import ABC, abstractmethod


# base class
class Activation(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args):
        pass


class ReLU(Activation):

    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        inputs = np.array(inputs)
        return np.maximum(np.zeros(shape=np.shape(inputs)),inputs)


class Sigmoid(Activation):

    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        inputs = np.array(inputs)
        return 1 / (1 + np.exp(np.multiply(-1, inputs)))


class LeakyReLU(Activation):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def __call__(self, inputs):
        inputs = np.array(inputs, dtype=float)
        shape = inputs.shape
        if (len(shape) == 2):
            rl = []
            for l in inputs:
                tl = []
                for e in l:
                    if (e < 0):
                        e = e * self.alpha
                    tl.append(e)
                rl.append(tl)
            return np.array(rl)
        elif (len(shape) == 1):
            for i in range(len(inputs)):
                if (inputs[i] < 0):
                    inputs[i] = inputs[i] * self.alpha
            return inputs

        return None


class tanh(Activation):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        inputs = np.array(inputs)
        return (1 - np.exp(-2 * inputs)) / (1 + np.exp(-2 * inputs))


class ExponentialLU(Activation):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def __call__(self, inputs):
        inputs = np.array(inputs, dtype=float)
        shape = inputs.shape
        if (len(shape) == 2):
            rl = []
            for l in inputs:
                tl = []
                for e in l:
                    if (e < 0):
                        e = self.alpha * (np.exp(e) - 1)
                    tl.append(e)
                rl.append(tl)
            return np.array(rl)
        elif (len(shape) == 1):
            for i in range(len(inputs)):
                if (inputs[i] < self.alpha):
                    inputs[i] = inputs[i] * self.alpha

            return inputs

        return None


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        inputs = np.array(inputs, dtype=float)
        return np.exp(inputs) / np.reshape(np.sum(np.exp(inputs), axis=1), (-1, 1))
