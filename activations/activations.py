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
    '''
    The Rectified Linear Unit Activation function.
    x' = max(0, x)

    :return: (inputs, outputs) tuple. Where output = max(0, inputs).
    '''
    def __init__(self):
        super().__init__()
        self.name = "ReLU"

    def __call__(self, inputs):
        '''
        :return: (inputs, outputs) tuple. Where output = max(0, inputs).
        '''
        inputs = np.array(inputs)
        outputs = np.maximum(np.zeros(shape=np.shape(inputs)),inputs)
        assert np.shape(inputs)==np.shape(outputs)
        return inputs, outputs

    def der(self, inputs):
        '''
        :param inputs: The activations from previous layers.
        :return: (inputs, outputs) tuple.
        '''
        inputs = np.array(inputs)
        outputs = inputs > np.zeros(np.shape(inputs))
        assert np.shape(inputs)==np.shape(outputs)
        return inputs, outputs


class Sigmoid(Activation):
    '''
    The sigmoid activation function.
    x' = 1/(1 + e^-x)

    :return (inputs, outputs) tuple.
    '''
    def __init__(self):
        super().__init__()
        self.name = "Sigmoid"

    def __call__(self, inputs):
        inputs = np.array(inputs)
        outputs = 1 / (1 + np.exp(np.multiply(-1, inputs)))
        assert inputs.shape==outputs.shape
        return inputs, outputs

    def der(self, inputs):
        inputs = np.array(inputs)
        outputs = np.multiply(inputs,(1-inputs))
        assert inputs.shape == outputs.shape
        return inputs, outputs



class LeakyReLU(Activation):
    def __init__(self, alpha) -> object:
        super().__init__()
        self.name = "LeakyReLU"
        self.alpha = alpha

    def __call__(self, inputs):
        inputs = np.array(inputs, dtype=float)
        shape = inputs.shape
        inp_copy = np.copy(inputs)
        if (len(shape) == 2):
            rl = []
            for l in inputs:
                tl = []
                for e in l:
                    if (e < 0):
                        e = e * self.alpha
                    tl.append(e)
                rl.append(tl)
            assert inputs.shape==np.shape(rl)
            return inputs, np.array(rl)
        elif (len(shape) == 1):
            for i in range(len(inputs)):
                if (inputs[i] < 0):
                    inputs[i] = inputs[i] * self.alpha
            return inp_copy, inputs

        return None

    def der(self, inputs):
        op = np.zeros(shape=np.shape(inputs))
        for i in range(len(inputs)):
            for j in range(len(inputs[0])):
                if inputs[i][j] > 0:
                    op[i][j] = 1
                else:
                    op[i][j] = self.alpha

        assert inputs.shape == op.shape
        return inputs, op


class tanh(Activation):
    def __init__(self):
        super().__init__()
        self.name = "tanh"

    def __call__(self, inputs):
        inputs = np.array(inputs)
        outputs = (1 - np.exp(-2 * inputs)) / (1 + np.exp(-2 * inputs))
        assert inputs.shape == outputs.shape
        return inputs, outputs

    def der(self, inputs):
        inputs = np.array(inputs)
        outputs = 1 - np.square(inputs)
        assert inputs.shape==outputs.shape
        return inputs, outputs


class ExponentialLU(Activation):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.name = "ExponentialLU"

    def __call__(self, inputs):
        inputs = np.array(inputs, dtype=float)
        inp_copy = np.copy(inputs)
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
            assert inp_copy.shape == np.shape(rl)
            return inp_copy, np.array(rl)
        elif (len(shape) == 1):
            for i in range(len(inputs)):
                if (inputs[i] < 0):
                    inputs[i] = (np.exp(inputs[i])-1) * self.alpha

            return inp_copy, inputs

        return None

    def der(self, inputs):
        inputs = np.array(inputs)
        shape = inputs.shape
        inp_copy = np.copy(inputs)
        if (len(shape) == 2):
            rl = []
            for l in inputs:
                tl = []
                for e in l:
                    if(e < 0):
                        e = e + self.alpha
                    else:
                        e = 1
                    tl.append(e)
                rl.append(tl)
            assert inp_copy.shape==np.shape(rl)
            return inp_copy, np.array(rl)
        elif (len(shape)==1):
            for i in range(len(inputs)):
                if(inputs[i] < 0):
                    inputs[i] = inputs[i] + self.alpha
                else:
                    inputs[i] = 1
            return inp_copy, inputs

        return None



class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self.name = "Softmax"

    def __call__(self, inputs):
        inputs = np.array(inputs, dtype=float)
        outputs = np.exp(inputs) / np.reshape(np.sum(np.exp(inputs), axis=1), (-1, 1))
        assert inputs.shape==outputs.shape
        return inputs, outputs


