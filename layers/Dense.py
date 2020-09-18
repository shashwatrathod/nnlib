import numpy as np
from initializers import initializers


class Dense:
    def __init__(self, units, input_dim=0, activation=None, kernel_initializer=initializers.RandomNormal(0, 0.35),
                 bias_initializer=initializers.RandomNormal(0, 0.35)):
        self.units = units
        self.input_dim = input_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        # Activation

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
