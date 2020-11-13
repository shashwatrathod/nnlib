import numpy as np
from initializers import initializers


class Dense:
    def __init__(self, units, input_dim=0, activation=None, kernel_initializer=initializers.XavierNormal(),
                 bias_initializer=initializers.XavierNormal()):
        self.units = units
        self.input_dim = input_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        # Activation

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim

    def set_activation(self, activation):
        self.activation = activation

    def set_kernel_initializer(self, initializer):
        self.kernel_initializer = initializer

    def set_bias_initializer(self, initializer):
        self.bias_initializer = initializer