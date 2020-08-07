import numpy as np

class Dense:
    def __init__(self, units, input_dim = 0, activation = None, kernel_initializer = None, bias_initializer = None):
        self.units = units
        #Activation