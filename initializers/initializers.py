import numpy as np
from Base import Initializer

class Gaussian(Initializer):
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    def __call__(self, units, dim_input=1):
            
        return np.random.normal(self.mean, self.std_dev, size=(dim_input, units))


class Random(Initializer):
    def __init__(self):
        pass

    def __call__(self, units, dim_input=1):
        
        return np.random.randn(dim_input, units)


class Zeros(Initializer):
    def __init__(self):
        pass

    
    def __call__(self, units, dim_input=1):
        
        return np.zeros((dim_input, units))

class Ones(Initializer):
    def __init__(self):
        pass

    def __call__(self, units, dim_input=1):

        return np.ones((dim_input, units))
