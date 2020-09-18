import numpy as np
from abc import ABC, abstractmethod


# base class
class Initializer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, units, dim_input=1):
        pass


class RandomNormal(Initializer):
    '''
    Generates RandomNormal distribution of weights of shape (current_layer_units, previous_layer_units).

    Attributes:
        mean (float): represents the mean around which the distribution will be centered
        std_dev (float): represents the standard deviation of the gaussian distribution
        scale (boolean): If true, scales the weights by a factor of SQRT(2/(number of units in previous layer))
    '''

    def __init__(self, mean, std_dev, scale=True):
        '''
        The constructor for RandomNormal Initializer

        :param mean (float): represents the mean around which the distribution will be centered
        :param std_dev (float): represents the standard deviation of the gaussian distribution
        :param scale (boolean): If true, scales the weights by a factor of SQRT(2/(number of units in previous layer))
        '''

        super().__init__()
        self.mean = mean
        self.std_dev = std_dev
        self.scale = scale

    def __call__(self, units, dim_input=1):
        '''
        The function to generate the distribution.


        :param units (int): Number of units in the current layer
        :param dim_input (int): Number of units in the previous layer. Default value 1 for biases
        :return: np.ndarray: shape=(units, dim_input)

        '''

        if (not self.scale):
            return np.random.normal(self.mean, self.std_dev, size=(units, dim_input))
        else:
            return np.random.normal(self.mean, self.std_dev, size=(units, dim_input)) * np.sqrt(2 / dim_input)


class Random(Initializer):
    '''
    Creates a (current_layer_units, previous_layer_units) matrix initialized with random values.

    Parameters:
        scale (boolean): If true, scales the weights by a factor of SQRT(2/(number of units in previous layer))
    '''

    def __init__(self, scale=True):
        '''
        Constructor for Random Initializer

        :param scale: If true, scales the weights by a factor of SQRT(2/(number of units in previous layer))
        '''
        super().__init__()
        self.scale = scale

    def __call__(self, units, dim_input=1):
        '''
        Creates the (units, dim_input) matrix.

        :param units: number of nodes in the current layer
        :param dim_input: number of nodes in the previous layer
        :return: a (units, dim_input) shaped matrix initialized with random values
        '''

        if (not self.scale):
            return np.random.randn(units, dim_input)
        else:
            return np.random.randn(units, dim_input) * np.sqrt(2 / dim_input)


class RandomUniform(Initializer):
    '''
    Creates a (current_layer_nodes, previous_layer_nodes) shaped matrix initialized with random values with uniform distribution in the range [low, high)

    :param low: Denotes the lower bound of the uniform distribution
    :param high: Denotes the higher bound of the uniform distribution
    :param scale: If true, scales the weights by a factor of SQRT(2/(number of units in previous layer))
    '''

    def __init__(self, low, high, scale=True):
        '''
        Constructor for RandomUniform.

        :param low: Denotes the lower bound of the uniform distribution
        :param high: Denotes the higher bound of the uniform distribution
        :param scale: If true, scales the weights by a factor of SQRT(2/(number of units in previous layer))
        '''

        super().__init__()
        self.low = low
        self.high = high
        self.scale = scale

    def __call__(self, units, dim_input=1):
        '''
        :param units: number of nodes in the current layer
        :param dim_input: number of nodes in the previous layer
        :returns : (units, dim_input) shaped matrix.

        '''

        if (not self.scale):
            return np.random.uniform(self.low, self.high, size=(units, dim_input))
        else:
            return np.random.uniform(self.low, self.high, size=(units, dim_input)) * np.sqrt(2 / dim_input)


class XavierUniform(Initializer):
    """
    Initializes the weights uniformly in the interval [-x,x) where x = sqrt(6 / (units + dim_input)).
    Shape = (units, dim_input)
    """

    def __init__(self):
        """
        Constructor for XavierUniform
        """
        super().__init__()

    def __call__(self, units, dim_input=1):
        """
        :param units: number of nodes in the current layer
        :param dim_input: number of nodes in the previous layer
        :return: (units, dim_input) shaped matrix of uniformly distributed weights.
        """
        x = np.sqrt(6 / (units + dim_input))
        return np.random.uniform(-x, x, size=(units, dim_input))


class XavierNormal(Initializer):
    """
    Initializes the weights normally with mean 0 and standard_deviation = sqrt(2/(units+dim_input)).
    Shape: (units, dim_input)
    """

    def __init__(self):
        """
        Constructor for XavierNormal
        """

        super().__init__()

    def __call__(self, units, dim_input=1):
        """
        :param units: number of nodes in the current layer
        :param dim_input: number of nodes in the previous layer
        :return: (units, dim_input) shaped matrix of normally distributed weights.
        """
        std_dev = np.sqrt(2/(units+dim_input))
        return np.random.normal(0, std_dev, size=(units, dim_input))



class Zeros(Initializer):
    """
    Creates a (units, dim_input) shape matrix initialiezed with zeros.

    """

    def __init__(self):
        super().__init__()

    def __call__(self, units, dim_input=1):
        return np.zeros((units, dim_input))


class Ones(Initializer):
    '''
    Creates a (units, dim_input) shape matrix initialized with ones.

    '''

    def __init__(self):
        super().__init__()

    def __call__(self, units, dim_input=1):
        return np.ones((units, dim_input))
