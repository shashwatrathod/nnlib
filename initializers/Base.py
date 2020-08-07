from abc import ABC,abstractmethod


class Initializer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, units, dim_input=1):
        pass