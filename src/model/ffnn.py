import numpy as np
from src.model.value import Value
from src.model.nn import MLP
from src.func.loss import LossFunction 
from src.utils.converter import Converter
from typing import Callable


class FFNN:
    """ Feedforward neural network Model with autograd. """

    def __init__(self, Xs: np.ndarray, y: np.ndarray , loss='mse', is_multi_func = False, active_funcs = None):
        """
        Args:
            Xs (np.ndarray): Data input. Array of vectors.
            y (np.ndarray): Target output. Array of value.
            loss (str, optional): Loss function to use. enum: ['mse', TODO ... ]
            is_multi_func (bool, optional): Does each layer have different activation function.
            active_funcs (list, optional): List of activation functions for each layer. enum: ['linier' 'relu', 'sigmoid', 'tanh', 'softmax', 'exp', 'log']

        """
        self.Xs = Xs                                 # Input data / Data Layer (batch ~ in form of a matrix)
        self.y = y                                   # Target output
        self.MLP = MLP()                             # Hidden Layers
        self.loss = self.__getLossFunction(loss)     # Loss function
        

    def __getLossFunction(self, loss: str) -> Callable[[list, list[Value]], Value]:
        """ Get the corresponding loss function from the string. """
        
        loss_functions = {
            "mse": LossFunction.mse
            # "xxx": xxx
            # "xxx": xxx
            }
        
        if loss not in loss_functions:
            raise ValueError(f"Loss function '{loss}' is not supported. Choose from {list(loss_functions.keys())}.")
        
        return loss_functions[loss]
        
