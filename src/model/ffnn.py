import numpy as np
from src.model.value import Value
from src.model.matrix import Matrix
from src.model.nn import MLP
from src.func.loss import LossFunction 


class FFNN:
    """ Feedforward neural network Model with autograd. """

    def __init__(self, X, y, loss='mse', is_multi_func = False, active_funcs = None):
        """
        Args:
            Xs (np.ndarray): Data input. Array of vectors.
            y (np.ndarray): Target output. Array of value.
            loss (str, optional): Loss function to use. enum: ['mse', TODO ... ]
            is_multi_func (bool, optional): Does each layer have different activation function.
            active_funcs (list, optional): List of activation functions for each layer. enum: ['linier' 'relu', 'sigmoid', 'tanh', 'softmax', 'exp', 'log']

        """
        self.X = Matrix(X)                              # Input data / Data Layer (batch ~ in form of a matrix)
        self.y = y                                      # Target output
        self.MLP = MLP()                                # Hidden Layers
        self.loss = self.__getLossFunction(loss)        # Loss function
        

    def __getLossFunction(self, loss: str) -> callable[[list, list[Value]], Value]:
        """ Get the corresponding loss function from the string. """
        
        loss_functions = {
            "mse": LossFunction.mse
            # "xxx": xxx
            # "xxx": xxx
            }
        
        if loss not in loss_functions:
            raise ValueError(f"Loss function '{loss}' is not supported. Choose from {list(loss_functions.keys())}.")
        
        return loss_functions[loss]
    
    def __getActivationFunction(self, func_name: str) -> callable[[Value], Value]:
        """ Get the corresponding activation function from the string. """
        
        activation_functions = {
            "linier": Value.linier,
            "relu": Value.relu,
            "sigmoid": Value.sigmoid,
            "tanh": Value.tanh,
            "softmax": Value.softmax,
            "exp": Value.exp,
            "log": Value.log
        }

        if func_name not in activation_functions:
            raise ValueError(f"Activation function '{func_name}' is not supported. "
                             f"Choose from {list(activation_functions.keys())}.")
        
        return activation_functions[func_name]
