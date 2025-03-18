import numpy as np
from src.model.value import Value
from src.model.matrix import Matrix
from src.model.nn import MLP
from src.func.loss import LossFunction 


class FFNN:
    """ Feedforward neural network Model with autograd. 
    
        Attributes:
            X (Matrix): Batch input.
            y (list): Target.
            loss (callable): Loss function.
            NN (MLP): Neural network of the model.
    """

    def __init__(self, X, y, hidden_layer: list[int] = [8,8], weight:str="zero", loss:str="mse", active:str|list[str]="relu"):
        """
        Args:
            Xs (np.ndarray): Data input. Array of vectors.
            y (np.ndarray): Target output. Array of value.
            hidden_layer (list[int]): The amount of neuron and amount of layer in the hidden layer.
            weight (str, optional): Weight initialization mode. enum: ['zero', 'uniform', 'normal']
            loss (str, optional): Loss function to use. enum: ['mse', TODO ... ]
            active_funcs (str|list[str], optional): List of activation functions for each layer. enum: ['linier' 'relu', 'sigmoid', 'tanh', 'softmax', 'exp', 'log']
        """
        valid_weights = {'zero', 'uniform', 'normal'}
        valid_losses = {'mse'}  
        valid_activations = {'linier', 'relu', 'sigmoid', 'tanh', 'softmax', 'exp', 'log'}

        assert weight in valid_weights, f"Weight initialization mode '{weight}' not recognized. Choose from {valid_weights}."
        assert loss in valid_losses, f"Loss function '{loss}' not recognized. Choose from {valid_losses}."

        if isinstance(active, str):  
            assert active in valid_activations, f"Activation function '{active}' not recognized. Choose from {valid_activations}."
        elif isinstance(active, list):  
            assert all(func in valid_activations for func in active), f"Some activation functions in {active} are not recognized. Choose from {valid_activations}."


        self.X = Matrix(X)                              # Input data / Data Layer (batch ~ in form of a matrix)
        self.y = y                                      # Target output
        self.loss = self.__getLossFunction(loss)        # Loss function
        
        nin = self.X.cols
        nout = hidden_layer + [1] # add  output layer which only has one neuron

        if type(active) != list:
            active_funcs = [active for _ in range(len(nout) - 1)] + ["linier"]
        else:
            active_funcs = active

        assert active_funcs[-1] == "linier", "Output Layer must use Linier Activation Function."

        self.NN = MLP(nin, nout, active_funcs, weight)                                

    def __repr__(self):
        return f"Fully Connected Feed Forward Neural Network\n> X = {self.X.rows} x {self.X.cols}\n> y = 1 x {len(self.y)}\n> {self.NN}"

    def __getLossFunction(self, loss: str) -> callable:
        """ Get the corresponding loss function from the string. """
        loss_functions = {
            "mse": LossFunction.mse
        }
        
        if loss not in loss_functions:
            raise ValueError(f"Loss function '{loss}' is not supported. Choose from {list(loss_functions.keys())}.")
        
        return loss_functions[loss]
    
    def train():
        pass
