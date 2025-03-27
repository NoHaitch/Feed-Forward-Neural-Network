import numpy as np
# import sys
from src.model.value import Value
from src.model.matrix import Matrix
from src.model.nn import MLP
from src.func.loss import LossFunction 
from tqdm import tqdm


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


        # self.X = Matrix(X)                              # Input data / Data Layer (batch ~ in form of a matrix)
        # self.y = Matrix([[int(val)] for val in y])      # Target output
        self.X = X
        # self.y = [[int(val)] for val in y]
        self.y = y.astype(int)
        self.loss = self.__getLossFunction(loss)        # Loss function
        
        nin = self.X.shape[1]
        nout = hidden_layer + [1] # add  output layer which only has one neuron

        if type(active) != list:
            active_funcs = [active for _ in range(len(nout) - 1)] + ["linier"]
        else:
            active_funcs = active

        assert active_funcs[-1] == "linier", "Output Layer must use Linier Activation Function."

        self.NN = MLP(nin, nout, active_funcs, weight)                                

    def __repr__(self):
        X_row, X_col = self.X.shape
        return f"Fully Connected Feed Forward Neural Network\n> X = {str(X_row)} x {str(X_col)}\n> y = {str(len(self.y))} x {str(1)}\n> {self.NN}"

    def __getLossFunction(self, loss: str) -> callable:
        """ Get the corresponding loss function from the string. """
        loss_functions = {
            "mse": LossFunction.mse
        }
        
        if loss not in loss_functions:
            raise ValueError(f"Loss function '{loss}' is not supported. Choose from {list(loss_functions.keys())}.")
        
        return loss_functions[loss]
    
    def forward(self, batch_X : np.ndarray, batch_y : np.ndarray) -> Value:
        """ Forward Pass of the neural network using loss function. """
        return self.loss(Matrix(batch_y).transpose(), self.NN(Matrix(batch_X)))
    
    def backpropagation(self, forward_loss : Value):
        """ Backward Pass of the Neural Network using loss Value """
        forward_loss.backward()

    def training(self, batch_size : int = 100, learning_rate : float = 0.01, max_epoch : int = 10, verbose : int = 0) -> Value:
        """ Train the model based on parameters """
        if (verbose == 1):
            pbar = tqdm(total=max_epoch)
        for epoch in range (max_epoch):
            print("Starting Epoch " + str(epoch))
            batches_X = [self.X[i:i+batch_size] for i in range(0, len(self.X), batch_size)]
            batches_y = [self.y[i:i+batch_size] for i in range(0, len(self.y), batch_size)]
            for batch_X, batch_y in zip(batches_X, batches_y):
                loss = self.forward(batch_X, batch_y)
                self.zero_grad()
                self.backpropagation(loss)
            for p in self.parameters():     # Update Parameters
                p.data -= learning_rate * p.grad
            if (verbose == 1):
                pbar.update(1)
                print("Epoch " + str(epoch) + " Done training")
                print("Training Loss " + str(loss.data))
        if (verbose == 1):
            pbar.close()
        return loss

    def parameters(self):
        return self.NN.parameters()

    def zero_grad(self):
        self.NN.zero_grad()
