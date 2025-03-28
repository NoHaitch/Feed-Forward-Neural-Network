import numpy as np

# import sys
from src.model.value import Value
from src.model.matrix import Matrix
from src.model.nn import MLP
from src.func.loss import LossFunction
from src.graph.visualize import Visualizer
from tqdm import tqdm


class FFNN:
    """Feedforward neural network Model with autograd.

    Attributes:
        X (Matrix): Batch input.
        y (list): Target.
        loss (callable): Loss function.
        NN (MLP): Neural network of the model.
    """

    def __init__(self, X, y, layers: list[int] = [8,8], loss:str="mse", active:str|list[str]="relu", seed:int = 42, weight:str="zero", **kwargs):
        """
        Args:
            Xs (np.ndarray): Data input. Array of vectors.
            y (np.ndarray): Target output. Array of value.
            layer (list[int]): The amount of neuron and amount of layer in all layer.
            weight (str, optional): Weight initialization mode. enum: ['zero', 'uniform', 'normal']
            loss (str, optional): Loss function to use. enum: ['mse', 'bce', 'cce' ]
            active_funcs (str|list[str], optional): List of activation functions for each layer. enum: ['linier' 'relu', 'sigmoid', 'tanh', 'softmax', 'exp', 'log']
        """
        valid_weights = {'zero', 'uniform', 'normal'}
        valid_losses = {'mse', 'bce', 'cce'}  
        valid_activations = {'linier', 'relu', 'sigmoid', 'tanh', 'softmax', 'exp', 'log'}

        assert weight in valid_weights, f"Weight initialization mode '{weight}' not recognized. Choose from {valid_weights}."
        assert loss in valid_losses, f"Loss function '{loss}' not recognized. Choose from {valid_losses}."

        assert X.shape[1] == layers[0], f"Input Layer must include same amount of neuron as input features"
        assert len(layers)-1 == len(active), f"Every layer's activation function must be specified"

        if isinstance(active, str):
            assert (
                active in valid_activations
            ), f"Activation function '{active}' not recognized. Choose from {valid_activations}."
        elif isinstance(active, list):
            assert all(
                func in valid_activations for func in active
            ), f"Some activation functions in {active} are not recognized. Choose from {valid_activations}."

        self.X = X  # Input data / Data Layer (batch ~ in form of a matrix)
        self.y = y.astype(int)  # Target output
        self.loss = self.__getLossFunction(loss)  # Loss function
        self.seed = seed

        nin = self.X.shape[1]
        nout = layers[1:]

        # if type(active) != list:
        #     active_funcs = [active for _ in range(len(nout) - 1)] + ["sigmoid"]
        # else:
        active_funcs = active

        # assert active_funcs[-1] == "linier", "Output Layer must use Linier Activation Function."

        self.NN = MLP(nin, nout, active_funcs, weight, seed, **kwargs)

    def __repr__(self):
        X_row, X_col = self.X.shape
        return f"Fully Connected Feed Forward Neural Network\n> X = {str(X_row)} x {str(X_col)}\n> y = {str(len(self.y))} x {str(1)}\n> {self.NN}"

    def __getLossFunction(self, loss: str) -> callable:
        """ Get the corresponding loss function from the string. """
        loss_functions = {
            "mse": LossFunction.mse,
            "bce": LossFunction.bce,
            "cce": LossFunction.cce
        }
        
        if loss not in loss_functions:
            raise ValueError(
                f"Loss function '{loss}' is not supported. Choose from {list(loss_functions.keys())}."
            )

        return loss_functions[loss]

    def forward(self, batch_X: np.ndarray, batch_y: np.ndarray) -> Value:
        """Forward Pass of the neural network using loss function."""
        return self.loss(Matrix(batch_y), self.NN(Matrix(batch_X)))

    def backpropagation(self, forward_loss: Value):
        """Backward Pass of the Neural Network using loss Value"""
        forward_loss.backward()

    def training(
        self,
        batch_size: int = 100,
        learning_rate: float = 0.01,
        max_epoch: int = 10,
        verbose: int = 0,
        split_point: float = 0.8,
    ) -> Value:
        """Train the model based on training hyperparameters"""
        if verbose == 1:
            pbar = tqdm(total=max_epoch)
        np.random.seed(self.seed)
        indices = np.arange(len(self.X))  # Split Validation Set
        np.random.shuffle(indices)
        split = int(split_point * len(self.X))
        X_train, X_val = self.X[indices[:split]], self.X[indices[split:]]
        y_train, y_val = self.y[indices[:split]], self.y[indices[split:]]
        for epoch in range(max_epoch):
            print("Starting Epoch " + str(epoch))
            batches_X = [
                X_train[i : i + batch_size] for i in range(0, len(X_train), batch_size)
            ]
            batches_y = [
                y_train[i : i + batch_size] for i in range(0, len(y_train), batch_size)
            ]
            for batch_X, batch_y in zip(batches_X, batches_y):
                loss = self.forward(batch_X, batch_y)
                self.zero_grad()
                self.backpropagation(loss)
            for p in self.parameters():  # Update Parameters
                p.data -= learning_rate * p.grad
            if verbose == 1:
                validation_loss = self.forward(
                    X_val, y_val
                )  # Feed Forward for Validation
                print("Training Loss " + str(loss.data))
                print("Validation Loss " + str(validation_loss.data))
                pbar.update(1)
                print()
        if verbose == 1:
            pbar.close()
        return loss

    def parameters(self):
        return self.NN.parameters()

    def zero_grad(self):
        self.NN.zero_grad()

    def visualize_model(self):
        """Visualize the network architecture."""
        return Visualizer.draw_ffnn(self)

    def visualize_computation_graph(self, output):
        """Visualize the computation graph for a given output."""
        return Visualizer.draw_dot(output)

    def plot_weight_distribution(self, layers):
        """
        Plot weight distribution for specified layers.

        Args:
            layers (list[int]): List of layer indices to plot.
        """
        Visualizer.plot_weight_distribution(self.NN, layers)

    def plot_gradient_distribution(self, layer_indices):
        """
        Plots the distribution of gradients of weights in specified layers of the FFNN.

        Args:
            layer_indices (list[int]): List of layer indices to plot.

        Raises:
            ValueError: If a layer index does not exist.
        """
        Visualizer.plot_gradient_distribution(self.NN, layer_indices)