import numpy as np
import json
import os
from src.model.value import Value
from src.model.matrix import Matrix
from src.model.nn import MLP
from src.func.loss import LossFunction
from src.func.regularization import RegularizationFunctions
from src.graph.visualize import Visualizer
from tqdm import tqdm


class FFNN:
    """Feedforward neural network Model with autograd."""

    def __init__(
        self,
        layers: list[int] = [8, 8],
        loss: str = "mse",
        regularization = "none",
        active: str | list[str] = "relu",
        seed: int = 42,
        weight: str = "zero",
        **kwargs,
    ):
        """
        Initializes the FFNN model **without requiring input data (X, y)**.

        Args:
            layers (list[int]): Number of neurons in each layer.
            weight (str, optional): Weight initialization mode. enum: ['zero', 'uniform', 'normal']
            loss (str, optional): Loss function. enum: ['mse', 'bce', 'cce']
            active (str|list[str], optional): List of activation functions per layer.
        """
        valid_weights = {"zero", "uniform", "normal"}
        valid_losses = {"mse", "bce", "cce"}
        valid_regularization = {"none", "l1", "l2"}
        valid_activations = {
            "linier",
            "relu",
            "sigmoid",
            "tanh",
            "softmax",
            "exp",
            "log",
        }

        assert (
            weight in valid_weights
        ), f"Invalid weight mode: {weight}. Choose from {valid_weights}."
        assert (
            loss in valid_losses
        ), f"Invalid loss function: {loss}. Choose from {valid_losses}."
        assert (
            regularization in valid_regularization
        ), f"Invalid regularization: {regularization}. Choose from {valid_regularization}."
        if isinstance(active, str):
            assert (
                active in valid_activations
            ), f"Invalid activation: {active}. Choose from {valid_activations}."
        elif isinstance(active, list):
            assert all(
                func in valid_activations for func in active
            ), f"Invalid activations: {active}."

        self.loss = self.__getLossFunction(loss)
        self.regularize = self.__getRegularizationFunction(regularization)
        self.seed = seed
        self.weight_mode = weight
        self.layers = layers
        self.active = (
            active if isinstance(active, list) else [active] * (len(layers) - 1)
        )

        # Initialize MLP without X, y
        self.NN = MLP(layers[0], layers[1:], self.active, weight, seed, **kwargs)

    def __repr__(self):
        return f"Feed Forward Neural Network\n> Layers: {self.layers}\n> {self.NN}"

    def __getLossFunction(self, loss: str) -> callable:
        """Returns the loss function."""
        loss_functions = {
            "mse": LossFunction.mse,
            "bce": LossFunction.bce,
            "cce": LossFunction.cce,
        }
        return loss_functions.get(loss, None)
    
    def __getRegularizationFunction(self, regularization: str) -> callable:
        """Returns the loss function."""
        regularization_functions = {
            "l1": RegularizationFunctions.l1,
            "l2": RegularizationFunctions.l2,
            "none": RegularizationFunctions.none,
        }
        return regularization_functions.get(regularization, None)

    def forward(self, batch_X: np.ndarray, batch_y: np.ndarray) -> Value:
        """Forward Pass of the neural network using loss function."""
        params = [v.data for v in self.parameters()]
        return self.loss(Matrix(batch_y), self.NN(Matrix(batch_X))) + self.regularize(params).data

    def backpropagation(self, forward_loss: Value):
        """Backward Pass of the Neural Network using loss Value."""
        forward_loss.backward()

    def training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 100,
        learning_rate: float = 0.01,
        max_epoch: int = 10,
        verbose: int = 0,
        split_point: float = 0.8,
    ) -> Value:
        """
        Train the model on the given dataset.
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target output.
        """
        if verbose == 1:
            pbar = tqdm(total=max_epoch)
        np.random.seed(self.seed)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split = int(split_point * len(X))
        X_train, X_val = X[indices[:split]], X[indices[split:]]
        y_train, y_val = y[indices[:split]], y[indices[split:]]

        loss_history = {"train_loss": [], "val_loss": []}

        for epoch in range(max_epoch):
            print(f"Starting Epoch {epoch}")
            batches_X = [
                X_train[i : i + batch_size] for i in range(0, len(X_train), batch_size)
            ]
            batches_y = [
                y_train[i : i + batch_size] for i in range(0, len(y_train), batch_size)
            ]
            for batch_X, batch_y in zip(batches_X, batches_y):
                loss = self.forward(batch_X, batch_y)
                loss_history["train_loss"].append(loss.data)
                self.zero_grad()
                self.backpropagation(loss)
            for p in self.parameters():
                p.data -= learning_rate * p.grad
            if verbose == 1:
                validation_loss = self.forward(X_val, y_val)
                loss_history["val_loss"].append(validation_loss.data)
                print(f"Training Loss: {loss.data}")
                print(f"Validation Loss: {validation_loss.data}")
                pbar.update(1)
                print()
        if verbose == 1:
            pbar.close()
        return loss, loss_history

    def parameters(self):
        return self.NN.parameters()

    def zero_grad(self):
        self.NN.zero_grad()

    def visualize_model(self):
        return Visualizer.draw_ffnn(self, self.layers[0])

    def visualize_computation_graph(self, output):
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

    def save(self, file_path):
        """Saves model parameters (including gradients) to a JSON file."""

        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        model_data = {
            "layers": self.layers,
            "weight_mode": self.weight_mode,
            "seed": self.seed,
            "active": self.active,
            "parameters": [
                {
                    "neurons": [
                        {
                            "weights": [
                                {"data": w.data, "grad": w.grad} for w in neuron.w
                            ],
                            "bias": {"data": neuron.b.data, "grad": neuron.b.grad},
                        }
                        for neuron in layer.neurons
                    ]
                }
                for layer in self.NN.layers
            ],
        }
        with open(file_path + ".json", "w") as f:
            json.dump(model_data, f, indent=4)
        print(f"Model saved to {file_path}.json")

    @staticmethod
    def load(file_path):
        """Loads a saved FFNN model from a JSON file, including gradients."""
        with open(file_path + ".json", "r") as f:
            model_data = json.load(f)

        # Create a new FFNN instance
        ffnn = FFNN(
            layers=model_data["layers"],
            weight=model_data["weight_mode"],
            seed=model_data["seed"],
            active=model_data["active"],
        )

        # Restore weights, biases, and gradients
        for layer, saved_layer in zip(ffnn.NN.layers, model_data["parameters"]):
            for neuron, saved_neuron in zip(layer.neurons, saved_layer["neurons"]):
                for w, saved_w in zip(neuron.w, saved_neuron["weights"]):
                    w.data = saved_w["data"]
                    w.grad = saved_w.get("grad", 0)  # Default gradient to 0 if missing
                neuron.b.data = saved_neuron["bias"]["data"]
                neuron.b.grad = saved_neuron["bias"].get(
                    "grad", 0
                )  # Default gradient to 0

        print(f"Model loaded from {file_path}.json")
        return ffnn
